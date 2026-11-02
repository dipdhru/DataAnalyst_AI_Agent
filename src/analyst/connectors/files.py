"""Uploaded CSV/Excel/Parquet files, one DuckDB table per file (or per Excel sheet)."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel

from analyst.config import QueryLimits
from analyst.connectors.base import Connector, QueryError, QueryTimeoutError
from analyst.models import ColumnInfo, SchemaInfo, TableInfo

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
PARQUET_SUFFIXES = {".parquet"}
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES | PARQUET_SUFFIXES


class UnsupportedFileError(ValueError):
    pass


class TableExistsError(ValueError):
    pass


class IngestResult(BaseModel):
    table: str
    row_count: int
    columns: list[str]
    renamed_columns: dict[str, str]  # original -> stored, only where they differ
    sheet: str | None = None


class FileConnector(Connector):
    kind = "files"
    dialect = "duckdb"
    default_schema = "main"

    def __init__(self, name: str, db_path: Path, *, limits: QueryLimits) -> None:
        super().__init__(name, limits=limits)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Queries can never touch the filesystem (read_csv, COPY, ATTACH...). Files are read by
        # a separate in-memory staging connection and handed over as Arrow tables.
        self._con = duckdb.connect(str(db_path), config={"enable_external_access": False})
        self._write_lock = threading.Lock()

    def ingest(
        self,
        path: Path,
        *,
        table_name: str | None = None,
        replace: bool = False,
        original_filename: str | None = None,
    ) -> list[IngestResult]:
        """Load a file into one table (Excel: one table per sheet)."""
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedFileError(
                f"unsupported file type {suffix or '(none)'}; use {sorted(SUPPORTED_SUFFIXES)}"
            )
        base = sanitize_identifier(table_name or Path(original_filename or path.name).stem)
        frames = _read_file(path, suffix)

        planned: list[tuple[str, str | None, pa.Table]] = []
        for sheet, data in frames:
            name = (
                base if len(frames) == 1 or table_name else f"{base}_{sanitize_identifier(sheet)}"
            )
            planned.append((name, sheet if suffix in EXCEL_SUFFIXES else None, data))

        with self._write_lock:
            existing = {t.name for t in self._introspect().tables}
            clashes = [name for name, _, _ in planned if name in existing]
            if clashes and not replace:
                raise TableExistsError(f"table(s) already exist: {clashes}; pass replace=true")
            results = [self._store(name, sheet, data) for name, sheet, data in planned]
        self.invalidate_schema()
        return results

    def drop_table(self, table: str) -> None:
        with self._write_lock:
            if table not in {t.name for t in self._introspect().tables}:
                raise KeyError(table)
            self._con.execute(f"DROP TABLE {_quote(table)}")
        self.invalidate_schema()

    def _store(self, name: str, sheet: str | None, data: pa.Table) -> IngestResult:
        stored = _dedupe([sanitize_identifier(c) for c in data.column_names])
        renamed = {o: s for o, s in zip(data.column_names, stored, strict=True) if o != s}
        data = data.rename_columns(stored)
        self._con.register("__ingest_src", data)
        try:
            self._con.execute(
                f"CREATE OR REPLACE TABLE {_quote(name)} AS SELECT * FROM __ingest_src"
            )
        finally:
            self._con.unregister("__ingest_src")
        return IngestResult(
            table=name,
            row_count=data.num_rows,
            columns=stored,
            renamed_columns=renamed,
            sheet=sheet,
        )

    def _introspect(self) -> SchemaInfo:
        rows = (
            self._con.cursor()
            .execute(
                """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
            )
            .fetchall()
        )
        tables: dict[str, list[ColumnInfo]] = {}
        for table, column, data_type, nullable in rows:
            tables.setdefault(table, []).append(
                ColumnInfo(name=column, data_type=data_type, nullable=nullable == "YES")
            )
        return SchemaInfo(
            source=self.name,
            dialect=self.dialect,
            tables=[TableInfo(name=t, columns=cols) for t, cols in tables.items()],
        )

    def _execute(self, sql: str, timeout_s: float) -> tuple[list[str], list[tuple[Any, ...]]]:
        cursor = self._con.cursor()
        timer = threading.Timer(timeout_s, cursor.interrupt)
        timer.start()
        try:
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            return columns, cursor.fetchall()
        except duckdb.InterruptException as e:
            raise QueryTimeoutError(f"query exceeded {timeout_s:g}s timeout") from e
        except duckdb.Error as e:
            raise QueryError(str(e).strip()) from e
        finally:
            timer.cancel()
            cursor.close()

    def close(self) -> None:
        self._con.close()


def sanitize_identifier(raw: str) -> str:
    """'Order Date (UTC)' -> 'order_date_utc'. Always a valid unquoted SQL identifier."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw).strip().lstrip("﻿")).strip("_").lower()
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        seen.setdefault(n, 0)
        out.append(n)
    return out


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _read_file(path: Path, suffix: str) -> list[tuple[str, pa.Table]]:
    if suffix in EXCEL_SUFFIXES:
        sheets = pd.read_excel(path, sheet_name=None)
        return [
            (str(sheet), _frame_to_arrow(df))
            for sheet, df in sheets.items()
            if not df.dropna(how="all").empty
        ]
    staging = duckdb.connect()  # in-memory, external access allowed, discarded after
    try:
        reader = "read_parquet" if suffix in PARQUET_SUFFIXES else "read_csv"
        relation = staging.sql(f"SELECT * FROM {reader}(?)", params=[str(path)])
        # duckdb 2.x renamed arrow() to to_arrow_table().
        data = (getattr(relation, "to_arrow_table", None) or relation.arrow)()
    except duckdb.Error as e:
        raise UnsupportedFileError(f"could not read {path.name}: {e}") from e
    finally:
        staging.close()
    return [(path.stem, data)]


def _frame_to_arrow(df: pd.DataFrame) -> pa.Table:
    df = df.dropna(how="all").copy()
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        # Excel columns that mix numbers and text can't become one Arrow type: keep them as text.
        if df[col].dtype == object and pd.api.types.infer_dtype(df[col], skipna=True) not in {
            "string",
            "empty",
            "date",
            "datetime",
            "boolean",
        }:
            df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v))
    return pa.Table.from_pandas(df, preserve_index=False)
