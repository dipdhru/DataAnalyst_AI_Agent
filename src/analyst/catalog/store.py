"""Persist catalogs in the app's SQLite database (state/app.db)."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from analyst.catalog.models import (
    Catalog,
    CatalogColumn,
    CatalogTable,
    ColumnProfile,
    DescriptionSource,
    Relationship,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_sources (
    source TEXT PRIMARY KEY,
    dialect TEXT NOT NULL,
    refreshed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog_tables (
    source TEXT NOT NULL,
    table_name TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    row_count INTEGER,
    description TEXT,
    description_source TEXT,
    PRIMARY KEY (source, table_name)
);
CREATE TABLE IF NOT EXISTS catalog_columns (
    source TEXT NOT NULL,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    data_type TEXT NOT NULL,
    nullable INTEGER NOT NULL,
    primary_key INTEGER NOT NULL,
    profile TEXT,
    description TEXT,
    description_source TEXT,
    PRIMARY KEY (source, table_name, column_name)
);
CREATE TABLE IF NOT EXISTS catalog_relationships (
    source TEXT NOT NULL,
    from_table TEXT NOT NULL,
    from_column TEXT NOT NULL,
    to_table TEXT NOT NULL,
    to_column TEXT NOT NULL,
    origin TEXT NOT NULL,
    PRIMARY KEY (source, from_table, from_column, to_table, to_column)
);
"""


class CatalogStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            with con:  # commit on success, roll back on error
                yield con
        finally:
            con.close()

    def save(self, catalog: Catalog) -> None:
        """Replace the stored catalog, keeping descriptions for tables/columns that still exist."""
        src = catalog.source
        with self._connect() as con:
            table_desc = {
                r["table_name"]: (r["description"], r["description_source"])
                for r in con.execute("SELECT * FROM catalog_tables WHERE source = ?", (src,))
            }
            col_desc = {
                (r["table_name"], r["column_name"]): (r["description"], r["description_source"])
                for r in con.execute("SELECT * FROM catalog_columns WHERE source = ?", (src,))
            }
            for tbl in ("catalog_tables", "catalog_columns", "catalog_relationships"):
                con.execute(f"DELETE FROM {tbl} WHERE source = ?", (src,))
            con.execute(
                "INSERT OR REPLACE INTO catalog_sources VALUES (?, ?, ?)",
                (src, catalog.dialect, catalog.refreshed_at.isoformat()),
            )
            for t_ord, table in enumerate(catalog.tables):
                desc, desc_src = table_desc.get(
                    table.name, (table.description, table.description_source)
                )
                con.execute(
                    "INSERT INTO catalog_tables VALUES (?, ?, ?, ?, ?, ?)",
                    (src, table.name, t_ord, table.row_count, desc, desc_src),
                )
                for c_ord, col in enumerate(table.columns):
                    cdesc, cdesc_src = col_desc.get(
                        (table.name, col.name), (col.description, col.description_source)
                    )
                    con.execute(
                        "INSERT INTO catalog_columns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            src,
                            table.name,
                            col.name,
                            c_ord,
                            col.data_type,
                            int(col.nullable),
                            int(col.primary_key),
                            col.profile.model_dump_json() if col.profile else None,
                            cdesc,
                            cdesc_src,
                        ),
                    )
            con.executemany(
                "INSERT OR IGNORE INTO catalog_relationships VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (src, r.from_table, r.from_column, r.to_table, r.to_column, r.origin)
                    for r in catalog.relationships
                ],
            )

    def load(self, source: str) -> Catalog | None:
        with self._connect() as con:
            head = con.execute(
                "SELECT * FROM catalog_sources WHERE source = ?", (source,)
            ).fetchone()
            if head is None:
                return None
            columns: dict[str, list[CatalogColumn]] = {}
            for r in con.execute(
                "SELECT * FROM catalog_columns WHERE source = ? ORDER BY table_name, ordinal",
                (source,),
            ):
                columns.setdefault(r["table_name"], []).append(
                    CatalogColumn(
                        name=r["column_name"],
                        data_type=r["data_type"],
                        nullable=bool(r["nullable"]),
                        primary_key=bool(r["primary_key"]),
                        description=r["description"],
                        description_source=r["description_source"],
                        profile=ColumnProfile.model_validate(json.loads(r["profile"]))
                        if r["profile"]
                        else None,
                    )
                )
            tables = [
                CatalogTable(
                    name=r["table_name"],
                    row_count=r["row_count"],
                    description=r["description"],
                    description_source=r["description_source"],
                    columns=columns.get(r["table_name"], []),
                )
                for r in con.execute(
                    "SELECT * FROM catalog_tables WHERE source = ? ORDER BY ordinal", (source,)
                )
            ]
            relationships = [
                Relationship(
                    **{
                        k: r[k]
                        for k in ("from_table", "from_column", "to_table", "to_column", "origin")
                    }
                )
                for r in con.execute(
                    "SELECT * FROM catalog_relationships WHERE source = ? "
                    "ORDER BY from_table, from_column",
                    (source,),
                )
            ]
        return Catalog(
            source=source,
            dialect=head["dialect"],
            refreshed_at=dt.datetime.fromisoformat(head["refreshed_at"]),
            tables=tables,
            relationships=relationships,
        )

    def set_table_description(
        self,
        source: str,
        table: str,
        description: str | None,
        origin: DescriptionSource = "user",
    ) -> None:
        with self._connect() as con:
            cur = con.execute(
                "UPDATE catalog_tables SET description = ?, description_source = ? "
                "WHERE source = ? AND table_name = ?",
                (description, origin if description else None, source, table),
            )
            if cur.rowcount == 0:
                raise KeyError(f"{source}.{table}")

    def set_column_description(
        self,
        source: str,
        table: str,
        column: str,
        description: str | None,
        origin: DescriptionSource = "user",
    ) -> None:
        with self._connect() as con:
            cur = con.execute(
                "UPDATE catalog_columns SET description = ?, description_source = ? "
                "WHERE source = ? AND table_name = ? AND column_name = ?",
                (description, origin if description else None, source, table, column),
            )
            if cur.rowcount == 0:
                raise KeyError(f"{source}.{table}.{column}")
