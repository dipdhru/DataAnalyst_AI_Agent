"""Import column descriptions that ship with a dataset (BIRD's database_description/*.csv).

Imported text is stored with `description_source="import"`: it never overwrites a user's
description, and a refresh keeps it like any other description.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from pydantic import BaseModel, Field

from analyst.catalog.models import Catalog
from analyst.catalog.store import CatalogStore

MAX_CHARS = 300  # descriptions travel in every describe_table result; keep them short


class ImportReport(BaseModel):
    imported: int = 0
    kept_user: int = 0  # the user had written their own description
    unmatched: list[str] = Field(default_factory=list)  # in the files, not in the database


def read_bird_descriptions(directory: Path) -> dict[tuple[str, str], str]:
    """(table, column) -> description, from one database's database_description folder.
    Table names come from file names; the original_column_name column names the column."""
    found: dict[tuple[str, str], str] = {}
    for path in sorted(directory.glob("*.csv")):
        for row in csv.DictReader(io.StringIO(_decode(path.read_bytes()))):
            # A trailing comma in a header or row leaves extra cells under the key None.
            row = {
                k.strip().lower(): (v or "").strip()
                for k, v in row.items()
                if isinstance(k, str) and not isinstance(v, list)
            }
            column = row.get("original_column_name", "")
            text = describe(column, row)
            if column and text:
                found[(path.stem, column)] = text
    return found


def describe(column: str, row: dict[str, str]) -> str:
    """One line from BIRD's fields: readable name, description, then value notes."""
    parts: list[str] = []
    for field in ("column_name", "column_description"):
        value = _clean(row.get(field, ""))
        if value and value.lower() not in {column.lower(), *(p.lower() for p in parts)}:
            parts.append(value)
    values = _clean(re.sub(r"(?i)commonsense evidence:", "", row.get("value_description", "")))
    if values and values.lower() not in {"", "not useful"}:
        parts.append(f"values: {values}")
    text = "; ".join(parts)
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1].rstrip() + "…"


def import_descriptions(
    store: CatalogStore, source: str, descriptions: dict[tuple[str, str], str]
) -> ImportReport:
    """Write descriptions into a built catalog, matching names without regard to case.
    Reads the stored catalog itself, so a description the user just wrote is never lost."""
    catalog = store.load(source)
    if catalog is None:
        raise KeyError(source)
    columns = {
        (t.name.lower(), c.name.lower()): (t.name, c) for t in catalog.tables for c in t.columns
    }
    report = ImportReport()
    for (table, column), text in sorted(descriptions.items()):
        match = columns.get((table.lower(), column.lower()))
        if match is None:
            report.unmatched.append(f"{table}.{column}")
            continue
        table_name, col = match
        if col.description_source == "user":
            report.kept_user += 1
            continue
        store.set_column_description(catalog.source, table_name, col.name, text, origin="import")
        report.imported += 1
    return report


def without_imported(catalog: Catalog) -> Catalog:
    """A copy with imported descriptions removed, to measure what they add."""
    copy = catalog.model_copy(deep=True)
    for table in copy.tables:
        for col in table.columns:
            if col.description_source == "import":
                col.description = col.description_source = None
    return copy


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")  # a few BIRD files are Windows-encoded


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.")
