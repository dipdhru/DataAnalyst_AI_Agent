"""Model-written table and column descriptions, from names, types and profiles.

One call per table. Results are stored with description_source="llm" and only fill gaps: a
description written by the user or imported with the dataset is never replaced.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from analyst.catalog.models import Catalog, CatalogTable
from analyst.catalog.store import CatalogStore

_SYSTEM = """\
You write short, factual descriptions of database tables and columns for a data analyst.
Base them only on the names, types, sample values and statistics given; don't guess beyond
the data. The table description is one sentence on what one row represents, not its name.
A column description says what the column means and how its values look (units, codes,
formats, which table it points to). Don't repeat counts or null shares: the analyst sees
those already. At most 20 words each.
Reply with JSON only: {"table": "...", "columns": {"<column name>": "...", ...}}"""
ATTEMPTS = 2  # a reply that isn't valid JSON is asked for once more


class DescribeReport(BaseModel):
    tables: int = 0
    columns: int = 0
    failed: list[str] = Field(default_factory=list)  # tables the model's reply couldn't fill


def describe_catalog(
    client: Any,
    model: str,
    store: CatalogStore,
    source: str,
    *,
    only_tables: set[str] | None = None,
    on_table: Callable[[str], None] | None = None,
) -> DescribeReport:
    catalog = store.load(source)
    if catalog is None:
        raise KeyError(source)
    report = DescribeReport()
    for table in catalog.tables:
        if only_tables and table.name not in only_tables:
            continue
        missing = [c.name for c in table.columns if not c.description]
        if table.description and not missing:
            continue
        if on_table:
            on_table(table.name)
        for attempt in range(ATTEMPTS):
            try:
                written = _ask(client, model, table, catalog)
            except ValueError as e:
                if attempt == ATTEMPTS - 1:
                    report.failed.append(f"{table.name}: {e}")
                continue
            _store(store, source, table, written, report)
            break
    return report


def table_brief(table: CatalogTable, catalog: Catalog) -> str:
    """What the model sees about one table."""
    lines = [f"Table {table.name} ({table.row_count} rows)"]
    joins = [
        f"{r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column}"
        for r in catalog.relationships
        if table.name in (r.from_table, r.to_table)
    ]
    if joins:
        lines.append("Joins: " + "; ".join(joins))
    for c in table.columns:
        p = c.profile
        stats = ""
        if p:
            samples = ", ".join(str(s)[:40] for s in p.samples[:5])
            stats = (
                f" distinct={p.distinct_count} nulls={(p.null_fraction or 0):.0%}"
                f" min={str(p.min)[:30]} max={str(p.max)[:30]} samples=[{samples}]"
            )
        lines.append(f"- {c.name} {c.data_type}{' PK' if c.primary_key else ''}{stats}")
    return "\n".join(lines)


def _ask(client: Any, model: str, table: CatalogTable, catalog: Catalog) -> dict[str, Any]:
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": table_brief(table, catalog)}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content if b.type == "text")
    return parse_reply(text)


def parse_reply(text: str) -> dict[str, Any]:
    """The JSON object in a reply, with or without a code fence around it."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("reply had no JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"reply was not valid JSON ({e.msg})") from e
    if not isinstance(data, dict) or not isinstance(data.get("columns", {}), dict):
        raise ValueError("reply JSON had the wrong shape")
    return data


def _store(
    store: CatalogStore,
    source: str,
    table: CatalogTable,
    written: dict[str, Any],
    report: DescribeReport,
) -> None:
    text = _short(written.get("table"))
    if text and not table.description:
        store.set_table_description(source, table.name, text, origin="llm")
        report.tables += 1
    by_name = {k.lower(): v for k, v in written.get("columns", {}).items()}
    for col in table.columns:
        text = _short(by_name.get(col.name.lower()))
        if text and not col.description:
            store.set_column_description(source, table.name, col.name, text, origin="llm")
            report.columns += 1


def _short(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.split())[:300]
