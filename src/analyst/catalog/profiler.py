"""Column profiling with plain aggregate SQL, so it works on SQLite and Postgres alike."""

from __future__ import annotations

import logging
from typing import Any

from sqlglot import exp

from analyst.catalog.models import ColumnProfile
from analyst.config import ProfilingSettings
from analyst.connectors.base import Connector, QueryError
from analyst.models import ColumnInfo, TableInfo

log = logging.getLogger(__name__)

# Types where MIN/MAX (and for json, DISTINCT) are meaningless or unsupported in some backends.
_UNORDERED_MARKERS = (
    "json",
    "bool",
    "bytea",
    "blob",
    "binary",
    "array",
    "[]",
    "struct",
    "map",
    "geometry",
    "uuid",
    "xml",
)
_NO_DISTINCT_MARKERS = ("json", "xml", "geometry")
_MAX_SAMPLE_CHARS = 120


def profile_table(
    connector: Connector, table: TableInfo, settings: ProfilingSettings
) -> tuple[int, dict[str, ColumnProfile]]:
    """Return (row_count, {column: profile}). Large tables are profiled on their first rows."""
    dialect = connector.dialect
    table_sql = exp.table_(table.name, db=table.schema_name, quoted=True).sql(dialect)

    row_count = _scalar(connector, f"SELECT COUNT(*) FROM {table_sql}", settings) or 0
    sampled = row_count > settings.max_rows
    source = (
        f"(SELECT * FROM {table_sql} LIMIT {settings.max_rows}) AS _p" if sampled else table_sql
    )
    profiled_rows = min(row_count, settings.max_rows)

    stats = _aggregate_stats(connector, table.columns, source, settings)
    profiles: dict[str, ColumnProfile] = {}
    for col in table.columns:
        s = stats.get(col.name, {})
        non_null = s.get("non_null")
        null_count = profiled_rows - non_null if non_null is not None else None
        distinct = s.get("distinct")
        profiles[col.name] = ColumnProfile(
            row_count=profiled_rows,
            null_count=null_count,
            null_fraction=round(null_count / profiled_rows, 4)
            if null_count is not None and profiled_rows
            else None,
            distinct_count=distinct,
            min=_clip(s.get("min")),
            max=_clip(s.get("max")),
            samples=_samples(connector, col, source, settings),
            is_unique=bool(profiled_rows and null_count == 0 and distinct == profiled_rows),
            sampled=sampled,
        )
    return row_count, profiles


def _aggregate_stats(
    connector: Connector, columns: list[ColumnInfo], source: str, settings: ProfilingSettings
) -> dict[str, dict[str, Any]]:
    """One scan for all columns; on failure fall back to one query per column."""
    plan = [(col, _aggregates(col, connector.dialect)) for col in columns]
    select = ", ".join(sql for _, aggs in plan for _, sql in aggs)
    try:
        row = _row(connector, f"SELECT {select} FROM {source}", settings)
        values = iter(row)
        return {col.name: {key: next(values) for key, _ in aggs} for col, aggs in plan}
    except QueryError as e:
        log.info("batched profile failed on %s (%s); profiling column by column", source, e)
    stats: dict[str, dict[str, Any]] = {}
    for col, aggs in plan:
        stats[col.name] = {}
        for key, sql in aggs:
            try:
                stats[col.name][key] = _scalar(connector, f"SELECT {sql} FROM {source}", settings)
            except QueryError:
                stats[col.name][key] = None
    return stats


def _aggregates(col: ColumnInfo, dialect: str) -> list[tuple[str, str]]:
    c = exp.column(col.name, quoted=True).sql(dialect)
    data_type = col.data_type.lower()
    aggs = [("non_null", f"COUNT({c})")]
    if not any(m in data_type for m in _NO_DISTINCT_MARKERS):
        aggs.append(("distinct", f"COUNT(DISTINCT {c})"))
    if not any(m in data_type for m in _UNORDERED_MARKERS):
        aggs += [("min", f"MIN({c})"), ("max", f"MAX({c})")]
    return aggs


def _samples(
    connector: Connector, col: ColumnInfo, source: str, settings: ProfilingSettings
) -> list[Any]:
    if settings.sample_values <= 0 or any(m in col.data_type.lower() for m in _NO_DISTINCT_MARKERS):
        return []
    c = exp.column(col.name, quoted=True).sql(connector.dialect)
    try:
        result = connector.run_query(
            f"SELECT DISTINCT {c} FROM {source} WHERE {c} IS NOT NULL",
            limit=settings.sample_values,
            timeout_s=settings.timeout_seconds,
        )
    except QueryError:
        return []
    return [_clip(r[0]) for r in result.rows]


def _row(connector: Connector, sql: str, settings: ProfilingSettings) -> list[Any]:
    result = connector.run_query(sql, limit=1, timeout_s=settings.timeout_seconds)
    return result.rows[0]


def _scalar(connector: Connector, sql: str, settings: ProfilingSettings) -> Any:
    return _row(connector, sql, settings)[0]


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_SAMPLE_CHARS:
        return value[:_MAX_SAMPLE_CHARS] + "…"
    return value
