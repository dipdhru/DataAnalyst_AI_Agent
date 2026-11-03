"""Validate SQL before it reaches a database: one read-only SELECT over known tables, capped rows.

This is the second line of defence. The first is the connection itself (read-only SQLite URI,
read-only Postgres transaction). File-reading functions of other dialects (DuckDB) stay blocked
in case such a source is added later.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class UnsafeQueryError(ValueError):
    """The SQL was rejected before execution."""


_ALLOWED_ROOTS = (exp.Select, exp.SetOperation)

# Looked up by name so the list survives sqlglot renames between versions.
_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "AlterTable",
        "TruncateTable",
        "Command",
        "Into",
        "Copy",
        "Set",
        "Pragma",
        "Attach",
        "Detach",
        "Grant",
        "Revoke",
        "Transaction",
        "Commit",
        "Rollback",
        "Use",
        "LoadData",
        "Analyze",
        "Install",
        "Export",
    )
    if hasattr(exp, name)
)

# Functions that read files, reach other databases, or stall the server.
_BLOCKED_FUNCTIONS = {
    # duckdb
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_ndjson_auto",
    "read_text",
    "read_blob",
    "read_xlsx",
    "glob",
    "sniff_csv",
    "sqlite_scan",
    "sqlite_attach",
    "postgres_scan",
    "postgres_attach",
    "query",
    "query_table",
    "getenv",
    # postgres
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "dblink",
    "dblink_exec",
    "pg_sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "set_config",
    "current_setting",
    # sqlite
    "load_extension",
    "readfile",
    "writefile",
}


def validate_select(
    sql: str,
    *,
    dialect: str,
    max_rows: int,
    allowed_tables: set[str] | None = None,
) -> str:
    """Return a safe, LIMIT-capped version of `sql` or raise UnsafeQueryError.

    `allowed_tables` holds lower-cased table names (qualified as `schema.table` outside the
    default schema). Pass None to skip the table check.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as e:
        raise UnsafeQueryError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeQueryError(f"expected exactly one statement, got {len(statements)}")
    stmt = statements[0]

    if not isinstance(stmt, _ALLOWED_ROOTS):
        raise UnsafeQueryError(f"only SELECT queries are allowed, got {stmt.key.upper()}")

    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeQueryError(f"{node.key.upper()} is not allowed")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in _BLOCKED_FUNCTIONS:
                raise UnsafeQueryError(f"function {name}() is not allowed")

    _check_tables(stmt, allowed_tables)
    return _cap_limit(stmt, max_rows).sql(dialect=dialect)


def _check_tables(stmt: exp.Expression, allowed_tables: set[str] | None) -> None:
    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise UnsafeQueryError("table functions are not allowed in FROM")
        if allowed_tables is None:
            continue
        name = table.name.lower()
        qualified = f"{table.db.lower()}.{name}" if table.db else name
        if not table.db and name in cte_names:
            continue
        if qualified not in allowed_tables:
            raise UnsafeQueryError(f"unknown table: {qualified}")


def _cap_limit(stmt: exp.Query, max_rows: int) -> exp.Query:
    existing = stmt.args.get("limit")
    if isinstance(existing, (exp.Limit, exp.Fetch)):
        value = (
            existing.expression if isinstance(existing, exp.Limit) else existing.args.get("count")
        )
        if isinstance(value, exp.Literal) and value.is_int and int(value.this) <= max_rows:
            return stmt
    return stmt.limit(max_rows, copy=False)
