"""The tools the agent can call. Every query goes through Connector.run_query."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from analyst.catalog import Catalog, CatalogTable
from analyst.connectors import Connector, QueryError
from analyst.models import QueryResult
from analyst.sql_guard import UnsafeQueryError


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOL_DEFINITIONS = [
    _tool(
        "list_tables",
        "List every table with its row count, column count and description.",
        {},
        [],
    ),
    _tool(
        "describe_table",
        "Show a table's columns (type, null share, distinct count, min/max, sample values, "
        "description) and the joins it takes part in.",
        {"table": {"type": "string", "description": "Table name, as listed by list_tables."}},
        ["table"],
    ),
    _tool(
        "run_sql",
        "Run one read-only SELECT and see the result. Use it to explore values and to check "
        "your final query. Only the first rows are shown; the full row count is reported.",
        {"sql": {"type": "string", "description": "A single SELECT in the source's dialect."}},
        ["sql"],
    ),
    _tool(
        "submit_answer",
        "Finish: give the answer for the user and the one query that produces it. Call this "
        "exactly once, at the end. If the data cannot answer the question, set answerable to "
        "false, explain why in answer, and set sql to null.",
        {
            "answer": {"type": "string", "description": "Short answer in plain language."},
            "sql": {
                "type": ["string", "null"],
                "description": "The final SELECT whose result supports the answer.",
            },
            "answerable": {"type": "boolean"},
        },
        ["answer", "sql", "answerable"],
    ),
]


class ToolCall(BaseModel):
    """One tool call, kept for the audit trail and the eval report."""

    tool: str
    input: dict[str, Any]
    ok: bool
    error: str | None = None
    sql: str | None = None  # SQL as executed (after the guard)
    row_count: int | None = None


class AgentTools:
    def __init__(self, connector: Connector, catalog: Catalog, *, rows_shown: int = 50) -> None:
        self.connector = connector
        self.catalog = catalog
        self.rows_shown = rows_shown

    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, ToolCall]:
        """Run a tool. Returns (text for the model, record). Never raises on bad input."""
        handler = {
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "run_sql": self._run_sql,
        }.get(name)
        if handler is None:
            return self._error(name, tool_input, f"unknown tool {name!r}")
        try:
            return handler(tool_input)
        except (KeyError, TypeError) as e:
            return self._error(name, tool_input, f"bad input: {e}")

    def run_final(self, sql: str) -> QueryResult:
        return self.connector.run_query(sql)

    def _list_tables(self, tool_input: dict[str, Any]) -> tuple[str, ToolCall]:
        rows = [
            {
                "table": t.name,
                "rows": t.row_count,
                "columns": len(t.columns),
                **({"description": t.description} if t.description else {}),
            }
            for t in self.catalog.tables
        ]
        return _json(rows), ToolCall(tool="list_tables", input=tool_input, ok=True)

    def _describe_table(self, tool_input: dict[str, Any]) -> tuple[str, ToolCall]:
        name = tool_input["table"]
        try:
            table = self.catalog.table(name)
        except KeyError:
            known = ", ".join(t.name for t in self.catalog.tables)
            return self._error("describe_table", tool_input, f"no table {name!r}. Tables: {known}")
        joins = [
            f"{r.from_table}.{r.from_column} = {r.to_table}.{r.to_column} ({r.origin})"
            for r in self.catalog.relationships
            if table.name in (r.from_table, r.to_table)
        ]
        body = {
            "table": table.name,
            "rows": table.row_count,
            **({"description": table.description} if table.description else {}),
            "columns": [_describe_column(c) for c in table.columns],
            "joins": joins,
        }
        return _json(body), ToolCall(tool="describe_table", input=tool_input, ok=True)

    def _run_sql(self, tool_input: dict[str, Any]) -> tuple[str, ToolCall]:
        sql = tool_input["sql"]
        try:
            result = self.connector.run_query(sql)
        except (UnsafeQueryError, QueryError) as e:
            return self._error("run_sql", tool_input, str(e))
        shown = result.rows[: self.rows_shown]
        body = {
            "columns": result.columns,
            "rows": shown,
            "row_count": result.row_count,
            **(
                {"rows_hidden": result.row_count - len(shown)}
                if len(shown) < result.row_count
                else {}
            ),
            **({"truncated_at_row_limit": True} if result.truncated else {}),
        }
        record = ToolCall(
            tool="run_sql", input=tool_input, ok=True, sql=result.sql, row_count=result.row_count
        )
        return _json(body), record

    @staticmethod
    def _error(name: str, tool_input: dict[str, Any], message: str) -> tuple[str, ToolCall]:
        return f"Error: {message}", ToolCall(tool=name, input=tool_input, ok=False, error=message)


def _describe_column(col) -> dict[str, Any]:
    out: dict[str, Any] = {"name": col.name, "type": col.data_type}
    if col.primary_key:
        out["primary_key"] = True
    if col.description:
        out["description"] = col.description
    p = col.profile
    if p:
        out.update(
            {
                "null_share": p.null_fraction,
                "distinct": p.distinct_count,
                "min": p.min,
                "max": p.max,
                "samples": p.samples,
            }
        )
        if p.is_unique:
            out["unique"] = True
    return out


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def table_summary(table: CatalogTable) -> str:
    cols = ", ".join(f"{c.name} {c.data_type}" for c in table.columns)
    return f"{table.name} ({table.row_count} rows): {cols}"
