"""The one interface every data source implements: `get_schema` and `run_query`."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from analyst.config import QueryLimits
from analyst.models import QueryResult, SchemaInfo, to_jsonable
from analyst.sql_guard import validate_select


class QueryError(RuntimeError):
    """The database rejected or failed the query."""


class QueryTimeoutError(QueryError):
    """The query ran past its timeout and was cancelled."""


class Connector(ABC):
    kind: ClassVar[str]
    dialect: str  # sqlglot dialect name
    default_schema: str | None = None  # accepted as a qualifier, e.g. main.orders

    def __init__(self, name: str, *, limits: QueryLimits) -> None:
        self.name = name
        self.limits = limits
        self._schema: SchemaInfo | None = None
        self._schema_lock = threading.Lock()

    def get_schema(self, *, refresh: bool = False) -> SchemaInfo:
        with self._schema_lock:
            if refresh or self._schema is None:
                self._schema = self._introspect()
            return self._schema

    def invalidate_schema(self) -> None:
        with self._schema_lock:
            self._schema = None

    def allowed_tables(self) -> set[str]:
        names: set[str] = set()
        for table in self.get_schema().tables:
            names.add(table.qualified_name.lower())
            if table.schema_name is None and self.default_schema:
                names.add(f"{self.default_schema}.{table.name}".lower())
        return names

    def run_query(
        self, sql: str, *, limit: int | None = None, timeout_s: float | None = None
    ) -> QueryResult:
        """Validate, cap and run a read-only query. Raises UnsafeQueryError or QueryError."""
        limit = self.limits.clamp(limit)
        timeout_s = timeout_s or self.limits.timeout_seconds
        # Ask for one extra row so we can tell the caller the result was cut off.
        safe_sql = validate_select(
            sql, dialect=self.dialect, max_rows=limit + 1, allowed_tables=self.allowed_tables()
        )
        start = time.perf_counter()
        columns, rows = self._execute(safe_sql, timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        truncated = len(rows) > limit
        rows = rows[:limit]
        return QueryResult(
            sql=safe_sql,
            columns=columns,
            rows=[[to_jsonable(v) for v in row] for row in rows],
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round(elapsed_ms, 2),
        )

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    @abstractmethod
    def _introspect(self) -> SchemaInfo: ...

    @abstractmethod
    def _execute(self, sql: str, timeout_s: float) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run already-validated SQL. Implementations must enforce `timeout_s`."""
