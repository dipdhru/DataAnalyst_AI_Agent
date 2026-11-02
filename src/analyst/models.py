"""Shared data shapes returned by every connector."""

from __future__ import annotations

import datetime as dt
import decimal
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False


class ForeignKey(BaseModel):
    columns: list[str]
    referred_table: str  # qualified name, see TableInfo.qualified_name
    referred_columns: list[str]


class TableInfo(BaseModel):
    name: str
    schema_name: str | None = None  # None = default schema (main / public)
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.name}" if self.schema_name else self.name


class SchemaInfo(BaseModel):
    source: str
    dialect: str  # sqlglot dialect name
    tables: list[TableInfo]

    def table(self, name: str) -> TableInfo:
        for t in self.tables:
            if t.qualified_name.lower() == name.lower():
                return t
        raise KeyError(name)


class QueryResult(BaseModel):
    sql: str  # the SQL actually executed, after validation and limit injection
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool  # True when more rows existed than the row limit
    elapsed_ms: float


def to_jsonable(value: Any) -> Any:
    """Convert driver values (Decimal, datetime, bytes, ...) into JSON-safe ones."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)
