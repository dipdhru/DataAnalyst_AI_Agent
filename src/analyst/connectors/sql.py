"""Live databases through SQLAlchemy: SQLite (Olist test data) and Postgres."""

from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import create_engine, event, inspect

from analyst.config import QueryLimits
from analyst.connectors.base import Connector, QueryError, QueryTimeoutError
from analyst.models import ColumnInfo, ForeignKey, SchemaInfo, TableInfo

_SQLGLOT_DIALECTS = {"postgresql": "postgres", "sqlite": "sqlite"}


class SqlAlchemyConnector(Connector):
    kind = "database"

    def __init__(
        self,
        name: str,
        url: str,
        *,
        limits: QueryLimits,
        schemas: list[str] | None = None,
    ) -> None:
        super().__init__(name, limits=limits)
        self.engine = create_engine(url, pool_pre_ping=True)
        backend = self.engine.dialect.name
        if backend not in _SQLGLOT_DIALECTS:
            raise ValueError(f"unsupported database backend: {backend}")
        self.dialect = _SQLGLOT_DIALECTS[backend]
        if self.dialect == "postgres":
            self.default_schema = "public"
            self.schemas = schemas or ["public"]
            # Belt and braces: the configured user should already be read-only.
            event.listen(self.engine, "connect", _make_session_read_only)
        else:
            self.default_schema = "main"
            self.schemas = [None]

    def _introspect(self) -> SchemaInfo:
        insp = inspect(self.engine)
        tables: list[TableInfo] = []
        for schema in self.schemas:
            schema_name = None if schema in (None, self.default_schema) else schema
            names = sorted(insp.get_table_names(schema=schema) + insp.get_view_names(schema=schema))
            for table in names:
                pk = set(
                    insp.get_pk_constraint(table, schema=schema).get("constrained_columns") or []
                )
                columns = [
                    ColumnInfo(
                        name=c["name"],
                        data_type=str(c["type"]),
                        nullable=bool(c.get("nullable", True)),
                        primary_key=c["name"] in pk,
                    )
                    for c in insp.get_columns(table, schema=schema)
                ]
                fks = [
                    ForeignKey(
                        columns=fk["constrained_columns"],
                        referred_table=_qualify(
                            fk.get("referred_schema"), fk["referred_table"], self.default_schema
                        ),
                        referred_columns=fk["referred_columns"],
                    )
                    for fk in insp.get_foreign_keys(table, schema=schema)
                ]
                tables.append(
                    TableInfo(
                        name=table, schema_name=schema_name, columns=columns, foreign_keys=fks
                    )
                )
        return SchemaInfo(source=self.name, dialect=self.dialect, tables=tables)

    def _execute(self, sql: str, timeout_s: float) -> tuple[list[str], list[tuple[Any, ...]]]:
        raw = self.engine.raw_connection()
        timer: threading.Timer | None = None
        try:
            cursor = raw.cursor()
            if self.dialect == "postgres":
                cursor.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
            else:
                timer = threading.Timer(timeout_s, raw.driver_connection.interrupt)
                timer.start()
            # Raw DB-API cursor, no parameters: '%' and ':name' in the SQL stay literal.
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchall()
            return columns, rows
        except Exception as e:
            if _is_timeout(e):
                raise QueryTimeoutError(f"query exceeded {timeout_s:g}s timeout") from e
            raise QueryError(str(e).strip()) from e
        finally:
            if timer:
                timer.cancel()
            try:
                raw.rollback()
            finally:
                raw.close()

    def close(self) -> None:
        self.engine.dispose()


def _make_session_read_only(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    cursor.close()
    dbapi_conn.commit()


def _qualify(schema: str | None, table: str, default_schema: str | None) -> str:
    return table if schema in (None, default_schema) else f"{schema}.{table}"


def _is_timeout(e: Exception) -> bool:
    name = type(e).__name__
    return name == "QueryCanceled" or "interrupted" in str(e).lower()
