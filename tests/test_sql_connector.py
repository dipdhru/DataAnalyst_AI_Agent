from pathlib import Path

import pytest

from analyst.config import QueryLimits
from analyst.connectors import QueryError, QueryTimeoutError, SqlAlchemyConnector
from analyst.sql_guard import UnsafeQueryError

OLIST_TABLES = {
    "customers",
    "geolocation",
    "leads_closed",
    "leads_qualified",
    "order_items",
    "order_payments",
    "order_reviews",
    "orders",
    "product_category_name_translation",
    "products",
    "sellers",
}


@pytest.fixture
def connector(sample_sqlite: Path):
    url = f"sqlite:///file:{sample_sqlite}?mode=ro&uri=true"
    c = SqlAlchemyConnector(
        "olist", url, limits=QueryLimits(default_row_limit=50, max_row_limit=200)
    )
    yield c
    c.close()


def test_schema_lists_all_olist_tables(connector: SqlAlchemyConnector) -> None:
    schema = connector.get_schema()
    assert schema.dialect == "sqlite"
    assert {t.name for t in schema.tables} == OLIST_TABLES
    orders = [c.name for c in schema.table("orders").columns]
    assert orders[:3] == ["order_id", "customer_id", "order_status"]


def test_run_query_returns_rows_and_executed_sql(connector: SqlAlchemyConnector) -> None:
    result = connector.run_query(
        "SELECT order_status, COUNT(*) AS n FROM orders GROUP BY order_status ORDER BY n DESC"
    )
    assert result.columns == ["order_status", "n"]
    assert result.rows[0][0] == "delivered"
    assert sum(r[1] for r in result.rows) == 2000
    assert "LIMIT" in result.sql
    assert not result.truncated


def test_row_limit_truncates_and_is_capped(connector: SqlAlchemyConnector) -> None:
    default = connector.run_query("SELECT * FROM orders")
    assert default.row_count == 50 and default.truncated
    capped = connector.run_query("SELECT * FROM orders", limit=10_000)
    assert capped.row_count == 200 and capped.truncated


def test_connection_is_read_only_even_without_the_guard(connector: SqlAlchemyConnector) -> None:
    with pytest.raises(QueryError, match="readonly"):
        connector._execute("DELETE FROM orders", timeout_s=5)


def test_unsafe_sql_never_reaches_the_database(connector: SqlAlchemyConnector) -> None:
    with pytest.raises(UnsafeQueryError):
        connector.run_query("DROP TABLE orders")


def test_long_query_times_out(connector: SqlAlchemyConnector) -> None:
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c"
    with pytest.raises(QueryTimeoutError):
        connector.run_query(endless, timeout_s=0.3)
    # The connection is still usable afterwards.
    assert connector.run_query("SELECT COUNT(*) FROM orders").rows == [[2000]]


def test_database_errors_are_wrapped(connector: SqlAlchemyConnector) -> None:
    with pytest.raises(QueryError, match="no such column"):
        connector.run_query("SELECT not_a_column FROM orders")
