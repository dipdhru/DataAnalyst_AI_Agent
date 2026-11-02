import pytest

from analyst.sql_guard import UnsafeQueryError, validate_select

TABLES = {"orders", "customers", "main.orders"}


def ok(sql: str, dialect: str = "sqlite", max_rows: int = 100) -> str:
    return validate_select(sql, dialect=dialect, max_rows=max_rows, allowed_tables=TABLES)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "SELECT o.order_id FROM orders o JOIN customers c USING (customer_id)",
        "WITH recent AS (SELECT * FROM orders) SELECT COUNT(*) FROM recent",
        "SELECT customer_id FROM orders UNION SELECT customer_id FROM customers",
        "SELECT * FROM main.orders",
        "SELECT * FROM orders WHERE order_status LIKE '%deliv%'",
    ],
)
def test_accepts_read_only_selects(sql: str) -> None:
    assert ok(sql).upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize(
    ("sql", "dialect", "reason"),
    [
        ("DELETE FROM orders", "sqlite", "only SELECT"),
        ("UPDATE orders SET order_status = 'x'", "sqlite", "only SELECT"),
        ("DROP TABLE orders", "sqlite", "only SELECT"),
        ("INSERT INTO orders VALUES (1)", "sqlite", "only SELECT"),
        ("SELECT 1; DROP TABLE orders", "sqlite", "exactly one statement"),
        ("PRAGMA table_info(orders)", "sqlite", "only SELECT"),
        ("ATTACH 'other.db' AS o", "duckdb", "only SELECT"),
        ("COPY orders TO 'out.csv'", "duckdb", "only SELECT"),
        ("SELECT * INTO backup FROM orders", "postgres", "INTO"),
        ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "postgres", "DELETE"),
        ("SELECT * FROM read_csv('/etc/passwd')", "duckdb", "read_csv"),
        ("SELECT * FROM '/etc/passwd'", "duckdb", "unknown table"),
        ("SELECT pg_read_file('/etc/passwd')", "postgres", "pg_read_file"),
        ("SELECT pg_sleep(100)", "postgres", "pg_sleep"),
        ("SELECT * FROM sqlite_master", "sqlite", "unknown table"),
        ("SELECT * FROM secrets", "sqlite", "unknown table"),
        ("SELECT FROM WHERE (", "sqlite", "parse"),
    ],
)
def test_rejects_unsafe_sql(sql: str, dialect: str, reason: str) -> None:
    with pytest.raises(UnsafeQueryError, match=reason):
        ok(sql, dialect)


def test_injects_limit_when_missing() -> None:
    assert ok("SELECT * FROM orders").endswith("LIMIT 100")


def test_keeps_smaller_limit_and_caps_larger_one() -> None:
    assert ok("SELECT * FROM orders LIMIT 5").endswith("LIMIT 5")
    assert ok("SELECT * FROM orders LIMIT 50000").endswith("LIMIT 100")
    assert ok("SELECT * FROM orders FETCH FIRST 3 ROWS ONLY", "postgres").endswith("3 ROWS ONLY")


def test_cte_names_are_not_mistaken_for_unknown_tables() -> None:
    sql = "WITH secrets AS (SELECT * FROM orders) SELECT * FROM secrets"
    assert "LIMIT 100" in ok(sql)
