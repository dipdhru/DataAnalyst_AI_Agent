from pathlib import Path

import pandas as pd
import pytest

from analyst.config import QueryLimits
from analyst.connectors import FileConnector, QueryError
from analyst.connectors.files import TableExistsError, UnsupportedFileError, sanitize_identifier
from analyst.sql_guard import UnsafeQueryError


@pytest.fixture
def files(tmp_path: Path):
    c = FileConnector("uploads", tmp_path / "uploads.duckdb", limits=QueryLimits())
    yield c
    c.close()


def test_each_olist_csv_becomes_a_table(files: FileConnector, sample_csv_dir: Path) -> None:
    for csv in sorted(sample_csv_dir.glob("*.csv")):
        files.ingest(csv)
    names = {t.name for t in files.get_schema().tables}
    assert len(names) == 9
    assert {"olist_orders_dataset", "product_category_name_translation"} <= names
    count = files.run_query("SELECT COUNT(*) FROM olist_orders_dataset").rows[0][0]
    assert count == 2000


def test_bom_header_and_leading_zero_zip_codes(files: FileConnector, olist: Path) -> None:
    # The translation CSV starts with a UTF-8 BOM; zip prefixes like '01037' must stay text.
    [translation] = files.ingest(olist / "csv" / "product_category_name_translation.csv")
    assert translation.columns == ["product_category_name", "product_category_name_english"]
    files.ingest(olist / "csv" / "olist_sellers_dataset.csv")
    zips = files.get_schema().table("olist_sellers_dataset").columns[1]
    assert zips.name == "seller_zip_code_prefix" and zips.data_type == "VARCHAR"
    zero = files.run_query(
        "SELECT seller_zip_code_prefix FROM olist_sellers_dataset "
        "WHERE seller_zip_code_prefix LIKE '0%' LIMIT 1"
    )
    assert zero.rows[0][0].startswith("0")


def test_table_name_override_and_replace(files: FileConnector, sample_csv_dir: Path) -> None:
    csv = sample_csv_dir / "olist_sellers_dataset.csv"
    files.ingest(csv, table_name="Sellers 2018")
    assert "sellers_2018" in {t.name for t in files.get_schema().tables}
    with pytest.raises(TableExistsError):
        files.ingest(csv, table_name="Sellers 2018")
    [result] = files.ingest(csv, table_name="Sellers 2018", replace=True)
    assert result.row_count == 2000


def test_excel_sheets_become_tables_with_clean_columns(
    files: FileConnector, tmp_path: Path
) -> None:
    path = tmp_path / "Q3 Sales.xlsx"
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame({"Order ID": [1, 2], "Amount (BRL)": [10.5, 20.0]}).to_excel(
            xl, sheet_name="July", index=False
        )
        pd.DataFrame({"Order ID": [3], "Mixed": ["a"], "Notes": [None]}).to_excel(
            xl, sheet_name="Aug", index=False
        )
    results = {r.table: r for r in files.ingest(path)}
    assert set(results) == {"q3_sales_july", "q3_sales_aug"}
    assert results["q3_sales_july"].columns == ["order_id", "amount_brl"]
    assert results["q3_sales_july"].renamed_columns == {
        "Order ID": "order_id",
        "Amount (BRL)": "amount_brl",
    }
    total = files.run_query("SELECT SUM(amount_brl) FROM q3_sales_july").rows[0][0]
    assert total == pytest.approx(30.5)


def test_rejects_unsupported_files(files: FileConnector, tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(UnsupportedFileError):
        files.ingest(path)


def test_queries_cannot_read_the_filesystem(files: FileConnector, olist: Path) -> None:
    target = olist / "csv" / "olist_sellers_dataset.csv"
    with pytest.raises(UnsafeQueryError):
        files.run_query(f"SELECT * FROM read_csv('{target}')")
    # Even if the guard were bypassed, DuckDB itself refuses.
    with pytest.raises(QueryError, match="Permission"):
        files._execute(f"SELECT * FROM read_csv('{target}')", timeout_s=5)


def test_drop_table(files: FileConnector, sample_csv_dir: Path) -> None:
    files.ingest(sample_csv_dir / "olist_sellers_dataset.csv")
    files.drop_table("olist_sellers_dataset")
    assert files.get_schema().tables == []
    with pytest.raises(KeyError):
        files.drop_table("olist_sellers_dataset")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Order Date (UTC)", "order_date_utc"),
        ("﻿name", "name"),
        ("2024 sales", "t_2024_sales"),
        ("---", "col"),
        ("customer_id", "customer_id"),
    ],
)
def test_sanitize_identifier(raw: str, expected: str) -> None:
    assert sanitize_identifier(raw) == expected
