from pathlib import Path

import pytest

from analyst.catalog import CatalogService, CatalogStore
from analyst.config import ProfilingSettings, Settings, SourceConfig
from analyst.connectors import SourceRegistry

# Joins a human would draw for Olist; none are declared in the SQLite file.
EXPECTED_JOINS = {
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    ("order_payments", "order_id", "orders", "order_id"),
    ("order_reviews", "order_id", "orders", "order_id"),
    (
        "products",
        "product_category_name",
        "product_category_name_translation",
        "product_category_name",
    ),
    ("leads_closed", "mql_id", "leads_qualified", "mql_id"),
    ("leads_closed", "seller_id", "sellers", "seller_id"),
}


def make_service(settings: Settings) -> CatalogService:
    registry = SourceRegistry(settings)
    return CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)


@pytest.fixture
def service(settings: Settings):
    svc = make_service(settings)
    yield svc
    svc.registry.close()


def joins(catalog) -> set[tuple[str, str, str, str]]:
    return {(r.from_table, r.from_column, r.to_table, r.to_column) for r in catalog.relationships}


def test_refresh_profiles_every_column(service: CatalogService) -> None:
    catalog = service.refresh("olist")
    assert len(catalog.tables) == 11
    orders = catalog.table("orders")
    assert orders.row_count == 2000

    order_id = orders.column("order_id").profile
    assert order_id.is_unique and order_id.null_count == 0 and order_id.distinct_count == 2000

    status = orders.column("order_status").profile
    assert "delivered" in status.samples and len(status.samples) <= 3
    assert not status.is_unique

    delivered = orders.column("order_delivered_customer_date").profile
    assert delivered.null_count > 0 and 0 < delivered.null_fraction < 1
    assert delivered.min < delivered.max

    price = catalog.table("order_items").column("price").profile
    assert isinstance(price.min, float) and price.min <= price.max


def test_infers_olist_joins(service: CatalogService) -> None:
    catalog = service.refresh("olist")
    assert joins(catalog) == EXPECTED_JOINS
    assert all(r.origin == "inferred" for r in catalog.relationships)


def test_user_descriptions_survive_refresh(service: CatalogService) -> None:
    service.refresh("olist")
    service.describe_table("olist", "orders", "One row per order placed on Olist.")
    col = service.describe_column("olist", "orders", "order_status", "Lifecycle state.")
    assert col.description_source == "user"

    catalog = service.refresh("olist")
    orders = catalog.table("orders")
    assert orders.description == "One row per order placed on Olist."
    assert orders.description_source == "user"
    assert orders.column("order_status").description == "Lifecycle state."

    service.describe_table("olist", "orders", None)
    assert service.get("olist").table("orders").description_source is None


def test_unknown_table_or_column_raises(service: CatalogService) -> None:
    service.refresh("olist")
    with pytest.raises(KeyError):
        service.describe_table("olist", "nope", "x")
    with pytest.raises(KeyError):
        service.describe_column("olist", "orders", "nope", "x")


def test_large_tables_are_profiled_on_a_sample(settings: Settings) -> None:
    settings.profiling = ProfilingSettings(sample_values=2, max_rows=500)
    svc = make_service(settings)
    try:
        orders = svc.refresh("olist").table("orders")
    finally:
        svc.registry.close()
    assert orders.row_count == 2000  # true count
    profile = orders.column("order_id").profile
    assert profile.sampled and profile.row_count == 500 and profile.distinct_count == 500


def test_catalog_persists_across_restarts(settings: Settings) -> None:
    first = make_service(settings)
    first.refresh("olist")
    first.registry.close()
    second = make_service(settings)
    try:
        catalog = second.get("olist")
    finally:
        second.registry.close()
    assert catalog is not None and len(catalog.tables) == 11
    assert joins(catalog) == EXPECTED_JOINS


@pytest.mark.full_olist
def test_full_olist_catalog(tmp_path: Path, olist: Path) -> None:
    settings = Settings(
        state_dir=tmp_path,
        sources=[SourceConfig(name="olist", type="sqlite", path=olist / "olist.sqlite")],
    )
    svc = make_service(settings)
    try:
        catalog = svc.refresh("olist")
    finally:
        svc.registry.close()
    counts = {t.name: t.row_count for t in catalog.tables}
    assert counts["orders"] == 99441
    assert counts["order_items"] == 112650
    assert counts["geolocation"] == 1000163
    assert catalog.table("geolocation").columns[0].profile.sampled  # > 1M rows
    assert joins(catalog) == EXPECTED_JOINS
