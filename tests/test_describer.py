"""Model-written catalog descriptions: fill gaps only, never replace user or imported text."""

from __future__ import annotations

import json

import pytest
from fake_claude import FakeClaude, reply, text

from analyst.catalog import CatalogService, CatalogStore
from analyst.catalog.describer import describe_catalog, parse_reply, table_brief
from analyst.config import Settings
from analyst.connectors import SourceRegistry


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings, only={"olist"})
    store = CatalogStore(settings.app_db_path)
    service = CatalogService(registry, store, settings.profiling)
    service.refresh("olist")
    yield service, store
    registry.close()


def written(table: str, **columns: str) -> object:
    body = {"table": f"All {table}.", "columns": columns}
    return reply(text(f"Here you go:\n```json\n{json.dumps(body)}\n```"))


def test_parse_reply_finds_fenced_json_and_rejects_junk() -> None:
    assert parse_reply('```json\n{"table": "t", "columns": {}}\n```')["table"] == "t"
    with pytest.raises(ValueError, match="no JSON"):
        parse_reply("Sorry, I can't.")
    with pytest.raises(ValueError, match="wrong shape"):
        parse_reply('{"columns": ["a"]}')


def test_fills_gaps_and_keeps_existing_descriptions(olist_env) -> None:
    service, store = olist_env
    service.describe_column("olist", "orders", "order_status", "My words")
    fake = FakeClaude(written("orders", order_id="Order key.", order_status="Model words."))
    report = describe_catalog(fake, "m", store, "olist", only_tables={"orders"})

    orders = store.load("olist").table("orders")
    assert orders.description == "All orders." and orders.description_source == "llm"
    assert orders.column("order_id").description_source == "llm"
    assert orders.column("order_status").description == "My words"
    assert (report.tables, report.columns, report.failed) == (1, 1, [])
    # The model saw the profile, not just the names.
    assert "samples=[" in fake.requests[0]["messages"][0]["content"]


def test_a_bad_reply_skips_that_table(olist_env) -> None:
    _, store = olist_env
    fake = FakeClaude(reply(text("no idea")), reply(text("still no idea")))
    report = describe_catalog(fake, "m", store, "olist", only_tables={"orders"})
    assert report.columns == 0 and report.failed[0].startswith("orders: reply had no JSON")
    assert len(fake.requests) == 2  # asked once more before giving up


def test_a_bad_reply_is_asked_again(olist_env) -> None:
    _, store = olist_env
    fake = FakeClaude(
        reply(text('{"table": "x", "columns": {')), written("orders", order_id="Key.")
    )
    report = describe_catalog(fake, "m", store, "olist", only_tables={"orders"})
    assert report.failed == [] and report.columns == 1


def test_fully_described_tables_cost_no_call(olist_env) -> None:
    service, store = olist_env
    sellers = store.load("olist").table("sellers")
    service.describe_table("olist", "sellers", "Sellers.")
    for col in sellers.columns:
        service.describe_column("olist", "sellers", col.name, "x")
    report = describe_catalog(FakeClaude(), "m", store, "olist", only_tables={"sellers"})
    assert report.tables == report.columns == 0


def test_brief_lists_joins(olist_env) -> None:
    _, store = olist_env
    catalog = store.load("olist")
    assert "Joins: " in table_brief(catalog.table("orders"), catalog)
