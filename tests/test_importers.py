"""Importing dataset-supplied column descriptions (BIRD's database_description CSVs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyst.agent import AgentTools
from analyst.catalog import CatalogService, CatalogStore
from analyst.catalog.importers import (
    describe,
    import_descriptions,
    read_bird_descriptions,
    without_imported,
)
from analyst.config import Settings
from analyst.connectors import SourceRegistry

HEADER = "original_column_name,column_name,column_description,data_format,value_description"


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings, only={"olist"})
    store = CatalogStore(settings.app_db_path)
    service = CatalogService(registry, store, settings.profiling)
    yield service, store, service.refresh("olist"), registry
    registry.close()


def test_reads_bom_windows_encoding_and_trailing_commas(tmp_path: Path) -> None:
    (tmp_path / "district.csv").write_bytes(
        ("﻿" + HEADER + ",\nA11,average salary,average salary,integer ,,\n").encode()
    )
    (tmp_path / "orders.csv").write_bytes(
        (
            HEADER + "\nstatus,,order status,text,\"commonsense evidence:\n'livré' = delivered\"\n"
        ).encode("cp1252")
    )
    assert read_bird_descriptions(tmp_path) == {
        ("district", "A11"): "average salary",
        ("orders", "status"): "order status; values: 'livré' = delivered",
    }


def test_describe_drops_repeats_and_useless_notes() -> None:
    row = {"column_name": "race ID", "column_description": "race id", "value_description": ""}
    assert describe("raceId", row) == "race ID"
    assert describe("A9", {"column_name": "", "value_description": "not useful"}) == ""
    assert describe("id", {"column_name": "id"}) == ""
    assert len(describe("x", {"column_description": "y" * 1000})) == 300


def test_import_matches_names_case_insensitively_and_keeps_user_text(olist_env) -> None:
    service, store, catalog, _ = olist_env
    service.describe_column("olist", "orders", "order_status", "My own words")
    report = import_descriptions(
        store,
        "olist",
        {
            ("ORDERS", "Order_Status"): "imported status",
            ("orders", "order_id"): "unique order id",
            ("orders", "no_such_column"): "x",
        },
    )
    assert (report.imported, report.kept_user, report.unmatched) == (
        1,
        1,
        ["orders.no_such_column"],
    )
    orders = store.load("olist").table("orders")
    assert orders.column("order_status").description == "My own words"
    assert orders.column("order_id").description_source == "import"


def test_imported_text_survives_a_refresh_and_reaches_the_agent(olist_env) -> None:
    service, store, catalog, registry = olist_env
    import_descriptions(store, "olist", {("orders", "order_id"): "unique order id"})
    refreshed = service.refresh("olist")
    assert refreshed.table("orders").column("order_id").description == "unique order id"

    tools = AgentTools(registry.get("olist"), refreshed)
    body = json.loads(tools.execute("describe_table", {"table": "orders"})[0])
    assert body["columns"][0]["description"] == "unique order id"

    bare = without_imported(refreshed)
    assert bare.table("orders").column("order_id").description is None
    assert refreshed.table("orders").column("order_id").description == "unique order id"
