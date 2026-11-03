import pytest
from fastapi.testclient import TestClient

from analyst.api.app import create_app
from analyst.config import Settings


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_sources(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    sources = {s["name"]: s for s in client.get("/sources").json()}
    assert sources["olist"]["dialect"] == "sqlite"
    assert list(sources) == ["olist"]


def test_schema_and_unknown_source(client: TestClient) -> None:
    tables = client.get("/sources/olist/schema").json()["tables"]
    assert len(tables) == 11
    assert client.get("/sources/nope/schema").status_code == 404


def test_query_endpoint(client: TestClient) -> None:
    r = client.post("/sources/olist/query", json={"sql": "SELECT COUNT(*) AS n FROM orders"})
    assert r.status_code == 200
    assert r.json()["rows"] == [[2000]]

    rejected = client.post("/sources/olist/query", json={"sql": "DELETE FROM orders"})
    assert rejected.status_code == 400 and "rejected" in rejected.json()["detail"]

    broken = client.post("/sources/olist/query", json={"sql": "SELECT nope FROM orders"})
    assert broken.status_code == 400 and "query failed" in broken.json()["detail"]


def test_catalog_lifecycle(client: TestClient) -> None:
    assert client.get("/sources/olist/catalog").status_code == 404
    built = client.post("/sources/olist/catalog/refresh").json()
    assert len(built["tables"]) == 11 and built["relationships"]

    r = client.patch(
        "/sources/olist/catalog/tables/orders", json={"description": "One row per order."}
    )
    assert r.status_code == 200 and r.json()["description_source"] == "user"
    r = client.patch(
        "/sources/olist/catalog/tables/orders/columns/order_status",
        json={"description": "Lifecycle state."},
    )
    assert r.json()["description"] == "Lifecycle state."
    assert (
        client.patch("/sources/olist/catalog/tables/nope", json={"description": "x"}).status_code
        == 404
    )

    catalog = client.get("/sources/olist/catalog").json()
    orders = next(t for t in catalog["tables"] if t["name"] == "orders")
    assert orders["description"] == "One row per order."
