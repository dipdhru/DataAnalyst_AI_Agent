import json
from pathlib import Path

import pytest
from fake_claude import FakeClaude, reply, text, tool_use

from analyst.agent import Agent
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import Settings
from analyst.connectors import SourceRegistry
from analyst.semantic import load_semantics

ROOT = Path(__file__).resolve().parents[1]
COUNT_SQL = "SELECT COUNT(*) AS orders FROM orders"


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings)
    catalog = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    built = catalog.refresh("olist")
    semantics = load_semantics(ROOT / "metrics.yaml")["olist"]
    yield registry.get("olist"), built, semantics, settings
    registry.close()


def make_agent(env, fake: FakeClaude, **agent_overrides) -> Agent:
    connector, catalog, semantics, settings = env
    agent_settings = settings.agent.model_copy(update=agent_overrides)
    return Agent(fake, connector, catalog, agent_settings, semantics)


def test_explore_query_and_answer(olist_env) -> None:
    fake = FakeClaude(
        reply(tool_use("list_tables")),
        reply(tool_use("describe_table", table="orders")),
        reply(tool_use("run_sql", sql=COUNT_SQL)),
        reply(
            text("There are 2000."),
            tool_use("submit_answer", answer="2,000 orders.", sql=COUNT_SQL, answerable=True),
        ),
    )
    answer = make_agent(olist_env, fake).ask("How many orders are there?")

    assert answer.status == "answered" and answer.answerable
    assert answer.result.rows == [[2000]] and "LIMIT" in answer.sql
    assert [s.tool for s in answer.steps] == [
        "list_tables",
        "describe_table",
        "run_sql",
        "submit_answer",
    ]
    assert answer.turns == 4 and answer.usage.input_tokens == 4000

    tables = json.loads(fake.tool_results(1)[0]["content"])
    assert {t["table"] for t in tables} >= {"orders", "customers"}
    described = json.loads(fake.tool_results(2)[0]["content"])
    assert "orders.customer_id = customers.customer_id (inferred)" in described["joins"]
    ran = json.loads(fake.tool_results(3)[0]["content"])
    assert ran["rows"] == [[2000]]


def test_request_shape(olist_env) -> None:
    fake = FakeClaude(reply(tool_use("submit_answer", answer="x", sql=COUNT_SQL, answerable=True)))
    make_agent(olist_env, fake).ask("q")
    req = fake.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "high"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tool_choice" not in req  # forced tool use is a 400 on Opus 5.5
    assert {t["name"] for t in req["tools"]} == {
        "list_tables",
        "describe_table",
        "run_sql",
        "submit_answer",
    }
    assert all(
        t["strict"] and t["input_schema"]["additionalProperties"] is False for t in req["tools"]
    )
    system = req["system"][0]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert "customer_unique_id" in system["text"]  # definitions are in the prompt
    assert "- orders (2000 rows)" in system["text"]


def test_no_fallback_for_models_without_it(olist_env) -> None:
    fake = FakeClaude(reply(tool_use("submit_answer", answer="x", sql=COUNT_SQL, answerable=True)))
    make_agent(olist_env, fake, model="claude-haiku-5-5").ask("q")
    assert "fallbacks" not in fake.requests[0] and "betas" not in fake.requests[0]


def test_unsafe_sql_is_returned_as_a_tool_error(olist_env) -> None:
    fake = FakeClaude(
        reply(tool_use("run_sql", sql="DELETE FROM orders")),
        reply(tool_use("submit_answer", answer="x", sql=COUNT_SQL, answerable=True)),
    )
    answer = make_agent(olist_env, fake).ask("q")
    [result] = fake.tool_results(1)
    assert result["is_error"] and "only SELECT" in result["content"]
    assert answer.status == "answered" and not answer.steps[0].ok


def test_failing_final_sql_is_fed_back(olist_env) -> None:
    fake = FakeClaude(
        reply(
            tool_use("submit_answer", answer="x", sql="SELECT nope FROM orders", answerable=True)
        ),
        reply(tool_use("submit_answer", answer="x", sql=COUNT_SQL, answerable=True)),
    )
    answer = make_agent(olist_env, fake).ask("q")
    [result] = fake.tool_results(1)
    assert result["is_error"] and "final SQL failed" in result["content"]
    assert answer.status == "answered" and answer.turns == 2


def test_unanswerable(olist_env) -> None:
    fake = FakeClaude(
        reply(tool_use("submit_answer", answer="No cost data.", sql=None, answerable=False))
    )
    answer = make_agent(olist_env, fake).ask("What is our profit margin?")
    assert answer.status == "unanswerable" and answer.result is None and answer.sql is None


def test_answerable_without_sql_is_rejected(olist_env) -> None:
    fake = FakeClaude(
        reply(tool_use("submit_answer", answer="2000", sql=None, answerable=True)),
        reply(tool_use("submit_answer", answer="2000", sql=COUNT_SQL, answerable=True)),
    )
    answer = make_agent(olist_env, fake).ask("q")
    assert "sql is empty" in fake.tool_results(1)[0]["content"]
    assert answer.status == "answered"


def test_plain_text_reply_gets_one_nudge(olist_env) -> None:
    fake = FakeClaude(
        reply(text("It's 2000.")),
        reply(tool_use("submit_answer", answer="2000", sql=COUNT_SQL, answerable=True)),
    )
    answer = make_agent(olist_env, fake).ask("q")
    assert "submit_answer" in fake.requests[1]["messages"][-1]["content"]
    assert answer.status == "answered"

    stubborn = FakeClaude(reply(text("2000")), reply(text("Still 2000")))
    answer = make_agent(olist_env, stubborn).ask("q")
    assert answer.status == "error" and answer.answer == "Still 2000"


def test_gives_up_after_max_steps(olist_env) -> None:
    fake = FakeClaude(*[reply(tool_use("list_tables")) for _ in range(3)])
    answer = make_agent(olist_env, fake, max_steps=3).ask("q")
    assert answer.status == "max_steps" and answer.turns == 3


def test_refusal_stops_the_run(olist_env) -> None:
    fake = FakeClaude(reply(text(""), stop="refusal"))
    answer = make_agent(olist_env, fake).ask("q")
    assert answer.status == "refusal" and "declined" in answer.detail


def test_history_is_append_only(olist_env) -> None:
    fake = FakeClaude(
        reply(tool_use("list_tables")),
        reply(tool_use("run_sql", sql=COUNT_SQL)),
        reply(tool_use("submit_answer", answer="x", sql=COUNT_SQL, answerable=True)),
    )
    make_agent(olist_env, fake).ask("q")
    first, second, third = (r["messages"] for r in fake.requests)
    assert second[: len(first)] == first and third[: len(second)] == second


def test_ask_endpoint(settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from analyst.api.app import create_app

    fake = FakeClaude(
        reply(tool_use("submit_answer", answer="2,000", sql=COUNT_SQL, answerable=True))
    )
    with TestClient(create_app(settings, llm_client=fake)) as client:
        early = client.post("/sources/olist/ask", json={"question": "How many orders?"})
        assert early.status_code == 409
        client.post("/sources/olist/catalog/refresh")
        r = client.post("/sources/olist/ask", json={"question": "How many orders?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered" and body["result"]["rows"] == [[2000]]
