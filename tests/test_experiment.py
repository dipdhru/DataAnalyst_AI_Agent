"""The controlled experiment's modes: A (direct), B (retrieval), C (retrieval + correction)."""

from pathlib import Path

import pytest
from fake_claude import FakeClaude, reply, tool_use

from analyst.agent import Agent
from analyst.agent.providers import QuestionFailedError
from analyst.agent.retrieval import SchemaRetriever
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import Settings
from analyst.connectors import SourceRegistry
from analyst.evals.runner import EvalResult, gold_tables, retrieval_recall, run_eval, summarize
from analyst.semantic import load_semantics

ROOT = Path(__file__).resolve().parents[1]
GOOD = "SELECT COUNT(*) AS orders FROM orders"
BAD = "SELECT COUNT(*) FROM orderz"
QUESTION = {
    "id": "q",
    "category": "orders",
    "question": "How many orders are there?",
    "sql": GOOD,
    "expected": {"columns": ["orders"], "rows": [[2000]]},
}


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings)
    service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    catalog = service.refresh("olist")
    semantics = load_semantics(ROOT / "metrics.yaml")["olist"]
    yield registry.get("olist"), catalog, semantics, settings
    registry.close()


def make_agent(env, fake: FakeClaude, mode: str, **overrides) -> Agent:
    connector, catalog, semantics, settings = env
    agent_settings = settings.agent.model_copy(update={"mode": mode, **overrides})
    return Agent(fake, connector, catalog, agent_settings, semantics)


def submit(sql: str):
    return reply(tool_use("submit_answer", answer="Counts orders.", sql=sql, answerable=True))


# --- retrieval -------------------------------------------------------------------------------


def test_revenue_question_retrieves_items_orders_and_definitions(olist_env) -> None:
    _, catalog, semantics, _ = olist_env
    got = SchemaRetriever(catalog, semantics).retrieve("What is our total revenue?")
    assert "revenue" in got.definitions
    # revenue's SQL names order_items; valid_order (if retrieved) names orders.
    assert "order_items" in got.tables
    assert "Business definitions" in got.text and "SUM(order_items.price)" in got.text


def test_retrieval_shows_only_joins_between_chosen_tables(olist_env) -> None:
    _, catalog, semantics, _ = olist_env
    got = SchemaRetriever(catalog, semantics, top_tables=2, top_definitions=0).retrieve(
        "review score of orders"
    )
    assert len(got.tables) <= 2
    for line in got.text.split("Joins:")[-1].splitlines():
        if " = " in line:
            left, right = line.lstrip("- ").split(" = ")
            assert left.split(".")[0] in got.tables and right.split(".")[0] in got.tables


def test_nothing_matched_falls_back_to_every_table(olist_env) -> None:
    _, catalog, semantics, _ = olist_env
    got = SchemaRetriever(catalog, semantics).retrieve("zzzz qqqq")
    assert len(got.tables) == len(catalog.tables)


# --- modes -----------------------------------------------------------------------------------


def test_direct_mode_offers_only_submit_and_the_bare_schema(olist_env) -> None:
    fake = FakeClaude(submit(GOOD))
    answer = make_agent(olist_env, fake, "direct").ask("How many orders are there?")

    assert answer.status == "answered" and answer.result.rows == [[2000]]
    request = fake.requests[0]
    assert [t["name"] for t in request["tools"]] == ["submit_answer"]
    system = request["system"][0]["text"]
    assert "order_purchase_timestamp" in system
    assert "Business definitions" not in system and "valid_order" not in system
    assert answer.retrieved_tables is None and answer.mode == "direct"


def test_retrieval_mode_shows_retrieved_schema_only(olist_env) -> None:
    fake = FakeClaude(submit(GOOD))
    answer = make_agent(olist_env, fake, "retrieval").ask("What is our total revenue?")

    system = fake.requests[0]["system"][0]["text"]
    assert "Business definitions" in system
    assert answer.retrieved_tables and "order_items" in answer.retrieved_tables
    _, catalog, _, _ = olist_env
    hidden = set(t.name for t in catalog.tables) - set(answer.retrieved_tables)
    assert hidden, "retrieval should leave some tables out"
    assert not any(f"- {t} (" in system for t in hidden)


@pytest.mark.parametrize("mode", ["direct", "retrieval"])
def test_a_and_b_get_no_correction(olist_env, mode: str) -> None:
    fake = FakeClaude(submit(BAD))  # a second request would fail the fake
    answer = make_agent(olist_env, fake, mode).ask("How many orders are there?")

    assert answer.status == "error" and "orderz" in (answer.detail or "")
    assert answer.turns == 1 and answer.corrections == 0
    assert len(fake.requests) == 1


def test_corrected_mode_feeds_the_error_back_and_recovers(olist_env) -> None:
    fake = FakeClaude(submit(BAD), submit(GOOD))
    answer = make_agent(olist_env, fake, "corrected").ask("How many orders are there?")

    assert answer.status == "answered" and answer.result.rows == [[2000]]
    assert answer.corrections == 1 and answer.turns == 2
    fed_back = fake.tool_results(1)[0]
    assert fed_back["is_error"] and "orderz" in fed_back["content"]


def test_corrected_mode_stops_after_max_corrections(olist_env) -> None:
    fake = FakeClaude(submit(BAD), submit(BAD), submit(BAD))  # 1 try + 2 corrections
    answer = make_agent(olist_env, fake, "corrected", max_corrections=2).ask("How many?")

    assert answer.status == "error" and answer.corrections == 2 and answer.turns == 3
    assert not fake.replies


def test_single_shot_modes_cannot_run_sql(olist_env) -> None:
    """A model calling a tool it was not offered gets an error, never a query result."""
    fake = FakeClaude(reply(tool_use("run_sql", sql=GOOD)), submit(GOOD))
    answer = make_agent(olist_env, fake, "direct").ask("How many orders are there?")

    result = fake.tool_results(1)[0]
    assert result["is_error"] and "not available" in result["content"]
    assert [s.tool for s in answer.steps][0] == "run_sql" and not answer.steps[0].ok


def test_agent_mode_is_unchanged(olist_env) -> None:
    fake = FakeClaude(submit(GOOD))
    make_agent(olist_env, fake, "agent").ask("How many orders are there?")
    names = [t["name"] for t in fake.requests[0]["tools"]]
    assert names == ["list_tables", "describe_table", "run_sql", "submit_answer"]


# --- scoring ---------------------------------------------------------------------------------


def test_gold_tables_ignores_ctes() -> None:
    sql = "WITH t AS (SELECT * FROM orders o JOIN order_items i USING (order_id)) SELECT * FROM t"
    assert gold_tables(sql) == {"orders", "order_items"}


def test_retrieval_recall_and_summary(olist_env) -> None:
    fake = FakeClaude(submit(BAD), submit(GOOD))
    results = run_eval(
        [QUESTION],
        lambda: make_agent(olist_env, fake, "corrected"),
        model="claude-opus-5-5",
    )
    r = results[0]
    assert r.passed and r.retrieval_recall in (0.0, 1.0)
    assert r.retrieval_recall == retrieval_recall(QUESTION, r.agent)

    s = summarize(results)
    assert s["accuracy"] == 1.0 and s["execution_rate"] == 1.0
    assert s["corrected_questions"] == 1 and s["correction_success"] == 1.0
    assert s["avg_input_tokens"] == 2 * (1000 + 800)


def test_provider_failure_keeps_retrieval_and_time(olist_env) -> None:
    """A question cut short by the provider still counts in retrieval recall and timing."""

    class Failing(FakeClaude):
        def _create(self, **kwargs):
            raise QuestionFailedError(400, "model wrote an invalid tool call")

    [r] = run_eval([QUESTION], lambda: make_agent(olist_env, Failing(), "retrieval"), model="x")
    assert r.agent.status == "error" and "invalid tool call" in r.agent.detail
    assert r.agent.retrieved_tables and r.retrieval_recall is not None
    assert r.agent.mode == "retrieval" and r.agent.elapsed_s >= 0


def test_empty_sql_counts_as_declined(olist_env) -> None:
    """OpenAI-compatible providers send "" instead of null when the model declines."""
    fake = FakeClaude(
        reply(tool_use("submit_answer", answer="No cost data.", sql="", answerable=False))
    )
    answer = make_agent(olist_env, fake, "retrieval").ask("What is our profit margin?")
    assert answer.status == "unanswerable" and answer.sql is None


def test_summary_of_older_results_without_new_fields() -> None:
    """Saved campaign results from before the experiment still load and summarize."""
    old = {
        "id": "q",
        "category": "orders",
        "passed": False,
        "reason": "x",
        "agent": {"question": "q", "status": "error"},
    }
    s = summarize([EvalResult.model_validate(old)])
    assert s["retrieval_recall"] is None and s["corrected_questions"] == 0
