"""Benchmark runner: sampling, scoring on full results, harness errors kept apart, resuming.

Uses the Olist sample as a stand-in database, so it runs offline without BIRD data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fake_claude import FakeClaude, reply, text, tool_use

from analyst.agent import Agent
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import Settings
from analyst.connectors import SourceRegistry
from analyst.evals.bench import (
    BenchQuestion,
    BenchResult,
    bird_predictions,
    load_bird,
    pick_sample,
    prompt_for,
    run_bench,
    summarize_bench,
)
from analyst.evals.campaign import Campaign

ROW_CAP = 500  # the settings fixture's max_row_limit; its chat limit is 100


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings, only={"olist"})
    service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    catalog = service.refresh("olist")
    yield registry.get("olist"), catalog, settings
    registry.close()


def q(id: str, gold_sql: str, **kw) -> BenchQuestion:
    return BenchQuestion(id=id, db_id="olist", question=f"question {id}", gold_sql=gold_sql, **kw)


def submits(sql: str | None, answerable: bool = True) -> FakeClaude:
    return FakeClaude(reply(tool_use("submit_answer", answer="x", sql=sql, answerable=answerable)))


def bench(olist_env, questions, clients, **kw) -> list[BenchResult]:
    connector, catalog, settings = olist_env
    clients = list(clients)
    return run_bench(
        questions,
        lambda db: Agent(clients.pop(0), connector, catalog, settings.agent),
        lambda db: connector,
        model="m",
        row_cap=ROW_CAP,
        **kw,
    )


def test_load_bird(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rows = [
        {
            "question_id": 3,
            "db_id": "f1",
            "question": "Q?",
            "evidence": "",
            "SQL": "SELECT 1",
            "difficulty": "simple",
        },
        {"question_id": 4, "db_id": "f1", "question": "R?", "evidence": "hint", "SQL": "SELECT 2"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loaded = load_bird(path)
    assert [(x.id, x.evidence, x.gold_sql, x.difficulty) for x in loaded] == [
        ("3", "", "SELECT 1", "simple"),
        ("4", "hint", "SELECT 2", None),
    ]


def test_sample_is_fixed_proportional_and_prefix_stable() -> None:
    pool = [BenchQuestion(id=f"a{i}", db_id="a", question="", gold_sql="") for i in range(30)]
    pool += [BenchQuestion(id=f"b{i}", db_id="b", question="", gold_sql="") for i in range(10)]
    s20, s8 = pick_sample(pool, 20), pick_sample(pool, 8)
    assert s8 == s20[:8]  # growing a run never changes the questions already asked
    assert pick_sample(pool, 20) == s20
    assert sum(x.db_id == "a" for x in s20) == 15  # 3:1, like the pool
    assert pick_sample(pool, 20, seed=1) != s20


def test_hint_goes_in_the_question_unless_ablated() -> None:
    question = q("1", "SELECT 1", evidence="late = after the estimate")
    assert prompt_for(question).endswith("Hint: late = after the estimate")
    assert prompt_for(question, with_evidence=False) == "question 1"


def test_scores_the_full_result_not_the_chat_limit(olist_env) -> None:
    # 300 rows: above the chat limit (100) the agent's own answer was cut to, below the cap.
    sql = "SELECT order_id FROM orders ORDER BY order_id LIMIT 300"
    [result] = bench(olist_env, [q("1", sql)], [submits(sql)])
    assert result.agent.result.truncated
    assert result.passed and result.soft_passed
    assert result.submitted_sql == sql  # as written, without the guard's added LIMIT


def test_rounding_difference_is_a_near_miss_not_ex(olist_env) -> None:
    gold = "SELECT ROUND(AVG(price), 2) FROM order_items"
    [result] = bench(olist_env, [q("1", gold)], [submits("SELECT AVG(price) FROM order_items")])
    assert not result.passed
    assert result.soft_passed
    assert result.reason.startswith("near miss")


def test_wrong_and_unanswered_fail(olist_env) -> None:
    gold = "SELECT COUNT(*) FROM orders"
    results = bench(
        olist_env,
        [q("1", gold), q("2", gold)],
        [
            submits("SELECT COUNT(*) FROM orders WHERE order_status = 'canceled'"),
            submits(None, answerable=False),
        ],
    )
    assert [(r.passed, r.soft_passed) for r in results] == [(False, False), (False, False)]
    assert results[1].reason.startswith("no answer: unanswerable")


def test_broken_gold_is_a_harness_error_left_out_of_the_score(olist_env) -> None:
    good = "SELECT COUNT(*) FROM orders"
    results = bench(
        olist_env,
        [q("1", good), q("2", "SELECT order_id FROM orders"), q("3", "SELECT nope FROM orders")],
        [submits(good), submits(good), submits(good)],
    )
    assert [r.harness_error is not None for r in results] == [False, True, True]
    assert "more than 500 rows" in results[1].harness_error  # 2000 rows > cap
    summary = summarize_bench(results)
    assert summary["ex"] == "1/1 (100%)"
    assert summary["harness_errors"] == ["2", "3"]


def test_results_survive_a_resume(olist_env, tmp_path: Path) -> None:
    sql = "SELECT COUNT(*) FROM orders"
    questions = [q("1", sql, difficulty="simple"), q("2", sql, difficulty="challenging")]
    path = tmp_path / "bench.json"
    campaign = Campaign(path, model="m", provider="p", result_type=BenchResult)
    bench(olist_env, questions[:1], [submits(sql)], on_result=campaign.add)

    again = Campaign(path, model="m", provider="p", result_type=BenchResult)
    assert again.remaining(questions) == questions[1:]
    [saved] = again.results()
    assert isinstance(saved, BenchResult) and saved.passed and saved.difficulty == "simple"
    assert summarize_bench([saved])["by_difficulty"] == {"simple": "1/1 (100%)"}


def test_predictions_use_birds_format(olist_env) -> None:
    sql = "SELECT COUNT(*) FROM orders"
    results = bench(olist_env, [q("10", sql), q("2", sql)], [submits(sql), submits(None, False)])
    assert bird_predictions(results) == {
        "2": "SELECT 0\t----- bird -----\tolist",
        "10": f"{sql}\t----- bird -----\tolist",
    }


class BadToolCalls:
    """A client whose model only ever writes unparseable tool calls."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        from analyst.agent.providers import BadToolCallError

        raise BadToolCallError("Failed to parse tool call arguments as JSON")


def test_a_question_with_broken_tool_calls_fails_and_the_run_goes_on(olist_env) -> None:
    sql = "SELECT COUNT(*) FROM orders"
    results = bench(olist_env, [q("1", sql), q("2", sql)], [BadToolCalls(), submits(sql)])
    assert [r.passed for r in results] == [False, True]
    assert "invalid tool call" in results[0].reason


GOOD = "SELECT order_id FROM orders ORDER BY order_id LIMIT 5"
SAME = "SELECT order_id FROM orders ORDER BY order_id ASC LIMIT 5"
OTHER = "SELECT order_id FROM orders ORDER BY order_id DESC LIMIT 5"
EXTRA_COLUMN = "SELECT order_id, 1 AS n FROM orders ORDER BY order_id LIMIT 5"


def test_trust_two_attempts_agree_is_confident(olist_env) -> None:
    second = submits(SAME)
    [r] = bench(olist_env, [q("1", GOOD)], [submits(GOOD), second], with_trust=True)
    assert r.trust.label == "confident" and r.passed
    assert "from scratch" in second.requests[0]["messages"][0]["content"]


def test_trust_extra_column_is_not_a_disagreement(olist_env) -> None:
    clients = [submits(GOOD), submits(EXTRA_COLUMN)]
    [r] = bench(olist_env, [q("1", GOOD)], clients, with_trust=True)
    assert r.trust.label == "confident"


def test_trust_disagreement_flags_a_wrong_answer_and_keeps_the_first_score(olist_env) -> None:
    clients = [submits(OTHER), submits(GOOD)]
    [r] = bench(olist_env, [q("1", GOOD)], clients, with_trust=True)
    assert r.trust.label == "check" and not r.passed  # the right second attempt doesn't rescue it
    assert r.submitted_sql == OTHER


def test_trust_second_attempt_failure_flags_the_answer(olist_env) -> None:
    second = FakeClaude(reply(text("I can't tell.")), reply(text("Still no.")))
    [r] = bench(olist_env, [q("1", GOOD)], [submits(GOOD), second], with_trust=True)
    assert r.trust.label == "check" and r.passed
    assert "second attempt" in r.trust.reason


def test_trust_spends_no_second_attempt_without_an_answer(olist_env) -> None:
    first = FakeClaude(reply(text("I can't tell.")), reply(text("Still no.")))
    [r] = bench(olist_env, [q("1", GOOD)], [first], with_trust=True)  # only one client scripted
    assert r.trust.label == "no_answer" and r.trust.second is None


def test_trust_off_makes_one_attempt(olist_env) -> None:
    [r] = bench(olist_env, [q("1", GOOD)], [submits(GOOD)])
    assert r.trust is None
    assert "trust" not in summarize_bench([r])


def test_trust_second_attempt_counts_toward_cost(olist_env) -> None:
    connector, catalog, settings = olist_env
    clients = [submits(GOOD), submits(SAME)]
    model = "claude-opus-5-5"
    make = lambda db: Agent(clients.pop(0), connector, catalog, settings.agent)  # noqa: E731
    [one] = run_bench(
        [q("1", GOOD)], make, lambda db: connector, model=model, row_cap=ROW_CAP, with_trust=False
    )
    clients[:] = [submits(GOOD), submits(SAME)]
    [two] = run_bench(
        [q("1", GOOD)], make, lambda db: connector, model=model, row_cap=ROW_CAP, with_trust=True
    )
    assert two.cost_usd == pytest.approx(2 * one.cost_usd)


def test_trust_summary_counts_labels_against_the_real_score(olist_env) -> None:
    results = bench(
        olist_env,
        [q("1", GOOD), q("2", GOOD), q("3", GOOD), q("4", GOOD)],
        [
            submits(GOOD),
            submits(SAME),  # right, confident
            submits(OTHER),
            submits(OTHER),  # wrong, confident: a shared mistake
            submits(OTHER),
            submits(GOOD),  # wrong, flagged
            submits(GOOD),
            submits(OTHER),  # right, flagged: a false alarm
        ],
        with_trust=True,
    )
    assert summarize_bench(results)["trust"] == {
        "confident_and_right": "1/2 (50%)",
        "wrong_answers_flagged": "1/2 (50%)",
        "false_alarms": "1/2 (50%)",
        "no_answer": 0,
    }


def test_trust_result_survives_a_campaign_reload(olist_env, tmp_path: Path) -> None:
    [r] = bench(olist_env, [q("1", GOOD)], [submits(GOOD), submits(SAME)], with_trust=True)
    campaign = Campaign(tmp_path / "c.json", model="m", provider="p", result_type=BenchResult)
    campaign.add(r)
    [loaded] = campaign.results()
    assert loaded.trust.label == "confident"
    assert loaded.trust.second.final_sql == SAME
