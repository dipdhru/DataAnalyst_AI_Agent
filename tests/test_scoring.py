import pytest
import yaml
from test_golden import GOLDEN, QUESTIONS

from analyst.agent import AgentAnswer
from analyst.evals.runner import cost, score
from analyst.evals.scoring import (
    answer_match,
    execution_match,
    has_top_level_order_by,
    values_match,
)
from analyst.models import QueryResult


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        (13494400.7412, 13494400.74, True),
        (13494400.76, 13494400.74, False),
        (12.5537, 12.6, True),
        (12.4, 12.6, False),
        (0.0677, 6.77, True),  # fraction vs percentage
        (96096, 96096, True),
        (960.96, 96096, False),  # no scaling for counts
        ("SP ", "sp", True),
        (None, None, True),
        (0, None, False),
    ],
)
def test_values_match(pred, gold, ok) -> None:
    assert values_match(pred, gold) is ok


def test_extra_and_reordered_columns_are_fine() -> None:
    gold = [["health_beauty", 1255695.13], ["watches_gifts", 1198185.21]]
    pred = [[1255695.1299, "health_beauty", 9670], [1198185.2101, "watches_gifts", 5991]]
    ok, reason = answer_match(["revenue", "category", "orders"], pred, gold, ordered=True)
    assert ok, reason


def test_row_order_counts_only_when_ordered() -> None:
    gold = [["a", 1], ["b", 2]]
    pred = [["b", 2], ["a", 1]]
    assert answer_match(["k", "v"], pred, gold, ordered=False)[0]
    assert not answer_match(["k", "v"], pred, gold, ordered=True)[0]


def test_wrong_answers_fail() -> None:
    assert not answer_match(["n"], [[99441]], [[96096]], ordered=False)[0]
    assert not answer_match(["n"], [[1], [2]], [[1]], ordered=False)[0]
    assert not answer_match(["a"], [[1]], [[1, 2]], ordered=False)[0]


@pytest.mark.parametrize("q", [q for q in QUESTIONS if "sql" in q], ids=lambda q: q["id"])
def test_every_golden_answer_matches_itself(q) -> None:
    e = q["expected"]
    ok, reason = answer_match(
        e["columns"], e["rows"], e["rows"], ordered=has_top_level_order_by(q["sql"])
    )
    assert ok, reason
    assert execution_match(e["rows"], e["rows"])


def test_order_by_detection() -> None:
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a")
    assert not has_top_level_order_by("SELECT * FROM (SELECT a FROM t ORDER BY a) x")


def test_score_and_cost() -> None:
    declined = AgentAnswer(question="q", status="unanswerable")
    assert score({"category": "unanswerable"}, declined)[0]
    assert not score(
        {"category": "revenue", "sql": "SELECT 1", "expected": {"rows": [[1]]}}, declined
    )[0]
    answered = AgentAnswer(
        question="q",
        status="answered",
        result=QueryResult(
            sql="", columns=["n"], rows=[[1]], row_count=1, truncated=False, elapsed_ms=1
        ),
    )
    assert score({"category": "orders", "sql": "SELECT 1", "expected": {"rows": [[1]]}}, answered)[
        0
    ]
    answered.usage.input_tokens, answered.usage.output_tokens = 1_000_000, 100_000
    assert cost("claude-opus-5-5", answered) == pytest.approx(6.0)


def test_golden_definitions_are_in_metrics_yaml() -> None:
    from pathlib import Path

    metrics = yaml.safe_load((Path(__file__).parents[1] / "metrics.yaml").read_text())
    assert set(GOLDEN["definitions"]) <= set(metrics[GOLDEN["source"]]["definitions"])
