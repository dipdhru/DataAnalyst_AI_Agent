"""Run the agent over a golden question set and score each answer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from analyst.agent import Agent, AgentAnswer
from analyst.evals.scoring import answer_match, has_top_level_order_by

# $ per million tokens: input, output, cache read, cache write (5-minute TTL).
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


class EvalResult(BaseModel):
    id: str
    category: str
    difficulty: str | None = None
    passed: bool
    reason: str
    agent: AgentAnswer
    cost_usd: float | None = None


def score(question: dict[str, Any], answer: AgentAnswer) -> tuple[bool, str]:
    if question["category"] == "unanswerable":
        if answer.status == "unanswerable":
            return True, "correctly declined"
        return False, f"should have declined, got status {answer.status}"
    if answer.status != "answered" or answer.result is None:
        return False, f"no answer: {answer.status} ({answer.detail or 'no detail'})"
    expected = question["expected"]
    return answer_match(
        answer.result.columns,
        answer.result.rows,
        expected["rows"],
        ordered=has_top_level_order_by(question["sql"]),
    )


def cost(model: str, answer: AgentAnswer) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_out, p_read, p_write = PRICES[model]
    u = answer.usage
    total = (
        u.input_tokens * p_in
        + u.output_tokens * p_out
        + u.cache_read_input_tokens * p_read
        + u.cache_creation_input_tokens * p_write
    )
    return round(total / 1_000_000, 4)


def run_eval(
    questions: list[dict[str, Any]],
    make_agent: Callable[[], Agent],
    *,
    model: str,
    on_result: Callable[[EvalResult], None] | None = None,
    on_question: Callable[[dict[str, Any]], None] | None = None,
) -> list[EvalResult]:
    results = []
    for q in questions:
        if on_question:
            on_question(q)
        answer = make_agent().ask(q["question"])
        passed, reason = score(q, answer)
        result = EvalResult(
            id=q["id"],
            category=q["category"],
            difficulty=q.get("difficulty"),
            passed=passed,
            reason=reason,
            agent=answer,
            cost_usd=cost(model, answer),
        )
        results.append(result)
        if on_result:
            on_result(result)
    return results


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    by_category: dict[str, list[int]] = {}
    for r in results:
        by_category.setdefault(r.category, [0, 0])
        by_category[r.category][0] += r.passed
        by_category[r.category][1] += 1
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    return {
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "accuracy": round(sum(r.passed for r in results) / len(results), 3) if results else None,
        "by_category": {k: f"{p}/{n}" for k, (p, n) in sorted(by_category.items())},
        "cost_usd": round(sum(costs), 2) if costs else None,
        "avg_turns": round(sum(r.agent.turns for r in results) / len(results), 1)
        if results
        else None,
    }
