"""Run the agent over a golden question set and score each answer."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import sqlglot
from pydantic import BaseModel
from sqlglot import exp

from analyst.agent import Agent, AgentAnswer
from analyst.agent.providers import QuestionFailedError
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
    # Retrieval modes: share of the tables the golden SQL uses that retrieval put in front of
    # the model. None when the mode retrieves nothing or the question has no golden SQL.
    retrieval_recall: float | None = None


def gold_tables(sql: str | None, dialect: str = "sqlite") -> set[str]:
    """Tables a golden query reads (CTE names excluded), lower-cased."""
    if not sql:
        return set()
    stmt = sqlglot.parse_one(sql, read=dialect)
    ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    return {t.name.lower() for t in stmt.find_all(exp.Table)} - ctes


def retrieval_recall(question: dict[str, Any], answer: AgentAnswer) -> float | None:
    needed = gold_tables(question.get("sql"))
    if answer.retrieved_tables is None or not needed:
        return None
    shown = {t.lower() for t in answer.retrieved_tables}
    return round(len(needed & shown) / len(needed), 3)


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
        agent = make_agent()
        start = time.perf_counter()
        try:
            answer = agent.ask(q["question"])
        except QuestionFailedError as e:  # a model failure on this question: score it, go on
            # Keep what the run had (retrieved tables, turns, tokens) so averages stay honest.
            partial = getattr(agent, "last_answer", None) or AgentAnswer(
                question=q["question"], status="error"
            )
            answer = partial.model_copy(
                update={
                    "status": "error",
                    "detail": e.message,
                    "elapsed_s": round(time.perf_counter() - start, 2),
                }
            )
        passed, reason = score(q, answer)
        result = EvalResult(
            id=q["id"],
            category=q["category"],
            difficulty=q.get("difficulty"),
            passed=passed,
            reason=reason,
            agent=answer,
            cost_usd=cost(model, answer),
            retrieval_recall=retrieval_recall(q, answer),
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
    answerable = [r for r in results if r.category != "unanswerable"]
    # Executed: the final SQL ran (whether or not its result was right).
    executed = [r for r in answerable if r.agent.status == "answered"]
    recalls = [r.retrieval_recall for r in results if r.retrieval_recall is not None]
    corrected = [r for r in results if r.agent.corrections > 0]
    return {
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "accuracy": _share(sum(r.passed for r in results), len(results)),
        "execution_rate": _share(len(executed), len(answerable)),
        "by_category": {k: f"{p}/{n}" for k, (p, n) in sorted(by_category.items())},
        "retrieval_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "retrieval_full_recall": _share(sum(x == 1.0 for x in recalls), len(recalls)),
        "corrected_questions": len(corrected),
        "correction_success": _share(sum(r.passed for r in corrected), len(corrected)),
        "cost_usd": round(sum(costs), 2) if costs else None,
        "avg_turns": _avg([r.agent.turns for r in results], 1),
        "avg_elapsed_s": _avg([r.agent.elapsed_s for r in results], 2),
        "avg_input_tokens": _avg([_input_tokens(r.agent) for r in results], 0),
        "avg_output_tokens": _avg([r.agent.usage.output_tokens for r in results], 0),
    }


def _input_tokens(answer: AgentAnswer) -> int:
    u = answer.usage
    return u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _avg(values: list[float], digits: int) -> float | None:
    return round(sum(values) / len(values), digits) if values else None
