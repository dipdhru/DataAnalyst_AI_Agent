"""Public text-to-SQL benchmarks (BIRD now, Spider later): load, sample, run the agent, score.

Scoring re-runs both the gold SQL and the SQL the agent submitted with the benchmark's own row
cap, so the chat UI's row limit never decides a score. A gold query that can't run cleanly is a
harness error, reported apart from the agent's failures. See docs/benchmarks.md.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from analyst.agent import Agent, AgentAnswer
from analyst.agent.providers import QuestionFailedError
from analyst.agent.trust import (
    TrustCheck,
    cross_check,
    needs_second_attempt,
    second_attempt_prompt,
)
from analyst.connectors import Connector, QueryError
from analyst.evals.runner import EvalResult, cost
from analyst.evals.scoring import answer_match, execution_match, has_top_level_order_by
from analyst.sql_guard import UnsafeQueryError


class BenchQuestion(BaseModel):
    id: str
    db_id: str
    question: str
    evidence: str = ""
    gold_sql: str
    difficulty: str | None = None


class BenchResult(EvalResult):
    """`category` holds the database; `passed` is strict EX."""

    soft_passed: bool = False  # our tolerant answer match
    harness_error: str | None = None  # the gold side failed: not the agent's fault
    submitted_sql: str | None = None
    trust: TrustCheck | None = None  # only on runs with the trust check on


def load_bird(path: Path) -> list[BenchQuestion]:
    """BIRD Mini-Dev jsonl (one question per line)."""
    questions = []
    for line in path.read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            questions.append(
                BenchQuestion(
                    id=str(q["question_id"]),
                    db_id=q["db_id"],
                    question=q["question"],
                    evidence=q.get("evidence") or "",
                    gold_sql=q["SQL"],
                    difficulty=q.get("difficulty"),
                )
            )
    return questions


def pick_sample(questions: Sequence[BenchQuestion], n: int, seed: int = 0) -> list[BenchQuestion]:
    """A fixed sample, proportional by database. Prefix-stable: the first 20 of a 50-question
    sample are the 20-question sample, so a free-tier run can grow without redoing work."""
    by_db: dict[str, list[BenchQuestion]] = {}
    for q in questions:
        by_db.setdefault(q.db_id, []).append(q)
    rng = random.Random(seed)
    keyed = []
    for db in sorted(by_db):
        group = sorted(by_db[db], key=lambda q: q.id)
        rng.shuffle(group)
        # Spread each database evenly over the order: its i-th question sits at (i + 0.5) / size.
        keyed += [((i + 0.5) / len(group), db, q) for i, q in enumerate(group)]
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [q for _, _, q in keyed[:n]]


def prompt_for(q: BenchQuestion, *, with_evidence: bool = True) -> str:
    if with_evidence and q.evidence:
        return f"{q.question}\n\nHint: {q.evidence}"
    return q.question


def submitted_sql(answer: AgentAnswer) -> str | None:
    """The SQL as the agent wrote it. `answer.sql` carries the guard's added LIMIT."""
    return answer.final_sql


def score_bench(
    q: BenchQuestion, answer: AgentAnswer, connector: Connector, *, row_cap: int
) -> tuple[bool, bool, str, str | None]:
    """Returns (ex, soft, reason, harness_error)."""
    try:
        gold = connector.run_query(q.gold_sql, limit=row_cap)
    except (UnsafeQueryError, QueryError) as e:
        return False, False, "gold SQL failed", str(e)
    if gold.truncated:
        return False, False, "gold result truncated", f"gold has more than {row_cap} rows"

    sql = submitted_sql(answer)
    if answer.status != "answered" or not sql:
        return False, False, f"no answer: {answer.status} ({answer.detail or 'no detail'})", None
    try:
        pred = connector.run_query(sql, limit=row_cap)
    except (UnsafeQueryError, QueryError) as e:
        return False, False, f"submitted SQL failed on re-run: {e}", None

    ex = execution_match(pred.rows, gold.rows)
    soft, soft_reason = answer_match(
        pred.columns, pred.rows, gold.rows, ordered=has_top_level_order_by(q.gold_sql)
    )
    if ex:
        return True, True, "EX match", None
    if soft:
        return False, True, f"near miss (answer match, not EX): {soft_reason}", None
    return False, False, soft_reason, None


def run_bench(
    questions: list[BenchQuestion],
    make_agent: Callable[[str], Agent],
    connector_for: Callable[[str], Connector],
    *,
    model: str,
    row_cap: int,
    with_evidence: bool = True,
    with_trust: bool = False,
    on_result: Callable[[BenchResult], None] | None = None,
    on_question: Callable[[BenchQuestion], None] | None = None,
) -> list[BenchResult]:
    """`make_agent(db_id)` and `connector_for(db_id)` give the agent and source per database.
    `with_trust` asks each answered question a second time; the score stays the first attempt's,
    so EX is comparable with runs that have the check off."""

    def attempt(q: BenchQuestion, prompt: str) -> AgentAnswer:
        try:
            return make_agent(q.db_id).ask(prompt)
        except QuestionFailedError as e:  # a model failure on this question: score it, go on
            return AgentAnswer(question=prompt, status="error", detail=e.message)

    results = []
    for q in questions:
        if on_question:
            on_question(q)
        prompt = prompt_for(q, with_evidence=with_evidence)
        answer = attempt(q, prompt)
        connector = connector_for(q.db_id)
        ex, soft, reason, harness = score_bench(q, answer, connector, row_cap=row_cap)
        trust = None
        cost_usd = cost(model, answer)
        if with_trust:
            second = (
                attempt(q, second_attempt_prompt(prompt)) if needs_second_attempt(answer) else None
            )
            trust = cross_check(answer, second, connector, row_cap=row_cap)
            if second is not None and cost_usd is not None:
                second_cost = cost(model, second)
                cost_usd = None if second_cost is None else cost_usd + second_cost
        result = BenchResult(
            id=q.id,
            category=q.db_id,
            difficulty=q.difficulty,
            passed=ex,
            soft_passed=soft,
            reason=reason,
            harness_error=harness,
            submitted_sql=submitted_sql(answer),
            trust=trust,
            agent=answer,
            cost_usd=cost_usd,
        )
        results.append(result)
        if on_result:
            on_result(result)
    return results


def summarize_trust(results: Sequence[BenchResult]) -> dict[str, Any]:
    """What the labels are worth. A label is only useful if `confident` beats the plain EX rate
    and `wrong_answers_flagged` is high without too many `false_alarms`."""
    checked = [r for r in results if r.trust is not None and r.harness_error is None]
    confident = [r for r in checked if r.trust.label == "confident"]
    flagged = [r for r in checked if r.trust.label == "check"]
    answered = confident + flagged
    right = [r for r in answered if r.passed]
    wrong = [r for r in answered if not r.passed]

    def rate(hits: int, total: int) -> str:
        return f"{hits}/{total} ({hits / total:.0%})" if total else "0/0"

    return {
        "confident_and_right": rate(sum(r.passed for r in confident), len(confident)),
        "wrong_answers_flagged": rate(sum(not r.passed for r in flagged), len(wrong)),
        "false_alarms": rate(sum(r.passed for r in flagged), len(right)),
        "no_answer": len(checked) - len(answered),
    }


def summarize_bench(results: Sequence[BenchResult]) -> dict[str, Any]:
    scored = [r for r in results if r.harness_error is None]

    def rate(rs: Sequence[BenchResult], attr: str = "passed") -> str:
        hits = sum(getattr(r, attr) for r in rs)
        return f"{hits}/{len(rs)} ({hits / len(rs):.0%})" if rs else "0/0"

    def grouped(key: Callable[[BenchResult], str | None]) -> dict[str, str]:
        groups: dict[str, list[BenchResult]] = {}
        for r in scored:
            groups.setdefault(key(r) or "unknown", []).append(r)
        return {k: rate(v) for k, v in sorted(groups.items())}

    costs = [r.cost_usd for r in scored if r.cost_usd is not None]
    summary: dict[str, Any] = {
        "ex": rate(scored),
        "answer_match": rate(scored, "soft_passed"),
        "by_difficulty": grouped(lambda r: r.difficulty),
        "by_database": grouped(lambda r: r.category),
        "harness_errors": [r.id for r in results if r.harness_error],
        "cost_usd": round(sum(costs), 2) if costs else None,
        "avg_turns": round(sum(r.agent.turns for r in scored) / len(scored), 1) if scored else None,
    }
    if any(r.trust is not None for r in scored):
        summary["trust"] = summarize_trust(scored)
    return summary


def bird_predictions(results: Sequence[BenchResult]) -> dict[str, str]:
    """Predictions in the shape BIRD's evaluation scripts read: id -> "SQL\\t----- bird -----\\tdb".
    Check against the official mini_dev scripts before publishing a number from them."""
    return {
        r.id: f"{(r.submitted_sql or 'SELECT 0').strip()}\t----- bird -----\t{r.category}"
        for r in sorted(results, key=lambda r: int(r.id) if r.id.isdigit() else r.id)
    }
