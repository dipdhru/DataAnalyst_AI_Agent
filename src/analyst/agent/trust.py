"""Trust check: answer a question a second, independent way and compare the two results.

A wrong answer looks just like a right one, so the label comes from disagreement, not from the
model's own opinion of itself. Agreement isn't proof: two attempts can share a mistake. How much
a "confident" label is worth is measured on benchmarks (see evals/bench.py), not assumed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from analyst.agent.loop import AgentAnswer
from analyst.connectors import Connector, QueryError
from analyst.evals.scoring import answer_match
from analyst.models import QueryResult
from analyst.sql_guard import UnsafeQueryError

TrustLabel = Literal["confident", "check", "no_answer"]

# Without this, a second attempt tends to repeat the first query and agree with its mistakes.
SECOND_ATTEMPT_NOTE = (
    "Work this out from scratch. If there is another reasonable way to write the query "
    "(different joins, a subquery, another aggregation), prefer it."
)


class TrustCheck(BaseModel):
    label: TrustLabel
    reason: str
    second: AgentAnswer | None = None  # None when the first attempt gave no answer to check


def second_attempt_prompt(question: str) -> str:
    return f"{question}\n\n{SECOND_ATTEMPT_NOTE}"


def results_agree(first: QueryResult, second: QueryResult) -> bool:
    """Tolerant both ways: one attempt returning an extra column isn't a disagreement."""
    return any(
        answer_match(a.columns, a.rows, b.rows, ordered=False)[0]
        for a, b in ((first, second), (second, first))
    )


def needs_second_attempt(first: AgentAnswer) -> bool:
    return first.status == "answered" and bool(first.final_sql)


def cross_check(
    first: AgentAnswer, second: AgentAnswer | None, connector: Connector, *, row_cap: int
) -> TrustCheck:
    """Both SQL statements are re-run with `row_cap`, so a chat row limit can't decide agreement."""
    if not needs_second_attempt(first):
        return TrustCheck(label="no_answer", reason="the first attempt gave no answer")
    if second is None or not needs_second_attempt(second):
        return TrustCheck(label="check", reason="the second attempt gave no answer", second=second)
    try:
        first_result = connector.run_query(first.final_sql, limit=row_cap)
    except (UnsafeQueryError, QueryError) as e:
        return TrustCheck(label="check", reason=f"the answer's SQL failed: {e}", second=second)
    try:
        second_result = connector.run_query(second.final_sql, limit=row_cap)
    except (UnsafeQueryError, QueryError) as e:
        return TrustCheck(label="check", reason=f"the second SQL failed: {e}", second=second)
    if results_agree(first_result, second_result):
        return TrustCheck(label="confident", reason="two attempts agree", second=second)
    return TrustCheck(label="check", reason="two attempts disagree", second=second)
