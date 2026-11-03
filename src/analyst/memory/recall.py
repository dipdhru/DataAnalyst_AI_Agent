"""Pick the memory that matters for one question and phrase it for the agent.

Similarity is word overlap weighted by rarity (a small TF-IDF), which needs no model and no
extra dependency. It finds "late deliveries by state" from "late-delivery rate per state";
it won't find synonyms. Swap in embeddings here if that becomes the limit.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from analyst.connectors import Connector, QueryError
from analyst.memory.store import Fact, MemoryStore, VerifiedQuery
from analyst.sql_guard import UnsafeQueryError

MAX_EXAMPLES = 3
MIN_SIMILARITY = 0.25
MAX_CORRECTIONS = 10

_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "by",
        "per",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "how",
        "many",
        "much",
        "do",
        "does",
        "did",
        "our",
        "we",
        "us",
        "my",
        "me",
        "show",
        "give",
        "list",
        "tell",
        "from",
        "with",
        "at",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "be",
        "been",
        "each",
        "all",
        "any",
        "please",
    ]
)


class Recalled(BaseModel):
    examples: list[VerifiedQuery] = Field(default_factory=list)
    glossary: list[Fact] = Field(default_factory=list)
    corrections: list[Fact] = Field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.examples or self.glossary or self.corrections)


def recall(store: MemoryStore, source: str, question: str) -> Recalled:
    facts = store.facts(source)
    words = set(tokens(question))
    return Recalled(
        examples=similar(question, store.verified(source)),
        # A glossary term counts when all its words appear in the question.
        glossary=[
            f
            for f in facts
            if f.kind == "glossary" and set(tokens(f.term or "")) <= words and tokens(f.term or "")
        ],
        # Corrections are general rules ("revenue excludes freight"): send the latest ones.
        corrections=[f for f in facts if f.kind == "correction"][-MAX_CORRECTIONS:],
    )


def similar(
    question: str, candidates: list[VerifiedQuery], k: int = MAX_EXAMPLES
) -> list[VerifiedQuery]:
    if not candidates:
        return []
    docs = [Counter(tokens(c.question)) for c in candidates]
    df = Counter(word for d in docs for word in d)
    n = len(docs)

    def vector(counts: Counter[str]) -> dict[str, float]:
        return {w: c * math.log(1 + (n + 1) / (df.get(w, 0) + 1)) for w, c in counts.items()}

    query = vector(Counter(tokens(question)))
    scored = []
    for candidate, doc in zip(candidates, docs, strict=True):
        score = _cosine(query, vector(doc))
        if score >= MIN_SIMILARITY:
            scored.append((score, candidate.id, candidate))
    scored.sort(key=lambda s: (-s[0], -s[1]))  # best first; newer wins a tie
    return [c for _, _, c in scored[:k]]


def render(recalled: Recalled) -> str:
    """The memory block placed before the question. Empty when there is nothing to say."""
    if recalled.empty:
        return ""
    lines = ["Notes from memory. The user approved all of these; follow them."]
    if recalled.glossary:
        lines.append("Glossary:")
        lines += [f"- {f.term}: {f.text}" for f in recalled.glossary]
    if recalled.corrections:
        lines.append("Corrections the user made to earlier answers:")
        lines += [f"- {f.text}" for f in recalled.corrections]
    if recalled.examples:
        lines.append(
            "Verified answers to similar questions. Reuse their SQL when it fits, adapting it "
            "to this question; check it still answers what is asked."
        )
        for ex in recalled.examples:
            lines += [f"- Question: {ex.question}", f"  SQL: {' '.join(ex.sql.split())}"]
    return "\n".join(lines)


def revalidate(store: MemoryStore, source: str, connector: Connector) -> list[VerifiedQuery]:
    """Re-run every verified query (one row is enough) after a schema change. Queries that
    fail are marked stale and stop being recalled; ones that run again are restored.
    Returns the queries that are stale now."""
    stale = []
    for q in store.verified(source, include_stale=True):
        try:
            connector.run_query(q.sql, limit=1)
            reason = None
        except (UnsafeQueryError, QueryError) as e:
            reason = str(e)[:300]
            stale.append(q.model_copy(update={"stale_reason": reason}))
        if reason != q.stale_reason:
            store.set_stale(q.id, reason)
    return stale


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in _STOP]


def _stem(word: str) -> str:
    # Just enough to match plurals and -ies/-ied forms: "deliveries" ~ "delivery".
    for suffix, repl in (("ies", "y"), ("ied", "y"), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + repl
    return word


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0
