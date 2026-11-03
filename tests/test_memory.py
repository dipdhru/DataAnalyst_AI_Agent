"""Phase 4 memory: verified queries, glossary, corrections, conversations, and recall."""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_claude import FakeClaude, reply, tool_use

from analyst.agent import Agent, Turn
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import Settings
from analyst.connectors import SourceRegistry
from analyst.memory import MemoryStore, recall, render, revalidate
from analyst.memory.recall import similar, tokens

LATE_SQL = "SELECT COUNT(*) FROM orders WHERE order_status = 'delivered'"


@pytest.fixture
def memory(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "app.db")


def test_verified_queries_round_trip_per_source(memory: MemoryStore) -> None:
    qid = memory.add_verified("olist", " Late-delivery rate? ", LATE_SQL, "6.8%")
    memory.add_verified("shop", "Other source", "SELECT 1", None)
    [saved] = memory.verified("olist")
    assert (saved.id, saved.question, saved.answer) == (qid, "Late-delivery rate?", "6.8%")
    memory.delete_verified(qid)
    assert memory.verified("olist") == []
    with pytest.raises(KeyError):
        memory.delete_verified(qid)


def test_glossary_needs_a_term(memory: MemoryStore) -> None:
    with pytest.raises(ValueError, match="needs a term"):
        memory.add_fact("olist", "glossary", "meaning only")


def test_tokens_ignore_filler_and_plurals() -> None:
    assert tokens("What are the late deliveries per state?") == ["late", "delivery", "state"]


def test_similar_questions_are_found_and_unrelated_ones_are_not(memory: MemoryStore) -> None:
    memory.add_verified("olist", "What is our late-delivery rate by state?", LATE_SQL, None)
    memory.add_verified("olist", "Top 5 product categories by revenue", "SELECT 1", None)
    memory.add_verified("olist", "How many sellers are active?", "SELECT 2", None)
    candidates = memory.verified("olist")

    hits = similar("late deliveries for each state", candidates)
    assert [h.question for h in hits] == ["What is our late-delivery rate by state?"]
    assert similar("average review score", candidates) == []


def test_recall_picks_matching_glossary_and_all_corrections(memory: MemoryStore) -> None:
    memory.add_fact("olist", "glossary", "orders that arrived after the estimate", term="late")
    memory.add_fact("olist", "glossary", "price plus freight", term="gross revenue")
    memory.add_fact("olist", "correction", "Revenue excludes freight.")
    got = recall(memory, "olist", "How many late orders?")
    assert [f.term for f in got.glossary] == ["late"]
    assert [f.text for f in got.corrections] == ["Revenue excludes freight."]
    assert recall(memory, "other", "late").empty


def test_render_phrases_memory_for_the_agent(memory: MemoryStore) -> None:
    memory.add_verified("olist", "Late-delivery rate", f"SELECT\n  {LATE_SQL[7:]}", None)
    memory.add_fact("olist", "correction", "Only count delivered orders.")
    text = render(recall(memory, "olist", "late delivery rate in 2017"))
    assert "The user approved all of these" in text
    assert "- Only count delivered orders." in text
    assert f"  SQL: {LATE_SQL}" in text  # whitespace folded to one line
    assert render(recall(memory, "olist", "unrelated question")) == (
        "Notes from memory. The user approved all of these; follow them.\n"
        "Corrections the user made to earlier answers:\n- Only count delivered orders."
    )


def test_conversations_keep_messages_in_order(memory: MemoryStore) -> None:
    cid = memory.start_conversation("olist", "Late deliveries")
    memory.add_message(cid, {"role": "user", "text": "q1"})
    memory.add_message(cid, {"role": "assistant", "answer": {"answer": "a1"}})
    assert [c.title for c in memory.conversations("olist")] == ["Late deliveries"]
    assert memory.messages(cid) == [
        {"role": "user", "text": "q1"},
        {"role": "assistant", "answer": {"answer": "a1"}},
    ]
    memory.delete_conversation(cid)
    assert memory.conversations("olist") == [] and memory.messages(cid) == []


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings, only={"olist"})
    service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    yield registry.get("olist"), service.refresh("olist"), settings
    registry.close()


def test_agent_gets_memory_and_history_before_the_question(olist_env) -> None:
    connector, catalog, settings = olist_env
    fake = FakeClaude(reply(tool_use("submit_answer", answer="x", sql=LATE_SQL, answerable=True)))
    Agent(fake, connector, catalog, settings.agent).ask(
        "And by state?",
        memory="Notes from memory. ...",
        history=[Turn(question="Late-delivery rate?", answer="6.8%", sql=LATE_SQL)],
    )
    messages = fake.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == f"6.8%\n\nSQL I used:\n{LATE_SQL}"
    assert messages[2]["content"] == "Notes from memory. ...\n\nQuestion: And by state?"


def test_revalidate_marks_broken_queries_stale_and_restores_fixed_ones(
    olist_env, memory: MemoryStore
) -> None:
    connector, _, _ = olist_env
    good = memory.add_verified("olist", "ok", LATE_SQL, None)
    bad = memory.add_verified("olist", "renamed column", "SELECT gone FROM orders", None)
    stale = revalidate(memory, "olist", connector)
    assert [q.id for q in stale] == [bad]
    assert [q.id for q in memory.verified("olist")] == [good]  # stale ones aren't recalled
    assert memory.verified("olist", include_stale=True)[1].stale_reason

    memory.set_stale(good, "pretend it broke")
    revalidate(memory, "olist", connector)
    assert memory.verified("olist", include_stale=True)[0].stale_reason is None
