"""The Streamlit chat UI, run headless with a scripted model (no API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fake_claude import FakeClaude, reply, tool_use
from streamlit.testing.v1 import AppTest

from analyst.catalog import CatalogStore

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")
COUNT_SQL = "SELECT COUNT(*) AS orders FROM orders"
MONTHLY_SQL = (
    "SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, COUNT(*) AS orders "
    "FROM orders GROUP BY 1 ORDER BY 1"
)


@pytest.fixture
def app_config(tmp_path: Path, sample_sqlite: Path, monkeypatch) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "state_dir": str(tmp_path / "state"),
                "metrics_path": str(Path(APP).parents[1] / "metrics.yaml"),
                "agent": {
                    "provider": "openai_compatible",
                    "model": "gemini-3.6-flash",
                    "base_url": "https://example.invalid/",
                    "api_key_env": "UNUSED",
                },
                "sources": [{"name": "olist", "type": "sqlite", "path": str(sample_sqlite)}],
            }
        )
    )
    monkeypatch.setenv("ANALYST_CONFIG", str(config))
    return config


def start(fake: FakeClaude) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["llm_client"] = fake
    return at.run()


def test_builds_the_catalog_then_answers(app_config: Path) -> None:
    fake = FakeClaude(
        reply(tool_use("describe_table", table="orders")),
        reply(tool_use("run_sql", sql=COUNT_SQL)),
        reply(
            tool_use(
                "submit_answer",
                answer="There are **2,000** orders.",
                sql=COUNT_SQL,
                answerable=True,
            )
        ),
    )
    at = start(fake)
    assert not at.exception
    assert at.chat_input[0].disabled  # no catalog yet
    next(b for b in at.button if b.label.startswith("Build catalog")).click().run()
    assert not at.chat_input[0].disabled

    at.chat_input[0].set_value("How many orders are there?").run()
    assert not at.exception
    assert any("2,000" in m.value for m in at.markdown)
    assert at.metric[0].value == "2,000"
    assert at.dataframe[0].value.iloc[0, 0] == 2000
    sql_shown = [c.value for c in at.code]
    assert any("COUNT(*)" in s for s in sql_shown)
    steps = " ".join(m.value for m in at.markdown)
    assert "Looked at the **orders** table" in steps and "Ran a query: **1 rows**" in steps
    assert len(at.session_state["messages"]) == 2


def test_example_question_and_line_chart(app_config: Path) -> None:
    fake = FakeClaude(
        reply(
            tool_use("submit_answer", answer="Orders per month.", sql=MONTHLY_SQL, answerable=True)
        ),
    )
    at = start(fake)
    next(b for b in at.button if b.label.startswith("Build catalog")).click().run()
    next(b for b in at.button if b.label == "Show monthly revenue for 2017.").click().run()
    assert not at.exception
    charts = at.get("plotly_chart")
    assert charts and '"type":"scatter"' in charts[0].proto.spec.replace(" ", "")


def test_unanswerable_and_quota_messages(app_config: Path) -> None:
    from analyst.agent.providers import ProviderError

    class OutOfQuota(FakeClaude):
        def _create(self, **kwargs):
            raise ProviderError(429, "m: daily free quota used up (20 requests/day)")

    at = start(
        FakeClaude(
            reply(
                tool_use(
                    "submit_answer", answer="There is no cost data.", sql=None, answerable=False
                )
            )
        )
    )
    next(b for b in at.button if b.label.startswith("Build catalog")).click().run()
    at.chat_input[0].set_value("What is our profit margin?").run()
    assert any("no cost data" in i.value for i in at.info)

    at.session_state["llm_client"] = OutOfQuota()
    at.chat_input[0].set_value("Anything else?").run()
    assert any("free quota" in e.value for e in at.error)


def built(fake: FakeClaude) -> AppTest:
    at = start(fake)
    next(b for b in at.button if b.label.startswith("Build catalog")).click().run()
    return at


def submits(answer: str, sql: str = COUNT_SQL):
    return reply(tool_use("submit_answer", answer=answer, sql=sql, answerable=True))


def test_marked_answers_are_recalled_with_the_conversation(app_config: Path) -> None:
    fake = FakeClaude(submits("2,000 orders."), submits("Still 2,000."))
    at = built(fake)
    at.chat_input[0].set_value("How many orders are there?").run()
    next(b for b in at.button if b.label.startswith("✅")).click().run()
    assert not at.exception
    assert any("Saved as a verified answer" in s.value for s in at.success)

    at.chat_input[0].set_value("How many orders are there in total?").run()
    assert not at.exception
    sent = fake.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]  # earlier turn replayed
    assert "Verified answers to similar questions" in sent[-1]["content"]
    assert "SQL: SELECT COUNT(*) AS orders FROM orders" in sent[-1]["content"]
    assert any("Used from memory (1)" in e.label for e in at.expander)


def test_memory_view_adds_glossary_and_corrections_that_reach_the_agent(
    app_config: Path,
) -> None:
    fake = FakeClaude(submits("2,000 orders."))
    at = built(fake)
    at.radio[0].set_value("Memory").run()
    term, meaning = at.text_input[0], at.text_input[1]
    term.set_value("late order")
    meaning.set_value("delivered after the estimated date")
    next(b for b in at.button if b.label == "Add term").click().run()
    at.text_input[2].set_value("Count only delivered orders.")
    next(b for b in at.button if b.label == "Add correction").click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "**late order**: delivered after the estimated date" in page
    assert "Count only delivered orders." in page

    at.radio[0].set_value("Chat").run()
    at.chat_input[0].set_value("How many late orders?").run()
    content = fake.requests[0]["messages"][-1]["content"]
    assert "- late order: delivered after the estimated date" in content
    assert "- Count only delivered orders." in content


def test_catalog_view_saves_user_descriptions(app_config: Path) -> None:
    at = built(FakeClaude())
    at.radio[0].set_value("Catalog").run()
    assert not at.exception
    table_select = next(s for s in at.selectbox if s.label == "Table")
    table_select.set_value("orders").run()
    at.text_input[0].set_value("One row per order.").run()
    next(b for b in at.button if b.label == "Save descriptions").click().run()
    assert not at.exception
    assert any("Saved 1 descriptions" in s.value for s in at.success)

    store = CatalogStore(Path(yaml.safe_load(app_config.read_text())["state_dir"]) / "app.db")
    orders = store.load("olist").table("orders")
    assert (orders.description, orders.description_source) == ("One row per order.", "user")


def test_past_conversations_can_be_reopened(app_config: Path) -> None:
    at = built(FakeClaude(submits("2,000 orders.")))
    at.chat_input[0].set_value("How many orders are there?").run()
    next(b for b in at.button if b.label == "New conversation").click().run()
    assert at.session_state["messages"] == []
    next(b for b in at.button if b.label == "How many orders are there?").click().run()
    assert not at.exception
    assert len(at.session_state["messages"]) == 2
    assert any("2,000 orders." in m.value for m in at.markdown)
