"""The Streamlit chat UI, run headless with a scripted model (no API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fake_claude import FakeClaude, reply, tool_use
from streamlit.testing.v1 import AppTest

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
