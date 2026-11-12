"""Demo mode: visitors are isolated from each other and from the server's own credentials."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import yaml
from fake_claude import FakeClaude, reply, tool_use
from streamlit.testing.v1 import AppTest

from analyst.agent.providers import make_llm_client
from analyst.config import AgentSettings, load_settings
from analyst.demo import ConnectError, Workspace, agent_settings_for, parse_schemas
from analyst.netguard import UnsafeHostError

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")
COUNT_SQL = "SELECT COUNT(*) AS orders FROM orders"


@pytest.fixture
def demo_config(tmp_path: Path, sample_sqlite: Path, monkeypatch) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "state_dir": str(tmp_path / "state"),
                "metrics_path": str(Path(APP).parents[1] / "metrics.yaml"),
                "agent": {
                    "provider": "anthropic",  # the owner's billed provider: must never be used
                    "model": "claude-opus-5-5",
                },
                "sources": [
                    {"name": "olist", "type": "sqlite", "path": str(sample_sqlite)},
                    # The owner's database: a demo visitor must not be able to reach it.
                    {"name": "owner_pg", "type": "postgres", "url": "postgresql://o:p@10.0.0.9/x"},
                    {"name": "missing", "type": "sqlite", "path": str(tmp_path / "gone.sqlite")},
                ],
            }
        )
    )
    monkeypatch.setenv("ANALYST_CONFIG", str(config))
    monkeypatch.setenv("ANALYST_DEMO", "1")
    return config


def test_workspaces_do_not_share_state(demo_config: Path) -> None:
    settings = load_settings(demo_config)
    a, b = Workspace(settings), Workspace(settings)
    assert a.settings.state_dir != b.settings.state_dir
    a.memory.start_conversation("olist", "A's private question")
    assert a.memory.conversations("olist", limit=5)
    assert b.memory.conversations("olist", limit=5) == []
    assert a.settings.state_dir != settings.state_dir  # not the owner's state/app.db


def test_workspace_folder_is_deleted_with_the_session(demo_config: Path) -> None:
    ws = Workspace(load_settings(demo_config))
    folder = ws.settings.state_dir
    assert folder.exists()
    del ws
    gc.collect()
    assert not folder.exists()


def test_only_sample_sqlite_sources_are_offered(demo_config: Path) -> None:
    ws = Workspace(load_settings(demo_config))
    assert ws.source_names() == ["olist"]  # no owner_pg, no source whose file is missing


def test_private_hosts_are_refused_before_any_connection(demo_config: Path) -> None:
    ws = Workspace(load_settings(demo_config))
    for url in [
        "postgresql://u:p@127.0.0.1/app",
        "postgresql://u:p@169.254.169.254/app",
        "postgresql://u:p@localhost/app",
        "postgresql://u:p@10.0.0.9/x",  # the owner's database address, from the config above
    ]:
        with pytest.raises(UnsafeHostError):
            ws.connect_postgres(url)
    assert ws.source_names() == ["olist"]


def test_failed_connection_does_not_leak_the_password(demo_config: Path, monkeypatch) -> None:
    from sqlalchemy.engine import make_url

    # Skip the host vetting to reach the connection error path; nothing listens on port 1.
    monkeypatch.setattr(
        "analyst.demo.safe_postgres_url",
        lambda raw: make_url("postgresql+psycopg://u:hunter2@127.0.0.1:1/db"),
    )
    ws = Workspace(load_settings(demo_config))
    with pytest.raises(ConnectError) as err:
        ws.connect_postgres("ignored")
    assert "hunter2" not in str(err.value)
    assert ws.source_names() == ["olist"]  # a failed connection is not registered


@pytest.mark.parametrize("bad", ["public; DROP", "a b", "x" * 80, "1abc", "a,b,c,d,e,f", '"x"'])
def test_schema_names_are_plain_identifiers(bad: str) -> None:
    with pytest.raises(ConnectError):
        parse_schemas(bad)


def test_schema_default_and_list() -> None:
    assert parse_schemas("") == ["public"]
    assert parse_schemas("public, sales") == ["public", "sales"]


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_visitors_never_use_the_owners_provider_or_key(provider: str) -> None:
    owner = AgentSettings(provider="anthropic", model="claude-opus-5-5")
    s = agent_settings_for(provider, owner)
    assert s.provider == "openai_compatible"
    assert s.api_key_env is None  # nothing is read from the server's environment
    assert s.base_url and "anthropic" not in s.base_url


def test_supplied_key_wins_over_the_servers_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "owner-secret")
    settings = AgentSettings(
        provider="openai_compatible", base_url="https://example.invalid/", api_key_env=None
    )
    client = make_llm_client(settings, api_key="visitor-key")
    assert client._openai.api_key == "visitor-key"


# ---- the Streamlit app in demo mode --------------------------------------------------------


def start(fake: FakeClaude | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    if fake:
        at.session_state["llm_client"] = fake
    return at.run()


def set_key(at: AppTest, key: str = "visitor-key") -> AppTest:
    next(t for t in at.sidebar.text_input if t.label == "Your API key").set_value(key)
    return at.run()


def test_landing_page_until_a_key_is_given(demo_config: Path) -> None:
    at = start()
    assert not at.exception
    assert not at.chat_input  # nothing to chat with yet
    assert any("To start" in m.value for m in at.markdown)
    at = set_key(at)
    assert not at.exception
    assert at.chat_input  # sample source is available, so the chat opens


def test_connect_form_shows_a_clear_error_for_a_private_host(demo_config: Path) -> None:
    at = set_key(start())
    next(t for t in at.sidebar.text_input if t.label.startswith("Postgres")).set_value(
        "postgresql://u:secret@127.0.0.1:5432/app"
    )
    next(b for b in at.sidebar.button if b.label == "Connect").click().run()
    assert not at.exception
    assert any("private or local" in e.value for e in at.sidebar.error)
    assert "secret" not in " ".join(e.value for e in at.sidebar.error)


def test_two_visitors_do_not_see_each_others_catalog_or_chat(demo_config: Path) -> None:
    fake = FakeClaude(
        reply(tool_use("describe_table", table="orders")),
        reply(tool_use("run_sql", sql=COUNT_SQL)),
        reply(tool_use("submit_answer", answer="2,000 orders.", sql=COUNT_SQL, answerable=True)),
    )
    one = set_key(start(fake))
    next(b for b in one.sidebar.button if b.label.startswith("Build catalog")).click().run()
    one.chat_input[0].set_value("How many orders are there?").run()
    assert not one.exception
    assert len(one.session_state["messages"]) == 2

    two = set_key(start())
    assert two.chat_input[0].disabled  # visitor two has no catalog: nothing was shared
    assert two.session_state["messages"] == []
    assert not any("Past conversations" in s.value for s in two.sidebar.subheader)
