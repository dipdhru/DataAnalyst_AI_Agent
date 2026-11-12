"""Chat UI: ask a question, watch the agent work, see the answer, chart, table and SQL.

    uv run streamlit run ui/app.py

Three views:
- Chat: answers use memory (verified answers, glossary, corrections) and the earlier turns of
  the conversation, so follow-ups work. Mark an answer correct, or correct it, to teach it.
  Conversations are saved and can be reopened. "Double-check answers" asks every question a
  second, independent time and labels the answer (agent/trust.py).
- Catalog: read and edit table and column descriptions; refresh; let the model fill gaps.
- Memory: everything the agent has been taught, with delete.

Only the user writes memory; the agent only reads it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from analyst.agent import Agent, AgentAnswer, ToolCall, Turn
from analyst.agent.providers import ProviderError, make_llm_client
from analyst.agent.trust import cross_check, needs_second_attempt, second_attempt_prompt
from analyst.catalog import CatalogService, CatalogStore
from analyst.catalog.describer import describe_catalog
from analyst.charts import make_figure, suggest_chart
from analyst.config import load_settings
from analyst.connectors import SourceRegistry
from analyst.demo import (
    PROVIDERS,
    ConnectError,
    Workspace,
    agent_settings_for,
    demo_enabled,
)
from analyst.memory import MemoryStore, recall, render, revalidate
from analyst.netguard import UnsafeHostError
from analyst.semantic import load_semantics

log = logging.getLogger("analyst.ui")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get("ANALYST_CONFIG", ROOT / "config.yaml"))
# Free-tier models that handled tool calls in testing, by provider; each has its own quota.
FREE_MODELS = {
    "generativelanguage.googleapis.com": [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ],
    "api.groq.com": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
}
EXAMPLES = [
    "What is our late-delivery rate?",
    "Show monthly revenue for 2017.",
    "Which 5 product categories bring in the most revenue?",
    "What share of customers ordered more than once?",
]
DEMO = demo_enabled()  # public host: isolated sessions, visitors bring their database and key
HISTORY_TURNS = 3  # earlier answered questions replayed to the agent for follow-ups

st.set_page_config(page_title="DataAnalyst Agent", page_icon=":bar_chart:", layout="wide")


@st.cache_resource(show_spinner=False)
def backend(config_path: str, source: str) -> dict[str, Any]:
    settings = load_settings(config_path)
    registry = SourceRegistry(settings, only={source})
    store = CatalogStore(settings.app_db_path)
    return {
        "settings": settings,
        "connector": registry.get(source),
        "store": store,
        "service": CatalogService(registry, store, settings.profiling),
        "memory": MemoryStore(settings.app_db_path),
        "semantics": load_semantics(settings.metrics_path).get(source),
    }


def client_for(model: str, b: dict[str, Any], on_wait=None) -> Any:
    agent_settings = b["settings"].agent.model_copy(update={"model": model})
    return st.session_state.get("llm_client") or make_llm_client(
        agent_settings, on_wait=on_wait, api_key=b.get("api_key")
    )


def step_text(step: ToolCall) -> str:
    if not step.ok:
        return f":red[{step.tool} failed:] {step.error}"
    if step.tool == "describe_table":
        return f"Looked at the **{step.input.get('table')}** table"
    if step.tool == "list_tables":
        return "Listed the tables"
    if step.tool == "run_sql":
        return f"Ran a query: **{step.row_count} rows**"
    if step.tool == "submit_answer":
        return "Re-ran the final query to double-check the answer"
    return step.tool


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# ---- chat ----------------------------------------------------------------------------------
def show_trust(trust: dict[str, str] | None) -> None:
    """Agreement is evidence, not proof: two attempts can share a mistake, so "confident" says
    what was checked rather than "correct"."""
    if not trust or trust["label"] == "no_answer":
        return
    if trust["label"] == "confident":
        st.markdown(
            ":green-badge[✅ Double-checked] A second, independent attempt got the same result."
        )
    else:
        st.markdown(
            f":orange-badge[⚠️ Double-check this] A second attempt didn't confirm it: "
            f"{trust['reason']}. Look at the SQL before relying on this answer."
        )


def show_answer(
    a: AgentAnswer, model: str, used: list[str], trust: dict[str, str] | None = None
) -> None:
    if a.status == "answered":
        show_trust(trust)
        st.markdown(a.answer or "")
    elif a.status == "unanswerable":
        st.info(a.answer or "The data can't answer this question.")
    else:
        st.warning(f"No answer: {a.detail or a.status}")
        if a.answer:
            st.markdown(a.answer)

    if a.result and a.result.rows:
        columns, rows = a.result.columns, a.result.rows
        spec = suggest_chart(columns, rows)
        if spec and spec.kind == "metric":
            for col, box in zip(spec.y, st.columns(len(spec.y)), strict=True):
                box.metric(col.replace("_", " "), fmt(rows[0][columns.index(col)]))
        elif spec:
            st.plotly_chart(make_figure(columns, rows, spec), use_container_width=True)
        st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True, use_container_width=True)
        if a.result.truncated:
            st.caption(f"Showing the first {a.result.row_count} rows.")

    if a.sql:
        with st.expander("SQL used"):
            st.code(a.sql, language="sql")
    if used:
        with st.expander(f"Used from memory ({len(used)})"):
            for item in used:
                st.markdown(f"- {item}")
    with st.expander(f"How it got there ({len(a.steps)} steps)"):
        for i, step in enumerate(a.steps, 1):
            st.markdown(f"{i}. {step_text(step)}")
            if step.tool == "run_sql" and step.input.get("sql"):
                st.code(step.input["sql"], language="sql")
    st.caption(f"{model} · {a.turns} model turns · {a.elapsed_s:.0f}s")


def feedback(i: int, m: dict[str, Any], source: str, memory: MemoryStore) -> None:
    """Teach the agent from one answer: keep it as verified, or record a correction."""
    a = AgentAnswer.model_validate(m["answer"])
    if a.status != "answered" or not a.final_sql:
        return
    left, right, _ = st.columns([2, 1, 3])
    if m.get("verified"):
        left.success("Saved as a verified answer")
    elif left.button("✅ Correct", key=f"ok-{i}", help="Remember this answer and its SQL"):
        memory.add_verified(source, m["question"], a.final_sql, a.answer)
        m["verified"] = True
        st.rerun()
    with right.popover("✏️ Fix it"):
        note = st.text_area(
            "What should it do differently?",
            key=f"fix-text-{i}",
            placeholder="e.g. Revenue should exclude freight.",
        )
        if st.button("Save correction", key=f"fix-save-{i}") and note.strip():
            memory.add_fact(source, "correction", note)
            st.toast("Saved. It will apply to the next questions.")


def history(messages: list[dict[str, Any]]) -> list[Turn]:
    turns = []
    for m in messages:
        if m["role"] == "assistant" and "answer" in m:
            a = AgentAnswer.model_validate(m["answer"])
            if a.status in ("answered", "unanswerable"):
                turns.append(Turn(question=m["question"], answer=a.answer, sql=a.final_sql))
    return turns[-HISTORY_TURNS:]


def ask(
    question: str,
    b: dict[str, Any],
    catalog: Any,
    model: str,
    memory_text: str,
    turns: list[Turn],
    double_check: bool = False,
) -> tuple[AgentAnswer | None, dict[str, str] | None, str | None]:
    """Run the agent with a live status box. Returns (answer, trust label or None, None) or
    (None, None, error text)."""
    agent_settings = b["settings"].agent.model_copy(update={"model": model})
    trust = None
    with st.status("Working on it...", expanded=True) as status:

        def on_step(step: ToolCall) -> None:
            status.write(step_text(step))

        def on_wait(message: str) -> None:
            status.write(":orange[The model is busy, waiting a moment...]")

        def new_agent() -> Agent:
            return Agent(
                client_for(model, b, on_wait),
                b["connector"],
                catalog,
                agent_settings,
                b["semantics"],
                on_step=on_step,
            )

        try:
            answer = new_agent().ask(question, memory=memory_text, history=turns)
        except ProviderError as e:
            log.warning("provider error for %s: %s", model, e.message)
            status.update(label="The model could not answer", state="error", expanded=False)
            if "daily free quota" in e.message:
                return (
                    None,
                    None,
                    (
                        f"Today's free quota for **{model}** is used up. Pick another "
                        "model in the sidebar, or try again tomorrow."
                    ),
                )
            return None, None, f"Model provider error: {e.message}"
        except Exception as e:  # show anything unexpected instead of failing silently
            log.exception("agent run failed")
            status.update(label="Something went wrong", state="error", expanded=False)
            return None, None, f"Unexpected error: {type(e).__name__}: {e}"
        if double_check and needs_second_attempt(answer):
            status.write("**Double-checking:** a second, independent attempt")
            try:
                second = new_agent().ask(
                    second_attempt_prompt(question), memory=memory_text, history=turns
                )
            except ProviderError as e:  # the first answer stands; it just isn't confirmed
                log.warning("second attempt failed for %s: %s", model, e.message)
                second = None
            check = cross_check(
                answer, second, b["connector"], row_cap=b["settings"].query.max_row_limit
            )
            trust = {"label": check.label, "reason": check.reason}
        state = "complete" if answer.status in ("answered", "unanswerable") else "error"
        status.update(
            label=f"Done: {len(answer.steps)} steps, {answer.elapsed_s:.0f}s",
            state=state,
            expanded=False,
        )
    return answer, trust, None


def save(memory: MemoryStore, source: str, message: dict[str, Any]) -> None:
    """Keep the conversation in state/app.db so it can be reopened later."""
    if "conversation_id" not in st.session_state:
        title = message.get("text", "Conversation")
        st.session_state.conversation_id = memory.start_conversation(source, title)
    memory.add_message(st.session_state.conversation_id, message)


def chat_view(
    source: str, b: dict[str, Any], catalog: Any, model: str, double_check: bool = False
) -> None:
    st.title("Ask your data")
    st.caption(
        "Ask a business question in plain English. The AI explores the data with read-only "
        "queries, and every answer comes with the SQL behind it. Follow-up questions work."
    )
    memory: MemoryStore = b["memory"]
    messages: list[dict[str, Any]] = st.session_state.setdefault("messages", [])
    for i, m in enumerate(messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["text"])
            elif "error" in m:
                st.error(m["error"])
            else:
                show_answer(
                    AgentAnswer.model_validate(m["answer"]),
                    m["model"],
                    m.get("used", []),
                    m.get("trust"),
                )
                feedback(i, m, source, memory)

    question = st.chat_input("Ask a question about your data", disabled=catalog is None)
    question = question or st.session_state.pop("pending", None)
    if not question or catalog is None:
        return
    user_message = {"role": "user", "text": question}
    messages.append(user_message)
    save(memory, source, user_message)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        recalled = recall(memory, source, question)
        used = [f"Verified answer: {e.question}" for e in recalled.examples]
        used += [f"Glossary: {f.term}: {f.text}" for f in recalled.glossary]
        used += [f"Correction: {f.text}" for f in recalled.corrections]
        answer, trust, error = ask(
            question, b, catalog, model, render(recalled), history(messages[:-1]), double_check
        )
        if answer:
            show_answer(answer, model, used, trust)
            reply = {
                "role": "assistant",
                "question": question,
                "answer": answer.model_dump(mode="json"),
                "model": model,
                "used": used,
                "trust": trust,
            }
        else:
            st.error(error)
            reply = {"role": "assistant", "question": question, "error": error}
        messages.append(reply)
        save(memory, source, reply)
    st.rerun()  # redraw with the feedback buttons under the new answer


# ---- catalog -------------------------------------------------------------------------------
def catalog_view(source: str, b: dict[str, Any], catalog: Any, model: str) -> None:
    st.title("Catalog")
    st.caption(
        "What the AI knows about each table. Good descriptions are the cheapest way to better "
        "answers: say what a column means, its units, and what its codes stand for."
    )
    if catalog is None:
        st.info("Build the catalog first (sidebar).")
        return
    service: CatalogService = b["service"]
    left, right = st.columns(2)
    if left.button("Refresh catalog", help="Re-read the schema and re-profile every table."):
        with st.spinner("Profiling every table..."):
            service.refresh(source)
            stale = revalidate(b["memory"], source, b["connector"])
        msg = "Catalog refreshed."
        if stale:
            msg += f" {len(stale)} verified answers no longer run; see Memory."
        st.session_state.flash = msg
        st.rerun()
    if right.button(
        "Write missing descriptions with AI",
        help=f"One call to {model} per table with gaps. Your own descriptions are kept.",
    ):
        with st.spinner("Writing descriptions..."):
            try:
                report = describe_catalog(client_for(model, b), model, b["store"], source)
                msg = f"Wrote {report.tables} table and {report.columns} column descriptions."
                if report.failed:
                    msg += f" Skipped {len(report.failed)} tables."
            except ProviderError as e:
                msg = f"Model provider error: {e.message}"
        st.session_state.flash = msg
        st.rerun()
    if flash := st.session_state.pop("flash", None):
        st.success(flash)

    name = st.selectbox("Table", [t.name for t in catalog.tables])
    table = catalog.table(name)
    st.caption(f"{table.row_count:,} rows · {len(table.columns)} columns")
    table_desc = st.text_input("Table description", table.description or "", key=f"td-{name}")
    frame = pd.DataFrame(
        [
            {
                "column": c.name,
                "type": c.data_type,
                "description": c.description or "",
                "written by": c.description_source or "",
                "examples": ", ".join(str(s) for s in (c.profile.samples if c.profile else [])[:3]),
            }
            for c in table.columns
        ]
    )
    edited = st.data_editor(
        frame,
        key=f"cols-{name}",
        hide_index=True,
        use_container_width=True,
        disabled=["column", "type", "written by", "examples"],
    )
    if st.button("Save descriptions", type="primary"):
        changed = 0
        if table_desc.strip() != (table.description or ""):
            service.describe_table(source, name, table_desc.strip() or None)
            changed += 1
        for before, after in zip(
            frame.itertuples(index=False), edited.itertuples(index=False), strict=True
        ):
            if str(after.description).strip() != before.description:
                text = str(after.description).strip() or None
                service.describe_column(source, name, before.column, text)
                changed += 1
        st.session_state.flash = f"Saved {changed} descriptions." if changed else "No changes."
        st.rerun()


# ---- memory --------------------------------------------------------------------------------
def memory_view(source: str, b: dict[str, Any]) -> None:
    memory: MemoryStore = b["memory"]
    st.title("Memory")
    st.caption(
        "What you have taught the AI for this source. It recalls the relevant items with every "
        "question; only you can add or remove them."
    )

    st.subheader("Verified answers")
    verified = memory.verified(source, include_stale=True)
    if not verified:
        st.caption("None yet: press ✅ under a correct answer in the chat.")
    for q in verified:
        with st.expander(("⚠️ " if q.stale_reason else "") + q.question):
            if q.stale_reason:
                st.warning(f"No longer runs, so it isn't used: {q.stale_reason}")
            if q.answer:
                st.markdown(q.answer)
            st.code(q.sql, language="sql")
            if st.button("Delete", key=f"del-v-{q.id}"):
                memory.delete_verified(q.id)
                st.rerun()

    st.subheader("Glossary")
    with st.form("glossary", clear_on_submit=True):
        term = st.text_input("Term", placeholder="e.g. active customer")
        meaning = st.text_input("Meaning", placeholder="ordered in the last 90 days")
        if st.form_submit_button("Add term") and term.strip() and meaning.strip():
            memory.add_fact(source, "glossary", meaning, term=term)
            st.rerun()
    for f in memory.facts(source, "glossary"):
        left, right = st.columns([5, 1])
        left.markdown(f"**{f.term}**: {f.text}")
        if right.button("Delete", key=f"del-f-{f.id}"):
            memory.delete_fact(f.id)
            st.rerun()

    st.subheader("Corrections")
    with st.form("correction", clear_on_submit=True):
        text = st.text_input("Rule", placeholder="e.g. Revenue excludes freight.")
        if st.form_submit_button("Add correction") and text.strip():
            memory.add_fact(source, "correction", text)
            st.rerun()
    for f in memory.facts(source, "correction"):
        left, right = st.columns([5, 1])
        left.markdown(f.text)
        if right.button("Delete", key=f"del-c-{f.id}"):
            memory.delete_fact(f.id)
            st.rerun()


# ---- demo mode -----------------------------------------------------------------------------
def workspace() -> Workspace:
    """This browser session's own state, connections and memory (never shared)."""
    if "workspace" not in st.session_state:
        st.session_state.workspace = Workspace(load_settings(CONFIG))
    return st.session_state.workspace


def demo_controls(ws: Workspace) -> tuple[Any, str]:
    """Sidebar for the public demo: model key and the visitor's own database."""
    st.header("Try it on your data")
    st.warning(
        "Use a **read-only** database user, ideally on a copy, not production. Your connection "
        "string and API key are held in this session's memory and never written to disk. Column "
        "samples and chat history sit in a temporary folder deleted when your session ends. "
        "Query results are sent to the model provider you pick.",
        icon="🔒",
    )
    provider = PROVIDERS[st.selectbox("Model provider", list(PROVIDERS))]
    api_key = st.text_input(
        "Your API key",
        type="password",
        help="Both providers have a free tier. The key stays in this session and is never "
        "shared or stored by this app.",
    ).strip()

    with st.form("connect", border=False):
        raw_url = st.text_input(
            "Postgres connection string",
            type="password",
            placeholder="postgresql://user:password@host:5432/dbname",
        )
        schemas = st.text_input("Schemas", "public", help="Comma-separated, up to 5.")
        if st.form_submit_button("Connect", use_container_width=True):
            try:
                with st.spinner("Connecting..."):
                    st.session_state.demo_source = ws.connect_postgres(raw_url, schemas)
            except (UnsafeHostError, ConnectError) as e:
                st.error(str(e))
    if ws.source_names():
        mine = [n for n in ws.source_names() if n.startswith("your_database")]
        if mine and st.button("Disconnect my database", use_container_width=True):
            for name in mine:
                ws.disconnect(name)
            st.session_state.pop("demo_source", None)
            st.rerun()
    return agent_settings_for(provider, load_settings(CONFIG).agent), api_key


def landing() -> None:
    st.title("Ask your data")
    st.markdown(
        "Ask a database a business question in plain English and get an answer with the SQL, a "
        "table and a chart. Every number comes with the query that produced it.\n\n"
        "**To start**, in the sidebar:\n"
        "1. Pick a model provider and paste your own free API key.\n"
        "2. Paste a Postgres connection string for a **read-only** user and press Connect.\n\n"
        "Queries are checked twice before they run: the connection is read-only and a SQL guard "
        "rejects anything but a single SELECT."
    )


# ---- sidebar -------------------------------------------------------------------------------
settings = load_settings(CONFIG)
ws: Workspace | None = None
demo_agent = None
demo_key = ""
if DEMO:
    ws = workspace()
    with st.sidebar:
        demo_agent, demo_key = demo_controls(ws)
    if not demo_key or not ws.source_names():
        landing()
        st.stop()
with st.sidebar:
    st.header("Settings")
    names = ws.source_names() if ws else [s.name for s in settings.sources]
    default = (
        names.index(st.session_state["demo_source"])
        if (DEMO and st.session_state.get("demo_source") in names)
        else 0
    )
    source = st.selectbox("Data source", names, index=default)
    b = ws.backend(source, demo_agent, demo_key) if ws else backend(str(CONFIG), source)
    agent_cfg = b["settings"].agent
    if agent_cfg.provider == "openai_compatible":
        known = next(
            (models for host, models in FREE_MODELS.items() if host in (agent_cfg.base_url or "")),
            [],
        )
        model = st.selectbox(
            "Model",
            [agent_cfg.model] + [m for m in known if m != agent_cfg.model],
            help="Free tiers have daily limits per model. Switch models when one runs out.",
        )
    else:
        model = st.text_input("Model", agent_cfg.model)
    st.caption(f"Provider: {agent_cfg.provider}")

    catalog = b["service"].get(source)
    if catalog is None:
        st.warning("This source has no catalog yet.")
        if st.button("Build catalog (about 10 s)", type="primary"):
            with st.spinner("Profiling every table..."):
                catalog = b["service"].refresh(source)
            st.rerun()
    else:
        st.caption(f"Catalog: {len(catalog.tables)} tables, {len(catalog.relationships)} joins")

    view = st.radio("View", ["Chat", "Catalog", "Memory"], horizontal=True)
    if view == "Chat":
        double_check = st.toggle(
            "Double-check answers",
            help="Ask each question a second, independent time and label the answer "
            "double-checked or not. Uses about twice the model calls.",
        )
        if st.button("New conversation"):
            st.session_state.messages = []
            st.session_state.pop("conversation_id", None)
            st.rerun()
        past = b["memory"].conversations(source, limit=8)
        if past:
            st.subheader("Past conversations")
            for c in past:
                if st.button(c.title, key=f"conv-{c.id}", use_container_width=True):
                    st.session_state.messages = b["memory"].messages(c.id)
                    st.session_state.conversation_id = c.id
                    st.rerun()
        st.subheader("Try asking")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example

if view == "Chat":
    chat_view(source, b, catalog, model, double_check)
elif view == "Catalog":
    catalog_view(source, b, catalog, model)
else:
    memory_view(source, b)
