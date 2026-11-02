"""Chat UI: ask a question, watch the agent work, see the answer, chart, table and SQL.

    uv run streamlit run ui/app.py

Each question is a fresh agent run (memory of earlier questions comes in phase 4). Uses the
provider and model in config.yaml; on the free Gemini tier the sidebar lets you switch models
when one model's daily quota is used up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from analyst.agent import Agent, AgentAnswer, ToolCall
from analyst.agent.providers import ProviderError, make_llm_client
from analyst.catalog import CatalogService, CatalogStore
from analyst.charts import make_figure, suggest_chart
from analyst.config import load_settings
from analyst.connectors import SourceRegistry
from analyst.semantic import load_semantics

log = logging.getLogger("analyst.ui")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get("ANALYST_CONFIG", ROOT / "config.yaml"))
# Free-tier Gemini models that handled tool calls in testing; each has its own daily quota.
FREE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
EXAMPLES = [
    "What is our late-delivery rate?",
    "Show monthly revenue for 2017.",
    "Which 5 product categories bring in the most revenue?",
    "What share of customers ordered more than once?",
]

st.set_page_config(page_title="DataAnalyst Agent", page_icon=":bar_chart:", layout="wide")


@st.cache_resource(show_spinner=False)
def backend(config_path: str, source: str) -> dict[str, Any]:
    settings = load_settings(config_path)
    # Only this source: the API server may hold the uploads database open.
    registry = SourceRegistry(settings, only={source})
    service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    return {
        "settings": settings,
        "connector": registry.get(source),
        "service": service,
        "semantics": load_semantics(settings.metrics_path).get(source),
    }


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


def show_answer(a: AgentAnswer, model: str) -> None:
    if a.status == "answered":
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
    with st.expander(f"How it got there ({len(a.steps)} steps)"):
        for i, step in enumerate(a.steps, 1):
            st.markdown(f"{i}. {step_text(step)}")
            if step.tool == "run_sql" and step.input.get("sql"):
                st.code(step.input["sql"], language="sql")
    st.caption(f"{model} · {a.turns} model turns · {a.elapsed_s:.0f}s")


def ask(
    question: str, b: dict[str, Any], catalog: Any, model: str
) -> tuple[AgentAnswer | None, str | None]:
    """Run the agent with a live status box. Returns (answer, None) or (None, error text)."""
    agent_settings = b["settings"].agent.model_copy(update={"model": model})
    with st.status("Working on it...", expanded=True) as status:

        def on_step(step: ToolCall) -> None:
            status.write(step_text(step))

        def on_wait(message: str) -> None:
            status.write(":orange[The model is busy, waiting a moment...]")

        try:
            client = st.session_state.get("llm_client") or make_llm_client(
                agent_settings, on_wait=on_wait
            )
            agent = Agent(
                client, b["connector"], catalog, agent_settings, b["semantics"], on_step=on_step
            )
            answer = agent.ask(question)
        except ProviderError as e:
            log.warning("provider error for %s: %s", model, e.message)
            status.update(label="The model could not answer", state="error", expanded=False)
            if "daily free quota" in e.message:
                return None, (
                    f"Today's free quota for **{model}** is used up. Pick another "
                    "model in the sidebar, or try again tomorrow."
                )
            return None, f"Model provider error: {e.message}"
        except Exception as e:  # show anything unexpected instead of failing silently
            log.exception("agent run failed")
            status.update(label="Something went wrong", state="error", expanded=False)
            return None, f"Unexpected error: {type(e).__name__}: {e}"
        state = "complete" if answer.status in ("answered", "unanswerable") else "error"
        status.update(
            label=f"Done: {len(answer.steps)} steps, {answer.elapsed_s:.0f}s",
            state=state,
            expanded=False,
        )
    return answer, None


# ---- sidebar -------------------------------------------------------------------------------
settings = load_settings(CONFIG)
with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Data source", [s.name for s in settings.sources])
    b = backend(str(CONFIG), source)
    agent_cfg = b["settings"].agent
    if agent_cfg.provider == "openai_compatible":
        options = [agent_cfg.model] + [m for m in FREE_MODELS if m != agent_cfg.model]
        model = st.selectbox(
            "Model",
            options,
            help="Free tier: about 20 requests per model per day, and one "
            "question takes 3-5. Switch models when one runs out.",
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

    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example

# ---- main ----------------------------------------------------------------------------------
st.title("Ask your data")
st.caption(
    "Ask a business question in plain English. The AI explores the data with read-only "
    "queries, and every answer comes with the SQL behind it."
)

messages: list[dict[str, Any]] = st.session_state.setdefault("messages", [])
for m in messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["text"])
        elif "error" in m:
            st.error(m["error"])
        else:
            show_answer(AgentAnswer.model_validate(m["answer"]), m["model"])

question = st.chat_input("Ask a question about your data", disabled=catalog is None)
question = question or st.session_state.pop("pending", None)
if question and catalog is not None:
    messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer, error = ask(question, b, catalog, model)
        if answer:
            show_answer(answer, model)
            messages.append(
                {"role": "assistant", "answer": answer.model_dump(mode="json"), "model": model}
            )
        else:
            st.error(error)
            messages.append({"role": "assistant", "error": error})
