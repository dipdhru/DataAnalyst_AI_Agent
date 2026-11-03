# DataAnalyst Agent: Build Brief

## Goal
A plug-and-play data analyst chatbot for databases. Connect an existing SQL database (later MongoDB), ask business questions in chat, get answers with the query used, tables and charts. It keeps its own memory of verified answers.

## Confirmed decisions (v1)
- Deployment: single-user, local tool (Docker Compose). Multi-user hosting comes later.
- Data sources: SQL databases: Postgres (live DB) and SQLite (Olist test data). MongoDB later.
- Out of scope (removed 2026-11-03 to focus on chat over databases): CSV/Excel uploads, daily briefing.
- UI: Streamlit for the MVP. Next.js later.
- LLM: Claude API with tool use.
- Test dataset: Olist (9 CSVs plus olist.sqlite, in `data/olist/`).

## Stack
- Backend: Python 3.11+, FastAPI
- Data access: SQLAlchemy for SQL databases with read-only credentials. MongoDB later (read-only aggregation pipelines, same connector interface)
- SQL safety: sqlglot parse, SELECT-only allowlist, row limit, query timeout
- Charts: Plotly (JSON spec returned to UI)
- Memory store: SQLite. Tables: conversations, facts/glossary, verified_queries, preferences
- Packaging: Docker Compose, one `config.yaml` plus `.env`

## Architecture
1. Connectors: every database connector exposes one interface (`get_schema`, `run_query`).
2. Catalog: introspect tables/columns/keys, profile columns (nulls, distinct, min/max, samples), LLM-written table and column descriptions, editable by the user.
3. Semantic layer: `metrics.yaml` with business definitions (revenue, churn, AOV). The agent must use these before inventing its own SQL.
4. Agent tools: `list_tables`, `describe_table`, `run_sql`, `make_chart`, `remember`, `recall`. Loop: plan, query, check result, answer, show SQL used.
5. Memory: store glossary terms, corrections from the user, and queries the user marked as correct. Retrieve relevant ones into the prompt each turn.
6. Guardrails: read-only DB user, PII column masking, audit log of every query, per-query cost/row caps.

## Build phases
0. Scaffold: repo, CLAUDE.md, folder layout, Olist test data, Docker skeleton.
1. Connectors and catalog (Postgres/SQLite).
2. Agent core with tool loop, SQL validator, and a golden test set of 20-30 questions with expected answers (run as an eval on every change).
3. Chat UI (Streamlit) with tables, charts, and "show SQL".
4. Memory (glossary, corrections, verified queries).
5. MongoDB connector: schema sampling for collections, read-only aggregation queries, its own guard.
6. Hardening and packaging: auth, secrets, Docker Compose, README, demo video.

## User to-do
Write the golden question set (20-30 Olist questions with correct answers) before phase 2. Examples: top categories by revenue, late-delivery rate by state, repeat-customer share.
