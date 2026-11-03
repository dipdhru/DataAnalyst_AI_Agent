# CLAUDE.md

Conventions for working on the DataAnalyst Agent. The product plan lives in `build-brief.md`;
read it before starting a new phase.

## Status

- Done: phase 0 (scaffold) and phase 1 (SQL connectors, schema catalog with column profiling,
  FastAPI endpoints).
- Phase 2 in progress: agent core built (tool loop, semantic layer, eval runner, `/ask`
  endpoint) and tested offline with a scripted fake model; not yet run against the live API.
  Golden questions: Claude may draft questions, SQL and answers, but an expected answer only
  counts once the user has approved it (definitions are approved; questions are still draft).
- Phase 3 started: Streamlit chat UI with live steps, charts, table and SQL. Not yet: catalog
  editing in the UI.
- Not yet: memory (phase 4), MongoDB (5), auth (6). Don't build ahead of the phase.
- Out of scope (user decision 2026-11-03): CSV/Excel uploads and the daily briefing. Both were
  removed; don't add them back unless the user asks.

### Deferred on purpose
- LLM-written table/column descriptions: still to do in phase 2. The store already supports
  `description_source="llm"`.
- Memory tables (conversations, glossary, verified_queries, preferences): phase 4. Put them in
  `state/app.db` next to the catalog tables.
- PII masking: phase 6. Until then, column profile `samples` store raw values.
- Audit log of every query and auth: phase 6.

## Commands

```bash
uv sync                                   # install (Python 3.11+)
uv run pytest -q                          # fast suite, uses sampled Olist data (~10s)
uv run pytest -q -m full_olist            # full Olist dataset
uv run ruff check . && uv run ruff format --check .
uv run uvicorn analyst.api.app:app --reload
docker compose up --build
uv run python scripts/build_golden.py     # recompute golden answers (--check: verify only)
uv run python scripts/setup_bird.py       # unpack BIRD Mini-Dev into data/bird (see docs/)
uv run pytest -q -m bird                  # BIRD gold SQL through run_query (~2 min)
uv run python scripts/run_eval.py --resume  # live eval, resumes daily on the free tier
uv run python scripts/run_bench.py --sample 20 --resume  # BIRD score, resumes daily
uv run python scripts/run_bench.py --groq openai/gpt-oss-120b --sample 20 --resume  # free, fast
uv run python scripts/import_bird_descriptions.py   # BIRD column descriptions -> catalogs
uv run streamlit run ui/app.py            # chat UI on http://localhost:8501
```

Before calling a change done: tests and ruff both pass.

## Layout

```
src/analyst/
  config.py          config.yaml + .env -> typed Settings (pydantic)
  models.py          shared shapes: SchemaInfo, TableInfo, QueryResult, to_jsonable
  sql_guard.py       sqlglot validator: one SELECT, no DDL/DML, blocked functions, known tables, LIMIT cap
  connectors/
    base.py          Connector ABC: get_schema(), run_query(); subclasses implement _introspect/_execute
    sql.py           SQLAlchemy connector (SQLite read-only URI, Postgres read-only session)
    registry.py      builds all configured connectors; looks them up by name
  catalog/
    profiler.py      column profiles with portable aggregate SQL (nulls, distinct, min/max, samples)
    relationships.py infers joins the DB doesn't declare (Olist declares none)
    store.py         persists catalogs in state/app.db (SQLite)
    service.py       refresh / edit descriptions; keeps user descriptions across refreshes
    importers.py     dataset-supplied descriptions (BIRD CSVs), stored as description_source="import"
  semantic.py        metrics.yaml loader: business definitions per source
  agent/
    tools.py         list_tables, describe_table, run_sql, submit_answer (strict schemas)
    prompt.py        system prompt: tables, definitions, rules (stable per source, cached)
    loop.py          manual tool loop; final SQL is executed before an answer is accepted
    providers.py     model clients: Claude (Anthropic SDK) or any OpenAI-compatible API
  evals/
    scoring.py       answer_match (tolerant) and execution_match (BIRD EX)
    runner.py        runs the agent over golden questions; cost per question
    bench.py         public benchmarks (BIRD): fixed sample, EX on full results, harness errors apart
    campaign.py      multi-day runs: saves each scored question, resumes after a quota stop
  charts.py          picks a Plotly chart (metric / line / bar) from a result table's shape
  api/app.py         FastAPI routes; create_app(settings, llm_client) for tests
ui/app.py            Streamlit chat UI (calls the agent in-process; tests: tests/test_ui.py)
tests/               pytest, against Olist in data/olist/; fake_claude.py scripts the model
evals/golden.yaml    golden questions: definitions, SQL, expected answers, draft/approved status
docs/benchmarks.md       plan for rating the agent on BIRD and Spider; free-tier budget
docs/bird-alignment.md   BIRD details: what distorts a score, step-by-step status
scripts/build_golden.py  computes expected answers from the SQL; never hand-edit them
```

## Rules

- **Agent tests never call the API.** Use `tests/fake_claude.py` to script replies. Live runs
  (`scripts/run_eval.py`) cost money: only run them when the user asks.
- **Providers:** `agent.provider` in `config.yaml`. The loop speaks the Anthropic Messages
  shape; `providers.OpenAICompatClient` translates for Gemini/Groq/OpenRouter/Ollama (free
  tiers, used while the user doesn't want to spend). Presets: `groq_settings` (Groq free
  plan, best free option for benchmarks) and `ollama_settings` (local; too slow on the user's
  8 GB M1 for the agent). Keep the loop provider-neutral.
- **Claude API usage:** model and effort come from `config.yaml` (`agent:`). No forced
  `tool_choice` (400 on Opus 5.5); tools use `strict: true`; history is append-only (append the
  full `response.content`); refusals use server-side `fallbacks: "default"`.

- **Every query goes through `Connector.run_query`.** It validates with `sql_guard`, clamps the
  row limit, enforces the timeout and converts values to JSON-safe types. Never call a
  connector's `_execute` or a raw engine from new code, including the profiler and the agent.
  The one exception is tests that deliberately bypass the guard to prove the connection-level
  defences hold (read-only SQLite).
- **Read-only, two layers.** The connection is read-only (SQLite `mode=ro`, Postgres
  `TRANSACTION READ ONLY`), and `sql_guard` rejects anything that isn't a single SELECT. Keep
  both; neither is enough alone. MongoDB (phase 5) needs the same two layers: a read-only user
  and its own guard that allows only read pipeline stages (no `$out`/`$merge`).
- **New data source = new `Connector` subclass** with `kind`, `dialect` (sqlglot name),
  `_introspect` and `_execute`. Register it in `SourceRegistry` and `SourceConfig`. MongoDB
  will not fit the SQL-only parts (sqlglot dialect, `sql_guard`); extend the interface then,
  not before.
- **Portable SQL in shared code.** Build identifiers with `sqlglot.exp` and render with the
  connector's dialect; the same code runs on SQLite and Postgres.
- **User edits win.** Descriptions with `description_source="user"` survive catalog refreshes.
  Imported ones (`"import"`, e.g. BIRD's CSVs) and LLM-written ones (`"llm"`) must never
  overwrite user ones.
- **Config:** settings in `config.yaml`, secrets only in `.env` (referenced via `url_env`).
  Relative paths resolve against the config file's directory. Runtime state goes in `state/`.

## Code style

- Python 3.11+, `from __future__ import annotations`, full type hints, pydantic v2 models for
  anything crossing an API or storage boundary.
- Ruff (line length 100, rules E F I B UP SIM). Format with `ruff format`.
- Module docstring saying what the module is for. Comments explain *why*, not *what*.
- Errors: raise specific exceptions (`UnsafeQueryError`, `QueryError`, `QueryTimeoutError`,
  `KeyError` for unknown names); map them to HTTP status codes only in `api/app.py`. Current
  mapping: `UnsafeQueryError` and `QueryError` 400, `QueryTimeoutError` 408, unknown
  source/table/column 404, catalog not built yet 409, model provider errors 429/502/503.
- No new dependency without a reason; add it with `uv add` so `uv.lock` stays in sync.

## Tests

- Use the fixtures in `tests/conftest.py`: `sample_sqlite` is the first 2000 rows of each
  Olist table, `settings` points at a temp `state/`. Tests skip cleanly when
  `data/olist/` is missing.
- Mark anything that needs the full dataset `@pytest.mark.full_olist`.
- Every guardrail change gets a test that tries to break it (see `tests/test_sql_guard.py`).
- Phase 2 onward: the golden question set runs as an eval on every agent change.

## Data

`data/olist/` (git-ignored, ~230 MB): `olist.sqlite` plus the 9 CSVs in `csv/`. SQLite tables
are named without the `olist_` prefix and `_dataset` suffix (`orders`, `order_items`, ...).
