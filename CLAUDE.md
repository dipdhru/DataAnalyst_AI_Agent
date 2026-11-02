# CLAUDE.md

Conventions for working on the DataAnalyst Agent. The product plan lives in `build-brief.md`;
read it before starting a new phase.

## Status

- Done: phase 0 (scaffold) and phase 1 (connectors, DuckDB file ingestion, schema catalog with
  column profiling, FastAPI endpoints).
- Next: phase 2 (agent core: tool loop, golden question eval). It needs the golden question
  set first. Claude may draft questions, SQL and answers from the data, but an expected answer
  only counts once the user has reviewed and approved it (including the metric definitions
  behind it, e.g. what "revenue" or "late delivery" means).
- Not yet: UI (Streamlit, phase 3), memory, briefing, auth. Don't build ahead of the phase.

### Deferred on purpose
- LLM-written table/column descriptions: phase 2, once the Claude client exists. The store
  already supports `description_source="llm"`.
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
    files.py         DuckDB connector for uploaded CSV/TSV/Excel/Parquet, one table per file/sheet
    registry.py      builds all configured connectors; looks them up by name
  catalog/
    profiler.py      column profiles with portable aggregate SQL (nulls, distinct, min/max, samples)
    relationships.py infers joins the DB doesn't declare (Olist declares none)
    store.py         persists catalogs in state/app.db (SQLite)
    service.py       refresh / edit descriptions; keeps user descriptions across refreshes
  api/app.py         FastAPI routes; create_app(settings) for tests
tests/               pytest, against Olist in data/olist/
evals/golden.yaml    golden questions: definitions, SQL, expected answers, draft/approved status
docs/bird-alignment.md   plan for scoring the agent on the BIRD text-to-SQL benchmark
scripts/build_golden.py  computes expected answers from the SQL; never hand-edit them
```

## Rules

- **Every query goes through `Connector.run_query`.** It validates with `sql_guard`, clamps the
  row limit, enforces the timeout and converts values to JSON-safe types. Never call a
  connector's `_execute` or a raw engine from new code, including the profiler and the agent.
  The one exception is tests that deliberately bypass the guard to prove the connection-level
  defences hold (read-only SQLite, DuckDB file access blocked).
- **Read-only, two layers.** The connection is read-only (SQLite `mode=ro`, Postgres
  `TRANSACTION READ ONLY`, DuckDB `enable_external_access=False`), and `sql_guard` rejects
  anything that isn't a single SELECT. Keep both; neither is enough alone.
- **DuckDB never reads files from SQL.** Ingestion reads files on a separate in-memory staging
  connection and hands over Arrow tables. Don't enable external access on the main connection.
- **New data source = new `Connector` subclass** with `kind`, `dialect` (sqlglot name),
  `_introspect` and `_execute`. Register it in `SourceRegistry` and `SourceConfig`.
- **Portable SQL in shared code.** Build identifiers with `sqlglot.exp` and render with the
  connector's dialect; the same code runs on SQLite, Postgres and DuckDB.
- **Uploaded names are sanitized** with `sanitize_identifier` (lowercase snake_case). Report
  renames back to the caller (`IngestResult.renamed_columns`).
- **User edits win.** Descriptions with `description_source="user"` survive catalog refreshes.
  LLM-written ones (phase 2+) use `"llm"` and must never overwrite user ones.
- **Config:** settings in `config.yaml`, secrets only in `.env` (referenced via `url_env`).
  Relative paths resolve against the config file's directory. Runtime state goes in `state/`.

## Code style

- Python 3.11+, `from __future__ import annotations`, full type hints, pydantic v2 models for
  anything crossing an API or storage boundary.
- Ruff (line length 100, rules E F I B UP SIM). Format with `ruff format`.
- Module docstring saying what the module is for. Comments explain *why*, not *what*.
- Errors: raise specific exceptions (`UnsafeQueryError`, `QueryError`, `QueryTimeoutError`,
  `UnsupportedFileError`, `TableExistsError`, `KeyError` for unknown names); map them to HTTP
  status codes only in `api/app.py`. Current mapping: `UnsafeQueryError` and `QueryError` 400,
  `QueryTimeoutError` 408, unknown source/table/column 404, `TableExistsError` 409,
  `UnsupportedFileError` 415, upload over `max_upload_mb` 413.
- No new dependency without a reason; add it with `uv add` so `uv.lock` stays in sync.

## Tests

- Use the fixtures in `tests/conftest.py`: `sample_sqlite` and `sample_csv_dir` are the first
  2000 rows of each Olist table, `settings` points at a temp `state/`. Tests skip cleanly when
  `data/olist/` is missing.
- Mark anything that needs the full dataset `@pytest.mark.full_olist`.
- Every guardrail change gets a test that tries to break it (see `tests/test_sql_guard.py`).
- Phase 2 onward: the golden question set runs as an eval on every agent change.

## Data

`data/olist/` (git-ignored, ~230 MB): `olist.sqlite` plus the 9 CSVs in `csv/`. SQLite tables
are named without the `olist_` prefix and `_dataset` suffix (`orders`, `order_items`, ...).
