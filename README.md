# DataAnalyst Agent

Plug-and-play data analyst chatbot. Connect a database or upload CSV/Excel, ask business
questions, get answers with SQL, tables and charts. See [build-brief.md](build-brief.md) for
the full plan.

Current state: **phases 0–1 done, phase 2 in progress**. Connectors, schema catalog with column
profiling, a FastAPI backend, and the Claude agent core (not yet evaluated live). No UI yet.

## Quick start

```bash
uv sync
cp .env.example .env            # add keys / DB URLs as needed
uv run uvicorn analyst.api.app:app --reload
```

Open http://localhost:8000/docs, then:

1. `POST /sources/olist/catalog/refresh` builds the catalog (about 7 s for full Olist).
2. `GET /sources/olist/catalog` returns tables, column profiles and inferred joins.
3. `POST /sources/olist/query` with `{"sql": "SELECT ..."}` runs a read-only query.
4. `POST /uploads` (multipart `file`) turns a CSV/Excel/Parquet file into tables in the
   `uploads` source.
5. `POST /sources/olist/ask` with `{"question": "What is our late-delivery rate?"}` asks the
   agent. Needs Claude credentials: `ANTHROPIC_API_KEY` in `.env`.

## Agent eval

`uv run python scripts/run_eval.py` runs the agent on the golden questions in
`evals/golden.yaml` and scores each answer (calls the Claude API, so it costs money; start with
`--ids rev_total,del_late_rate`). Results are saved in `evals/runs/`.

## Olist test data

Tests and the default config expect the Olist dataset in `data/olist/`:

```
data/olist/olist.sqlite
data/olist/csv/*.csv     # the 9 Olist CSVs
```

It is gitignored (~230 MB). Get it from Kaggle ("Brazilian E-Commerce Public Dataset by Olist"
and the "olist.sqlite" variant) or copy it from an existing checkout.

## Configuration

`config.yaml` sets sources, row limits, timeouts and profiling. Secrets stay in `.env`.
To add Postgres, use a **read-only** database user:

```yaml
sources:
  - name: warehouse
    type: postgres
    url_env: ANALYST_PG_URL   # postgresql+psycopg://analyst_ro:...@host:5432/db
    schemas: [public]
```

## Tests

```bash
uv run pytest                  # fast, on Olist samples
uv run pytest -m full_olist    # full dataset
```

## Docker

```bash
docker compose up --build
```
