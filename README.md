# DataAnalyst Agent

Ask a database a business question in plain English and get an answer with the SQL, a table and
a chart. The agent explores the schema with read-only queries, runs its own SQL, and re-runs the
final query before it answers, so every number comes with the query that produced it.

**What's different: it tells you when not to trust it.** A wrong answer looks just like a right
one. With "Double-check answers" on, the agent answers each question a second, independent time
and compares the two results. Agreement gets a ✅ *Double-checked* label; disagreement gets
⚠️ *Double-check this*. How much that label is worth is measured on a public benchmark, below,
not assumed.

![The chat answering "Which 5 product categories bring in the most revenue?" with a Double-checked label, the answer and a bar chart](docs/images/chat-double-checked.jpg)

Built as a learning and portfolio project. The plan and the phases are in
[build-brief.md](build-brief.md).

## Results

Measured on [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) (2026 revision), the standard
benchmark for this job: a fixed sample of 20 questions (18 challenging), scored by execution
accuracy (EX: the result must match the reference query's result). Model: `gpt-oss-120b`, on free
tiers. Details and per-question notes: [docs/bird-alignment.md](docs/bird-alignment.md).

| Run | Provider | EX |
|---|---|---|
| Without column descriptions | Groq | 11/20 (55%) |
| With BIRD's column descriptions | Groq | 12/20 (60%) |
| With descriptions and the trust check | Cerebras | 11/20 (55%) |

The trust check, on the 15 questions it answered:

| | Right | Wrong |
|---|---|---|
| ✅ Double-checked | 9 | 2 |
| ⚠️ Double-check this | 2 | 2 |

- Answers labelled double-checked were right 9 times out of 11 (82%), against 11 out of 15 (73%)
  for all answers.
- Half the wrong answers were flagged. The two that weren't are mistakes both attempts made the
  same way, which agreement can't catch.

**Read these as early numbers.** 20 questions is a small sample (4 wrong answers is too few to
measure a flag rate), one run per setup, and public benchmarks may have been seen by the model in
training. The differences between the first three rows are within noise.

## How it works

1. **Catalog.** Reads the schema, profiles every column (nulls, distinct values, ranges, samples),
   infers joins the database doesn't declare, and keeps descriptions: yours, imported, or written
   by the model for gaps. Your edits always win.
2. **Agent.** A tool loop with four tools: `list_tables`, `describe_table`, `run_sql` and
   `submit_answer`. Business definitions (`metrics.yaml`) and memory (answers you marked correct,
   a glossary, corrections) go into the prompt.
3. **Check.** The submitted SQL is executed again before the answer is accepted; with the trust
   check, a second attempt is compared with the first ([trust.py](src/analyst/agent/trust.py)).

**Safety: read-only in two layers.** The connection itself is read-only (SQLite `mode=ro`,
Postgres `READ ONLY` transactions), and a SQL guard built on sqlglot rejects anything but a single
`SELECT`, blocks risky functions, allows only known tables and caps the row count. Either layer
alone would be one bug away from a write.

**Any model.** Claude through the Anthropic SDK, or any OpenAI-compatible API through one adapter:
Gemini, Groq, Cerebras, NVIDIA, OpenRouter or a local Ollama.

## Quick start

```bash
uv sync
cp .env.example .env            # add an API key, e.g. GEMINI_API_KEY (free)
uv run streamlit run ui/app.py  # chat UI on http://localhost:8501
```

The default config uses Gemini's free tier, about 20 requests a day per model: switch models in
the sidebar when one runs out. Turning on "Double-check answers" uses about twice the requests.

The chat shows each step live, then the answer, a chart when one fits, the result table and the
SQL. Mark an answer ✅ correct or ✏️ fix it to teach the agent; conversations are saved.

### API

```bash
uv run uvicorn analyst.api.app:app --reload   # http://localhost:8000/docs
```

1. `POST /sources/olist/catalog/refresh` builds the catalog (about 7 s for full Olist).
2. `GET /sources/olist/catalog` returns tables, column profiles and inferred joins.
3. `POST /sources/olist/query` with `{"sql": "SELECT ..."}` runs a read-only query.
4. `POST /sources/olist/ask` with `{"question": "What is our late-delivery rate?"}` asks the agent.

## Data

Tests and the default config use the Olist e-commerce dataset in `data/olist/olist.sqlite`
(git-ignored, about 230 MB). Get it from Kaggle ("Brazilian E-Commerce Public Dataset by Olist",
the `olist.sqlite` variant).

To add Postgres, use a **read-only** database user:

```yaml
sources:
  - name: warehouse
    type: postgres
    url_env: ANALYST_PG_URL   # postgresql+psycopg://analyst_ro:...@host:5432/db
    schemas: [public]
```

Settings live in `config.yaml`; secrets only in `.env`.

## Benchmarks and evals

```bash
uv run python scripts/setup_bird.py   # unpack BIRD Mini-Dev into data/bird (see docs/)
uv run python scripts/run_bench.py --cerebras gpt-oss-120b --sample 20 --resume --trust
uv run python scripts/run_bench.py --groq openai/gpt-oss-120b --sample 20 --resume
uv run python scripts/run_eval.py --resume   # Olist golden questions (evals/golden.yaml)
```

`--resume` saves each scored question, so a run stopped by a daily free quota continues the next
day. Benchmarks run without memory, so scores stay comparable.

## Tests

```bash
uv run pytest -q                 # fast, on Olist samples; the model is scripted, no API calls
uv run pytest -q -m full_olist   # full dataset
uv run pytest -q -m bird         # every BIRD gold query through the guard (~2 min)
```

## Docker

```bash
docker compose up --build
```

## Not built yet

MongoDB (phase 5) and authentication, PII masking and an audit log (phase 6).
