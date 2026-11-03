# Rating the agent on public benchmarks: BIRD and Spider

Our own Olist golden set (`evals/golden.yaml`) checks *our* business definitions and stays the
product eval. Public benchmarks answer a different question: how good is the agent at
text-to-SQL in general, on databases it has never seen? BIRD details are in
[bird-alignment.md](bird-alignment.md).

## The three benchmarks

| | Spider 1.0 | BIRD Mini-Dev | Spider 2.0-Lite |
|---|---|---|---|
| Tests | SQL across many small, clean databases | Messy real data plus a per-question hint ("evidence") | Enterprise work: huge schemas, multi-step |
| Databases | SQLite | SQLite (also MySQL, Postgres) | BigQuery, Snowflake, and a local SQLite part |
| Metric | EX (and test-suite accuracy) | EX, Soft F1, R-VES | EX on result tables |
| Here | not downloaded | **in `data/bird/`, runner built** | not downloaded |
| Closest to our product | basic SQL skill | evidence = our `metrics.yaml` | where the product could go |

Check sizes, licences and official scripts in each repo before publishing a number:
Spider 1.0 at yale-lily.github.io/spider, BIRD at bird-sql.github.io and
github.com/bird-bench/mini_dev, Spider 2.0 at github.com/xlang-ai/Spider2.

## Budget: what the free tier allows

Gemini's free tier gives about 20 requests a day per model, and one benchmark question takes
about 4 to 6 model calls. That is **3 to 5 questions a day**. So:

- Score on a **fixed sample**, not the whole set. `pick_sample` (in `evals/bench.py`) is
  proportional by database and prefix-stable: the first 20 of the 50-question sample are the
  20-question sample. Start with 20, extend to 50 later without redoing anything.
- Every run resumes (`--resume`); a question is saved the moment it is scored.
- Never mix models in one score. The campaign file is per model and refuses another model.
- A paid model (Claude) would run 50 questions in minutes; ask before spending.

## Rules that keep the score honest

1. **Plumbing first.** Gold SQL must give the same result through `run_query` as raw
   `sqlite3` (BIRD: done, `pytest -m bird`). Otherwise we score our guard, not the agent.
2. **Score on full results.** The gold SQL and the agent's submitted SQL are both re-run with
   the benchmark's row cap (1,000,000), not the 1,000-row chat limit. A truncated gold result is
   a harness error, reported apart, never counted as a failure of the agent.
3. **Two scores.** Strict EX (BIRD's rule: equal sets of row tuples, exact values) for
   comparison with others; our tolerant answer match for spotting near misses (rounding, extra
   columns, percentages as fractions).
4. **Examples never come from the test split.** When few-shot retrieval arrives (phase 4),
   examples come from BIRD train, scores from Mini-Dev.
5. **The data is public.** Models have probably seen it. Use the score to compare our own
   versions; be careful claiming it against others.
6. **Read failures before blaming the agent.** Benchmarks have annotation errors.

## Plan

### BIRD (now)
1. Done: data, plumbing check, catalogs for the 11 databases.
2. **Done in this step: benchmark runner.** `scripts/run_bench.py` asks the agent each sampled
   question (question + evidence), re-runs its SQL, scores EX and answer match, saves progress
   daily and writes predictions in the official BIRD format.
3. Next: first score on 20 questions, then 50. Report EX overall, per difficulty, per database.
4. Next: import `database_description/*.csv` into the catalog (`description_source="import"`)
   and compare: with/without descriptions, with/without evidence (`--no-evidence`). These four
   numbers show what our catalog and semantic layer add.
5. Later: check our EX against the official `evaluation.py` on our predictions file.

### Spider 1.0 (after the first BIRD score)
1. Download the dev split (about 1,000 questions, 20 SQLite databases); needs your OK.
2. Add a loader next to `load_bird` (fields `db_id`, `question`, `query`; no evidence) and a
   generated `config.spider.yaml`; run the same plumbing test.
3. Same runner, same sampling. Expect a much higher score than BIRD: Spider is the warm-up.

### Spider 2.0-Lite (later)
1. Only the local SQLite part runs without cloud accounts; start there.
2. Its schemas are large, so it tests what our catalog sends the model (likely needs table
   retrieval instead of listing every table in the prompt).
3. Gold answers are result tables; scoring compares those, not SQL.

## Commands

```bash
uv run python scripts/run_bench.py --sample 20 --resume        # BIRD, a few questions a day
uv run python scripts/run_bench.py --sample 50 --resume        # extend the same run to 50
uv run python scripts/run_bench.py --sample 20 --resume --no-evidence   # ablation
uv run python scripts/run_bench.py --ids 0,7 --provider anthropic --model claude-haiku-5-5
```
