# Matching BIRD: notes and first steps

BIRD ("Big Bench for Large-scale Database Grounded Text-to-SQL") is the standard public benchmark
for the job our agent does: question in, SQL out, judged by executing it. Running our agent on
BIRD gives a number we can compare with published systems. Our own Olist golden set
(`evals/golden.yaml`) stays the product eval, because it checks *our* definitions.

Status: step 1 done (2026-11-02). Runner for step 4 built (2026-11-03, `scripts/run_bench.py`;
gold SQL submitted as the prediction scores 20/20 on the default sample); no agent score yet.
Agent scores (2026-11-06, Groq `openai/gpt-oss-120b`, with evidence, default sample of 20):
**11/20 EX (55%)** without descriptions, **12/20 EX (60%)** with imported descriptions. That gap
is within noise; details under "Step 4 results".
Trust check (2026-11-07, Cerebras `gpt-oss-120b`, same sample): 11/20 EX; 9 of 11 answers
labelled confident were right, and 2 of 4 wrong answers were labelled check. Details under
"Trust check results".
Step 2 done (2026-11-03): `scripts/import_bird_descriptions.py` describes 620 of 798 columns.
Steps 3 and 5 not started. Overall plan, including Spider: [benchmarks.md](benchmarks.md).

## Step 1 results

- Data: databases from `minidev.zip` (bird-bench README link, 801 MB), questions from the
  2026 Hugging Face revision (`data/mini_dev_sqlite.jsonl`, sha256 matches its manifest).
  `uv run python scripts/setup_bird.py` unpacks them and writes `config.bird.yaml`.
- The 2026 revision is mostly hard: 460 challenging, 32 moderate, 8 simple. Scores on it are
  not comparable with results on the original Mini-Dev.
- **Plumbing is faithful**: all 500 gold queries return the same result through `run_query` as
  through raw `sqlite3` (`uv run pytest -m bird`, about 2 minutes). The guard re-rendered the SQL
  text in 305 of them, so this is a real check. No result is empty; the largest has under 1,000
  rows (the 10,000-row cap is safe); the slowest gold query takes 16 s (under the 30 s timeout).
- Catalogs for all 11 databases build in about 35 s.
- Bug found and fixed: join inference produced 219 false joins on BIRD (`id` to `id`, `url` to
  `url`, members of composite keys, columns that already had declared FKs). It now infers 3,
  all correct. Rules and tests are in `catalog/relationships.py` and `tests/test_catalog.py`.

## Step 4 results: baseline

`run_bench.py --groq openai/gpt-oss-120b --sample 20 --no-descriptions` (with evidence, max 10
steps), run 2026-11-03 to 2026-11-04 on Groq's free plan (about one question an hour under its
200k tokens per day). Sample: 18 challenging, 1 moderate, 1 simple.

- **EX 11/20 (55%)**; the tolerant answer match agrees (11/20). Average 4.9 turns.
- Passed: 449, 48, 134, 170, 454, 58, 260, 412, 353, 234, 151.
- No answer (4):
  - 99, 54: conversation over Groq's 8k-token request cap (a free-plan limit, not the model).
  - 310: the model wrote an invalid tool call twice.
  - 496: no answer within 10 steps.
- Wrong result (5):
  - 263: right race, but the time as milliseconds instead of BIRD's `33:31.266` string.
  - 370: wrong value (32.25 instead of 77.4).
  - 222 (`financial`): 14 of 17 columns right; missed the cryptic `A11` column, which the
    imported descriptions explain. The with-descriptions run tests this.
  - 1: returned a different column than the gold SQL.
  - 327: 3 rows instead of 5.
- Raw results: `evals/runs/bench-bird-evidence-nodesc-openai-gpt-oss-120b.json` (git-ignored).

## Step 4 results: with descriptions vs without

Same sample, model and step limit, with the imported descriptions, finished 2026-11-06.

| | Without | With |
|---|---|---|
| EX | 11/20 | 12/20 |
| Hit the 10-step limit | 1 | 5 |
| Average turns | 5.0 | 6.4 |
| Tokens per question (those with a recorded count) | 12,118 | 15,030 |

- Gained (5): 263, 222, 99, 310, 54. Lost (4): 449, 454, 260, 234.
- Four of those nine swings are provider errors in one run only, not answers: 99 and 54
  (request too large, without), 310 and 454 (invalid tool call, 310 without, 454 with). On the
  other 16 questions it is 10/16 without and 9/16 with.
- Descriptions clearly helped on 222, the cryptic `A11` column, and probably on 263 (time format).
- Against that, the agent explores more: 449 took 2 turns without and hit the limit with, using
  53,247 tokens. 234 also passed without and hit the limit with.
- Conclusion: no measurable overall gain on this sample. 20 questions and one run per arm are
  too few to separate a small effect from noise. Untested: whether a higher step limit recovers
  the runs that hit it.
- Raw results: `evals/runs/bench-bird-evidence-desc-openai-gpt-oss-120b.json` (git-ignored).

## Trust check results

`run_bench.py --cerebras gpt-oss-120b --sample 20 --resume --trust` (with evidence and imported
descriptions, max 10 steps), run 2026-11-07 on Cerebras's free trial in about 50 minutes. Each
answered question gets a second, independent attempt (`agent/trust.py`): the two results agree =
`confident`, disagree = `check`. The score is always the first attempt's.

- **EX 11/20 (55%)**. Same model as the Groq runs but another provider, so treat it as close to,
  not equal to, the 12/20 with descriptions on Groq.
- 15 answered, 5 not (10-step limit: 449, 222, 496, 58, 353).

| | Right | Wrong |
|---|---|---|
| `confident` | 9 | 2 (370, 327) |
| `check` | 2 (260, 234) | 2 (263, 1) |

- Of confident answers, 9/11 (82%) were right, against 11/15 (73%) of all answers given.
- Of wrong answers, 2/4 were flagged. The two that weren't (370: wrong value; 327: 3 rows
  instead of 5) are mistakes both attempts made alike, the known limit of agreement checks.
- Of right answers, 2/11 were flagged (false alarms).
- Conclusion: the label points the right way, but 4 wrong answers are far too few to measure
  it. Next: a larger sample (50+) now that Cerebras makes a run take about an hour.
- Raw results: `evals/runs/bench-bird-evidence-desc-trust-gpt-oss-120b.json` (git-ignored).

## What BIRD is (check details against the repo before building)

- Question / SQL pairs over real SQLite databases, split into train, dev and a hidden test set.
- **Mini-Dev**: 500 pairs from 11 databases, in SQLite, MySQL and PostgreSQL. Cheap enough to
  run on every agent change. Canonical copy: Hugging Face `birdsql/bird_mini_dev`; code and
  scripts: https://github.com/bird-bench/mini_dev. Licence CC BY-SA 4.0.
- Every question comes with **evidence**: a short piece of external knowledge such as a formula or
  a value mapping ("late = delivered after estimated date"). Supplying it raises accuracy by
  roughly 20 points in the original paper. This is the same role our `definitions` play.
- Every database ships `database_description/<table>.csv`: column descriptions, data formats and
  value meanings. This is the same role our catalog descriptions play.
- Difficulty labels: `simple`, `moderate`, `challenging`.
- Metrics:
  - **EX (execution accuracy)**: the predicted SQL's result set equals the gold SQL's result set.
    The official script compares sets of row tuples, so row order and duplicates are ignored but
    column order and exact values are not. No numeric tolerance.
  - **Soft F1**: partial credit on the result table; tolerates column order and missing values.
  - **R-VES**: rewards correct SQL that runs about as fast as, or faster than, the gold SQL.
- Reference point: human performance is reported at about 93% EX. Published system scores move
  fast and differ by split; read them from the official leaderboard, not from papers.

## How we line up today

| BIRD | Ours | Gap |
|---|---|---|
| SQLite databases | `SqlAlchemyConnector` (sqlite) | none: add each BIRD db as a source |
| PostgreSQL version of Mini-Dev | `SqlAlchemyConnector` (postgres) | untested; good first real Postgres test |
| `database_description/*.csv` | catalog descriptions | need an importer |
| evidence per question | `definitions` + `uses` in golden.yaml | need an `evidence` text field |
| simple / moderate / challenging | easy / medium / hard | rename to BIRD's labels |
| one final SQL per question | agent is conversational | need a non-interactive "SQL only" mode |
| EX, Soft F1, R-VES | exact match on rounded expected rows | need a scorer with BIRD semantics |
| gold SQL runs unmodified | `run_query` re-renders SQL and adds LIMIT | must prove this doesn't change results |

## Things in our design that would distort a BIRD score

1. **LIMIT injection.** `run_query` caps rows (`max_row_limit`, 10,000). A gold query that returns
   more rows would then "fail" EX. The benchmark runner must score on full results: run with a
   row cap above any gold result size, and treat `truncated: true` as a harness error, not a
   wrong answer.
2. **sqlglot re-rendering.** The guard parses and re-emits the SQL. For SQLite that is normally
   identical in meaning, but backtick identifiers, integer division, `||`, `CAST` and date
   functions are where transpilers slip. Run every gold SQL both raw and through `run_query` and
   require identical results (see step 1).
3. **Table allowlist and blocked functions.** BIRD uses tables and columns with spaces and mixed
   case (quoted with backticks or double quotes). The allowlist lower-cases names; check that
   every gold query passes the guard.
4. **Timeouts.** Default 30 s. A few BIRD databases are large; a gold query timing out is a harness
   problem. Use a benchmark-specific timeout and report timeouts separately.
5. **Rounding in our golden answers.** Our Olist SQL rounds (`ROUND(x, 2)`). Under strict EX an
   agent that returns 13494400.7412 instead of 13494400.74 fails. Keep two scores: BIRD-strict EX,
   and our own "answer match" (numeric tolerance, column order ignored, close to Soft F1).

## First steps, in order

### 1. Load Mini-Dev and prove the plumbing is faithful (no agent needed)
- Download Mini-Dev (SQLite) into `data/bird/` (gitignored, like Olist).
- Add each of the 11 databases as a source (one `config.bird.yaml`, so the main config stays
  small).
- New test `tests/test_bird_plumbing.py`, marked `bird` and skipped when data is missing: for every
  gold SQL, check that `run_query` accepts it and returns the same result set as a raw `sqlite3`
  execution. Any failure is a guard or connector bug to fix before scoring anything.
- Build catalogs for all 11 databases and time them (profiling uses `max_rows` sampling).

### 2. Import BIRD's descriptions into the catalog
- `scripts/import_bird_descriptions.py`: read `database_description/*.csv` and write column
  descriptions and value meanings into the catalog store.
- Store them with a new `description_source = "import"`, so they never overwrite user edits and
  can be switched off for ablations.

### 3. Give the eval one format for both datasets
- Golden format gains `evidence` (free text, built from the `definitions` a question `uses`) and
  `db_id` (the source name); difficulty uses `simple` / `moderate` / `challenging`.
- A converter loads BIRD's question JSON into the same structure, so one runner handles Olist and
  BIRD.
- `src/analyst/evals/scoring.py` implements EX (set of tuples, BIRD semantics), Soft F1 and our
  tolerant answer match. Sanity check: gold vs gold scores 100%, and our EX agrees with the
  official `evaluation.py` on a sample of predictions.

### 4. Run the agent on it (phase 2, once the agent exists)
- Add a non-interactive mode: question + evidence in, one final SQL out, no clarifying questions.
- `scripts/run_bird.py` writes predictions in the official format so the official scripts can
  score them too. Our scorer is for fast iteration; the official one is for numbers we publish.
- Report EX overall and per difficulty, and four variants: with/without evidence and
  with/without imported descriptions. These show what our catalog and semantic layer add.
- Track the score per agent change alongside the Olist golden set.

### 5. Later
- PostgreSQL Mini-Dev, against the Docker Compose Postgres: the first end-to-end Postgres test.
- R-VES once correctness is stable.
- Few-shot retrieval from the BIRD train split, the same mechanism as phase 4 "verified queries".

## Caveats

- BIRD dev and Mini-Dev are public, and current models have probably seen them. Use the score to
  compare our own versions, and be careful claiming it against others.
- BIRD has known annotation errors (the 2025 Mini-Dev revision fixed some). Look at failures
  before concluding the agent was wrong.
- BIRD measures one-shot SQL. Our product also needs clarifying questions, explanations and
  refusals (like `x_profit_margin`), which BIRD doesn't measure; the Olist set must keep covering
  those.
