"""Score the agent on the BIRD Mini-Dev benchmark (Spider later). See docs/benchmarks.md.

    uv run python scripts/run_bench.py --sample 20 --resume       # free tier: run daily
    uv run python scripts/run_bench.py --sample 50 --resume       # grow the same run to 50
    uv run python scripts/run_bench.py --sample 20 --resume --no-evidence   # ablation
    uv run python scripts/run_bench.py --sample 20 --resume --no-descriptions   # ablation
    uv run python scripts/run_bench.py --ids 0,7 --provider anthropic --model claude-haiku-5-5
    uv run python scripts/run_bench.py --ollama qwen3:4b     # local model: free, no quota
    uv run python scripts/run_bench.py --groq openai/gpt-oss-120b   # Groq free plan

The model comes from config.yaml (`agent:`); databases and limits from config.bird.yaml
(written by scripts/setup_bird.py). --resume keeps progress in
evals/runs/bench-bird-<variant>-<model>.json and stops cleanly when the daily quota runs out.
Calls the model provider: on Claude this costs money, on a free tier it is rate-limited.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import anthropic

from analyst.agent import Agent, ToolCall
from analyst.agent.providers import (
    ProviderError,
    groq_settings,
    make_llm_client,
    ollama_settings,
)
from analyst.catalog import Catalog, CatalogService, CatalogStore
from analyst.catalog.importers import without_imported
from analyst.config import load_settings
from analyst.connectors import SourceRegistry
from analyst.evals.bench import (
    BenchQuestion,
    BenchResult,
    bird_predictions,
    load_bird,
    pick_sample,
    run_bench,
    summarize_bench,
)
from analyst.evals.campaign import Campaign, CampaignMismatchError, file_tag

ROOT = Path(__file__).resolve().parents[1]
BIRD_QUESTIONS = ROOT / "data" / "bird" / "questions" / "mini_dev_sqlite.jsonl"
BIRD_CONFIG = ROOT / "config.bird.yaml"
RUNS = ROOT / "evals" / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the agent on BIRD Mini-Dev")
    parser.add_argument("--sample", type=int, default=20, help="questions in the fixed sample")
    parser.add_argument("--seed", type=int, default=0, help="sample seed (keep it fixed)")
    parser.add_argument("--ids", help="comma-separated question ids instead of a sample")
    parser.add_argument("--no-evidence", action="store_true", help="hide BIRD's hints")
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="hide column descriptions imported from BIRD (scripts/import_bird_descriptions.py)",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="ask each answered question twice and label it confident/check (twice the tokens)",
    )
    parser.add_argument("--provider", choices=["anthropic", "openai_compatible"])
    parser.add_argument("--model", help="override agent.model from config.yaml")
    parser.add_argument("--ollama", metavar="MODEL", help="use a local Ollama model")
    parser.add_argument("--groq", metavar="MODEL", help="use Groq (GROQ_API_KEY in .env)")
    parser.add_argument("--think", action="store_true", help="with --ollama: keep thinking on")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="model calls per question before giving up (keeps one hard question from"
        " using a whole day's free quota)",
    )
    parser.add_argument("--resume", action="store_true", help="continue a multi-day run")
    parser.add_argument("--restart", action="store_true", help="with --resume: start over")
    args = parser.parse_args()

    if not BIRD_QUESTIONS.exists() or not BIRD_CONFIG.exists():
        print("BIRD data missing: see docs/bird-alignment.md, then run scripts/setup_bird.py")
        return 2
    everything = load_bird(BIRD_QUESTIONS)
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in everything if q.id in wanted]
        if missing := wanted - {q.id for q in questions}:
            print(f"unknown ids: {sorted(missing)}")
            return 2
    else:
        questions = pick_sample(everything, args.sample, seed=args.seed)

    agent_settings = load_settings(ROOT / "config.yaml").agent
    if args.provider:
        agent_settings.provider = args.provider
    if args.model:
        agent_settings.model = args.model
    if args.groq:
        agent_settings = groq_settings(agent_settings, args.groq)
    if args.ollama:
        agent_settings = ollama_settings(agent_settings, args.ollama, think=args.think)
    agent_settings.max_steps = args.max_steps
    bird = load_settings(BIRD_CONFIG)
    row_cap = bird.query.max_row_limit
    variant = "noevidence" if args.no_evidence else "evidence"
    variant += "-nodesc" if args.no_descriptions else "-desc"
    variant += "-trust" if args.trust else ""

    campaign = None
    if args.resume:
        path = RUNS / f"bench-bird-{variant}-{file_tag(agent_settings.model)}.json"
        if args.restart:
            path.unlink(missing_ok=True)
        try:
            campaign = Campaign(
                path,
                model=agent_settings.model,
                provider=agent_settings.provider,
                result_type=BenchResult,
            )
        except CampaignMismatchError as e:
            print(e)
            return 2
        total = len(questions)
        questions = campaign.remaining(questions)
        print(f"resuming {path.relative_to(ROOT)}: {total - len(questions)}/{total} done")
        if not questions:
            return _report(campaign.results(), agent_settings.model, variant)

    registry = SourceRegistry(bird, only={q.db_id for q in questions})
    try:
        service = CatalogService(registry, CatalogStore(bird.app_db_path), bird.profiling)
        catalogs: dict[str, Catalog] = {}

        def catalog_for(db: str) -> Catalog:
            if db not in catalogs:
                catalog = service.get(db) or service.refresh(db)
                catalogs[db] = without_imported(catalog) if args.no_descriptions else catalog
            return catalogs[db]

        try:
            client = make_llm_client(agent_settings, on_wait=lambda m: print(f"      ... {m}"))
        except ProviderError as e:
            print(e.message)
            return 1

        def show_step(step: ToolCall) -> None:
            detail = f"{step.row_count} rows" if step.row_count is not None else ""
            if not step.ok:
                detail = f"error: {(step.error or '')[:80]}"
            print(f"      {step.tool:15} {detail}", flush=True)

        def show_question(q: BenchQuestion) -> None:
            print(f"\n[{q.id} {q.db_id}, {q.difficulty}] {q.question[:110]}", flush=True)

        def show(r: BenchResult) -> None:
            mark = "HARNESS" if r.harness_error else ("PASS" if r.passed else "FAIL")
            label = f"  [{r.trust.label}]" if r.trust else ""
            print(f"{mark} {r.id:>5} {r.agent.turns:2} turns  {r.reason[:100]}{label}")
            if campaign:
                campaign.add(r)

        print(
            f"{len(questions)} BIRD questions ({variant}), "
            f"{agent_settings.provider} {agent_settings.model}"
        )
        try:
            results = run_bench(
                questions,
                lambda db: Agent(
                    client, registry.get(db), catalog_for(db), agent_settings, on_step=show_step
                ),
                registry.get,
                model=agent_settings.model,
                row_cap=row_cap,
                with_evidence=not args.no_evidence,
                with_trust=args.trust,
                on_result=show,
                on_question=show_question,
            )
        except ProviderError as e:
            if campaign and "daily free quota" in e.message:
                print(f"\nStopped for today: {e.message}.")
                print("Progress saved. Run the same command tomorrow to continue.")
                _print_summary(campaign.results(), partial=True)
                return 0
            print(f"Model provider error {e.status_code}: {e.message}")
            return 1
        except anthropic.APIStatusError as e:
            print(f"Claude API error {e.status_code}: {e.message}")
            return 1
        except anthropic.APIConnectionError:
            print("Claude API: could not connect")
            return 1
    finally:
        registry.close()

    return _report(campaign.results() if campaign else results, agent_settings.model, variant)


def _print_summary(results: list[BenchResult], *, partial: bool) -> None:
    label = "Score so far" if partial else "Score"
    print(f"\n{label}:")
    print(json.dumps(summarize_bench(results), indent=2))


def _report(results: list[BenchResult], model: str, variant: str) -> int:
    _print_summary(results, partial=False)
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RUNS / f"bench-bird-{variant}-{file_tag(model)}-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "benchmark": "bird-mini-dev-2026",
                "variant": variant,
                "model": model,
                "summary": summarize_bench(results),
                "results": [r.model_dump(mode="json") for r in results],
            },
            indent=2,
        )
    )
    preds = RUNS / f"bird-predictions-{variant}-{file_tag(model)}.json"
    preds.write_text(json.dumps(bird_predictions(results), indent=2))
    print(f"saved {out.relative_to(ROOT)} and {preds.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
