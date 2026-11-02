"""Run the agent on the Olist golden questions and score it.

Calls the model provider set in config.yaml (`agent:`). On Claude this costs money; on a free
tier (Gemini) it is rate-limited instead.

    uv run python scripts/run_eval.py --ids rev_total,del_late_rate   # a couple first
    uv run python scripts/run_eval.py                                 # every question
    uv run python scripts/run_eval.py --status approved               # approved only
    uv run python scripts/run_eval.py --provider anthropic --model claude-haiku-5-5
    uv run python scripts/run_eval.py --resume    # free tier: run daily until all are done

--resume keeps progress in evals/runs/campaign-<model>.json. Each run asks only the questions
not answered yet and stops cleanly when the daily free quota runs out.

Needs the provider's key in .env (GEMINI_API_KEY, or ANTHROPIC_API_KEY for Claude) and a
built catalog for the golden set's source. Results go to evals/runs/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import anthropic
import yaml

from analyst.agent import Agent, ToolCall
from analyst.agent.providers import ProviderError, make_llm_client
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import load_settings
from analyst.connectors import SourceRegistry
from analyst.evals.campaign import Campaign, CampaignMismatchError, campaign_path
from analyst.evals.runner import EvalResult, run_eval, summarize
from analyst.semantic import load_semantics

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the agent on evals/golden.yaml")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--status", choices=["all", "approved"], default="all")
    parser.add_argument("--golden", default=str(ROOT / "evals" / "golden.yaml"))
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai_compatible"],
        help="override agent.provider from config.yaml",
    )
    parser.add_argument("--model", help="override agent.model from config.yaml")
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="override agent.effort from config.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a multi-day run, skipping questions already answered",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="with --resume: throw away saved progress and start over",
    )
    args = parser.parse_args()

    golden = yaml.safe_load(Path(args.golden).read_text())
    questions = golden["questions"]
    if args.status == "approved":
        questions = [q for q in questions if q["status"] == "approved"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            print(f"unknown ids: {sorted(missing)}")
            return 2
    if not questions:
        print("no questions selected")
        return 2

    settings = load_settings(ROOT / "config.yaml")
    if args.provider:
        settings.agent.provider = args.provider
    if args.model:
        settings.agent.model = args.model
    if args.effort:
        settings.agent.effort = args.effort
    out_dir = ROOT / "evals" / "runs"
    campaign = None
    if args.resume:
        path = campaign_path(out_dir, settings.agent.model)
        if args.restart:
            path.unlink(missing_ok=True)
        try:
            campaign = Campaign(path, model=settings.agent.model, provider=settings.agent.provider)
        except CampaignMismatchError as e:
            print(e)
            return 2
        total = len(questions)
        questions = campaign.remaining(questions)
        print(f"resuming {path.relative_to(ROOT)}: {total - len(questions)}/{total} done")
        if not questions:
            return _report(campaign.results(), settings, out_dir, final=True)

    source = golden["source"]
    registry = SourceRegistry(settings, only={source})
    try:
        service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
        catalog = service.get(source) or service.refresh(source)
        semantics = load_semantics(settings.metrics_path).get(source)
        try:
            client = make_llm_client(settings.agent, on_wait=lambda m: print(f"      ... {m}"))
        except ProviderError as e:
            print(e.message)
            return 1
        connector = registry.get(source)

        def show_step(step: ToolCall) -> None:
            detail = f"{step.row_count} rows" if step.row_count is not None else ""
            if not step.ok:
                detail = f"error: {(step.error or '')[:80]}"
            print(f"      {step.tool:15} {detail}", flush=True)

        def show(r: EvalResult) -> None:
            mark = "PASS" if r.passed else "FAIL"
            print(f"{mark} {r.id:24} {r.agent.turns:2} turns ${r.cost_usd or 0:.3f}  {r.reason}")
            if campaign:
                campaign.add(r)

        effort = (
            f", effort {settings.agent.effort}" if settings.agent.provider == "anthropic" else ""
        )
        print(
            f"{len(questions)} questions, {settings.agent.provider} {settings.agent.model}{effort}"
        )
        try:
            results = run_eval(
                questions,
                lambda: Agent(
                    client, connector, catalog, settings.agent, semantics, on_step=show_step
                ),
                model=settings.agent.model,
                on_result=show,
                on_question=lambda q: print(f"\n[{q['id']}] {q['question']}", flush=True),
            )
        except ProviderError as e:
            if campaign and "daily free quota" in e.message:
                done = len(campaign.data["results"])
                print(f"\nStopped for today: {e.message}.")
                print(
                    f"Progress saved: {done} questions done, {len(golden['questions']) - done}"
                    " to go. Run the same command tomorrow to continue."
                )
                return 0
            print(f"Model provider error {e.status_code}: {e.message}")
            if campaign:
                print("Progress so far is saved; run the same command again to continue.")
            return 1
        except anthropic.AuthenticationError:
            print("Claude API: invalid or missing key (ANTHROPIC_API_KEY in .env)")
            return 1
        except anthropic.APIStatusError as e:
            print(f"Claude API error {e.status_code}: {e.message}")
            return 1
        except anthropic.APIConnectionError:
            print("Claude API: could not connect")
            return 1
    finally:
        registry.close()

    if campaign:
        return _report(campaign.results(), settings, out_dir, final=True)
    return _report(results, settings, out_dir, final=False)


def _report(results: list[EvalResult], settings, out_dir: Path, *, final: bool) -> int:
    summary = summarize(results)
    if final:
        print("\nAll questions answered.")
    print(json.dumps(summary, indent=2))
    out_dir.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "model": settings.agent.model,
                "provider": settings.agent.provider,
                "summary": summary,
                "results": [r.model_dump(mode="json") for r in results],
            },
            indent=2,
        )
    )
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
