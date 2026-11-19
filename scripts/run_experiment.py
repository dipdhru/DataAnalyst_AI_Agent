"""The controlled experiment: the same model, database and questions under each configuration.

    A  direct     whole schema (names, types); no definitions; one try
    B  retrieval  retrieved tables, descriptions, definitions; one try
    C  corrected  as B, plus failed SQL fed back (up to agent.max_corrections retries)
    agent         the app's tool-using agent, as a reference point (--with-agent)

    uv run python scripts/run_experiment.py --groq openai/gpt-oss-120b --resume
    uv run python scripts/run_experiment.py --groq openai/gpt-oss-120b --ids rev_total
    uv run python scripts/run_experiment.py --report-only --groq openai/gpt-oss-120b

Questions run one at a time through every configuration, so a run stopped by a free-tier quota
leaves the configurations with the same questions done. --resume keeps each configuration's
progress in evals/runs/experiment-<mode>-<model>.json. Memory is off, as in every eval.
The report (markdown + JSON) goes to evals/runs/experiment-<model>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from analyst.agent import Agent
from analyst.agent.providers import (
    ProviderError,
    cerebras_settings,
    groq_settings,
    make_llm_client,
)
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import load_settings
from analyst.connectors import SourceRegistry
from analyst.evals.campaign import Campaign, CampaignMismatchError, file_tag
from analyst.evals.runner import EvalResult, run_eval, summarize
from analyst.semantic import load_semantics

ROOT = Path(__file__).resolve().parents[1]
MODES = {"A": "direct", "B": "retrieval", "C": "corrected"}
ROWS = [
    ("Answer accuracy (matches golden result)", "accuracy", "pct"),
    ("SQL execution rate (answerable questions)", "execution_rate", "pct"),
    ("Retrieval recall (gold tables shown)", "retrieval_recall", "pct"),
    ("Questions with full retrieval recall", "retrieval_full_recall", "pct"),
    ("Questions needing a correction", "corrected_questions", "int"),
    ("Correction success (passed after correcting)", "correction_success", "pct"),
    ("Average model calls per question", "avg_turns", "num"),
    ("Average response time (s)", "avg_elapsed_s", "num"),
    ("Average input tokens per question", "avg_input_tokens", "int"),
    ("Average output tokens per question", "avg_output_tokens", "int"),
    ("Total cost (USD)", "cost_usd", "num"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configurations A, B, C on the golden set")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--golden", default=str(ROOT / "evals" / "golden.yaml"))
    parser.add_argument("--groq", metavar="MODEL", help="Groq free plan (GROQ_API_KEY)")
    parser.add_argument("--cerebras", metavar="MODEL", help="Cerebras (CEREBRAS_API_KEY)")
    parser.add_argument("--anthropic", metavar="MODEL", help="Claude (costs money)")
    parser.add_argument("--with-agent", action="store_true", help="also run the full agent")
    parser.add_argument("--resume", action="store_true", help="skip questions already scored")
    parser.add_argument("--restart", action="store_true", help="with --resume: start over")
    parser.add_argument("--report-only", action="store_true", help="report saved results")
    args = parser.parse_args()

    golden = yaml.safe_load(Path(args.golden).read_text())
    questions = golden["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
        if missing := wanted - {q["id"] for q in questions}:
            print(f"unknown ids: {sorted(missing)}")
            return 2

    settings = load_settings(ROOT / "config.yaml")
    if args.groq:
        settings.agent = groq_settings(settings.agent, args.groq)
    elif args.cerebras:
        settings.agent = cerebras_settings(settings.agent, args.cerebras)
    elif args.anthropic:
        settings.agent.provider, settings.agent.model = "anthropic", args.anthropic
        settings.agent.base_url = settings.agent.api_key_env = None
    model = settings.agent.model

    modes = dict(MODES)
    if args.with_agent:
        modes["agent"] = "agent"
    out_dir = ROOT / "evals" / "runs"
    campaigns: dict[str, Campaign] = {}
    for label, mode in modes.items():
        path = out_dir / f"experiment-{mode}-{file_tag(model)}.json"
        if args.restart:
            path.unlink(missing_ok=True)
        if not (args.resume or args.report_only):
            path.unlink(missing_ok=True)  # a fresh run starts clean
        try:
            campaigns[label] = Campaign(path, model=model, provider=settings.agent.provider)
        except CampaignMismatchError as e:
            print(e)
            return 2

    if not args.report_only:
        code = _run(questions, golden["source"], settings, modes, campaigns)
        if code:
            return code
    return _report(campaigns, model, settings.agent.provider, {q["id"] for q in questions})


def _run(questions, source, settings, modes, campaigns) -> int:
    registry = SourceRegistry(settings, only={source})
    try:
        service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
        catalog = service.get(source) or service.refresh(source)
        semantics = load_semantics(settings.metrics_path).get(source)
        connector = registry.get(source)
        try:
            client = make_llm_client(settings.agent, on_wait=lambda m: print(f"      ... {m}"))
        except ProviderError as e:
            print(e.message)
            return 1
        print(f"{len(questions)} questions x {len(modes)} configurations, {settings.agent.model}")
        for q in questions:
            for label, mode in modes.items():
                campaign = campaigns[label]
                if not campaign.remaining([q]):
                    continue
                agent_settings = settings.agent.model_copy(update={"mode": mode})
                try:
                    [r] = run_eval(
                        [q],
                        lambda s=agent_settings: Agent(client, connector, catalog, s, semantics),
                        model=settings.agent.model,
                    )
                except ProviderError as e:
                    print(f"\nStopped: {e.message}. Progress is saved; rerun with --resume.")
                    return 0 if "daily free quota" in e.message else 1
                campaign.add(r)
                _show(label, r)
    finally:
        registry.close()
    return 0


def _show(label: str, r: EvalResult) -> None:
    mark = "PASS" if r.passed else "FAIL"
    extra = f" fixes={r.agent.corrections}" if r.agent.corrections else ""
    print(f"{label:5} {mark} {r.id:24} {r.agent.elapsed_s:6.1f}s{extra}  {r.reason[:70]}")


def _report(campaigns: dict[str, Campaign], model: str, provider: str, ids: set[str]) -> int:
    # Compare on the questions every configuration has finished, so the columns are like for like.
    done = [{r.id for r in c.results()} & ids for c in campaigns.values()]
    common = set.intersection(*done) if done else set()
    if not common:
        print("no question has been scored in every configuration yet")
        return 1
    summaries = {
        label: summarize([r for r in c.results() if r.id in common])
        for label, c in campaigns.items()
    }
    labels = list(summaries)
    header = f"| Metric | {' | '.join(_title(x) for x in labels)} |"
    lines = [
        f"# Experiment: {model} ({provider})",
        "",
        f"{len(common)} golden questions, same model and settings in every configuration.",
        "",
        header,
        "|" + "---|" * (len(labels) + 1),
    ]
    for name, key, kind in ROWS:
        cells = [_fmt(summaries[x][key], kind) for x in labels]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "Per category (passed/total):", ""]
    cats = sorted({k for s in summaries.values() for k in s["by_category"]})
    lines += [f"| Category | {' | '.join(labels)} |", "|" + "---|" * (len(labels) + 1)]
    for cat in cats:
        cells = [summaries[x]["by_category"].get(cat, "-") for x in labels]
        lines.append(f"| {cat} | {' | '.join(cells)} |")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    out = ROOT / "evals" / "runs" / f"experiment-{file_tag(model)}"
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(
        json.dumps({"model": model, "provider": provider, "summaries": summaries}, indent=2)
    )
    print(f"saved {out.relative_to(ROOT)}.md and .json")
    return 0


def _title(label: str) -> str:
    return {"A": "A: direct", "B": "B: retrieval", "C": "C: + correction"}.get(label, label)


def _fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value:g}"


if __name__ == "__main__":
    sys.exit(main())
