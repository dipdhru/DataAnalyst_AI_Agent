"""Compute expected answers for evals/golden.yaml by running each question's SQL.

    uv run python scripts/build_golden.py           # fill in / refresh draft answers
    uv run python scripts/build_golden.py --check   # verify, change nothing (exit 1 on drift)
    uv run python scripts/build_golden.py --force   # also overwrite approved answers

SQL runs through the source's Connector.run_query, so it passes the same guard the agent uses.
Approved answers are the user's: they are never overwritten without --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from analyst.config import Settings, load_settings
from analyst.connectors import Connector, SqlAlchemyConnector

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden.yaml"


class _Dumper(yaml.SafeDumper):
    pass


def _str_representer(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_representer)


def load_golden(path: Path = GOLDEN) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def open_source(settings: Settings, name: str) -> Connector:
    """Open only the golden set's source, so this runs while the API holds uploads.duckdb."""
    src = next(s for s in settings.sources if s.name == name)
    return SqlAlchemyConnector(
        src.name, src.resolve_url(), limits=settings.query, schemas=src.schemas
    )


def compute(golden: dict[str, Any], connector: Connector) -> dict[str, dict[str, Any]]:
    answers = {}
    for q in golden["questions"]:
        if "sql" not in q:
            continue
        result = connector.run_query(q["sql"], limit=1000, timeout_s=120)
        if result.truncated:
            raise SystemExit(f"{q['id']}: result truncated; golden answers must be small")
        answers[q["id"]] = {"columns": result.columns, "rows": result.rows}
    return answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report drift, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite approved answers too")
    args = parser.parse_args()

    golden = load_golden()
    connector = open_source(load_settings(ROOT / "config.yaml"), golden["source"])
    try:
        answers = compute(golden, connector)
    finally:
        connector.close()

    updated, held = [], []
    for q in golden["questions"]:
        new = answers.get(q["id"])
        if new is None or q.get("expected") == new:
            continue
        if q.get("status") == "approved" and not args.force:
            held.append(q["id"])
        else:
            updated.append(q["id"])
            q["expected"] = new

    for qid in held:
        print(f"approved answer differs from the data, left unchanged: {qid}")
    if args.check:
        for qid in updated:
            print(f"out of date: {qid}")
        return 1 if updated or held else 0

    GOLDEN.write_text(
        yaml.dump(golden, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
    )
    print(f"{len(answers)} answers checked, {len(updated)} updated in {GOLDEN.relative_to(ROOT)}")
    return 1 if held else 0


if __name__ == "__main__":
    sys.exit(main())
