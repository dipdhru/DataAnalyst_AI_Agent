"""The golden question set in evals/golden.yaml: well-formed, and its answers match the data."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from analyst.config import Settings, SourceConfig
from analyst.sql_guard import validate_select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_golden import compute, load_golden, open_source  # noqa: E402

GOLDEN = load_golden()
QUESTIONS = GOLDEN["questions"]
STATUSES = {"draft", "approved"}


def test_ids_are_unique_and_well_formed() -> None:
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z]+_[a-z0-9_]+", i) for i in ids)
    assert 20 <= len(QUESTIONS) <= 30


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_question_shape(q: dict) -> None:
    assert q["status"] in STATUSES
    assert q["question"].strip()
    assert set(q.get("uses", [])) <= set(GOLDEN["definitions"]), "unknown definition"
    if q["category"] == "unanswerable":
        assert "sql" not in q and q["expected_behaviour"].strip()
    else:
        assert q["sql"].strip() and q["expected"]["columns"] and q["expected"]["rows"]


@pytest.mark.parametrize("q", [q for q in QUESTIONS if "sql" in q], ids=lambda q: q["id"])
def test_golden_sql_passes_the_guard(q: dict) -> None:
    validate_select(q["sql"], dialect="sqlite", max_rows=1000)


@pytest.mark.full_olist
def test_expected_answers_match_the_data(tmp_path: Path, olist: Path) -> None:
    settings = Settings(
        state_dir=tmp_path,
        sources=[SourceConfig(name="olist", type="sqlite", path=olist / "olist.sqlite")],
    )
    connector = open_source(settings, GOLDEN["source"])
    try:
        answers = compute(GOLDEN, connector)
    finally:
        connector.close()
    stale = [q["id"] for q in QUESTIONS if "sql" in q and answers[q["id"]] != q["expected"]]
    assert not stale, f"run scripts/build_golden.py: {stale}"
