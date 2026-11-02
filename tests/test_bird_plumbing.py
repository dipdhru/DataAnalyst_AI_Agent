"""BIRD Mini-Dev gold SQL must give the same result through run_query as through raw sqlite3.

If this fails, the guard or the connector changes query meaning, and any BIRD score would be
measuring our plumbing, not the agent. Run with `uv run pytest -m bird` after
`scripts/setup_bird.py`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from analyst.config import QueryLimits
from analyst.connectors import SqlAlchemyConnector
from analyst.models import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "bird" / "questions" / "mini_dev_sqlite.jsonl"
DB_DIR = ROOT / "data" / "bird" / "dev_databases"
LIMITS = QueryLimits(default_row_limit=1_000_000, max_row_limit=1_000_000, timeout_seconds=120)

pytestmark = pytest.mark.bird


def _questions() -> list[dict[str, Any]]:
    if not QUESTIONS.exists():
        return []
    return [json.loads(line) for line in QUESTIONS.open()]


def _db_path(db_id: str) -> Path:
    return DB_DIR / db_id / f"{db_id}.sqlite"


@pytest.fixture(scope="module")
def connectors():
    opened: dict[str, SqlAlchemyConnector] = {}

    def get(db_id: str) -> SqlAlchemyConnector:
        if db_id not in opened:
            url = f"sqlite:///file:{_db_path(db_id).resolve()}?mode=ro&uri=true"
            opened[db_id] = SqlAlchemyConnector(db_id, url, limits=LIMITS)
        return opened[db_id]

    yield get
    for c in opened.values():
        c.close()


def _raw(db_id: str, sql: str) -> list[tuple[Any, ...]]:
    con = sqlite3.connect(f"file:{_db_path(db_id).resolve()}?mode=ro", uri=True)
    try:
        return [tuple(to_jsonable(v) for v in row) for row in con.execute(sql).fetchall()]
    finally:
        con.close()


def _top_level_order_by(sql: str) -> bool:
    depth = 0
    for token in re.findall(r"\(|\)|\bORDER\s+BY\b", sql, flags=re.IGNORECASE):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            return True
    return False


@pytest.mark.parametrize("q", _questions(), ids=lambda q: f"{q['question_id']}-{q['db_id']}")
def test_gold_sql_survives_run_query(q: dict[str, Any], connectors) -> None:
    if not _db_path(q["db_id"]).exists():
        pytest.skip("BIRD databases missing: run scripts/setup_bird.py")
    expected = _raw(q["db_id"], q["SQL"])
    result = connectors(q["db_id"]).run_query(q["SQL"])
    assert not result.truncated
    got = [tuple(row) for row in result.rows]
    if _top_level_order_by(q["SQL"]):
        assert got == expected, f"order or values differ\nran: {result.sql}"
    else:
        assert Counter(got) == Counter(expected), f"rows differ\nran: {result.sql}"
