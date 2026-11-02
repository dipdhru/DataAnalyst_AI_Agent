"""Fixtures built from the Olist data in data/olist/.

Most tests use small samples (first rows of each table) so the suite runs in seconds.
Tests marked `full_olist` use the full dataset: run them with `uv run pytest -m full_olist`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from analyst.config import ProfilingSettings, QueryLimits, Settings, SourceConfig

OLIST_DIR = Path(__file__).resolve().parents[1] / "data" / "olist"
OLIST_SQLITE = OLIST_DIR / "olist.sqlite"
OLIST_CSV_DIR = OLIST_DIR / "csv"
SAMPLE_ROWS = 2000


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not OLIST_SQLITE.exists():
        skip = pytest.mark.skip(reason=f"Olist data not found in {OLIST_DIR} (see README)")
        for item in items:
            if "olist" in item.fixturenames or item.get_closest_marker("full_olist"):
                item.add_marker(skip)
    selected = config.getoption("-m") or ""
    for marker, why in (("full_olist", "full dataset"), ("bird", "BIRD benchmark data")):
        if marker in selected:
            continue
        skip = pytest.mark.skip(reason=f"{why}: run with -m {marker}")
        for item in items:
            if item.get_closest_marker(marker):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def olist() -> Path:
    return OLIST_DIR


@pytest.fixture(scope="session")
def sample_sqlite(olist: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of olist.sqlite with only the first SAMPLE_ROWS rows of each table."""
    path = tmp_path_factory.mktemp("olist") / "olist_sample.sqlite"
    con = sqlite3.connect(f"file:{path}", uri=True)
    con.execute("ATTACH DATABASE ? AS full", (f"file:{OLIST_SQLITE}?mode=ro",))
    tables = [r[0] for r in con.execute("SELECT name FROM full.sqlite_master WHERE type = 'table'")]
    for t in tables:
        con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM full."{t}" LIMIT {SAMPLE_ROWS}')
    con.commit()
    con.execute("DETACH DATABASE full")
    con.close()
    return path


@pytest.fixture(scope="session")
def sample_csv_dir(olist: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Olist CSVs cut to their header plus SAMPLE_ROWS lines (byte-for-byte otherwise)."""
    out = tmp_path_factory.mktemp("olist_csv")
    for src in sorted(OLIST_CSV_DIR.glob("*.csv")):
        with src.open("rb") as f:
            lines = [line for _, line in zip(range(SAMPLE_ROWS + 1), f, strict=False)]
        (out / src.name).write_bytes(b"".join(lines))
    return out


@pytest.fixture
def settings(sample_sqlite: Path, tmp_path: Path) -> Settings:
    return Settings(
        state_dir=tmp_path / "state",
        query=QueryLimits(default_row_limit=100, max_row_limit=500, timeout_seconds=10),
        profiling=ProfilingSettings(sample_values=3, max_rows=100_000, timeout_seconds=30),
        sources=[SourceConfig(name="olist", type="sqlite", path=sample_sqlite)],
    )
