"""Load `config.yaml` (plus `.env`) into typed settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class QueryLimits(BaseModel):
    default_row_limit: int = 1000
    max_row_limit: int = 10_000
    timeout_seconds: float = 30.0

    def clamp(self, limit: int | None) -> int:
        if limit is None or limit <= 0:
            limit = self.default_row_limit
        return min(limit, self.max_row_limit)


class ProfilingSettings(BaseModel):
    sample_values: int = 5
    max_rows: int = 1_000_000
    timeout_seconds: float = 120.0


class UploadsConfig(BaseModel):
    name: str = "uploads"
    max_upload_mb: int = 200


class SourceConfig(BaseModel):
    name: str
    type: Literal["sqlite", "postgres"]
    path: Path | None = None  # sqlite
    url: str | None = None  # postgres; prefer url_env so secrets stay in .env
    url_env: str | None = None
    schemas: list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> SourceConfig:
        if self.type == "sqlite" and self.path is None:
            raise ValueError(f"source {self.name!r}: sqlite needs `path`")
        if self.type == "postgres" and not (self.url or self.url_env):
            raise ValueError(f"source {self.name!r}: postgres needs `url_env` or `url`")
        return self

    def resolve_url(self) -> str:
        if self.type == "sqlite":
            # Read-only at the driver level, whatever the SQL says.
            return f"sqlite:///file:{Path(self.path).resolve()}?mode=ro&uri=true"
        url = self.url or os.environ.get(self.url_env or "", "")
        if not url:
            raise ValueError(f"source {self.name!r}: environment variable {self.url_env} is empty")
        return url


class Settings(BaseModel):
    state_dir: Path = Path("state")
    query: QueryLimits = Field(default_factory=QueryLimits)
    profiling: ProfilingSettings = Field(default_factory=ProfilingSettings)
    uploads: UploadsConfig = Field(default_factory=UploadsConfig)
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> Settings:
        names = [s.name for s in self.sources] + [self.uploads.name]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate source names: {sorted(dupes)}")
        return self

    @property
    def app_db_path(self) -> Path:
        return self.state_dir / "app.db"

    @property
    def uploads_db_path(self) -> Path:
        return self.state_dir / "uploads.duckdb"

    @property
    def uploads_raw_dir(self) -> Path:
        return self.state_dir / "uploads"


def load_settings(path: str | Path | None = None) -> Settings:
    """Read settings from `path`, $ANALYST_CONFIG, or ./config.yaml (in that order)."""
    load_dotenv()
    path = Path(path or os.environ.get("ANALYST_CONFIG", "config.yaml"))
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    settings = Settings.model_validate(raw or {})
    return _resolve_paths(settings, path.resolve().parent)


def _resolve_paths(settings: Settings, base: Path) -> Settings:
    if not settings.state_dir.is_absolute():
        settings.state_dir = base / settings.state_dir
    for src in settings.sources:
        if src.path is not None and not src.path.is_absolute():
            src.path = base / src.path
    return settings
