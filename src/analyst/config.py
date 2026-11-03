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


class AgentSettings(BaseModel):
    # "anthropic": Claude through the Anthropic SDK. "openai_compatible": any provider with an
    # OpenAI-style chat API (Gemini, Groq, OpenRouter, Ollama), set by base_url + api_key_env.
    provider: Literal["anthropic", "openai_compatible"] = "anthropic"
    base_url: str | None = None
    api_key_env: str | None = None
    min_seconds_between_calls: float = 0.0  # throttle for free tiers' requests-per-minute
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = 16_000
    max_steps: int = 15  # model turns per question before giving up
    result_rows_shown: int = 50  # rows of each query result the model sees


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
    agent: AgentSettings = Field(default_factory=AgentSettings)
    metrics_path: Path = Path("metrics.yaml")
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> Settings:
        names = [s.name for s in self.sources]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate source names: {sorted(dupes)}")
        return self

    @property
    def app_db_path(self) -> Path:
        return self.state_dir / "app.db"


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
    if not settings.metrics_path.is_absolute():
        settings.metrics_path = base / settings.metrics_path
    for src in settings.sources:
        if src.path is not None and not src.path.is_absolute():
            src.path = base / src.path
    return settings
