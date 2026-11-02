"""Business definitions per source (metrics.yaml), handed to the agent as ground truth."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Definition(BaseModel):
    description: str
    sql: str | None = None


class SourceSemantics(BaseModel):
    definitions: dict[str, Definition] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Plain-text block for the system prompt; empty when nothing is defined."""
        lines = []
        for name, d in self.definitions.items():
            lines.append(f"- {name}: {d.description}")
            if d.sql:
                lines.append(f"  SQL: {d.sql}")
        if self.notes:
            lines.append("Notes:")
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def load_semantics(path: Path) -> dict[str, SourceSemantics]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {source: SourceSemantics.model_validate(body or {}) for source, body in raw.items()}
