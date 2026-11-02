"""An eval spread over several runs (days), for free tiers with small daily quotas.

Each scored question is saved to one JSON file as soon as it finishes, so a run can stop at any
point (quota used up, crash, Ctrl+C) and the next run carries on with the questions not yet done.
A question interrupted by a provider error is not saved, so it is asked again next time.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from analyst.evals.runner import EvalResult


class CampaignMismatchError(ValueError):
    pass


class Campaign:
    def __init__(self, path: Path, *, model: str, provider: str) -> None:
        self.path = path
        if path.exists():
            self.data: dict[str, Any] = json.loads(path.read_text())
            if self.data["model"] != model:
                raise CampaignMismatchError(
                    f"{path.name} was started with {self.data['model']}, not {model}; "
                    "use --restart to begin again with the new model"
                )
        else:
            self.data = {
                "model": model,
                "provider": provider,
                "started": dt.datetime.now().isoformat(timespec="seconds"),
                "results": {},
            }

    def remaining(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [q for q in questions if q["id"] not in self.data["results"]]

    def add(self, result: EvalResult) -> None:
        self.data["results"][result.id] = result.model_dump(mode="json")
        self.data["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)  # never leave a half-written file behind

    def results(self) -> list[EvalResult]:
        return [EvalResult.model_validate(r) for r in self.data["results"].values()]


def campaign_path(runs_dir: Path, model: str) -> Path:
    return runs_dir / f"campaign-{model}.json"
