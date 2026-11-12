"""Assemble the files for the Hugging Face Space into a folder you can push.

    uv run python scripts/build_space.py build/space

Spaces need the Dockerfile and a front-matter README at the repository root, which the GitHub
repo can't have, so the Space is a separate, generated copy. See docs/deploy-huggingface.md.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "huggingface"
FROM_REPO = ["pyproject.toml", "uv.lock", "metrics.yaml"]
TREES = ["src", "ui"]


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in FROM_REPO:
        shutil.copy2(ROOT / name, out / name)
    for tree in TREES:
        shutil.copytree(
            ROOT / tree, out / tree, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("Dockerfile", "README.md", "config.yaml"):
        shutil.copy2(DEPLOY / name, out / name)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "build/space")
    build(target)
    print(f"Space files written to {target.resolve()}")
