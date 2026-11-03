"""Load BIRD's column descriptions into the catalogs of the BIRD databases.

    uv run python scripts/import_bird_descriptions.py

Reads data/bird/dev_databases/<db>/database_description/*.csv and stores each column's text
with description_source="import" (user-written descriptions are never overwritten). Builds a
database's catalog first if it doesn't exist yet. Safe to run again.
"""

from __future__ import annotations

import sys
from pathlib import Path

from analyst.catalog import CatalogService, CatalogStore
from analyst.catalog.importers import import_descriptions, read_bird_descriptions
from analyst.config import load_settings
from analyst.connectors import SourceRegistry

ROOT = Path(__file__).resolve().parents[1]
BIRD_CONFIG = ROOT / "config.bird.yaml"
DB_DIR = ROOT / "data" / "bird" / "dev_databases"


def main() -> int:
    if not BIRD_CONFIG.exists():
        print("config.bird.yaml missing: run scripts/setup_bird.py first")
        return 2
    settings = load_settings(BIRD_CONFIG)
    registry = SourceRegistry(settings)
    store = CatalogStore(settings.app_db_path)
    service = CatalogService(registry, store, settings.profiling)
    try:
        for source in settings.sources:
            folder = DB_DIR / source.name / "database_description"
            if not folder.is_dir():
                print(f"{source.name:26} no description folder, skipped")
                continue
            if service.get(source.name) is None:
                service.refresh(source.name)
            report = import_descriptions(store, source.name, read_bird_descriptions(folder))
            line = f"{source.name:26} {report.imported:4} columns described"
            if report.kept_user:
                line += f", {report.kept_user} kept your own"
            if report.unmatched:
                line += f", {len(report.unmatched)} not in the database: "
                line += ", ".join(report.unmatched[:4]) + (
                    " ..." if len(report.unmatched) > 4 else ""
                )
            print(line)
    finally:
        registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
