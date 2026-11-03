"""Have the model write descriptions for tables and columns that have none.

    uv run python scripts/describe_catalog.py --source olist --groq openai/gpt-oss-120b
    uv run python scripts/describe_catalog.py --source olist      # model from config.yaml

One model call per table that still has gaps. User-written and imported descriptions are
never replaced; edit or clear any result in the UI's Catalog view. Builds the catalog first if
needed. Calls the model provider (free on Groq/Gemini, paid on Claude).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyst.agent.providers import ProviderError, groq_settings, make_llm_client
from analyst.catalog import CatalogService, CatalogStore
from analyst.catalog.describer import describe_catalog
from analyst.config import load_settings
from analyst.connectors import SourceRegistry

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write missing catalog descriptions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--groq", metavar="MODEL", help="use Groq (GROQ_API_KEY in .env)")
    parser.add_argument("--tables", help="comma-separated table names (default: all)")
    args = parser.parse_args()

    settings = load_settings(args.config)
    agent = groq_settings(settings.agent, args.groq) if args.groq else settings.agent
    registry = SourceRegistry(settings, only={args.source})
    store = CatalogStore(settings.app_db_path)
    try:
        service = CatalogService(registry, store, settings.profiling)
        if service.get(args.source) is None:
            service.refresh(args.source)
        client = make_llm_client(agent, on_wait=lambda m: print(f"   ... {m}"))
        report = describe_catalog(
            client,
            agent.model,
            store,
            args.source,
            only_tables=set(args.tables.split(",")) if args.tables else None,
            on_table=lambda t: print(f"describing {t}", flush=True),
        )
    except ProviderError as e:
        print(f"Model provider error: {e.message}")
        return 1
    finally:
        registry.close()
    print(f"wrote {report.tables} table and {report.columns} column descriptions")
    for failure in report.failed:
        print(f"  skipped {failure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
