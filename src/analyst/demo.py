"""Demo mode for a public host: every visitor gets an isolated workspace.

Without this the UI would share one connector and one `state/app.db` across all visitors, so one
person's catalog (which stores sampled column values), memory and conversations would show up for
the next. A Workspace holds everything one browser session owns: a temp state folder, its own
database connections, catalog and memory. It is deleted when the session is dropped.

Visitors bring their own database (vetted by `netguard`) and their own model API key. The server's
own keys and any Postgres sources in its config are never reachable from a demo session.
"""

from __future__ import annotations

import os
import re
import tempfile
import weakref
from pathlib import Path
from typing import Any

from analyst.agent.providers import GROQ_URL, groq_settings
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import AgentSettings, ProfilingSettings, Settings
from analyst.connectors import SourceRegistry, SqlAlchemyConnector
from analyst.memory import MemoryStore
from analyst.netguard import safe_postgres_url
from analyst.semantic import load_semantics

DEMO_ENV = "ANALYST_DEMO"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Profiling reads a stranger's database, so keep it lighter than for our own data.
DEMO_PROFILING = ProfilingSettings(sample_values=5, max_rows=100_000, timeout_seconds=60.0)
SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")
MAX_SCHEMAS = 5

PROVIDERS = {
    "Gemini (free key at aistudio.google.com)": "gemini",
    "Groq (free key at console.groq.com)": "groq",
}


class ConnectError(Exception):
    """The database could not be reached; the message is safe to show (no password)."""


def demo_enabled() -> bool:
    return os.environ.get(DEMO_ENV, "").lower() in ("1", "true", "yes")


def agent_settings_for(provider: str, base: AgentSettings) -> AgentSettings:
    """Model settings for a visitor's own key. Always an OpenAI-compatible free-tier provider:
    the server's configured provider (possibly Claude, billed to the owner) is never used."""
    if provider == "groq":
        settings = groq_settings(base, "openai/gpt-oss-120b")
        return settings.model_copy(update={"api_key_env": None, "base_url": GROQ_URL})
    return base.model_copy(
        update={
            "provider": "openai_compatible",
            "base_url": GEMINI_URL,
            "api_key_env": None,
            "model": "gemini-3.6-flash",
            "min_seconds_between_calls": 5.0,
        }
    )


def parse_schemas(text: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()] or ["public"]
    if len(names) > MAX_SCHEMAS:
        raise ConnectError(f"List at most {MAX_SCHEMAS} schemas.")
    for name in names:
        if not SCHEMA_NAME.match(name):
            raise ConnectError(f"{name!r} is not a plain schema name.")
    return names


class Workspace:
    """Everything one visitor's session owns."""

    def __init__(self, settings: Settings) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="analyst-demo-")
        self.settings = settings.model_copy(
            update={
                "state_dir": Path(self._tmp.name),
                "profiling": DEMO_PROFILING,
                # Sample databases only. A configured Postgres is the owner's, not the visitor's.
                "sources": [s for s in settings.sources if s.type == "sqlite" and s.path.exists()],
            }
        )
        self.registry = SourceRegistry(self.settings)
        self.store = CatalogStore(self.settings.app_db_path)
        self.memory = MemoryStore(self.settings.app_db_path)
        self.service = CatalogService(self.registry, self.store, self.settings.profiling)
        self._semantics = load_semantics(self.settings.metrics_path)
        self._connections = 0
        # Close database connections when the session goes; the registry holds no reference
        # back to the workspace, so this can run. The temp folder removes itself.
        weakref.finalize(self, self.registry.close)

    def source_names(self) -> list[str]:
        return [c.name for c in self.registry.all()]

    def connect_postgres(self, raw_url: str, schemas_text: str = "public") -> str:
        """Vet, connect and register a visitor's Postgres. Returns the new source's name.
        Raises UnsafeHostError (bad or forbidden URL) or ConnectError (could not connect)."""
        url = safe_postgres_url(raw_url)
        schemas = parse_schemas(schemas_text)
        self._connections += 1
        # A new name per connection, so a catalog or memory of a previous database never
        # shows up against the next one.
        name = "your_database" if self._connections == 1 else f"your_database_{self._connections}"
        password = url.password or ""
        connector = SqlAlchemyConnector(
            name,
            url.render_as_string(hide_password=False),
            limits=self.settings.query,
            schemas=schemas,
        )
        try:
            schema = connector.get_schema(refresh=True)
        except Exception as e:
            connector.close()
            text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            if password:
                text = text.replace(password, "***")
            raise ConnectError(f"Could not connect: {text[:300]}") from None
        if not schema.tables:
            connector.close()
            raise ConnectError(
                "Connected, but no tables were found in "
                f"{', '.join(schemas)}. Check the schema name and that the user can read it."
            )
        self.registry.add(connector)
        return name

    def disconnect(self, name: str) -> None:
        self.registry.remove(name)

    def backend(
        self, source: str, agent: AgentSettings, api_key: str | None = None
    ) -> dict[str, Any]:
        """The same shape the UI's `backend()` returns, bound to this session."""
        return {
            "settings": self.settings.model_copy(update={"agent": agent}),
            "connector": self.registry.get(source),
            "store": self.store,
            "service": self.service,
            "memory": self.memory,
            "semantics": self._semantics.get(source),
            "api_key": api_key,
        }
