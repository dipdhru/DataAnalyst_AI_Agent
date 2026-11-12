"""Build every configured connector once and look them up by name."""

from __future__ import annotations

from analyst.config import Settings
from analyst.connectors.base import Connector
from analyst.connectors.sql import SqlAlchemyConnector


class SourceRegistry:
    def __init__(self, settings: Settings, only: set[str] | None = None) -> None:
        """`only` limits which sources are opened, so a script or the UI connects to just the
        source it needs."""
        self._sources: dict[str, Connector] = {}
        for src in settings.sources:
            if only is None or src.name in only:
                self._sources[src.name] = SqlAlchemyConnector(
                    src.name, src.resolve_url(), limits=settings.query, schemas=src.schemas
                )

    def add(self, connector: Connector) -> None:
        """Register a connector built outside the config (the demo's visitor-supplied database)."""
        self._sources[connector.name] = connector

    def remove(self, name: str) -> None:
        connector = self._sources.pop(name, None)
        if connector is not None:
            connector.close()

    def get(self, name: str) -> Connector:
        try:
            return self._sources[name]
        except KeyError:
            raise KeyError(f"unknown source: {name}") from None

    def all(self) -> list[Connector]:
        return list(self._sources.values())

    def close(self) -> None:
        for connector in self._sources.values():
            connector.close()
