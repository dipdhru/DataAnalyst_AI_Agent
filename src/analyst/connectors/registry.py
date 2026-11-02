"""Build every configured connector once and look them up by name."""

from __future__ import annotations

from analyst.config import Settings
from analyst.connectors.base import Connector
from analyst.connectors.files import FileConnector
from analyst.connectors.sql import SqlAlchemyConnector


class SourceRegistry:
    def __init__(self, settings: Settings, only: set[str] | None = None) -> None:
        """`only` limits which sources are opened. Scripts use it so they don't need the
        uploads DuckDB file, which a running API server holds locked."""
        self._sources: dict[str, Connector] = {}
        for src in settings.sources:
            if only is None or src.name in only:
                self._sources[src.name] = SqlAlchemyConnector(
                    src.name, src.resolve_url(), limits=settings.query, schemas=src.schemas
                )
        self.uploads: FileConnector | None = None
        if only is None or settings.uploads.name in only:
            self.uploads = FileConnector(
                settings.uploads.name, settings.uploads_db_path, limits=settings.query
            )
            self._sources[self.uploads.name] = self.uploads

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
