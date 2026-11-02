"""Build, refresh and edit catalogs for configured sources."""

from __future__ import annotations

import datetime as dt
import logging

from analyst.catalog.models import Catalog, CatalogColumn, CatalogTable, Relationship
from analyst.catalog.profiler import profile_table
from analyst.catalog.relationships import infer_relationships
from analyst.catalog.store import CatalogStore
from analyst.config import ProfilingSettings
from analyst.connectors.registry import SourceRegistry

log = logging.getLogger(__name__)


class CatalogService:
    def __init__(
        self, registry: SourceRegistry, store: CatalogStore, profiling: ProfilingSettings
    ) -> None:
        self.registry = registry
        self.store = store
        self.profiling = profiling

    def get(self, source: str) -> Catalog | None:
        self.registry.get(source)  # raises KeyError for unknown sources
        return self.store.load(source)

    def refresh(self, source: str) -> Catalog:
        """Re-introspect and re-profile every table. User-written descriptions are kept."""
        connector = self.registry.get(source)
        schema = connector.get_schema(refresh=True)
        tables: list[CatalogTable] = []
        declared: list[Relationship] = []
        for info in schema.tables:
            log.info("profiling %s.%s", source, info.qualified_name)
            row_count, profiles = profile_table(connector, info, self.profiling)
            tables.append(
                CatalogTable(
                    name=info.qualified_name,
                    row_count=row_count,
                    columns=[
                        CatalogColumn(
                            name=c.name,
                            data_type=c.data_type,
                            nullable=c.nullable,
                            primary_key=c.primary_key,
                            profile=profiles.get(c.name),
                        )
                        for c in info.columns
                    ],
                )
            )
            for fk in info.foreign_keys:
                declared += [
                    Relationship(
                        from_table=info.qualified_name,
                        from_column=fc,
                        to_table=fk.referred_table,
                        to_column=rc,
                        origin="declared",
                    )
                    for fc, rc in zip(fk.columns, fk.referred_columns, strict=False)
                ]
        catalog = Catalog(
            source=source,
            dialect=connector.dialect,
            refreshed_at=dt.datetime.now(dt.UTC),
            tables=tables,
            relationships=declared + infer_relationships(tables, declared),
        )
        self.store.save(catalog)
        return self.store.load(source)  # type: ignore[return-value]

    def describe_table(self, source: str, table: str, description: str | None) -> CatalogTable:
        self.store.set_table_description(source, table, description)
        return self._table(source, table)

    def describe_column(
        self, source: str, table: str, column: str, description: str | None
    ) -> CatalogColumn:
        self.store.set_column_description(source, table, column, description)
        return self._table(source, table).column(column)

    def _table(self, source: str, table: str) -> CatalogTable:
        catalog = self.store.load(source)
        if catalog is None:
            raise KeyError(source)
        return catalog.table(table)
