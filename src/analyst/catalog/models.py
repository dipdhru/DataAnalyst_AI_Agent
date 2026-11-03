"""What the catalog knows about a source: tables, columns, profiles, relationships."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

# user: typed by the user (always wins); import: shipped with the dataset; llm: model-written.
DescriptionSource = Literal["user", "import", "llm"]


class ColumnProfile(BaseModel):
    row_count: int
    null_count: int | None = None
    null_fraction: float | None = None
    distinct_count: int | None = None
    min: Any = None
    max: Any = None
    samples: list[Any] = Field(default_factory=list)
    is_unique: bool = False  # every row has a distinct, non-null value: a candidate key
    sampled: bool = False  # computed on the first `profiling.max_rows` rows only


class CatalogColumn(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    description: str | None = None
    description_source: DescriptionSource | None = None
    profile: ColumnProfile | None = None


class CatalogTable(BaseModel):
    name: str  # qualified name, as used in SQL
    row_count: int | None = None
    description: str | None = None
    description_source: DescriptionSource | None = None
    columns: list[CatalogColumn]

    def column(self, name: str) -> CatalogColumn:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


class Relationship(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    origin: Literal["declared", "inferred"]


class Catalog(BaseModel):
    source: str
    dialect: str
    refreshed_at: dt.datetime
    tables: list[CatalogTable]
    relationships: list[Relationship] = Field(default_factory=list)

    def table(self, name: str) -> CatalogTable:
        for t in self.tables:
            if t.name.lower() == name.lower():
                return t
        raise KeyError(name)
