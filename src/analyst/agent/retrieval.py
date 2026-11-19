"""Schema-aware retrieval: pick the tables and business definitions relevant to one question.

Used by the experiment modes (config B and C): instead of the whole schema, the model sees only
the retrieved tables (with column descriptions and the joins between them) and the retrieved
definitions. Similarity is the same small TF-IDF as memory recall (no model, no dependency);
an embedding model could replace `_score` without changing the callers.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from analyst.catalog import Catalog, CatalogTable
from analyst.memory.recall import tokens
from analyst.semantic import Definition, SourceSemantics

TOP_TABLES = 4
TOP_DEFINITIONS = 4


class RetrievedSchema(BaseModel):
    tables: list[str] = Field(default_factory=list)
    definitions: list[str] = Field(default_factory=list)
    text: str = ""


class SchemaRetriever:
    def __init__(
        self,
        catalog: Catalog,
        semantics: SourceSemantics | None = None,
        *,
        top_tables: int = TOP_TABLES,
        top_definitions: int = TOP_DEFINITIONS,
    ) -> None:
        self.catalog = catalog
        self.semantics = semantics or SourceSemantics()
        self.top_tables = top_tables
        self.top_definitions = top_definitions
        self._table_docs = {t.name: Counter(tokens(_table_text(t))) for t in catalog.tables}
        self._def_docs = {
            name: Counter(tokens(f"{name.replace('_', ' ')} {d.description} {d.sql or ''}"))
            for name, d in self.semantics.definitions.items()
        }
        docs = [*self._table_docs.values(), *self._def_docs.values()]
        self._df = Counter(word for d in docs for word in d)
        self._n = len(docs)

    def retrieve(self, question: str) -> RetrievedSchema:
        query = self._vector(Counter(tokens(question)))
        definitions = _top(
            {n: self._score(query, d) for n, d in self._def_docs.items()}, self.top_definitions
        )
        tables = _top(
            {n: self._score(query, d) for n, d in self._table_docs.items()}, self.top_tables
        )
        # A retrieved definition names the tables its SQL needs ("SUM(order_items.price)").
        known = {t.name.lower(): t.name for t in self.catalog.tables}
        for name in definitions:
            for ref in _table_refs(self.semantics.definitions[name], known):
                if ref not in tables:
                    tables.append(ref)
        if not tables:  # nothing matched: better the whole schema than none
            tables = [t.name for t in self.catalog.tables]
        order = {t.name: i for i, t in enumerate(self.catalog.tables)}
        tables.sort(key=order.__getitem__)
        return RetrievedSchema(
            tables=tables, definitions=definitions, text=self._render(tables, definitions)
        )

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        n, df = self._n, self._df
        return {w: c * math.log(1 + (n + 1) / (df.get(w, 0) + 1)) for w, c in counts.items()}

    def _score(self, query: dict[str, float], doc: Counter[str]) -> float:
        vec = self._vector(doc)
        dot = sum(v * vec.get(w, 0.0) for w, v in query.items())
        norm = math.sqrt(sum(v * v for v in query.values())) * math.sqrt(
            sum(v * v for v in vec.values())
        )
        return dot / norm if norm else 0.0

    def _render(self, tables: list[str], definitions: list[str]) -> str:
        chosen = set(tables)
        lines = ["Tables:"]
        for name in tables:
            t = self.catalog.table(name)
            lines.append(f"- {t.name} ({t.row_count} rows)" + _desc(t.description))
            lines += [f"    {c.name} {c.data_type}{_desc(c.description)}" for c in t.columns]
        joins = [
            f"- {r.from_table}.{r.from_column} = {r.to_table}.{r.to_column}"
            for r in self.catalog.relationships
            if r.from_table in chosen and r.to_table in chosen
        ]
        if joins:
            lines += ["Joins:", *joins]
        if definitions:
            lines.append("Business definitions (agreed with the user; use them):")
            for name in definitions:
                d = self.semantics.definitions[name]
                lines.append(f"- {name}: {d.description}")
                if d.sql:
                    lines.append(f"  SQL: {d.sql}")
        if self.semantics.notes:
            lines += ["Notes:", *(f"- {n}" for n in self.semantics.notes)]
        return "\n".join(lines)


def full_schema_text(catalog: Catalog) -> str:
    """Config A's context: every table and column with its type, nothing else."""
    lines = ["Tables:"]
    for t in catalog.tables:
        cols = ", ".join(f"{c.name} {c.data_type}" for c in t.columns)
        lines.append(f"- {t.name}: {cols}")
    return "\n".join(lines)


def _table_text(table: CatalogTable) -> str:
    parts = [table.name.replace("_", " "), table.description or ""]
    for c in table.columns:
        parts += [c.name.replace("_", " "), c.description or ""]
    return " ".join(parts)


def _table_refs(definition: Definition, known: dict[str, str]) -> list[str]:
    refs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_]", definition.sql or "")
    return list(dict.fromkeys(known[r.lower()] for r in refs if r.lower() in known))


def _top(scores: dict[str, float], k: int) -> list[str]:
    ranked = sorted((s, n) for n, s in scores.items() if s > 0)
    return [n for _, n in sorted(ranked, key=lambda x: -x[0])[:k]]


def _desc(text: str | None) -> str:
    return f" -- {' '.join(text.split())}" if text else ""
