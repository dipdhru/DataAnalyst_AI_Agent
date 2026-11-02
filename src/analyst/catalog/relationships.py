"""Guess joins that the database doesn't declare (Olist declares none).

Rule: column `x` in table A joins table B when B has a unique column with the same name and a
compatible type. If `x` is unique in both tables (1:1), the direction is decided by the column
name (`customer_id` -> `customers`) or, failing that, from the smaller table to the larger.
"""

from __future__ import annotations

from analyst.catalog.models import CatalogColumn, CatalogTable, Relationship

_NUMERIC = ("int", "numeric", "decimal", "real", "double", "float", "number")


def infer_relationships(
    tables: list[CatalogTable], declared: list[Relationship]
) -> list[Relationship]:
    known = {(r.from_table, r.from_column, r.to_table, r.to_column) for r in declared}
    known |= {(r.to_table, r.to_column, r.from_table, r.from_column) for r in declared}

    keys: dict[str, list[tuple[CatalogTable, CatalogColumn]]] = {}
    for table in tables:
        for col in table.columns:
            if col.primary_key or (col.profile and col.profile.is_unique):
                keys.setdefault(col.name.lower(), []).append((table, col))

    inferred: list[Relationship] = []
    for table in tables:
        for col in table.columns:
            targets = [
                (target, target_col)
                for target, target_col in keys.get(col.name.lower(), [])
                if target is not table and _compatible(col.data_type, target_col.data_type)
            ]
            # Several candidates (sellers, leads_closed): keep the ones named after the column.
            named = [t for t in targets if _stem(col) in t[0].name.lower()]
            for target, target_col in named or targets:
                if _is_key(col) and not _points_to(table, col, target):
                    continue
                edge = (table.name, col.name, target.name, target_col.name)
                if edge in known:
                    continue
                known.add(edge)
                inferred.append(
                    Relationship(
                        from_table=table.name,
                        from_column=col.name,
                        to_table=target.name,
                        to_column=target_col.name,
                        origin="inferred",
                    )
                )
    return inferred


def _is_key(col: CatalogColumn) -> bool:
    return col.primary_key or bool(col.profile and col.profile.is_unique)


def _points_to(source: CatalogTable, col: CatalogColumn, target: CatalogTable) -> bool:
    """For a 1:1 pair, decide whether the edge runs source -> target."""
    stem = _stem(col)
    source_match, target_match = stem in source.name.lower(), stem in target.name.lower()
    if source_match != target_match:
        return target_match
    if source.row_count is not None and target.row_count is not None:
        return source.row_count < target.row_count
    return False


def _stem(col: CatalogColumn) -> str:
    return col.name.lower().removesuffix("_id")


def _family(data_type: str) -> str:
    t = data_type.lower()
    return "number" if any(m in t for m in _NUMERIC) else "text" if t else "unknown"


def _compatible(a: str, b: str) -> bool:
    return _family(a) == _family(b)
