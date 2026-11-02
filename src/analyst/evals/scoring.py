"""Compare an agent's result table with a golden one.

Two scores:
- answer_match (ours): column order and extra columns are ignored, numbers match to the
  golden value's own precision (13494400.74 matches 13494400.7412), a percentage matches the
  same value as a fraction (6.77 matches 0.0677), strings ignore case and outer spaces. Row
  order counts only when the golden SQL has a top-level ORDER BY.
- execution_match (BIRD's EX): the sets of row tuples are exactly equal.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

Row = Sequence[Any]


def has_top_level_order_by(sql: str) -> bool:
    depth = 0
    for token in re.findall(r"\(|\)|\bORDER\s+BY\b", sql, flags=re.IGNORECASE):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            return True
    return False


def execution_match(pred_rows: Sequence[Row], gold_rows: Sequence[Row]) -> bool:
    return {tuple(r) for r in pred_rows} == {tuple(r) for r in gold_rows}


def values_match(pred: Any, gold: Any) -> bool:
    if gold is None or pred is None:
        return gold is None and pred is None
    if isinstance(gold, bool) or isinstance(pred, bool):
        return pred == gold
    if isinstance(gold, (int, float)):
        try:
            value = float(pred)
        except (TypeError, ValueError):
            return False
        tolerance = 0.5 * 10 ** -_decimals(gold) + 1e-9
        scales = (1.0,) if isinstance(gold, int) else (1.0, 100.0, 0.01)
        return any(abs(value * s - gold) <= tolerance for s in scales)
    return str(pred).strip().lower() == str(gold).strip().lower()


def answer_match(
    pred_columns: Sequence[str],
    pred_rows: Sequence[Row],
    gold_rows: Sequence[Row],
    *,
    ordered: bool,
) -> tuple[bool, str]:
    """Return (passed, reason)."""
    if len(pred_rows) != len(gold_rows):
        return False, f"expected {len(gold_rows)} rows, got {len(pred_rows)}"
    if not gold_rows:
        return True, "both empty"
    n_gold = len(gold_rows[0])
    if pred_rows and len(pred_rows[0]) < n_gold:
        return False, f"expected at least {n_gold} columns, got {len(pred_rows[0])}"

    # Map each golden column to a distinct predicted column holding the same values.
    mapping: list[int] = []
    for g in range(n_gold):
        gold_col = [r[g] for r in gold_rows]
        match = next(
            (
                p
                for p in range(len(pred_rows[0]))
                if p not in mapping
                and _columns_match([r[p] for r in pred_rows], gold_col, ordered=ordered)
            ),
            None,
        )
        if match is None:
            return False, f"no column matches golden column {g + 1} ({gold_col[:3]}...)"
        mapping.append(match)

    projected = [[r[p] for p in mapping] for r in pred_rows]
    gold = [list(r) for r in gold_rows]
    if not ordered:
        projected, gold = sorted(projected, key=_sort_key), sorted(gold, key=_sort_key)
    for i, (p_row, g_row) in enumerate(zip(projected, gold, strict=True)):
        if not all(values_match(p, g) for p, g in zip(p_row, g_row, strict=True)):
            return False, f"row {i + 1} differs: {p_row} vs {g_row}"
    names = ", ".join(pred_columns[p] for p in mapping) if pred_columns else ""
    return True, f"matched on columns: {names}"


def _columns_match(pred: list[Any], gold: list[Any], *, ordered: bool) -> bool:
    if not ordered:
        pred, gold = sorted(pred, key=_value_key), sorted(gold, key=_value_key)
    return all(values_match(p, g) for p, g in zip(pred, gold, strict=True))


def _decimals(value: int | float) -> int:
    if isinstance(value, int):
        return 0
    exponent = Decimal(repr(value)).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _value_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, 0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (1, round(float(value), 2))
    return (2, str(value).strip().lower())


def _sort_key(row: list[Any]) -> tuple:
    return tuple(_value_key(v) for v in row)
