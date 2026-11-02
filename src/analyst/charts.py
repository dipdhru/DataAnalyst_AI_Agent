"""Pick a sensible chart for a result table (Plotly), or none when a table says it better."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

ChartKind = Literal["metric", "line", "bar"]
_DATE_LIKE = re.compile(r"^\d{4}(-\d{2}){1,2}([ T].*)?$")  # 2017-11, 2017-11-24, timestamps
MAX_BARS = 30


class ChartSpec(BaseModel):
    kind: ChartKind
    x: str | None = None
    y: list[str] = []


def suggest_chart(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> ChartSpec | None:
    if not rows or not columns:
        return None
    numeric = [c for i, c in enumerate(columns) if _all_numeric(r[i] for r in rows)]
    labels = [c for c in columns if c not in numeric]
    if len(rows) == 1 and numeric and not labels and len(numeric) <= 4:
        return ChartSpec(kind="metric", y=numeric)
    if not numeric or not labels or len(rows) < 2:
        return None
    x = labels[0]
    x_values = [r[columns.index(x)] for r in rows]
    if all(isinstance(v, str) and _DATE_LIKE.match(v) for v in x_values):
        return ChartSpec(kind="line", x=x, y=numeric[:3])
    if len(rows) <= MAX_BARS:
        return ChartSpec(kind="bar", x=x, y=numeric[:1])
    return None


def make_figure(columns: Sequence[str], rows: Sequence[Sequence[Any]], spec: ChartSpec) -> Any:
    """A Plotly figure for a line or bar spec (metrics are shown as numbers, not charts)."""
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(list(rows), columns=list(columns))
    if spec.kind == "line":
        fig = px.line(df, x=spec.x, y=spec.y, markers=True)
    elif spec.kind == "bar":
        fig = px.bar(df, x=spec.x, y=spec.y[0])
    else:
        raise ValueError(f"no figure for {spec.kind!r}")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, height=380, legend_title_text="")
    return fig


def _all_numeric(values: Any) -> bool:
    seen = False
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        seen = True
    return seen
