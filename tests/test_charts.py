from analyst.charts import make_figure, suggest_chart


def test_single_number_is_a_metric() -> None:
    assert suggest_chart(["revenue"], [[13494400.74]]).kind == "metric"
    assert suggest_chart(["delivered", "late", "late_pct"], [[96470, 6534, 6.77]]).y == [
        "delivered",
        "late",
        "late_pct",
    ]


def test_months_make_a_line() -> None:
    spec = suggest_chart(["month", "revenue"], [["2017-01", 120098.27], ["2017-02", 244959.35]])
    assert spec.kind == "line" and spec.x == "month" and spec.y == ["revenue"]


def test_categories_make_a_bar() -> None:
    rows = [["health_beauty", 1255695.13], ["watches_gifts", 1198185.21]]
    spec = suggest_chart(["category", "revenue"], rows)
    assert spec.kind == "bar" and spec.x == "category"
    assert make_figure(["category", "revenue"], rows, spec).data[0].type == "bar"


def test_no_chart_when_a_table_is_better() -> None:
    assert suggest_chart(["state"], [["SP"], ["RJ"]]) is None  # nothing numeric
    assert suggest_chart(["n"], []) is None
    many = [[f"city_{i}", i] for i in range(100)]
    assert suggest_chart(["city", "customers"], many) is None  # too many bars
    assert suggest_chart(["city", "state", "n"], [["sao paulo", "SP", 1]]) is None  # one row
