"""Helpers for 00_start_here.ipynb: they hide the setup so each notebook cell stays one line.

Everything here calls the project's real code (src/analyst/) on the real Olist data.
"""

from __future__ import annotations

import itertools
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyst.agent import Agent  # noqa: E402
from analyst.catalog import CatalogService, CatalogStore  # noqa: E402
from analyst.config import AgentSettings, QueryLimits, Settings, SourceConfig  # noqa: E402
from analyst.connectors import QueryError, SourceRegistry, SqlAlchemyConnector  # noqa: E402
from analyst.semantic import load_semantics  # noqa: E402
from analyst.sql_guard import UnsafeQueryError  # noqa: E402

OLIST = ROOT / "data" / "olist" / "olist.sqlite"
_db = SqlAlchemyConnector(
    "olist",
    f"sqlite:///file:{OLIST}?mode=ro&uri=true",  # read-only: nothing can change the data
    limits=QueryLimits(default_row_limit=20, max_row_limit=1000),
)
_catalog = None


def run_sql(sql: str) -> pd.DataFrame | str:
    """Run SQL the same way the AI does: through the safety guard, read-only."""
    try:
        result = _db.run_query(sql)
    except UnsafeQueryError as e:
        return f"BLOCKED by the safety guard: {e}"
    except QueryError as e:
        return f"The database said: {e}"
    return pd.DataFrame(result.rows, columns=result.columns)


def tables() -> pd.DataFrame:
    """The tables in the Olist database and how many rows each has."""
    rows = []
    for t in _db.get_schema().tables:
        count = _db.run_query(f'SELECT COUNT(*) FROM "{t.name}"').rows[0][0]
        rows.append({"table": t.name, "rows": count, "columns": len(t.columns)})
    return pd.DataFrame(rows)


def _get_catalog():
    global _catalog
    if _catalog is None:
        settings = Settings(
            state_dir=Path(tempfile.mkdtemp()),
            sources=[SourceConfig(name="olist", type="sqlite", path=OLIST)],
        )
        registry = SourceRegistry(settings, only={"olist"})
        service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
        _catalog = service.refresh("olist")
    return _catalog


def what_the_ai_knows(table: str) -> pd.DataFrame:
    """What the catalog tells the AI about one table's columns."""
    t = _get_catalog().table(table)
    return pd.DataFrame([
        {
            "column": c.name,
            "empty values": f"{(c.profile.null_fraction or 0):.0%}",
            "different values": c.profile.distinct_count,
            "every value unique?": "yes" if c.profile.is_unique else "no",
            "examples": ", ".join(str(s) for s in c.profile.samples[:3]),
        }
        for c in t.columns
    ])


def joins() -> pd.DataFrame:
    """How tables connect: the catalog works these out from the data."""
    return pd.DataFrame(
        [{"this column": f"{r.from_table}.{r.from_column}",
          "points to": f"{r.to_table}.{r.to_column}"} for r in _get_catalog().relationships]
    )


def definitions() -> pd.DataFrame:
    """The business definitions you approved (metrics.yaml)."""
    sem = load_semantics(ROOT / "metrics.yaml")["olist"]
    return pd.DataFrame(
        [{"term": name, "means": d.description} for name, d in sem.definitions.items()]
    )


_ids = itertools.count(1)
LATE_SQL = (
    "SELECT ROUND(100.0 * SUM(date(order_delivered_customer_date) > "
    "date(order_estimated_delivery_date)) / COUNT(*), 2) AS late_pct FROM orders "
    "WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL"
)


def _asks(tool: str, **inputs) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", id=f"call_{next(_ids)}", name=tool, input=inputs)
    usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage,
                           stop_details=None)


_PLAIN = {
    "list_tables": "Which tables are there?",
    "describe_table": "What's in the {table} table?",
    "run_sql": "Please run this SQL: {sql}",
    "submit_answer": "I'm done. My answer: {answer}",
}


class _PretendAI:
    """Plays the AI's part with pre-written requests, and narrates each step."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.step = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._reply))

    def _reply(self, **request):
        last = request["messages"][-1]
        if isinstance(last["content"], list):
            text = last["content"][0]["content"]
            print(f"   OUR CODE answers: {text[:110]}...\n")
        reply = self.replies.pop(0)
        call = reply.content[0]
        self.step += 1
        print(f"Step {self.step}")
        print(f"   AI asks: {_PLAIN[call.name].format(**call.input)[:110]}")
        return reply


def watch_the_ai(question: str = "What is our late-delivery rate?",
                 sql_to_try: str = "SELECT order_status, COUNT(*) FROM orders GROUP BY 1"):
    """Show, step by step, how the AI and our code work together on one question."""
    pretend = _PretendAI([
        _asks("list_tables"),
        _asks("describe_table", table="orders"),
        _asks("run_sql", sql=sql_to_try),
        _asks("submit_answer", answer="6.77% of delivered orders arrived late.",
              sql=LATE_SQL, answerable=True),
    ])
    semantics = load_semantics(ROOT / "metrics.yaml")["olist"]
    agent = Agent(pretend, _db, _get_catalog(), AgentSettings(), semantics)
    print(f"QUESTION: {question}\n")
    answer = agent.ask(question)
    print(f"\nFINAL ANSWER: {answer.answer}")
    print(f"Our code re-ran the AI's final SQL to double-check it and got: {answer.result.rows}")
    return answer


def check(answer) -> str:
    """Compare the AI's result with the known-correct answer from the golden set."""
    import yaml

    from analyst.evals.scoring import answer_match

    golden = yaml.safe_load((ROOT / "evals" / "golden.yaml").read_text())
    expected = next(q for q in golden["questions"] if q["id"] == "del_late_rate")
    passed, _ = answer_match(answer.result.columns, answer.result.rows,
                             expected["expected"]["rows"], ordered=False)
    verdict = "CORRECT" if passed else "WRONG"
    return (f"{verdict}: the AI said {answer.result.rows[0][0]}, "
            f"the known answer is {expected['expected']['rows'][0][0]}")


def system_prompt_preview() -> None:
    """The first part of the instructions the AI receives before your question."""
    from analyst.agent.prompt import build_system_prompt

    sem = load_semantics(ROOT / "metrics.yaml")["olist"]
    print(build_system_prompt(_get_catalog(), sem)[:900], "\n...")


__all__ = ["check", "definitions", "joins", "run_sql", "system_prompt_preview", "tables",
           "watch_the_ai", "what_the_ai_knows"]
