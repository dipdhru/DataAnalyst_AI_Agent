"""Multi-day eval runs: progress survives a quota stop and resumes where it left off."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from fake_claude import FakeClaude, reply, tool_use

from analyst.agent import Agent
from analyst.agent.providers import ProviderError
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import Settings
from analyst.connectors import SourceRegistry
from analyst.evals.campaign import Campaign, CampaignMismatchError
from analyst.evals.runner import run_eval

QUESTIONS = [
    {
        "id": "q_orders",
        "category": "orders",
        "question": "How many orders?",
        "sql": "SELECT COUNT(*) FROM orders",
        "expected": {"columns": ["n"], "rows": [[2000]]},
    },
    {"id": "q_margin", "category": "unanswerable", "question": "What is our margin?"},
    {
        "id": "q_items",
        "category": "orders",
        "question": "How many order items?",
        "sql": "SELECT COUNT(*) FROM order_items",
        "expected": {"columns": ["n"], "rows": [[2000]]},
    },
]


class QuotaAfter(FakeClaude):
    """Scripted replies, then a daily-quota error, like the free tier at the end of the day."""

    def _create(self, **kwargs):
        if not self.replies:
            raise ProviderError(429, "m: daily free quota used up (20 requests/day)")
        return super()._create(**kwargs)


@pytest.fixture
def olist_env(settings: Settings):
    registry = SourceRegistry(settings, only={"olist"})
    service = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    catalog = service.refresh("olist")
    yield registry.get("olist"), catalog, settings
    registry.close()


def run_day(olist_env, campaign: Campaign, client) -> None:
    connector, catalog, settings = olist_env
    with contextlib.suppress(ProviderError):  # the quota ending the day
        run_eval(
            campaign.remaining(QUESTIONS),
            lambda: Agent(client, connector, catalog, settings.agent),
            model="m",
            on_result=campaign.add,
        )


def test_resumes_where_the_quota_stopped_it(olist_env, tmp_path: Path) -> None:
    path = tmp_path / "campaign-m.json"

    day1 = QuotaAfter(
        reply(
            tool_use(
                "submit_answer",
                answer="2000",
                sql="SELECT COUNT(*) AS n FROM orders",
                answerable=True,
            )
        )
    )
    run_day(olist_env, Campaign(path, model="m", provider="p"), day1)
    saved = Campaign(path, model="m", provider="p")
    assert [q["id"] for q in saved.remaining(QUESTIONS)] == ["q_margin", "q_items"]

    day2 = QuotaAfter(
        reply(tool_use("submit_answer", answer="No cost data.", sql=None, answerable=False)),
        reply(
            tool_use(
                "submit_answer",
                answer="2000",
                sql="SELECT COUNT(*) AS n FROM order_items",
                answerable=True,
            )
        ),
    )
    run_day(olist_env, saved, day2)
    final = Campaign(path, model="m", provider="p")
    assert final.remaining(QUESTIONS) == []
    assert {r.id: r.passed for r in final.results()} == {
        "q_orders": True,
        "q_margin": True,
        "q_items": True,
    }
    assert not path.with_suffix(".tmp").exists()


def test_an_interrupted_question_is_asked_again(olist_env, tmp_path: Path) -> None:
    path = tmp_path / "campaign-m.json"
    # The quota runs out in the middle of the first question: nothing is saved for it.
    run_day(
        olist_env,
        Campaign(path, model="m", provider="p"),
        QuotaAfter(reply(tool_use("describe_table", table="orders"))),
    )
    assert len(Campaign(path, model="m", provider="p").remaining(QUESTIONS)) == 3


def test_models_are_never_mixed_in_one_campaign(tmp_path: Path, olist_env) -> None:
    path = tmp_path / "campaign-m.json"
    run_day(
        olist_env,
        Campaign(path, model="m", provider="p"),
        QuotaAfter(
            reply(
                tool_use(
                    "submit_answer",
                    answer="x",
                    sql="SELECT COUNT(*) AS n FROM orders",
                    answerable=True,
                )
            )
        ),
    )
    with pytest.raises(CampaignMismatchError, match="started with m"):
        Campaign(path, model="other-model", provider="p")
