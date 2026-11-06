"""The OpenAI-compatible adapter (Gemini, Groq, OpenRouter, Ollama), tested offline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from analyst.agent import TOOL_DEFINITIONS, Agent
from analyst.agent.providers import (
    BadToolCallError,
    OpenAICompatClient,
    ProviderError,
    RequestTooLargeError,
    groq_settings,
    make_llm_client,
    nvidia_settings,
    ollama_settings,
    to_openai_tool,
)
from analyst.catalog import CatalogService, CatalogStore
from analyst.config import AgentSettings, Settings
from analyst.connectors import SourceRegistry
from analyst.semantic import load_semantics

COUNT_SQL = "SELECT COUNT(*) AS orders FROM orders"


def call(name: str, call_id: str, **arguments) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def completion(
    *calls: SimpleNamespace, content: str | None = None, finish: str = "tool_calls"
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=list(calls) or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=500, completion_tokens=40),
    )


class FakeOpenAI:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def olist_agent_parts(settings: Settings):
    registry = SourceRegistry(settings)
    catalog = CatalogService(registry, CatalogStore(settings.app_db_path), settings.profiling)
    built = catalog.refresh("olist")
    semantics = load_semantics(Path(__file__).parents[1] / "metrics.yaml")["olist"]
    yield registry.get("olist"), built, semantics
    registry.close()


def test_agent_runs_end_to_end_through_the_adapter(olist_agent_parts) -> None:
    fake = FakeOpenAI(
        completion(call("list_tables", "c1")),
        completion(call("run_sql", "c2", sql=COUNT_SQL), content="Counting orders."),
        completion(
            call("submit_answer", "c3", answer="2,000 orders", sql=COUNT_SQL, answerable=True),
            finish="stop",
        ),
    )
    client = OpenAICompatClient(fake)
    settings = AgentSettings(provider="openai_compatible", model="gemini-flash-latest")
    answer = Agent(client, *olist_agent_parts[:2], settings, olist_agent_parts[2]).ask(
        "How many orders?"
    )

    assert answer.status == "answered" and answer.result.rows == [[2000]]
    assert answer.usage.input_tokens == 1500 and answer.turns == 3

    first, _, last = fake.requests
    assert first["model"] == "gemini-flash-latest" and first["tool_choice"] == "auto"
    assert first["messages"][0]["role"] == "system"
    assert "customer_unique_id" in first["messages"][0]["content"]
    assert first["messages"][1] == {"role": "user", "content": "How many orders?"}
    assert not any(k in first for k in ("betas", "fallbacks", "output_config"))

    roles = [m["role"] for m in last["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assistant = last["messages"][4]
    assert assistant["content"] == "Counting orders."
    assert assistant["tool_calls"][0]["id"] == "c2"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"sql": COUNT_SQL}
    tool = last["messages"][5]
    assert tool["tool_call_id"] == "c2" and json.loads(tool["content"])["rows"] == [[2000]]


def test_thought_signatures_are_sent_back(olist_agent_parts) -> None:
    signed = call("list_tables", "c1")
    signed.model_extra = {"extra_content": {"google": {"thought_signature": "sig-123"}}}
    fake = FakeOpenAI(
        completion(signed),
        completion(call("submit_answer", "c2", answer="x", sql=COUNT_SQL, answerable=True)),
    )
    settings = AgentSettings(provider="openai_compatible", model="m")
    Agent(OpenAICompatClient(fake), *olist_agent_parts[:2], settings).ask("q")
    echoed = fake.requests[1]["messages"][2]["tool_calls"][0]
    assert echoed["extra_content"] == {"google": {"thought_signature": "sig-123"}}
    assert echoed["id"] == "c1"


def test_overloads_and_timeouts_are_retried() -> None:
    overloaded = Exception.__new__(openai.APIStatusError)
    overloaded.status_code, overloaded.message, overloaded.body = 503, "high demand", None
    timeout = Exception.__new__(openai.APITimeoutError)
    sleeps: list[float] = []
    fake = FakeOpenAI(overloaded, timeout, completion(content="ok", finish="stop"))
    client = OpenAICompatClient(fake, sleep=sleeps.append)
    reply = client.beta.messages.create(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.content[0].text == "ok" and sleeps == [20.0, 40.0]

    bad_request = Exception.__new__(openai.APIStatusError)
    bad_request.status_code, bad_request.message, bad_request.body = 400, "bad schema", None
    client = OpenAICompatClient(FakeOpenAI(bad_request), sleep=sleeps.append)
    with pytest.raises(ProviderError, match="bad schema"):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )


def test_daily_quota_is_not_retried() -> None:
    daily = Exception.__new__(openai.APIStatusError)
    daily.status_code, daily.message = 429, "You exceeded your current quota"
    daily.body = [
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota",
                "details": [
                    {
                        "violations": [
                            {
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                "quotaValue": "20",
                            }
                        ]
                    },
                    {"retryDelay": "46702s"},
                ],
            }
        }
    ]
    sleeps: list[float] = []
    fake = FakeOpenAI(daily, completion(content="never reached", finish="stop"))
    client = OpenAICompatClient(fake, sleep=sleeps.append)
    with pytest.raises(
        ProviderError,
        match=r"daily free quota used up \(20 requests/day\)"
        r"; resets in about 13h",
    ):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    assert sleeps == [] and len(fake.requests) == 1


def test_tool_schemas_are_made_portable() -> None:
    tools = {t["name"]: to_openai_tool(t) for t in TOOL_DEFINITIONS}
    assert "parameters" not in tools["list_tables"]["function"]  # no-argument tool
    submit = tools["submit_answer"]["function"]["parameters"]
    assert "additionalProperties" not in submit and "strict" not in tools["submit_answer"]
    assert submit["properties"]["sql"]["type"] == "string"  # was ["string", "null"]
    assert submit["required"] == ["answer", "sql", "answerable"]
    # The originals (used for Claude) are untouched.
    claude_submit = next(t for t in TOOL_DEFINITIONS if t["name"] == "submit_answer")
    assert claude_submit["input_schema"]["properties"]["sql"]["type"] == ["string", "null"]


def test_unparseable_arguments_become_a_tool_error(olist_agent_parts) -> None:
    broken = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="run_sql", arguments="{not json")
    )
    fake = FakeOpenAI(
        completion(broken),
        completion(call("submit_answer", "c2", answer="x", sql=COUNT_SQL, answerable=True)),
    )
    settings = AgentSettings(provider="openai_compatible", model="m")
    answer = Agent(OpenAICompatClient(fake), *olist_agent_parts[:2], settings).ask("q")
    assert answer.status == "answered" and not answer.steps[0].ok
    assert fake.requests[1]["messages"][-1]["content"].startswith("Error: bad input")


def test_requests_are_spaced_out() -> None:
    sleeps: list[float] = []
    times = iter([0.0, 0.0, 1.0, 5.0])
    fake = FakeOpenAI(
        completion(content="a", finish="stop"), completion(content="b", finish="stop")
    )
    client = OpenAICompatClient(
        fake, min_seconds_between_calls=5, sleep=sleeps.append, clock=lambda: next(times)
    )
    for _ in range(2):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    assert sleeps == [4.0]


def _rate_limit_error() -> openai.RateLimitError:
    error = Exception.__new__(openai.RateLimitError)
    error.message = "quota exceeded"
    return error


def test_rate_limits_are_waited_out_then_reported() -> None:
    sleeps: list[float] = []
    fake = FakeOpenAI(_rate_limit_error(), completion(content="ok", finish="stop"))
    client = OpenAICompatClient(fake, sleep=sleeps.append)
    reply = client.beta.messages.create(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.content[0].text == "ok" and sleeps == [20.0]

    always = FakeOpenAI(*[_rate_limit_error() for _ in range(6)])
    client = OpenAICompatClient(always, sleep=lambda _s: None)
    with pytest.raises(ProviderError, match="gave up after 6 tries: quota exceeded"):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )


def test_missing_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = AgentSettings(
        provider="openai_compatible",
        api_key_env="GEMINI_API_KEY",
        base_url="https://example.invalid/v1/",
    )
    with pytest.raises(ProviderError, match="GEMINI_API_KEY is not set"):
        make_llm_client(settings)


def test_ollama_needs_no_key_and_turns_thinking_off(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    local = ollama_settings(AgentSettings(api_key_env="GEMINI_API_KEY"), "qwen3:4b")
    assert (local.api_key_env, local.min_seconds_between_calls) == (None, 0.0)
    assert isinstance(make_llm_client(local), OpenAICompatClient)  # no key error

    fake = FakeOpenAI(completion(content="ok", finish="stop"))
    client = OpenAICompatClient(fake, reasoning_effort=local.reasoning_effort)
    client.beta.messages.create(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert fake.requests[0]["reasoning_effort"] == "none"
    assert ollama_settings(AgentSettings(), "m", think=True).reasoning_effort is None


def test_groq_preset_reads_its_own_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq = groq_settings(AgentSettings(), "openai/gpt-oss-120b")
    assert groq.base_url.startswith("https://api.groq.com/") and groq.min_seconds_between_calls > 0
    with pytest.raises(ProviderError, match="GROQ_API_KEY is not set"):
        make_llm_client(groq)


def test_nvidia_preset_reads_its_own_key(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    nvidia = nvidia_settings(AgentSettings(), "openai/gpt-oss-20b")
    assert nvidia.base_url.startswith("https://integrate.api.nvidia.com/")
    assert nvidia.min_seconds_between_calls > 0
    with pytest.raises(ProviderError, match="NVIDIA_API_KEY is not set"):
        make_llm_client(nvidia)


def _bad_tool_call_error() -> openai.BadRequestError:
    error = Exception.__new__(openai.BadRequestError)
    error.status_code = 400
    error.message = "Failed to parse tool call arguments as JSON"
    error.body = {"error": {"message": error.message, "code": "tool_use_failed"}}
    return error


def test_a_malformed_tool_call_is_asked_again_then_reported() -> None:
    fake = FakeOpenAI(_bad_tool_call_error(), completion(content="ok", finish="stop"))
    client = OpenAICompatClient(fake, sleep=lambda _s: None)
    reply = client.beta.messages.create(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.content[0].text == "ok"

    always = FakeOpenAI(*[_bad_tool_call_error() for _ in range(3)])
    client = OpenAICompatClient(always, sleep=lambda _s: None)
    with pytest.raises(BadToolCallError, match="invalid tool call"):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )


def test_a_request_over_the_size_limit_is_not_retried() -> None:
    error = Exception.__new__(openai.APIStatusError)
    error.status_code = 413
    error.message = "Request too large for model on tokens per minute (TPM): Limit 8000"
    error.body = None
    sleeps: list[float] = []
    client = OpenAICompatClient(FakeOpenAI(error), sleep=sleeps.append)
    with pytest.raises(RequestTooLargeError, match="too large"):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    assert sleeps == []


def test_unparseable_output_in_any_wording_is_a_bad_tool_call() -> None:
    error = Exception.__new__(openai.BadRequestError)
    error.status_code = 400
    error.message = "Parsing failed. The model generated output that could not be parsed."
    error.body = {"error": {"message": error.message, "failed_generation": "{bad"}}
    client = OpenAICompatClient(FakeOpenAI(*[error] * 3), sleep=lambda _s: None)
    with pytest.raises(BadToolCallError):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )


def _groq_daily_limit(retry: str) -> openai.RateLimitError:
    error = Exception.__new__(openai.RateLimitError)
    error.status_code = 429
    error.message = (
        "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 197818, "
        f"Requested 3077. Please try again in {retry}. Need more tokens?"
    )
    error.body = {"error": {"message": error.message}}
    return error


def test_a_rolling_daily_limit_is_waited_out_when_soon() -> None:
    sleeps: list[float] = []
    fake = FakeOpenAI(_groq_daily_limit("6m26.5s"), completion(content="ok", finish="stop"))
    client = OpenAICompatClient(fake, sleep=sleeps.append, max_quota_wait_s=1800)
    reply = client.beta.messages.create(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.content[0].text == "ok" and sleeps == [391.5]

    far = FakeOpenAI(_groq_daily_limit("2h10m0s"))
    client = OpenAICompatClient(far, sleep=sleeps.append, max_quota_wait_s=1800)
    with pytest.raises(ProviderError, match="daily free quota"):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    # Without the setting (Gemini), a daily limit still stops at once.
    client = OpenAICompatClient(FakeOpenAI(_groq_daily_limit("1m0s")), sleep=sleeps.append)
    with pytest.raises(ProviderError, match="daily free quota"):
        client.beta.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
