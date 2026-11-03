"""Model providers. The agent loop speaks the Anthropic Messages shape; other providers get an
adapter that translates to and from an OpenAI-style chat API (Gemini, Groq, OpenRouter, Ollama).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from analyst.config import AgentSettings


class ProviderError(RuntimeError):
    """A provider call failed in a way the caller should report, not retry."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class QuestionFailedError(ProviderError):
    """The provider can't go on with this one question (the model's fault or its size), but
    the next question may be fine: an eval run scores it as failed and carries on."""


class BadToolCallError(QuestionFailedError):
    """The model kept writing tool calls the provider couldn't parse."""

    def __init__(self, message: str) -> None:
        super().__init__(400, f"model wrote an invalid tool call: {message}")


class RequestTooLargeError(QuestionFailedError):
    """The conversation outgrew the provider's per-request size limit (Groq free plan: 8k)."""

    def __init__(self, message: str) -> None:
        super().__init__(413, f"conversation too large for the provider: {message}")


def make_llm_client(settings: AgentSettings, on_wait: Callable[[str], None] | None = None) -> Any:
    """`on_wait` is told when a busy provider makes us pause (for progress output)."""
    if settings.provider == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    from openai import OpenAI

    if settings.api_key_env is None:
        key = "not-needed"  # a local server such as Ollama; the SDK still wants a value
    else:
        key = os.environ.get(settings.api_key_env, "")
        if not key:
            raise ProviderError(401, f"{settings.api_key_env} is not set (add it to .env)")
    return OpenAICompatClient(
        # Our own retry loop handles busy free tiers; the SDK's would stack on top of it.
        OpenAI(
            base_url=settings.base_url,
            api_key=key,
            max_retries=0,
            timeout=settings.request_timeout_s,
        ),
        min_seconds_between_calls=settings.min_seconds_between_calls,
        on_wait=on_wait,
        reasoning_effort=settings.reasoning_effort,
        max_quota_wait_s=settings.max_quota_wait_s,
    )


OLLAMA_URL = "http://localhost:11434/v1"


def ollama_settings(settings: AgentSettings, model: str, *, think: bool = False) -> AgentSettings:
    """Settings for a model served by a local Ollama: no key, no throttle, patient timeout.
    Thinking is off by default: on a laptop it can make one call take over ten minutes."""
    return settings.model_copy(
        update={
            "provider": "openai_compatible",
            "base_url": OLLAMA_URL,
            "api_key_env": None,
            "model": model,
            "min_seconds_between_calls": 0.0,
            "request_timeout_s": 900.0,
            "reasoning_effort": None if think else "none",
        }
    )


GROQ_URL = "https://api.groq.com/openai/v1"


def groq_settings(settings: AgentSettings, model: str) -> AgentSettings:
    """Settings for Groq's free plan (key in GROQ_API_KEY). Its tightest limit is tokens per
    minute, so calls are spaced out; per-minute 429s are waited out, per-day ones stop the run."""
    return settings.model_copy(
        update={
            "provider": "openai_compatible",
            "base_url": GROQ_URL,
            "api_key_env": "GROQ_API_KEY",
            "model": model,
            "min_seconds_between_calls": 10.0,
            "request_timeout_s": 120.0,
            "reasoning_effort": None,
            # Fewer result rows per query keep each request under the 8k-token limit.
            "result_rows_shown": 20,
            # Groq's daily limit is a rolling window that names a short wait: sit it out.
            "max_quota_wait_s": 1800.0,
        }
    )


class OpenAICompatClient:
    """Looks like anthropic.Anthropic to the agent: exposes `beta.messages.create(...)`."""

    RATE_LIMIT_RETRIES = 5
    RATE_LIMIT_WAIT_S = 20.0
    BAD_TOOL_CALL_RETRIES = 2

    def __init__(
        self,
        openai_client: Any,
        *,
        min_seconds_between_calls: float = 0.0,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
        on_wait: Callable[[str], None] | None = None,
        reasoning_effort: str | None = None,
        max_quota_wait_s: float = 0.0,
    ) -> None:
        self._openai = openai_client
        self._reasoning_effort = reasoning_effort
        self._max_quota_wait = max_quota_wait_s
        self._on_wait = on_wait
        self._min_gap = min_seconds_between_calls
        self._sleep, self._clock = sleep, clock
        self._last_call: float | None = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: Any = None,
        tools: list[dict[str, Any]] | None = None,
        **_claude_only: Any,
    ) -> SimpleNamespace:
        # output_config, betas, fallbacks and cache_control are Claude features: ignored here.
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            request["tools"] = [to_openai_tool(t) for t in tools]
            request["tool_choice"] = "auto"
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort
        return from_openai_response(self._send(request))

    def _send(self, request: dict[str, Any]) -> Any:
        """Free tiers are busy: rate limits (429), overloads (5xx) and timeouts are waited out
        with growing pauses; anything else is reported at once."""
        import openai

        bad_tool_calls = quota_waits = attempt = 0
        while True:
            self._throttle()
            try:
                return self._openai.chat.completions.create(**request)
            except openai.APIStatusError as e:
                if _bad_tool_call(e):
                    # The model wrote a malformed tool call (Groq rejects it with a 400).
                    # Sampling again usually works; after that it's the model's answer.
                    bad_tool_calls += 1
                    if bad_tool_calls > self.BAD_TOOL_CALL_RETRIES:
                        raise BadToolCallError(_message(e)) from e
                    continue
                if e.status_code == 413:  # the same request will never fit
                    raise RequestTooLargeError(_message(e)[:160]) from e
                if e.status_code != 429 and e.status_code < 500:
                    raise ProviderError(e.status_code, _message(e)) from e
                daily = _daily_quota(e)
                if daily:
                    # A rolling daily window (Groq) frees tokens bit by bit and says when to
                    # try again: wait if that's soon. Otherwise retrying can't help.
                    wait = _retry_after(e)
                    if wait is not None and wait <= self._max_quota_wait and quota_waits < 100:
                        quota_waits += 1
                        if self._on_wait:
                            self._on_wait(f"daily token limit; waiting {wait / 60:.0f} min")
                        self._sleep(wait + 5)
                        continue
                    raise ProviderError(429, f"{request['model']}: {daily}") from e
                failure: tuple[int | None, str] = (e.status_code, _message(e))
            except openai.APIConnectionError as e:  # includes timeouts
                failure = (None, f"could not reach the {request['model']} provider ({e})")
            if attempt == self.RATE_LIMIT_RETRIES:
                status, message = failure
                raise ProviderError(status, f"gave up after {attempt + 1} tries: {message}")
            attempt += 1
            wait = self.RATE_LIMIT_WAIT_S * attempt
            if self._on_wait:
                self._on_wait(
                    f"provider busy ({failure[1][:70]}); waiting {wait:.0f}s, "
                    f"retry {attempt}/{self.RATE_LIMIT_RETRIES}"
                )
            self._sleep(wait)

    def _throttle(self) -> None:
        now = self._clock()
        if self._last_call is not None and self._min_gap > 0:
            wait = self._min_gap - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()


def to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool["name"], "description": tool["description"]}
    schema = _portable_schema(tool["input_schema"])
    if schema.get("properties"):
        function["parameters"] = schema
    return {"type": "function", "function": function}


def _portable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop JSON-Schema features some providers reject (Gemini's schema is an OpenAPI subset):
    `additionalProperties`, and union types like ["string", "null"] (kept as the first type)."""
    out = copy.deepcopy(schema)
    out.pop("additionalProperties", None)
    for prop in out.get("properties", {}).values():
        if isinstance(prop.get("type"), list):
            prop["type"] = next(t for t in prop["type"] if t != "null")
    return out


def to_openai_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        text = system if isinstance(system, str) else "\n".join(b["text"] for b in system)
        out.append({"role": "system", "content": text})
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            out.append(_assistant_message(content))
        else:
            for block in content:
                if _get(block, "type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": _get(block, "tool_use_id"),
                            "content": _get(block, "content"),
                        }
                    )
                elif _get(block, "type") == "text":
                    out.append({"role": "user", "content": _get(block, "text")})
    return out


def _assistant_message(blocks: list[Any]) -> dict[str, Any]:
    text = "\n".join(_get(b, "text") for b in blocks if _get(b, "type") == "text").strip()
    calls = [
        {
            "id": _get(b, "id"),
            "type": "function",
            "function": {"name": _get(b, "name"), "arguments": json.dumps(_get(b, "input"))},
            # Provider data that must travel back unchanged, e.g. Gemini's thought_signature.
            **(_get(b, "provider_extra") or {}),
        }
        for b in blocks
        if _get(b, "type") == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    return message


_STOP_REASONS = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def from_openai_response(response: Any) -> SimpleNamespace:
    choice = response.choices[0]
    message = choice.message
    blocks: list[SimpleNamespace] = []
    if message.content:
        blocks.append(SimpleNamespace(type="text", text=message.content))
    for call in message.tool_calls or []:
        try:
            tool_input = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            tool_input = {"_unparsed_arguments": call.function.arguments}
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                id=call.id or f"call_{uuid.uuid4().hex}",
                name=call.function.name,
                input=tool_input,
                provider_extra=dict(getattr(call, "model_extra", None) or {}),
            )
        )
    stop = _STOP_REASONS.get(choice.finish_reason or "", "end_turn")
    if stop == "end_turn" and any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"  # some providers report "stop" even when they called a tool
    usage = getattr(response, "usage", None)
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def _get(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def _bad_tool_call(error: Any) -> bool:
    body = getattr(error, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    detail = body.get("error", body) if isinstance(body, dict) else {}
    if error.status_code != 400 or not isinstance(detail, dict):
        return False
    # Groq rejects unparseable model output with a 400 and echoes it back as failed_generation,
    # under codes such as tool_use_failed or output_parse_failed.
    message = str(detail.get("message", "")).lower()
    return (
        "failed_generation" in detail
        or detail.get("code") in {"tool_use_failed", "output_parse_failed"}
        or "could not be parsed" in message
        or "failed to parse tool call" in message
    )


def _retry_after(error: Any) -> float | None:
    """Seconds until a limit frees up, from "try again in 6m26.6s" (Groq) or a retry-after
    header; None when the provider doesn't say."""
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", _message(error))
    if match and any(match.groups()):
        h, m, sec = (float(g or 0) for g in match.groups())
        return h * 3600 + m * 60 + sec
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _daily_quota(error: Any) -> str | None:
    """If a 429 is a per-day quota (not a short burst limit), describe it; else None."""
    body = getattr(error, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    detail = body.get("error", body) if isinstance(body, dict) else {}
    text = json.dumps(detail) if isinstance(detail, dict) else str(detail)
    if "PerDay" not in text and "per day" not in text.lower():
        return None
    limit = re.search(r'"quotaValue":\s*"(\d+)"', text)
    retry = re.search(r'"retryDelay":\s*"(\d+)s"', text)
    text = "daily free quota used up"
    if limit:
        text += f" ({limit.group(1)} requests/day)"
    if retry:
        text += f"; resets in about {int(retry.group(1)) / 3600:.0f}h"
    return text


def _message(error: Any) -> str:
    """The provider's own error text: OpenAI-compatible APIs put the useful part in the body."""
    body = getattr(error, "body", None)
    if isinstance(body, list) and body:  # Gemini wraps the error object in a list
        body = body[0]
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict) and detail.get("message"):
            return str(detail["message"])
    return str(getattr(error, "message", None) or error)
