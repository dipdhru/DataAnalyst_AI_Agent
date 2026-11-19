"""The agent loop: Claude plans, calls tools, checks results and submits one answer.

A manual loop rather than the SDK tool runner, so that every tool call is recorded, the final
answer arrives as a structured submit_answer call, and the final SQL is executed (and fed back
on failure) before the answer is accepted.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from analyst.agent.prompt import build_pipeline_prompt, build_system_prompt
from analyst.agent.retrieval import SchemaRetriever, full_schema_text
from analyst.agent.tools import TOOL_DEFINITIONS, AgentTools, ToolCall
from analyst.catalog import Catalog
from analyst.config import AgentSettings
from analyst.connectors import Connector, QueryError
from analyst.models import QueryResult
from analyst.semantic import SourceSemantics
from analyst.sql_guard import UnsafeQueryError

# Server-side refusal fallback: a declined request is retried on a suitable model. Only these
# models accept it (Haiku 5.5 has none); others get the refusal back as is.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_NUDGE = "Please finish by calling submit_answer with your answer and final SQL."

Status = Literal["answered", "unanswerable", "max_steps", "refusal", "error"]


class MessagesClient(Protocol):
    """The slice of anthropic.Anthropic the agent uses (lets tests pass a fake)."""

    @property
    def beta(self) -> Any: ...


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        for field in type(self).model_fields:
            setattr(self, field, getattr(self, field) + (getattr(usage, field, 0) or 0))


class Turn(BaseModel):
    """One earlier question and its answer, replayed so follow-up questions have context."""

    question: str
    answer: str | None = None
    sql: str | None = None


class AgentAnswer(BaseModel):
    question: str
    status: Status
    answer: str | None = None
    sql: str | None = None
    answerable: bool | None = None
    result: QueryResult | None = None
    steps: list[ToolCall] = Field(default_factory=list)
    turns: int = 0
    elapsed_s: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    detail: str | None = None  # why the run ended, when not answered
    mode: str = "agent"
    retrieved_tables: list[str] | None = None  # retrieval modes: tables the model was shown
    corrections: int = 0  # failed submissions fed back to the model

    @property
    def final_sql(self) -> str | None:
        """The accepted SQL as the agent wrote it; `sql` carries the guard's added LIMIT."""
        for step in reversed(self.steps):
            if step.tool == "submit_answer" and step.ok:
                return step.input.get("sql")
        return None


class Agent:
    def __init__(
        self,
        client: MessagesClient,
        connector: Connector,
        catalog: Catalog,
        settings: AgentSettings,
        semantics: SourceSemantics | None = None,
        on_step: Callable[[ToolCall], None] | None = None,
    ) -> None:
        """`on_step` is called after every tool call, e.g. to show progress."""
        self.client = client
        self.settings = settings
        self.on_step = on_step
        self.tools = AgentTools(connector, catalog, rows_shown=settings.result_rows_shown)
        self.mode = settings.mode
        self.retriever = None
        self.last_answer: AgentAnswer | None = None
        if self.mode == "agent":
            self.system = build_system_prompt(catalog, semantics)
            self.tool_definitions = TOOL_DEFINITIONS
        else:
            # Single-shot pipelines: the model may only submit; it cannot explore.
            self.tool_definitions = [t for t in TOOL_DEFINITIONS if t["name"] == "submit_answer"]
            if self.mode == "direct":
                self.system = build_pipeline_prompt(catalog.dialect, full_schema_text(catalog))
            else:
                self.retriever = SchemaRetriever(catalog, semantics)
                self.system = ""  # built per question from what is retrieved

    def ask(self, question: str, *, memory: str = "", history: Sequence[Turn] = ()) -> AgentAnswer:
        """`memory`: notes recalled for this question (memory.render). `history`: earlier turns
        of the conversation, so follow-ups like "now by state" make sense."""
        start = time.perf_counter()
        answer = AgentAnswer(question=question, status="error", mode=self.mode)
        self.last_answer = answer  # kept so a caller can score a run a provider error cut short
        if self.retriever:
            retrieved = self.retriever.retrieve(question)
            answer.retrieved_tables = retrieved.tables
            self.system = build_pipeline_prompt(self.tools.catalog.dialect, retrieved.text)
        messages: list[dict[str, Any]] = _history_messages(history)
        content = f"{memory}\n\nQuestion: {question}" if memory else question
        messages.append({"role": "user", "content": content})
        nudged = False

        while answer.turns < self.settings.max_steps:
            response = self._call(messages)
            answer.turns += 1
            answer.usage.add(response.usage)

            if response.stop_reason == "refusal":
                answer.status, answer.detail = "refusal", _refusal_detail(response)
                break
            if response.stop_reason == "max_tokens":
                answer.detail = "response hit max_tokens"
                break

            # Append the full content (thinking, fallback and tool_use blocks): append-only.
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                if nudged:
                    answer.answer, answer.detail = _text(response), "never called submit_answer"
                    break
                nudged = True
                messages.append({"role": "user", "content": _NUDGE})
                continue

            results, finished = self._run_tools(tool_uses, answer)
            if finished:
                break
            messages.append({"role": "user", "content": results})
        else:
            answer.status, answer.detail = "max_steps", f"no answer after {answer.turns} turns"

        answer.elapsed_s = round(time.perf_counter() - start, 2)
        return answer

    def _call(self, messages: list[dict[str, Any]]) -> Any:
        fallback: dict[str, Any] = {}
        if self.settings.model in _FALLBACK_MODELS:
            fallback = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}
        return self.client.beta.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=self.tool_definitions,
            messages=messages,
            output_config={"effort": self.settings.effort},
            **fallback,
        )

    def _run_tools(
        self, tool_uses: list[Any], answer: AgentAnswer
    ) -> tuple[list[dict[str, Any]], bool]:
        """Execute one turn's tool calls. Returns (tool_result blocks, finished)."""
        results = []
        for block in tool_uses:
            if block.name == "submit_answer":
                error = self._accept(block.input, answer)
                if error is None:
                    return [], True
                self._record(
                    answer,
                    ToolCall(tool="submit_answer", input=dict(block.input), ok=False, error=error),
                )
                if not self._may_correct(answer):
                    answer.status, answer.detail = "error", error
                    return [], True
                answer.corrections += 1
                content, is_error = f"Error: {error}. Fix the query and submit again.", True
            elif block.name not in {t["name"] for t in self.tool_definitions}:
                # A model may call a tool it was not offered; single-shot modes must not explore.
                content, record = self.tools._error(
                    block.name, dict(block.input), f"tool {block.name!r} is not available"
                )
                is_error = True
                self._record(answer, record)
            else:
                content, record = self.tools.execute(block.name, dict(block.input))
                is_error = not record.ok
                self._record(answer, record)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    **({"is_error": True} if is_error else {}),
                }
            )
        return results, False

    def _may_correct(self, answer: AgentAnswer) -> bool:
        """Whether a failed submission goes back to the model. A and B get no second chance;
        C gets max_corrections; the agent is bounded only by max_steps."""
        if self.mode == "agent":
            return True
        if self.mode == "corrected":
            return answer.corrections < self.settings.max_corrections
        return False

    def _accept(self, tool_input: dict[str, Any], answer: AgentAnswer) -> str | None:
        """Validate and execute the submitted answer. Returns an error to feed back, or None."""
        sql = tool_input.get("sql")
        answerable = bool(tool_input.get("answerable", True))
        result = None
        if answerable:
            if not sql:
                return "answerable is true but sql is empty"
            try:
                result = self.tools.run_final(sql)
            except (UnsafeQueryError, QueryError) as e:
                return f"final SQL failed: {e}"
        answer.status = "answered" if answerable else "unanswerable"
        answer.answer = tool_input.get("answer")
        answer.sql = result.sql if result else None
        answer.answerable = answerable
        answer.result = result
        self._record(
            answer,
            ToolCall(
                tool="submit_answer",
                input=dict(tool_input),
                ok=True,
                sql=answer.sql,
                row_count=result.row_count if result else None,
            ),
        )
        return None

    def _record(self, answer: AgentAnswer, record: ToolCall) -> None:
        answer.steps.append(record)
        if self.on_step:
            self.on_step(record)


def _history_messages(history: Sequence[Turn]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in history:
        reply = turn.answer or "(no answer)"
        if turn.sql:
            reply += f"\n\nSQL I used:\n{turn.sql}"
        messages += [
            {"role": "user", "content": turn.question},
            {"role": "assistant", "content": reply},
        ]
    return messages


def _text(response: Any) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def _refusal_detail(response: Any) -> str:
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) if details else None
    return f"declined by the model (category: {category or 'unknown'})"
