"""A scripted stand-in for anthropic.Anthropic, so agent tests run offline and free."""

from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace
from typing import Any

_ids = itertools.count(1)


def tool_use(name: str, **tool_input: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=tool_input)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def reply(*blocks: SimpleNamespace, stop: str | None = None) -> SimpleNamespace:
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage, stop_details=None)


class FakeClaude:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(copy.deepcopy(kwargs))
        if not self.replies:
            raise AssertionError("agent made more requests than scripted")
        return self.replies.pop(0)

    def tool_results(self, request: int) -> list[dict[str, Any]]:
        """tool_result blocks sent in the given request (0-based)."""
        last = self.requests[request]["messages"][-1]
        return [
            b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
