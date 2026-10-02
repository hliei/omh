"""Application messages and their conversion to model input."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from omh.llm.types import AbortSignal, ImageContent, JsonValue, Message, TextContent


@runtime_checkable
class LoopApplicationMessage(Protocol):
    """Open application message accepted by the standalone loop."""

    @property
    def role(self) -> str: ...


@dataclass(slots=True)
class CustomAgentMessage:
    """Agent history data; only content participates in default model input."""

    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool = True
    details: JsonValue = None
    timestamp: int = field(default_factory=lambda: time.time_ns() // 1_000_000)
    role: Literal["custom"] = field(default="custom", init=False)


AgentMessage = Message | CustomAgentMessage
LoopMessage = Message | LoopApplicationMessage

#: Converts application messages to LLM messages before transcript normalization.
ConvertToLlm = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]

#: Transforms application history before conversion; runs before ``convert_to_llm``.
TransformContext = Callable[
    [list[AgentMessage], AbortSignal | None],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]

LoopConvertToLlm = Callable[[list[LoopMessage]], list[Message] | Awaitable[list[Message]]]
LoopTransformContext = Callable[
    [list[LoopMessage], AbortSignal | None],
    list[LoopMessage] | Awaitable[list[LoopMessage]],
]
