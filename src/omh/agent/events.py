"""Agent lifecycle events and ordered delivery to synchronous or async sinks."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from omh.agent.history import AgentHistoryEntry
from omh.agent.messages import LoopMessage
from omh.agent.tools import AgentToolResult
from omh.llm.types import AbortSignal, AssistantMessageEvent, ToolResultMessage


@dataclass(slots=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    messages: list[LoopMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass(slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    message: LoopMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class MessageStartEvent:
    message: LoopMessage
    type: Literal["message_start"] = "message_start"


@dataclass(slots=True)
class MessageUpdateEvent:
    message: LoopMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass(slots=True)
class MessageEndEvent:
    message: LoopMessage
    type: Literal["message_end"] = "message_end"


@dataclass(frozen=True, slots=True)
class HistoryCommitEvent:
    conversation_id: str
    entries: tuple[AgentHistoryEntry, ...]
    leaf_id: str
    type: Literal["history_commit"] = "history_commit"


@dataclass(slots=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    partial_result: AgentToolResult
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(slots=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | HistoryCommitEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)


AgentEventListener = Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]


AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


async def emit_event(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result
