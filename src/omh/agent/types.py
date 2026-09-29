from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from omh.llm.types import (
    AbortSignal,
    AssistantMessageEvent,
    ImageContent,
    JsonValue,
    Message,
    Model,
    ModelCost,
    ModelThinkingLevel,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream
from omh.llm.utils.transcript import (
    create_initial_system_message,
    get_current_system_prompt,
)
from omh.llm.utils.transcript import (
    to_tool_declaration as _to_llm_tool_declaration,
)

#: Requested reasoning level for a turn. ``off`` disables reasoning.
ThinkingLevel = ModelThinkingLevel

#: Application message union. The traditional Agent starts with the standard LLM
#: message roles; application-specific roles are added by the conversion boundary.
AgentMessage = Message

#: A single tool call emitted by an assistant message.
AgentToolCall = ToolCall

#: Low-level model-stream function. It receives the normalized transcript whose
#: prompt and tool declarations are carried by system messages, never by a
#: separate prompt/tool field.
StreamFn = Callable[
    [Model, TranscriptContext, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]

#: Fallback model used when an Agent is created without an explicit model. Mirrors
#: the baseline's placeholder so callers can inspect identity before configuring one.
DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="openai-completions",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=(),
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=0,
    max_tokens=0,
)


@dataclass(slots=True)
class AgentToolResult:
    """Result returned by an :class:`AgentTool` execution."""

    content: list[TextContent | ImageContent]
    details: JsonValue = None
    usage: Usage | None = None


AgentToolExecute = Callable[[str, dict[str, object], AbortSignal | None], Awaitable[AgentToolResult] | AgentToolResult]

#: Optional compatibility shim applied to raw tool-call arguments before schema
#: conversion and validation. It must return the argument object to validate.
AgentToolPrepareArguments = Callable[[dict[str, object]], dict[str, object]]


@dataclass(slots=True)
class AgentTool:
    """Executable tool held by the Agent.

    The declaration sent to the model is derived from ``name``, ``description``
    and ``parameters`` only; ``label``, ``execute`` and ``prepare_arguments`` stay
    on the execution side.
    """

    name: str
    description: str
    parameters: dict[str, object]
    label: str
    execute: AgentToolExecute
    prepare_arguments: AgentToolPrepareArguments | None = None


def to_tool_declaration(tool: AgentTool) -> Tool:
    """Build the model-facing declaration for an executable tool."""
    return _to_llm_tool_declaration(
        Tool(name=tool.name, description=tool.description, parameters=copy.deepcopy(tool.parameters))
    )


@dataclass(slots=True)
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    messages: list[AgentMessage]
    tools: list[AgentTool]


@dataclass(slots=True)
class AgentInitialState:
    """Seed values for a new :class:`~omh.agent.agent.Agent`."""

    system_prompt: str | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] | None = None
    messages: list[AgentMessage] | None = None


@dataclass(slots=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    messages: list[AgentMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass(slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass(slots=True)
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass(slots=True)
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass(slots=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    type: Literal["tool_execution_start"] = "tool_execution_start"


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
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
)


AgentEventListener = Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]


class AgentState:
    """Public Agent state with copy-on-assign message and tool lists.

    Assigning :attr:`messages` or :attr:`tools` copies the top-level list; the
    message and tool objects themselves are shared, not deep-copied. The prompt
    is replayed from the transcript's system messages and is read-only here.
    """

    def __init__(self, initial: AgentInitialState | None = None) -> None:
        initial = initial or AgentInitialState()
        self._tools: list[AgentTool] = list(initial.tools or [])
        self._messages: list[AgentMessage] = list(initial.messages or [])
        initial_message = create_initial_system_message(
            initial.system_prompt,
            [to_tool_declaration(tool) for tool in self._tools],
        )
        if (not self._messages or self._messages[0].role != "system") and initial_message is not None:
            self._messages.insert(0, initial_message)
        self.model: Model = initial.model if initial.model is not None else DEFAULT_MODEL
        self.thinking_level: ThinkingLevel = initial.thinking_level
        self.is_streaming: bool = False
        self.streaming_message: AgentMessage | None = None
        self.error_message: str | None = None

    @property
    def system_prompt(self) -> str:
        return get_current_system_prompt(self._messages)

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)


@dataclass(slots=True)
class AgentOptions:
    """Options for constructing a traditional :class:`~omh.agent.agent.Agent`."""

    stream_fn: StreamFn
    initial_state: AgentInitialState | None = None
