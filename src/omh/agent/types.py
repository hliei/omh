from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    JsonValue,
    Message,
    Model,
    ModelCost,
    ModelThinkingLevel,
    ProviderResponse,
    SimpleStreamOptions,
    TextContent,
    ThinkingBudgets,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Transport,
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

#: How the tool calls of one assistant message are executed.
#:
#: - ``sequential``: each call is prepared, executed, and finalized before the
#:   next one starts.
#: - ``parallel``: calls are prepared in source order, then the allowed calls
#:   execute concurrently. ``tool_execution_end`` is emitted in completion
#:   order, while result messages keep assistant source order.
ToolExecutionMode = Literal["sequential", "parallel"]

#: Application message union. The traditional Agent starts with the standard LLM
#: message roles; applications add their own roles by extending the conversion
#: boundary rather than by widening the LLM message union.
@runtime_checkable
class CustomAgentMessage(Protocol):
    """A message recognised by its ``role`` but outside the standard LLM roles.

    Custom messages live in the application history. The default model conversion
    filters them out; applications that want them in a request provide
    ``AgentOptions.convert_to_llm``.
    """

    role: str


AgentMessage = Message | CustomAgentMessage

#: Converts application messages to LLM messages before transcript normalization.
ConvertToLlm = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]

#: Transforms application history before conversion; runs before ``convert_to_llm``.
TransformContext = Callable[
    [list[AgentMessage], AbortSignal | None],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]

#: Resolves a provider API key per request. Returning a falsey value keeps the
#: static fallback configured on the request options.
GetApiKey = Callable[[str], str | None | Awaitable[str | None]]

OnPayload = Callable[[dict[str, object], Model], Awaitable[dict[str, object] | None] | dict[str, object] | None]
OnResponse = Callable[[ProviderResponse, Model], Awaitable[None] | None]
OnProviderStreamEvent = Callable[[object, Model], Awaitable[None] | None]

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
    terminate: bool | None = None


#: Progress callback for one ``execute`` invocation. Calls made after the tool
#: settles are ignored. The loop settles accepted progress before finalizing.
AgentToolUpdateCallback = Callable[[AgentToolResult], None]

AgentToolExecute = Callable[
    [str, dict[str, object], AbortSignal | None, AgentToolUpdateCallback],
    Awaitable[AgentToolResult] | AgentToolResult,
]

#: Optional compatibility shim applied to raw tool-call arguments before schema
#: conversion and validation. It must return the argument object to validate.
AgentToolPrepareArguments = Callable[[dict[str, object]], dict[str, object]]


@dataclass(slots=True)
class AgentTool:
    """Executable tool held by the Agent.

    The declaration sent to the model is derived from ``name``, ``description``
    and ``parameters`` only; ``label``, ``execute``, ``prepare_arguments`` and
    ``execution_mode`` stay on the execution side.
    """

    name: str
    description: str
    parameters: dict[str, object]
    label: str
    execute: AgentToolExecute
    prepare_arguments: AgentToolPrepareArguments | None = None
    execution_mode: ToolExecutionMode | None = None


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
class BeforeToolCallResult:
    """Result returned from a ``before_tool_call`` hook.

    ``block=True`` prevents execution and produces an error tool result whose
    text is ``reason`` (or a default message). ``terminate=True`` participates in
    the batch early-termination rule when this call is blocked.
    """

    block: bool = False
    reason: str | None = None
    terminate: bool | None = None


@dataclass(slots=True)
class BeforeToolCallContext:
    """Context passed to a ``before_tool_call`` hook after argument validation."""

    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, object]
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallResult:
    """Field-by-field override returned from an ``after_tool_call`` hook.

    Omitted or ``None`` fields keep the executed result's values. Provided
    values replace the corresponding field in full; there is no deep merge.
    """

    content: list[TextContent | ImageContent] | None = None
    details: JsonValue = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass(slots=True)
class AfterToolCallContext:
    """Context passed to an ``after_tool_call`` hook before result finalization."""

    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, object]
    result: AgentToolResult
    is_error: bool
    context: AgentContext


BeforeToolCall = Callable[
    [BeforeToolCallContext, AbortSignal | None],
    Awaitable[BeforeToolCallResult | None] | BeforeToolCallResult | None,
]
AfterToolCall = Callable[
    [AfterToolCallContext, AbortSignal | None],
    Awaitable[AfterToolCallResult | None] | AfterToolCallResult | None,
]


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
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
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
        self._pending_tool_calls: set[str] = set()

    @property
    def system_prompt(self) -> str:
        return get_current_system_prompt(self._messages)

    @property
    def pending_tool_calls(self) -> frozenset[str]:
        """Tool call ids currently executing, tracked from tool execution events."""
        return frozenset(self._pending_tool_calls)

    def _add_pending_tool_call(self, tool_call_id: str) -> None:
        self._pending_tool_calls.add(tool_call_id)

    def _remove_pending_tool_call(self, tool_call_id: str) -> None:
        self._pending_tool_calls.discard(tool_call_id)

    def _clear_pending_tool_calls(self) -> None:
        self._pending_tool_calls.clear()

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
    """Options for constructing a traditional :class:`~omh.agent.agent.Agent`.

    ``stream_fn`` is optional: when omitted the host-installed default is used,
    and construction fails clearly if the host has not installed one.
    """

    stream_fn: StreamFn | None = None
    initial_state: AgentInitialState | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    convert_to_llm: ConvertToLlm | None = None
    transform_context: TransformContext | None = None
    get_api_key: GetApiKey | None = None
    api_key: str | None = None
    on_payload: OnPayload | None = None
    on_response: OnResponse | None = None
    on_provider_stream_event: OnProviderStreamEvent | None = None
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport | None = None
    max_retry_delay_ms: float | None = None
