"""Request, turn, tool, and queue extension contracts for the Agent loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from omh.agent.context import AgentContext
from omh.agent.messages import LoopMessage
from omh.agent.tools import AgentToolResult
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from omh.llm.types import ModelThinkingLevel as ThinkingLevel

#: Resolves a provider API key per request. Returning a falsey value keeps the
#: static fallback configured on the request options.
GetApiKey = Callable[[str], str | None | Awaitable[str | None]]


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
    details: object | None = None
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

#: Decision returned by ``finish_turn``. ``None`` preserves normal scheduling.
#: ``"end"`` stops without another request; ``"continue"`` ensures one more.
AgentTurnDecision = Literal["continue", "end"]


@dataclass(slots=True)
class AgentTurnContext:
    """Completed-turn state passed to ``finish_turn`` and turn preparation.

    ``context`` is the runtime context after the assistant message and its tool
    results were appended; ``new_messages`` are the messages this run would
    return if it stops at this point. Both are the loop's live objects and may
    grow after the hook returns, so snapshot them when retaining them.
    Agent callbacks receive isolated snapshots instead of the loop's live data.
    """

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[LoopMessage]


FinishTurn = Callable[
    [AgentTurnContext, AbortSignal | None],
    AgentTurnDecision | None | Awaitable[AgentTurnDecision | None],
]


@dataclass(slots=True)
class AgentLoopTurnUpdate:
    """Replacement runtime state used before the next provider request.

    ``messages`` are appended to the next request's context with the normal
    message lifecycle and tool-declaration coordination. Omitted fields keep
    the current runtime values.
    """

    context: AgentContext | None = None
    messages: list[LoopMessage] | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True)
class PrepareRequestContext:
    """Runtime state available immediately before a conversation request.

    Pending messages, including a prepared turn's additions, have already been
    appended to ``context.messages`` and emitted when this is built.
    """

    context: AgentContext
    model: Model
    thinking_level: ThinkingLevel


@dataclass(slots=True)
class AgentRequestUpdate:
    """Replacement runtime state for the request being prepared.

    Unlike :class:`AgentLoopTurnUpdate`, a request update cannot append
    messages. It applies to this request and later requests in the run.
    """

    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


PrepareRequest = Callable[
    [PrepareRequestContext, AbortSignal | None],
    AgentRequestUpdate | None | Awaitable[AgentRequestUpdate | None],
]

#: Loop-level next-turn preparation. Receives the completed turn's context.
PrepareNextTurn = Callable[
    [AgentTurnContext],
    AgentLoopTurnUpdate | None | Awaitable[AgentLoopTurnUpdate | None],
]

#: Loop-level steering poll. Returns the messages to inject at a queue drain
#: point; the default Agent binding drains the steering queue by its mode.
GetSteeringMessages = Callable[[], list[LoopMessage] | Awaitable[list[LoopMessage]]]

#: Loop-level follow-up poll. Runs when the loop would otherwise stop and no
#: steering or natural tool continuation is pending.
GetFollowUpMessages = Callable[[], list[LoopMessage] | Awaitable[list[LoopMessage]]]

#: Agent-level next-turn preparation without loop context. Receives the run signal.
PrepareNextTurnWithSignal = Callable[
    [AbortSignal | None],
    AgentLoopTurnUpdate | None | Awaitable[AgentLoopTurnUpdate | None],
]

#: Agent-level next-turn preparation with the completed-turn context. Takes
#: priority over :data:`PrepareNextTurnWithSignal` when both are configured.
PrepareNextTurnWithContext = Callable[
    [AgentTurnContext, AbortSignal | None],
    AgentLoopTurnUpdate | None | Awaitable[AgentLoopTurnUpdate | None],
]
