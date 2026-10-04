"""Configuration consumed by the standalone loop and its execution modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omh.agent.conversation.messages import LoopConvertToLlm, LoopTransformContext
from omh.agent.execution.context import AgentContext
from omh.agent.execution.hooks import (
    AfterToolCall,
    BeforeToolCall,
    FinishTurn,
    GetApiKey,
    GetFollowUpMessages,
    GetSteeringMessages,
    PrepareNextTurn,
    PrepareRequest,
)
from omh.agent.execution.tools import ToolExecutionMode
from omh.llm.types import (
    AssistantMessage,
    Model,
    OnPayload,
    OnProviderStreamEvent,
    OnResponse,
    ThinkingBudgets,
    Transport,
)
from omh.llm.types import ThinkingLevel as ReasoningLevel


@dataclass(slots=True)
class AgentLoopConfig:
    model: Model
    reasoning: ReasoningLevel | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    convert_to_llm: LoopConvertToLlm | None = None
    transform_context: LoopTransformContext | None = None
    finish_turn: FinishTurn | None = None
    prepare_request: PrepareRequest | None = None
    #: Rebuilds the request context (messages and executable tools) from
    #: authoritative state before each request. The Agent uses this for live
    #: configuration; the standalone loop leaves it unset.
    refresh_request: Callable[[], AgentContext] | None = None
    prepare_next_turn: PrepareNextTurn | None = None
    get_steering_messages: GetSteeringMessages | None = None
    get_follow_up_messages: GetFollowUpMessages | None = None
    get_api_key: GetApiKey | None = None
    api_key: str | None = None
    on_payload: OnPayload | None = None
    on_response: OnResponse | None = None
    on_provider_stream_event: OnProviderStreamEvent | None = None
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport | None = None
    max_retry_delay_ms: float | None = None
    _snapshot_response: Callable[[AssistantMessage], AssistantMessage] | None = None
