"""Construction options for the stateful Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omh.agent.hooks import (
    AfterToolCall,
    BeforeToolCall,
    FinishTurn,
    GetApiKey,
    PrepareNextTurnWithContext,
    PrepareNextTurnWithSignal,
    PrepareRequest,
)
from omh.agent.messages import ConvertToLlm, TransformContext
from omh.agent.state import AgentInitialState
from omh.agent.stream_fn import StreamFn
from omh.agent.tools import ToolExecutionMode
from omh.llm.types import (
    OnPayload,
    OnProviderStreamEvent,
    OnResponse,
    ThinkingBudgets,
    Transport,
)

#: How many queued messages a drain point injects.
#:
#: - ``one-at-a-time`` (default): take only the oldest queued message.
#: - ``all``: take every message currently queued.
QueueMode = Literal["all", "one-at-a-time"]


@dataclass(slots=True)
class AgentOptions:
    """Options for constructing a traditional :class:`~omh.agent.agent.Agent`.

    ``stream_fn`` is optional: when omitted the host-installed default is used,
    and construction fails clearly if the host has not installed one.
    """

    stream_fn: StreamFn | None = None
    initial_state: AgentInitialState | None = None
    tool_execution: ToolExecutionMode = "parallel"
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    convert_to_llm: ConvertToLlm | None = None
    transform_context: TransformContext | None = None
    get_api_key: GetApiKey | None = None
    api_key: str | None = None
    on_payload: OnPayload | None = None
    on_response: OnResponse | None = None
    on_provider_stream_event: OnProviderStreamEvent | None = None
    finish_turn: FinishTurn | None = None
    prepare_request: PrepareRequest | None = None
    prepare_next_turn: PrepareNextTurnWithSignal | None = None
    prepare_next_turn_with_context: PrepareNextTurnWithContext | None = None
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport | None = None
    max_retry_delay_ms: float | None = None
    conversation_id: str | None = None
