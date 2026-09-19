from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omh.agent.agent_harness import OperationResultRecord, ToolReplayPolicy
from omh.agent.types import ThinkingLevel


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model_id: str


@dataclass(frozen=True, slots=True)
class LaneConfiguration:
    model: ModelIdentity
    thinking_level: ThinkingLevel
    active_tool_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InboxItem:
    entry_id: str
    kind: Literal["steer", "followUp", "nextRun", "write"]


@dataclass(frozen=True, slots=True)
class LaneState:
    current_operation_id: str | None = None
    last_operation_id: str | None = None
    inbox: tuple[InboxItem, ...] = ()


@dataclass(frozen=True, slots=True)
class RunIntent:
    prompt_entry_ids: tuple[str, ...]
    kind: Literal["run"] = "run"


@dataclass(frozen=True, slots=True)
class OperationMeta:
    operation_id: str
    lane: str
    source_tip_id: str | None
    started_at: int
    intent: RunIntent


@dataclass(frozen=True, slots=True)
class RunningControl:
    status: Literal["running"] = "running"


@dataclass(frozen=True, slots=True)
class CancelRequestedControl:
    requested_at: int
    status: Literal["cancel_requested"] = "cancel_requested"


type RunControl = RunningControl | CancelRequestedControl


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass(frozen=True, slots=True)
class RunSettings:
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    steering_mode: Literal["all", "one-at-a-time"] = "all"
    follow_up_mode: Literal["all", "one-at-a-time"] = "all"
    tool_execution: Literal["sequential", "parallel"] = "parallel"


@dataclass(frozen=True, slots=True)
class NeedAssistant:
    overflow_recovery_used: bool = False
    kind: Literal["need_assistant"] = "need_assistant"


@dataclass(frozen=True, slots=True)
class MayFinish:
    include_final_assistant: bool = True
    kind: Literal["may_finish"] = "may_finish"


type RunContinuation = NeedAssistant | MayFinish


@dataclass(frozen=True, slots=True)
class GenerationRetryPolicy:
    max_attempts: int
    base_delay_ms: int
    max_agent_delay_ms: int


@dataclass(frozen=True, slots=True)
class GenerationContext:
    step_id: str
    trigger_entry_id: str
    configuration: LaneConfiguration
    retry_policy: GenerationRetryPolicy
    overflow_recovery_used: bool = False


@dataclass(frozen=True, slots=True)
class StartingOperation:
    latest_assistant_entry_id: str | None
    control: RunControl = field(default_factory=RunningControl)
    settings: RunSettings = field(default_factory=RunSettings)
    at: Literal["starting"] = "starting"


@dataclass(frozen=True, slots=True)
class CheckpointOperation:
    latest_assistant_entry_id: str | None
    continuation: RunContinuation
    trigger_entry_id: str
    control: RunControl
    settings: RunSettings
    at: Literal["checkpoint"] = "checkpoint"


@dataclass(frozen=True, slots=True)
class AssistantReadyOperation:
    latest_assistant_entry_id: str | None
    generation_context: GenerationContext
    next_attempt: int
    control: RunControl
    settings: RunSettings
    at: Literal["assistant.ready"] = "assistant.ready"


@dataclass(frozen=True, slots=True)
class AssistantEffectPendingOperation:
    latest_assistant_entry_id: str | None
    generation_context: GenerationContext
    attempt: int
    response_entry_id: str
    usage_id: str
    intended_output_limit: int
    context_window: int
    control: RunControl
    settings: RunSettings
    at: Literal["assistant.effect_pending"] = "assistant.effect_pending"


@dataclass(frozen=True, slots=True)
class AssistantRetryWaitOperation:
    latest_assistant_entry_id: str | None
    generation_context: GenerationContext
    next_attempt: int
    not_before: int
    error_message: str
    control: RunControl
    settings: RunSettings
    at: Literal["assistant.retry_wait"] = "assistant.retry_wait"


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    source_index: int
    result_entry_id: str
    status: Literal["planned"] = "planned"


@dataclass(frozen=True, slots=True)
class EffectPendingToolCall:
    source_index: int
    result_entry_id: str
    replay: ToolReplayPolicy
    status: Literal["effect_pending"] = "effect_pending"


@dataclass(frozen=True, slots=True)
class OutcomeReadyToolCall:
    source_index: int
    result_entry_id: str
    terminate: bool
    status: Literal["outcome_ready"] = "outcome_ready"


@dataclass(frozen=True, slots=True)
class CompletedToolCall:
    source_index: int
    result_entry_id: str
    terminate: bool
    status: Literal["completed"] = "completed"


type ToolCallState = (
    PlannedToolCall
    | EffectPendingToolCall
    | OutcomeReadyToolCall
    | CompletedToolCall
)


@dataclass(frozen=True, slots=True)
class ToolBatch:
    assistant_entry_id: str
    configuration: LaneConfiguration
    turn_id: str
    calls: tuple[ToolCallState, ...]


@dataclass(frozen=True, slots=True)
class ToolsOperation:
    latest_assistant_entry_id: str | None
    batch: ToolBatch
    control: RunControl
    settings: RunSettings
    at: Literal["tools"] = "tools"


type OperationState = (
    StartingOperation
    | CheckpointOperation
    | AssistantReadyOperation
    | AssistantEffectPendingOperation
    | AssistantRetryWaitOperation
    | ToolsOperation
)


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    lane: LaneState
    meta: OperationMeta
    state: OperationState


type StoredOperationResult = OperationResultRecord
