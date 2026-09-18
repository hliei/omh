from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from omh.agent.context import Context
from omh.agent.numbers import MAX_SAFE_INTEGER
from omh.agent.result import Result
from omh.agent.session.types import Session
from omh.agent.types import AgentMessage, ThinkingLevel
from omh.llm.models import Models
from omh.llm.types import JsonValue, Model, ToolResultContent, Usage

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    content: list[ToolResultContent]
    details: JsonValue = None
    usage: Usage | None = None
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class ToolMemoUnset:
    pass


TOOL_MEMO_UNSET = ToolMemoUnset()
type ToolMemo = JsonValue | ToolMemoUnset


class AgentHarnessToolInvocation(Protocol):
    invocation_id: str
    operation_id: str
    turn_id: str

    async def get_memo(self, name: str) -> ToolMemo: ...
    async def set_memo(self, name: str, value: ToolMemo) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentHarnessToolUpdateOptions:
    checkpoint: bool = False


class AgentHarnessToolUpdateCallback(Protocol):
    def __call__(
        self,
        partial_result: AgentToolResult,
        options: AgentHarnessToolUpdateOptions | None = None,
    ) -> None: ...


type AgentHarnessToolExecute = Callable[
    [
        str,
        dict[str, object],
        AgentHarnessToolUpdateCallback,
        AgentHarnessToolInvocation,
        Context,
    ],
    Awaitable[AgentToolResult],
]
type ToolReplayPolicy = Literal["never", "safe"]


@dataclass(frozen=True, slots=True)
class AgentHarnessTool:
    name: str
    description: str
    parameters: dict[str, object]
    execute: AgentHarnessToolExecute
    replay: ToolReplayPolicy = "never"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 1_000
    max_agent_delay_ms: int = 60_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("retry.enabled must be a boolean")
        values = {
            "max_retries": self.max_retries,
            "base_delay_ms": self.base_delay_ms,
            "max_agent_delay_ms": self.max_agent_delay_ms,
        }
        for name, value in values.items():
            maximum = MAX_SAFE_INTEGER - 1 if name == "max_retries" else MAX_SAFE_INTEGER
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > maximum
            ):
                raise ValueError(f"retry.{name} must be a non-negative safe integer")


@dataclass(frozen=True, slots=True)
class AgentHarnessOptions:
    session: Session
    models: Models
    model: Model
    thinking_level: ThinkingLevel = "off"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    tools: tuple[AgentHarnessTool, ...] = ()
    active_tool_names: tuple[str, ...] | None = None
    tool_execution: Literal["sequential", "parallel"] = "parallel"


@dataclass(frozen=True, slots=True)
class AcquireLaneOptions:
    create_at: str | None = None


@dataclass(frozen=True, slots=True)
class PromptRequest:
    prompt: str | AgentMessage | list[AgentMessage]
    operation_id: str | None = None
    kind: Literal["prompt"] = "prompt"


@dataclass(frozen=True, slots=True)
class OperationAdmission:
    operation_id: str
    kind: Literal["run"]
    started_at: int


@dataclass(frozen=True, slots=True)
class LaneBusy:
    operation_id: str


@dataclass(frozen=True, slots=True)
class InvalidMessage:
    reason: str


type OperationAdmissionError = LaneBusy | InvalidMessage
type OperationAdmissionResult = Result[OperationAdmission, OperationAdmissionError]


@dataclass(frozen=True, slots=True)
class OperationError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OperationResultRecord:
    operation_id: str
    kind: Literal["run"]
    status: Literal["completed", "aborted", "failed"]
    from_tip_id: str | None
    tip_id: str | None
    started_at: int
    ended_at: int
    error: OperationError | None = None


@dataclass(frozen=True, slots=True)
class DriveOptions:
    operation_id: str
    wait_for_retry: bool = False


@dataclass(frozen=True, slots=True)
class SettledDriveOutcome:
    outcome: OperationResultRecord
    kind: Literal["settled"] = "settled"


@dataclass(frozen=True, slots=True)
class WaitingDriveOutcome:
    operation_id: str
    reason: Literal["retry"]
    not_before: int
    kind: Literal["waiting"] = "waiting"


type DriveOutcome = SettledDriveOutcome | WaitingDriveOutcome


@dataclass(frozen=True, slots=True)
class OperationMismatch:
    expected_operation_id: str | None
    operation_id: str


type DriveResult = Result[DriveOutcome, OperationMismatch]


@dataclass(frozen=True, slots=True)
class AbortRequest:
    operation_id: str
    newly_requested: bool
    steer: tuple[AgentMessage, ...] = ()
    follow_up: tuple[AgentMessage, ...] = ()


type AbortRequestResult = Result[AbortRequest, OperationMismatch]


@dataclass(frozen=True, slots=True)
class NoActiveOperation:
    pass


@dataclass(frozen=True, slots=True)
class AbortOutcome:
    operation_id: str
    steer: tuple[AgentMessage, ...] = ()
    follow_up: tuple[AgentMessage, ...] = ()


type AbortResult = Result[AbortOutcome, NoActiveOperation | OperationMismatch]


@dataclass(frozen=True, slots=True)
class NothingToResume:
    pass


type ResumeResult = Result[DriveOutcome, NothingToResume | OperationMismatch]
type RunResult = Result[
    OperationResultRecord, LaneBusy | InvalidMessage | OperationMismatch
]


@dataclass(frozen=True, slots=True)
class CurrentOperationInfo:
    operation_id: str
    kind: Literal["run"]
    started_at: int
    at: str


@dataclass(frozen=True, slots=True)
class LaneExecutionInfo:
    current: CurrentOperationInfo | None
    last_operation_id: str | None
    tip_id: str | None


@dataclass(frozen=True, slots=True)
class OpenOperation:
    lane: str
    operation_id: str
    kind: Literal["run"]
    started_at: int


@dataclass(frozen=True, slots=True)
class LaneInfo:
    name: str
    tip_id: str | None
    operation: CurrentOperationInfo | None


@dataclass(frozen=True, slots=True)
class AgentHarnessCreateResult:
    harness: AgentHarness
    open: list[OpenOperation]


class AgentHarness(Protocol):
    @staticmethod
    async def create(
        options: AgentHarnessOptions, context: Context
    ) -> AgentHarnessCreateResult:
        from omh.agent.runtime.harness import create_agent_harness

        return await create_agent_harness(options, context)

    async def lane(
        self,
        name: str,
        context: Context,
        options: AcquireLaneOptions | None = None,
    ) -> AgentLane: ...

    async def lanes(self, context: Context) -> list[LaneInfo]: ...

    async def get_tools(self, context: Context) -> tuple[AgentHarnessTool, ...]: ...

    async def set_tools(
        self, tools: tuple[AgentHarnessTool, ...], context: Context
    ) -> None: ...

    async def close(self, context: Context) -> None: ...


async def create_agent_harness(
    options: AgentHarnessOptions,
    context: Context,
) -> AgentHarnessCreateResult:
    return await AgentHarness.create(options, context)
