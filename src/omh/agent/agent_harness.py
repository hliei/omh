from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from omh.agent.context import Context
from omh.agent.result import Result
from omh.agent.session.types import Session
from omh.agent.types import AgentMessage, ThinkingLevel
from omh.llm.models import Models
from omh.llm.types import Model

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 1_000
    max_agent_delay_ms: int = 60_000


@dataclass(frozen=True, slots=True)
class AgentHarnessOptions:
    session: Session
    models: Models
    model: Model
    thinking_level: ThinkingLevel = "off"
    retry: RetryPolicy = field(default_factory=RetryPolicy)


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
class NothingToResume:
    pass


type ResumeResult = Result[DriveOutcome, NothingToResume | OperationMismatch]
type RunResult = Result[OperationResultRecord, LaneBusy | InvalidMessage | OperationMismatch]


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


@dataclass(frozen=True, slots=True)
class OpenOperation:
    lane: str
    operation_id: str
    kind: Literal["run"]
    started_at: int


@dataclass(frozen=True, slots=True)
class AgentHarnessCreateResult:
    harness: AgentHarness
    open: list[OpenOperation]


class AgentHarness:
    @staticmethod
    async def create(options: AgentHarnessOptions, context: Context) -> AgentHarnessCreateResult:
        from omh.agent.runtime.harness import create_agent_harness

        return await create_agent_harness(options, context)

    async def lane(self, name: str, context: Context) -> AgentLane:
        raise NotImplementedError

    async def close(self, context: Context) -> None:
        raise NotImplementedError


async def create_agent_harness(
    options: AgentHarnessOptions,
    context: Context,
) -> AgentHarnessCreateResult:
    return await AgentHarness.create(options, context)
