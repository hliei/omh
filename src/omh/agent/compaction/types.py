"""Configuration, prepared input, and outcomes for Agent compaction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from omh.agent.conversation.messages import AgentMessage
from omh.llm.types import (
    AssistantMessage,
    JsonValue,
    SimpleStreamOptions,
    TranscriptContext,
    Usage,
)

_MAX_SAFE_INTEGER = 2**53 - 1
_READ_FILES_KEY = "readFiles"
_MODIFIED_FILES_KEY = "modifiedFiles"


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Threshold and retention budgets for compaction.

    Defaults mirror the accepted baseline. Manual compaction runs even when
    ``enabled`` is false; the flag gates automatic threshold and response recovery.
    """

    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("compaction.enabled must be a boolean")
        for name, value in {
            "reserve_tokens": self.reserve_tokens,
            "keep_recent_tokens": self.keep_recent_tokens,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > _MAX_SAFE_INTEGER
            ):
                raise ValueError(f"compaction.{name} must be a non-negative safe integer")


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Public outcome of one committed compaction."""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    estimated_tokens_after: int
    usage: Usage | None = None
    details: JsonValue = None


@dataclass(slots=True)
class CompactionFailure(Exception):
    """A compaction that ended before committing a record."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    """Derived input for one compaction, taken from the canonical projection."""

    messages_to_summarize: tuple[AgentMessage, ...]
    turn_prefix_messages: tuple[AgentMessage, ...]
    retained_tail: tuple[AgentMessage, ...]
    first_kept_entry_id: str
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]


#: Runs one summarization request with an already captured stream function.
SummaryRequest = Callable[[TranscriptContext, SimpleStreamOptions], Awaitable[AssistantMessage]]
