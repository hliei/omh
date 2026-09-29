from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from omh.llm.types import Message, ModelThinkingLevel


@dataclass(slots=True)
class CompactionSummaryMessage:
    summary: str
    tokens_before: int
    timestamp: int
    role: Literal["compactionSummary"] = "compactionSummary"


@dataclass(slots=True)
class BranchSummaryMessage:
    summary: str
    from_id: str | None
    timestamp: int
    role: Literal["branchSummary"] = "branchSummary"


AgentMessage: TypeAlias = Message | CompactionSummaryMessage | BranchSummaryMessage
ThinkingLevel: TypeAlias = ModelThinkingLevel
