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


AgentMessage: TypeAlias = Message | CompactionSummaryMessage
ThinkingLevel: TypeAlias = ModelThinkingLevel
