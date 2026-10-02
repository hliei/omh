"""Conversation context shared by the loop and its hooks."""

from __future__ import annotations

from dataclasses import dataclass

from omh.agent.messages import LoopMessage
from omh.agent.tools import AgentTool


@dataclass(slots=True)
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    messages: list[LoopMessage]
    tools: list[AgentTool]
