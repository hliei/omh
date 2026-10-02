"""Append-only records and isolated snapshots for one Agent conversation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4, uuid7

from omh.agent.data import snapshot_messages
from omh.agent.messages import AgentMessage, CustomAgentMessage
from omh.llm.types import ImageContent, JsonValue, Message, Model, TextContent
from omh.llm.types import ModelThinkingLevel as ThinkingLevel


@dataclass(frozen=True, slots=True, kw_only=True)
class _Entry:
    id: str
    parent_id: str | None
    timestamp: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageHistoryEntry(_Entry):
    message: Message
    type: Literal["message"] = field(default="message", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class CustomMessageHistoryEntry(_Entry):
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    details: JsonValue = None
    type: Literal["custom_message"] = field(default="custom_message", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelChangeHistoryEntry(_Entry):
    provider: str
    model_id: str
    type: Literal["model_change"] = field(default="model_change", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ThinkingLevelChangeHistoryEntry(_Entry):
    thinking_level: ThinkingLevel
    type: Literal["thinking_level_change"] = field(default="thinking_level_change", init=False)


AgentHistoryEntry = (
    MessageHistoryEntry | CustomMessageHistoryEntry
    | ModelChangeHistoryEntry | ThinkingLevelChangeHistoryEntry
)


@dataclass(frozen=True, slots=True)
class AgentHistory:
    conversation_id: str
    created_at: datetime
    entries: tuple[AgentHistoryEntry, ...]
    leaf_id: str | None


class ConversationHistory:
    """Internal owner of records; only snapshots leave this boundary."""

    def __init__(self, conversation_id: str | None) -> None:
        self.conversation_id = str(uuid7()) if conversation_id is None else conversation_id
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise ValueError("conversation_id must be a non-empty string")
        self.created_at = datetime.now(UTC)
        self.entries: list[AgentHistoryEntry] = []
        self.leaf_id: str | None = None
        self._ids: set[str] = set()

    def _entry_fields(self) -> tuple[str, str | None, datetime]:
        entry_id = uuid4().hex[:8]
        while entry_id in self._ids:
            entry_id = uuid4().hex[:8]
        return entry_id, self.leaf_id, datetime.now(UTC)

    def _append(self, entry: AgentHistoryEntry) -> AgentHistoryEntry:
        self.entries.append(entry)
        self._ids.add(entry.id)
        self.leaf_id = entry.id
        return entry

    def append_message(self, message: AgentMessage) -> AgentHistoryEntry:
        message = snapshot_messages([message])[0]
        entry_id, parent_id, timestamp = self._entry_fields()
        if isinstance(message, CustomAgentMessage):
            return self._append(CustomMessageHistoryEntry(
                id=entry_id, parent_id=parent_id,
                timestamp=datetime.fromtimestamp(message.timestamp / 1000, UTC),
                custom_type=message.custom_type, content=message.content,
                display=message.display, details=message.details,
            ))
        return self._append(MessageHistoryEntry(
            id=entry_id, parent_id=parent_id, timestamp=timestamp, message=message,
        ))

    def append_model(self, model: Model) -> AgentHistoryEntry:
        entry_id, parent_id, timestamp = self._entry_fields()
        return self._append(ModelChangeHistoryEntry(
            id=entry_id, parent_id=parent_id, timestamp=timestamp,
            provider=model.provider, model_id=model.id,
        ))

    def append_thinking_level(self, level: ThinkingLevel) -> AgentHistoryEntry:
        entry_id, parent_id, timestamp = self._entry_fields()
        return self._append(ThinkingLevelChangeHistoryEntry(
            id=entry_id, parent_id=parent_id, timestamp=timestamp, thinking_level=level,
        ))

    def snapshot(self) -> AgentHistory:
        return AgentHistory(
            conversation_id=self.conversation_id, created_at=self.created_at,
            entries=copy.deepcopy(tuple(self.entries)), leaf_id=self.leaf_id,
        )
