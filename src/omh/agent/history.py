"""Append-only records and isolated snapshots for one Agent conversation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Literal, get_args
from uuid import uuid4, uuid7

from omh.agent.data import snapshot_messages, validate_typed_data
from omh.agent.messages import (
    AgentMessage,
    CompactionSummaryMessage,
    ContextEditReplacement,
    CustomAgentMessage,
)
from omh.llm.types import (
    AssistantMessage,
    ImageContent,
    JsonValue,
    Message,
    Model,
    SystemMessage,
    TextContent,
    ToolResultMessage,
    Usage,
)
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


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionHistoryEntry(_Entry):
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    system_message: SystemMessage | None
    usage: Usage | None = None
    details: JsonValue = None
    type: Literal["compaction"] = field(default="compaction", init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextEditHistoryEntry(_Entry):
    target_id: str
    replacement: ContextEditReplacement | None
    type: Literal["context_edit"] = field(default="context_edit", init=False)


AgentHistoryEntry = (
    MessageHistoryEntry | CustomMessageHistoryEntry
    | ModelChangeHistoryEntry | ThinkingLevelChangeHistoryEntry
    | CompactionHistoryEntry | ContextEditHistoryEntry
)


@dataclass(frozen=True, slots=True)
class AgentHistory:
    conversation_id: str
    created_at: datetime
    entries: tuple[AgentHistoryEntry, ...]
    leaf_id: str | None


@dataclass(frozen=True, slots=True)
class AgentHistorySettings:
    """Saved selections; the host resolves model identity to an executable Model."""

    provider: str | None = None
    model_id: str | None = None
    thinking_level: ThinkingLevel = "off"
    has_thinking_level: bool = False


def _timestamp_ms(timestamp: datetime) -> int:
    return (timestamp - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)


def _history_path(history: AgentHistory) -> list[AgentHistoryEntry]:
    by_id = {entry.id: entry for entry in history.entries}
    path: list[AgentHistoryEntry] = []
    current = history.leaf_id
    while current is not None:
        entry = by_id[current]
        path.append(entry)
        current = entry.parent_id
    return list(reversed(path))


def validate_history(history: AgentHistory) -> AgentHistorySettings:
    """Validate decoded records and relationships without executing or resolving models."""
    if type(history) is not AgentHistory:
        raise ValueError("history must be an AgentHistory")
    validate_typed_data(history.conversation_id, str, "history.conversation_id")
    if not history.conversation_id:
        raise ValueError("history.conversation_id must be non-empty")
    validate_typed_data(history.created_at, datetime, "history.created_at")
    validate_typed_data(history.leaf_id, str | None, "history.leaf_id")
    if type(history.entries) is not tuple:
        raise ValueError("history.entries must be a tuple of SDK records")
    by_id: dict[str, AgentHistoryEntry] = {}
    for index, entry in enumerate(history.entries):
        if type(entry) not in get_args(AgentHistoryEntry):
            raise ValueError(f"history.entries[{index}]: unknown record type")
        validate_typed_data(entry, type(entry), f"entry {entry.id}")
        if not entry.id:
            raise ValueError("entry id must be non-empty")
        if isinstance(entry, ModelChangeHistoryEntry) and (not entry.provider or not entry.model_id):
            raise ValueError(f"entry {entry.id}: provider and model_id must be non-empty")
        if isinstance(entry, CompactionHistoryEntry) and entry.tokens_before < 0:
            raise ValueError(f"entry {entry.id}: tokens_before must be non-negative")
        if entry.id in by_id:
            raise ValueError(f"entry {entry.id}: duplicate ID")
        by_id[entry.id] = entry
    resolved: set[str] = set()
    for entry in history.entries:
        seen: set[str] = set()
        current: str | None = entry.id
        while current is not None and current not in resolved:
            if current in seen:
                raise ValueError(f"entry {entry.id}: cycle in parent_id at {current}")
            seen.add(current)
            if current not in by_id:
                raise ValueError(f"entry {entry.id}: missing parent_id {current}")
            current = by_id[current].parent_id
        resolved.update(seen)
    for entry in history.entries:
        if not isinstance(entry, CompactionHistoryEntry | ContextEditHistoryEntry):
            continue
        relation = "first_kept_entry_id" if isinstance(entry, CompactionHistoryEntry) else "target_id"
        target_id = entry.first_kept_entry_id if isinstance(entry, CompactionHistoryEntry) else entry.target_id
        ancestor = entry.parent_id
        while ancestor is not None and ancestor != target_id:
            ancestor = by_id[ancestor].parent_id
        if ancestor is None:
            raise ValueError(f"entry {entry.id}: {relation} {target_id} must reference an ancestor")
        target = by_id[target_id]
        if isinstance(entry, ContextEditHistoryEntry) and not (
            isinstance(target, CustomMessageHistoryEntry)
            or isinstance(target, MessageHistoryEntry) and not isinstance(target.message, SystemMessage)
        ):
            raise ValueError(f"entry {entry.id}: target_id {target_id} must reference editable message content")
        if isinstance(entry, ContextEditHistoryEntry) and entry.replacement is not None:
            if isinstance(target, MessageHistoryEntry):
                message: Message | CustomAgentMessage = target.message
            else:
                assert isinstance(target, CustomMessageHistoryEntry)
                message = CustomAgentMessage(
                    custom_type=target.custom_type, content=target.content, display=target.display,
                    details=target.details,
                )
            content = entry.replacement.content
            if isinstance(message, AssistantMessage | ToolResultMessage) and isinstance(content, str):
                content = [TextContent(text=content)]
            validate_typed_data(
                replace(message, content=content),  # type: ignore[arg-type]
                type(message), f"entry {entry.id}.replacement",
            )
    if history.leaf_id is not None and history.leaf_id not in by_id:
        raise ValueError(f"leaf_id {history.leaf_id}: missing entry")
    provider: str | None = None
    model_id: str | None = None
    level: ThinkingLevel = "off"
    has_level = False
    for entry in _history_path(history):
        if isinstance(entry, ModelChangeHistoryEntry):
            provider, model_id = entry.provider, entry.model_id
        elif isinstance(entry, MessageHistoryEntry) and isinstance(entry.message, AssistantMessage):
            provider, model_id = entry.message.provider, entry.message.model
        elif isinstance(entry, ThinkingLevelChangeHistoryEntry):
            level, has_level = entry.thinking_level, True
    return AgentHistorySettings(provider, model_id, level, has_level)


def project_history(history: AgentHistory) -> list[AgentMessage]:
    """Build effective messages from an already validated history."""
    path = _history_path(history)
    compaction_index = next((
        index for index in range(len(path) - 1, -1, -1)
        if isinstance(path[index], CompactionHistoryEntry)
    ), None)
    messages: list[AgentMessage] = []
    if compaction_index is not None:
        compaction = path[compaction_index]
        assert isinstance(compaction, CompactionHistoryEntry)
        if compaction.system_message is not None:
            messages.append(compaction.system_message)
        messages.append(CompactionSummaryMessage(
            summary=compaction.summary, tokens_before=compaction.tokens_before,
            timestamp=_timestamp_ms(compaction.timestamp),
        ))
        first_kept_index = next(
            index for index, entry in enumerate(path) if entry.id == compaction.first_kept_entry_id
        )
        path = [
            entry for entry in path[first_kept_index:compaction_index]
            if not (isinstance(entry, MessageHistoryEntry) and isinstance(entry.message, SystemMessage))
        ] + path[compaction_index + 1:]
    edits = {entry.target_id: entry for entry in path if isinstance(entry, ContextEditHistoryEntry)}
    for entry in path:
        if isinstance(entry, MessageHistoryEntry):
            message: AgentMessage = copy.deepcopy(entry.message)
        elif isinstance(entry, CustomMessageHistoryEntry):
            message = CustomAgentMessage(
                custom_type=entry.custom_type, content=entry.content, display=entry.display,
                details=entry.details, timestamp=_timestamp_ms(entry.timestamp),
            )
        else:
            continue
        edit = edits.get(entry.id)
        if edit is not None:
            if edit.replacement is None:
                continue
            content = edit.replacement.content
            if isinstance(message, AssistantMessage | ToolResultMessage) and isinstance(content, str):
                content = [TextContent(text=content)]
            # Target-specific content types are checked by validate_history.
            message = replace(message, content=copy.deepcopy(content))  # type: ignore[arg-type]
        messages.append(message)
    return snapshot_messages(messages)


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

    @classmethod
    def from_snapshot(cls, history: AgentHistory) -> ConversationHistory:
        instance = cls.__new__(cls)
        instance.conversation_id = history.conversation_id
        instance.created_at = history.created_at
        instance.entries = copy.deepcopy(list(history.entries))
        instance.leaf_id = history.leaf_id
        instance._ids = {entry.id for entry in history.entries}
        return instance

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
        if isinstance(message, CompactionSummaryMessage):
            raise ValueError("CompactionSummaryMessage is derived context, not a history message")
        entry_id, parent_id, timestamp = self._entry_fields()
        if isinstance(message, CustomAgentMessage):
            return self._append(CustomMessageHistoryEntry(
                id=entry_id, parent_id=parent_id,
                timestamp=datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=message.timestamp),
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

    def append_omission(self, target_id: str) -> AgentHistoryEntry:
        entry_id, parent_id, timestamp = self._entry_fields()
        return self._append(ContextEditHistoryEntry(
            id=entry_id, parent_id=parent_id, timestamp=timestamp,
            target_id=target_id, replacement=None,
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
