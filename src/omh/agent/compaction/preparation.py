"""Budget estimates and compaction cuts derived from conversation history."""

from __future__ import annotations

import json
from collections.abc import Sequence

from omh.agent.compaction.types import (
    _MODIFIED_FILES_KEY,
    _READ_FILES_KEY,
    CompactionPreparation,
    CompactionSettings,
)
from omh.agent.conversation.history import (
    AgentHistory,
    CompactionHistoryEntry,
    ContextEditHistoryEntry,
    history_path,
    project_history_records,
)
from omh.agent.conversation.messages import (
    AgentMessage,
    CompactionSummaryMessage,
    CustomAgentMessage,
)
from omh.llm.types import (
    AssistantMessage,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from omh.llm.utils.estimate import (
    CHARS_PER_TOKEN,
    ESTIMATED_IMAGE_CHARS,
    calculate_context_tokens,
)


def _content_chars(content: str | Sequence[object]) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        if isinstance(block, TextContent):
            chars += len(block.text)
        else:
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    """Estimate one message's token cost with the accepted chars/4 heuristic."""
    chars = 0
    if isinstance(message, UserMessage):
        chars = _content_chars(message.content)
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextContent):
                chars += len(block.text)
            elif isinstance(block, ThinkingContent):
                chars += len(block.thinking)
            else:
                chars += len(block.name) + len(
                    json.dumps(block.arguments, ensure_ascii=False, default=str)
                )
    elif isinstance(message, ToolResultMessage):
        chars = _content_chars(message.content)
    elif isinstance(message, CustomAgentMessage):
        chars = _content_chars(message.content)
    elif isinstance(message, CompactionSummaryMessage):
        chars = len(message.summary)
    elif isinstance(message, SystemMessage):
        chars = _content_chars(message.content)
        for name, value in (message.sections or {}).items():
            chars += len(name) + len(value or "")
        for tool in message.tools_added or []:
            chars += len(tool.name) + len(tool.description) + len(
                json.dumps(tool.parameters, ensure_ascii=False, default=str)
            )
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _valid_usage(message: AgentMessage) -> Usage | None:
    if (
        isinstance(message, AssistantMessage)
        and message.stop_reason not in {"aborted", "error"}
        and calculate_context_tokens(message.usage) > 0
    ):
        return message.usage
    return None


def estimate_context_tokens(messages: Sequence[AgentMessage], *, usage_start: int = 0) -> int:
    """Use the latest valid assistant usage plus a trailing-message estimate."""
    usage_tokens = 0
    start = 0
    for index in range(len(messages) - 1, usage_start - 1, -1):
        usage = _valid_usage(messages[index])
        if usage is not None:
            usage_tokens = calculate_context_tokens(usage)
            start = index + 1
            break
    return usage_tokens + sum(estimate_tokens(message) for message in messages[start:])


def estimate_projection_tokens(messages: Sequence[AgentMessage]) -> int:
    """Estimate a rebuilt projection without reusing pre-compaction usage."""
    return sum(estimate_tokens(message) for message in messages)


def estimate_history_tokens(history: AgentHistory) -> int:
    """Budget the canonical projection using only usage after its last edit."""
    path = history_path(history)
    projected = project_history_records(history)
    context_change = next((
        index for index in range(len(path) - 1, -1, -1)
        if isinstance(path[index], CompactionHistoryEntry | ContextEditHistoryEntry)
    ), -1)
    fresh_ids = {entry.id for entry in path[context_change + 1:]}
    usage_start = next((
        index for index, (entry_id, _) in enumerate(projected) if entry_id in fresh_ids
    ), len(projected))
    return estimate_context_tokens(
        [message for _, message in projected], usage_start=usage_start,
    )


def _cut_points(records: Sequence[tuple[str | None, AgentMessage]]) -> list[int]:
    return [
        index
        for index, (_, message) in enumerate(records)
        if not isinstance(message, ToolResultMessage)
    ]


def _turn_start(records: Sequence[tuple[str | None, AgentMessage]], start: int) -> int:
    for index in range(start, -1, -1):
        if isinstance(records[index][1], UserMessage | CustomAgentMessage):
            return index
    return -1


def _cut_point(
    records: Sequence[tuple[str | None, AgentMessage]], keep_recent_tokens: int
) -> tuple[int, int, bool] | None:
    points = _cut_points(records)
    if not points:
        return None
    accumulated = 0
    cut_index = points[0]
    for index in range(len(records) - 1, -1, -1):
        accumulated += estimate_tokens(records[index][1])
        if accumulated >= keep_recent_tokens:
            cut_index = next((point for point in points if point >= index), points[-1])
            break
    cut_message = records[cut_index][1]
    is_turn_start = isinstance(cut_message, UserMessage | CustomAgentMessage)
    turn_start = -1 if is_turn_start else _turn_start(records, cut_index)
    return cut_index, turn_start, not is_turn_start and turn_start != -1


def _file_operations(
    messages: Sequence[AgentMessage],
    previous: CompactionHistoryEntry | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    read: set[str] = set()
    modified: set[str] = set()
    if previous is not None and isinstance(previous.details, dict):
        previous_read = previous.details.get(_READ_FILES_KEY)
        previous_modified = previous.details.get(_MODIFIED_FILES_KEY)
        if isinstance(previous_read, list):
            read.update(path for path in previous_read if isinstance(path, str))
        if isinstance(previous_modified, list):
            modified.update(path for path in previous_modified if isinstance(path, str))
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if not isinstance(block, ToolCall):
                continue
            path = block.arguments.get("path")
            if not isinstance(path, str):
                continue
            if block.name == "read":
                read.add(path)
            elif block.name in {"write", "edit"}:
                modified.add(path)
    return tuple(sorted(read - modified)), tuple(sorted(modified))


def prepare_compaction(
    history: AgentHistory, settings: CompactionSettings
) -> CompactionPreparation | None:
    """Derive the summary prefix and retained tail from the canonical projection.

    Returns ``None`` when there is no compactable prefix (including an empty
    history or a history whose leaf is already a compaction record).
    """
    path = history_path(history)
    if not path or isinstance(path[-1], CompactionHistoryEntry):
        return None
    previous = next(
        (
            entry
            for entry in reversed(path)
            if isinstance(entry, CompactionHistoryEntry)
        ),
        None,
    )
    projected = project_history_records(history)
    records = [
        (entry_id, message)
        for entry_id, message in projected
        if entry_id is not None and not isinstance(message, SystemMessage)
    ]
    cut = _cut_point(records, settings.keep_recent_tokens)
    if cut is None:
        return None
    cut_index, turn_start, is_split_turn = cut
    history_end = turn_start if is_split_turn else cut_index
    messages_to_summarize = tuple(message for _, message in records[:history_end])
    turn_prefix = (
        tuple(message for _, message in records[turn_start:cut_index]) if is_split_turn else ()
    )
    if not messages_to_summarize and not turn_prefix:
        return None
    first_kept_entry_id = records[cut_index][0]
    assert first_kept_entry_id is not None
    read_files, modified_files = _file_operations(
        (*messages_to_summarize, *turn_prefix), previous
    )
    return CompactionPreparation(
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix,
        retained_tail=tuple(message for _, message in records[cut_index:]),
        first_kept_entry_id=first_kept_entry_id,
        is_split_turn=is_split_turn,
        tokens_before=estimate_history_tokens(history),
        previous_summary=previous.summary if previous is not None else None,
        read_files=read_files,
        modified_files=modified_files,
    )
