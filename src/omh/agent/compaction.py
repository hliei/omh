"""Manual and automatic compaction support for one Agent conversation.

Compaction never edits the original records. It derives a summary of the older
effective context, keeps a recent tail, and appends a ``compaction`` record
whose checkpoint and first-kept boundary the projection uses.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from omh.agent.history import (
    AgentHistory,
    CompactionHistoryEntry,
    ContextEditHistoryEntry,
    history_path,
    project_history_records,
)
from omh.agent.messages import (
    AgentMessage,
    CompactionSummaryMessage,
    CustomAgentMessage,
)
from omh.llm.types import (
    AssistantMessage,
    JsonValue,
    Model,
    SimpleStreamOptions,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UsageCost,
    UserMessage,
)
from omh.llm.types import ModelThinkingLevel as ThinkingLevel
from omh.llm.utils.estimate import (
    CHARS_PER_TOKEN,
    ESTIMATED_IMAGE_CHARS,
    calculate_context_tokens,
)

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Read the conversation and produce "
    "only the requested structured summary; do not continue the conversation."
)
_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this format: Goal, Constraints & Preferences, Progress, Key Decisions, Next Steps, and Critical Context. Be concise and preserve exact paths, names, and errors."""
_UPDATE_SUMMARIZATION_PROMPT = """Update the existing structured summary with the new conversation messages. Preserve prior goals, constraints, decisions, and critical context; update progress and next steps. Be concise and preserve exact paths, names, and errors."""
_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the prefix of a turn whose suffix is retained. Summarize the original request, early progress, and context needed to understand the retained suffix."""

_MAX_SAFE_INTEGER = 2**53 - 1
_READ_FILES_KEY = "readFiles"
_MODIFIED_FILES_KEY = "modifiedFiles"


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Threshold and retention budgets for compaction.

    Defaults mirror the accepted baseline. Manual compaction runs even when
    ``enabled`` is false; the flag only gates automatic threshold compaction.
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


def _serialize(messages: Sequence[AgentMessage]) -> str:
    serialized: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            text = _text(message.content)
            if text:
                serialized.append(f"[User]: {text}")
        elif isinstance(message, CustomAgentMessage):
            text = _text(message.content)
            if text:
                serialized.append(f"[User]: {text}")
        elif isinstance(message, AssistantMessage):
            thinking = [
                block.thinking for block in message.content if isinstance(block, ThinkingContent)
            ]
            calls = [
                f"{block.name}("
                + ", ".join(
                    f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
                    for key, value in block.arguments.items()
                )
                + ")"
                for block in message.content
                if isinstance(block, ToolCall)
            ]
            text = _text(message.content)
            if thinking:
                serialized.append("[Assistant thinking]: " + "\n".join(thinking))
            if text:
                serialized.append(f"[Assistant]: {text}")
            if calls:
                serialized.append("[Assistant tool calls]: " + "; ".join(calls))
        elif isinstance(message, ToolResultMessage):
            text = _text(message.content)
            if len(text) > 2_000:
                text = f"{text[:2_000]}\n[... {len(text) - 2_000} more characters truncated]"
            serialized.append(f"[Tool result]: {text}")
    return "\n\n".join(serialized)


def _text(content: str | Sequence[object]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append("[image]")
    return "\n".join(parts)


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
        total_tokens=left.total_tokens + right.total_tokens,
        cost=UsageCost(
            input=left.cost.input + right.cost.input,
            output=left.cost.output + right.cost.output,
            cache_read=left.cost.cache_read + right.cost.cache_read,
            cache_write=left.cost.cache_write + right.cost.cache_write,
            total=left.cost.total + right.cost.total,
        ),
        cache_write_1h=(
            None
            if left.cache_write_1h is None and right.cache_write_1h is None
            else (left.cache_write_1h or 0) + (right.cache_write_1h or 0)
        ),
        reasoning=(
            None
            if left.reasoning is None and right.reasoning is None
            else (left.reasoning or 0) + (right.reasoning or 0)
        ),
    )


def _summary_prompt(previous_summary: str | None, custom_instructions: str | None, turn_prefix: bool) -> str:
    prompt = _TURN_PREFIX_SUMMARIZATION_PROMPT if turn_prefix else (
        _UPDATE_SUMMARIZATION_PROMPT if previous_summary is not None else _SUMMARIZATION_PROMPT
    )
    if custom_instructions:
        prompt += f"\n\nAdditional focus: {custom_instructions}"
    return prompt


async def _generate_summary(
    messages: Sequence[AgentMessage],
    model: Model,
    reserve_tokens: int,
    previous_summary: str | None,
    custom_instructions: str | None,
    thinking_level: ThinkingLevel,
    request: SummaryRequest,
    *,
    turn_prefix: bool = False,
) -> tuple[str, Usage]:
    prompt = _summary_prompt(previous_summary, custom_instructions, turn_prefix)
    text = f"<conversation>\n{_serialize(messages)}\n</conversation>\n\n"
    if previous_summary is not None and not turn_prefix:
        text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    text += prompt
    max_fraction = 0.5 if turn_prefix else 0.8
    reserve_max_tokens = math.floor(max_fraction * reserve_tokens)
    max_tokens = min(reserve_max_tokens, model.max_tokens) if model.max_tokens > 0 else reserve_max_tokens
    options = SimpleStreamOptions(max_tokens=max_tokens)
    if model.reasoning and thinking_level != "off":
        options.reasoning = thinking_level
    response = await request(
        TranscriptContext(
            messages=[
                SystemMessage(content=_SUMMARIZATION_SYSTEM_PROMPT, timestamp=0),
                UserMessage(content=[TextContent(text=text)], timestamp=0),
            ]
        ),
        options,
    )
    if response.stop_reason == "aborted":
        raise CompactionFailure("aborted", response.error_message or "Summarization aborted")
    if response.stop_reason == "error":
        prefix = "Turn prefix summarization failed" if turn_prefix else "Summarization failed"
        raise CompactionFailure(
            "summarization_failed",
            f"{prefix}: {response.error_message or 'Unknown error'}",
        )
    return "\n".join(
        block.text for block in response.content if isinstance(block, TextContent)
    ), response.usage


async def compact_with_request(
    preparation: CompactionPreparation,
    model: Model,
    custom_instructions: str | None,
    thinking_level: ThinkingLevel,
    system_message: SystemMessage | None,
    request: SummaryRequest,
    reserve_tokens: int,
) -> CompactionResult:
    """Run the summarization requests and build one committed compaction result."""
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = preparation.previous_summary or "No prior history."
        history_usage: Usage | None = None
        if preparation.messages_to_summarize:
            history_text, history_usage = await _generate_summary(
                preparation.messages_to_summarize,
                model,
                reserve_tokens,
                preparation.previous_summary,
                custom_instructions,
                thinking_level,
                request,
            )
        prefix_text, prefix_usage = await _generate_summary(
            preparation.turn_prefix_messages,
            model,
            reserve_tokens,
            None,
            custom_instructions,
            thinking_level,
            request,
            turn_prefix=True,
        )
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
        usage = prefix_usage if history_usage is None else _add_usage(history_usage, prefix_usage)
    else:
        summary, usage = await _generate_summary(
            preparation.messages_to_summarize,
            model,
            reserve_tokens,
            preparation.previous_summary,
            custom_instructions,
            thinking_level,
            request,
        )
    if preparation.read_files or preparation.modified_files:
        summary += "\n\n## Files\n"
        if preparation.read_files:
            summary += "\nRead: " + ", ".join(preparation.read_files)
        if preparation.modified_files:
            summary += "\nModified: " + ", ".join(preparation.modified_files)
    after: list[AgentMessage] = []
    if system_message is not None:
        after.append(system_message)
    after.append(CompactionSummaryMessage(
        summary=summary, tokens_before=preparation.tokens_before, timestamp=0,
    ))
    after.extend(preparation.retained_tail)
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        estimated_tokens_after=estimate_projection_tokens(after),
        usage=usage,
        details={
            _READ_FILES_KEY: list(preparation.read_files),
            _MODIFIED_FILES_KEY: list(preparation.modified_files),
        },
    )
