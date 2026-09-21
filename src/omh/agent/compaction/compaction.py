from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from omh.agent.context import Context
from omh.agent.messages import convert_to_llm
from omh.agent.numbers import MAX_SAFE_INTEGER
from omh.agent.session.context import (
    build_context_entries,
    session_entry_to_context_messages,
)
from omh.agent.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    Entry,
    MessageEntry,
)
from omh.agent.types import (
    AgentMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    ThinkingLevel,
)
from omh.agent.utils.usage import add_usage
from omh.llm.types import (
    AssistantMessage,
    JsonValue,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from omh.llm.types import Context as LlmContext

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Read the conversation and produce "
    "only the requested structured summary; do not continue the conversation."
)
SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this format: Goal, Constraints & Preferences, Progress, Key Decisions, Next Steps, and Critical Context. Be concise and preserve exact paths, names, and errors."""
UPDATE_SUMMARIZATION_PROMPT = """Update the existing structured summary with the new conversation messages. Preserve prior goals, constraints, decisions, and critical context; update progress and next steps. Be concise and preserve exact paths, names, and errors."""
TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the prefix of a turn whose suffix is retained. Summarize the original request, early progress, and context needed to understand the retained suffix."""


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("compaction.enabled must be a boolean")
        for name, value in {
            "reserve_tokens": self.reserve_tokens,
            "keep_recent_tokens": self.keep_recent_tokens,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > MAX_SAFE_INTEGER
            ):
                raise ValueError(f"compaction.{name} must be a non-negative safe integer")


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    messages_to_summarize: tuple[AgentMessage, ...]
    turn_prefix_messages: tuple[AgentMessage, ...]
    retained_tail: tuple[AgentMessage, ...]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    settings: CompactionSettings


@dataclass(frozen=True, slots=True)
class CompactResult:
    summary: str
    tokens_before: int
    retained_tail: tuple[AgentMessage, ...]
    usage: Usage | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class CompactionFailure(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


type SummaryRequest = Callable[
    [LlmContext, SimpleStreamOptions, Context], Awaitable[AssistantMessage]
]


def _context_tokens(usage: Usage) -> int:
    return usage.total_tokens or (
        usage.input + usage.output + usage.cache_read + usage.cache_write
    )


def _assistant_usage(message: AgentMessage) -> Usage | None:
    if (
        isinstance(message, AssistantMessage)
        and message.stop_reason not in {"aborted", "error"}
        and _context_tokens(message.usage) > 0
    ):
        return message.usage
    return None


def estimate_tokens(message: AgentMessage) -> int:
    chars = 0
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            chars = len(message.content)
        else:
            chars = sum(
                len(block.text) if isinstance(block, TextContent) else 4_800
                for block in message.content
            )
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
        chars = sum(
            len(block.text) if isinstance(block, TextContent) else 4_800
            for block in message.content
        )
    elif isinstance(message, CompactionSummaryMessage | BranchSummaryMessage):
        chars = len(message.summary)
    return math.ceil(chars / 4)


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    usage_index: int | None = None
    usage_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        usage = _assistant_usage(messages[index])
        if usage is not None:
            usage_index = index
            usage_tokens = _context_tokens(usage)
            break
    start = 0 if usage_index is None else usage_index + 1
    trailing = sum(estimate_tokens(message) for message in messages[start:])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=usage_index,
    )


def should_compact(
    context_tokens: int, context_window: int, settings: CompactionSettings
) -> bool:
    return (
        settings.enabled
        and context_window > settings.reserve_tokens
        and context_tokens > context_window - settings.reserve_tokens
    )


def _valid_cut_points(entries: list[Entry]) -> list[int]:
    points: list[int] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, MessageEntry) and not isinstance(
            entry.message, ToolResultMessage
        ):
            points.append(index)
        elif isinstance(entry, BranchSummaryEntry):
            points.append(index)
    return points


def _turn_start(entries: list[Entry], entry_index: int) -> int:
    for index in range(entry_index, -1, -1):
        entry = entries[index]
        if isinstance(entry, BranchSummaryEntry):
            return index
        if isinstance(entry, MessageEntry) and isinstance(entry.message, UserMessage):
            return index
    return -1


def _cut_point(entries: list[Entry], keep_recent_tokens: int) -> tuple[int, int, bool]:
    points = _valid_cut_points(entries)
    if not points:
        return 0, -1, False
    accumulated = 0
    cut_index = points[0]
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if not isinstance(entry, MessageEntry):
            continue
        accumulated += estimate_tokens(entry.message)
        if accumulated >= keep_recent_tokens:
            cut_index = next((point for point in points if point >= index), points[-1])
            break
    while cut_index > 0 and not isinstance(
        entries[cut_index - 1], MessageEntry | CompactionEntry
    ):
        cut_index -= 1
    cut = entries[cut_index]
    is_user = isinstance(cut, MessageEntry) and isinstance(cut.message, UserMessage)
    turn_start = -1 if is_user else _turn_start(entries, cut_index)
    return cut_index, turn_start, not is_user and turn_start != -1


def _message(entry: Entry) -> AgentMessage | None:
    if isinstance(entry, MessageEntry):
        return entry.message
    if isinstance(entry, BranchSummaryEntry):
        return BranchSummaryMessage(
            summary=entry.summary,
            from_id=entry.from_id,
            timestamp=entry.timestamp,
        )
    return None


def _file_operations(
    messages: tuple[AgentMessage, ...],
    previous: CompactionEntry | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    read: set[str] = set()
    modified: set[str] = set()
    if previous is not None and isinstance(previous.details, dict):
        previous_read = previous.details.get("readFiles")
        previous_modified = previous.details.get("modifiedFiles")
        if isinstance(previous_read, list):
            read.update(path for path in previous_read if isinstance(path, str))
        if isinstance(previous_modified, list):
            modified.update(
                path for path in previous_modified if isinstance(path, str)
            )
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
    path_entries: list[Entry], settings: CompactionSettings
) -> CompactionPreparation | None:
    if not path_entries or isinstance(path_entries[-1], CompactionEntry):
        return None
    previous_index = next(
        (
            index
            for index in range(len(path_entries) - 1, -1, -1)
            if isinstance(path_entries[index], CompactionEntry)
        ),
        -1,
    )
    previous_summary: str | None = None
    previous_compaction: CompactionEntry | None = None
    compactable = list(path_entries)
    if previous_index >= 0:
        previous = path_entries[previous_index]
        assert isinstance(previous, CompactionEntry)
        previous_compaction = previous
        previous_summary = previous.summary
        virtual = [
            MessageEntry(
                id=f"{previous.id}:retained:{index}",
                parent_id=(previous.id if index == 0 else f"{previous.id}:retained:{index - 1}"),
                seq=previous.seq,
                timestamp=message.timestamp,
                message=message,
            )
            for index, message in enumerate(previous.retained_tail)
        ]
        compactable = [*virtual, *path_entries[previous_index + 1 :]]

    context_messages = [
        message
        for entry in build_context_entries(path_entries)
        for message in session_entry_to_context_messages(entry)
    ]
    tokens_before = estimate_context_tokens(context_messages).tokens
    first_kept, turn_start, split = _cut_point(
        compactable, settings.keep_recent_tokens
    )
    history_end = turn_start if split else first_kept
    history = tuple(
        message
        for entry in compactable[:history_end]
        if (message := _message(entry)) is not None
    )
    turn_prefix = tuple(
        message
        for entry in compactable[turn_start:first_kept]
        if (message := _message(entry)) is not None
    ) if split else ()
    retained = tuple(
        message
        for entry in compactable[first_kept:]
        if (message := _message(entry)) is not None
    )
    read_files, modified_files = _file_operations(
        (*history, *turn_prefix), previous_compaction
    )
    return CompactionPreparation(
        messages_to_summarize=history,
        turn_prefix_messages=turn_prefix,
        retained_tail=retained,
        is_split_turn=split,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        read_files=read_files,
        modified_files=modified_files,
        settings=settings,
    )


def _serialize(messages: list[Message]) -> str:
    serialized: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            text = message.content if isinstance(message.content, str) else _text_content(message.content)
            if text:
                serialized.append(f"[User]: {text}")
        elif isinstance(message, AssistantMessage):
            thinking = [
                block.thinking
                for block in message.content
                if isinstance(block, ThinkingContent)
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
            text = _text_content(message.content)
            if thinking:
                serialized.append("[Assistant thinking]: " + "\n".join(thinking))
            if text:
                serialized.append(f"[Assistant]: {text}")
            if calls:
                serialized.append("[Assistant tool calls]: " + "; ".join(calls))
        else:
            text = _text_content(message.content)
            if len(text) > 2_000:
                text = f"{text[:2_000]}\n[... {len(text) - 2_000} more characters truncated]"
            serialized.append(f"[Tool result]: {text}")
    return "\n\n".join(serialized)


async def _generate_summary(
    messages: tuple[AgentMessage, ...],
    model: Model,
    reserve_tokens: int,
    previous_summary: str | None,
    custom_instructions: str | None,
    thinking_level: ThinkingLevel,
    request: SummaryRequest,
    context: Context,
    *,
    turn_prefix: bool = False,
) -> tuple[str, Usage]:
    prompt = TURN_PREFIX_SUMMARIZATION_PROMPT if turn_prefix else (
        UPDATE_SUMMARIZATION_PROMPT if previous_summary is not None else SUMMARIZATION_PROMPT
    )
    if custom_instructions:
        prompt += f"\n\nAdditional focus: {custom_instructions}"
    text = f"<conversation>\n{_serialize(convert_to_llm(list(messages)))}\n</conversation>\n\n"
    if previous_summary is not None and not turn_prefix:
        text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    text += prompt
    max_fraction = 0.5 if turn_prefix else 0.8
    reserve_max_tokens = math.floor(max_fraction * reserve_tokens)
    max_tokens = (
        min(reserve_max_tokens, model.max_tokens)
        if model.max_tokens > 0
        else reserve_max_tokens
    )
    options = SimpleStreamOptions(max_tokens=max_tokens)
    if model.reasoning and thinking_level != "off":
        options.reasoning = thinking_level
    response = await request(
        LlmContext(
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            messages=[
                UserMessage(
                    content=[TextContent(text=text)],
                    timestamp=0,
                )
            ],
        ),
        options,
        context,
    )
    if response.stop_reason == "aborted":
        raise CompactionFailure("aborted", response.error_message or "Summarization aborted")
    if response.stop_reason == "error":
        prefix = "Turn prefix summarization failed" if turn_prefix else "Summarization failed"
        raise CompactionFailure(
            "summarization_failed",
            f"{prefix}: {response.error_message or 'Unknown error'}",
        )
    return _text_content(response.content), response.usage


def _text_content(content: Sequence[object]) -> str:
    return "\n".join(
        block.text for block in content if isinstance(block, TextContent)
    )


async def compact_with_request(
    preparation: CompactionPreparation,
    model: Model,
    custom_instructions: str | None,
    thinking_level: ThinkingLevel,
    request: SummaryRequest,
    context: Context,
) -> CompactResult:
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None
        if preparation.messages_to_summarize:
            history_text, history_usage = await _generate_summary(
                preparation.messages_to_summarize,
                model,
                preparation.settings.reserve_tokens,
                preparation.previous_summary,
                custom_instructions,
                thinking_level,
                request,
                context,
            )
        prefix_text, prefix_usage = await _generate_summary(
            preparation.turn_prefix_messages,
            model,
            preparation.settings.reserve_tokens,
            None,
            None,
            thinking_level,
            request,
            context,
            turn_prefix=True,
        )
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
        usage = prefix_usage if history_usage is None else add_usage(history_usage, prefix_usage)
    else:
        summary, usage = await _generate_summary(
            preparation.messages_to_summarize,
            model,
            preparation.settings.reserve_tokens,
            preparation.previous_summary,
            custom_instructions,
            thinking_level,
            request,
            context,
        )
    if preparation.read_files or preparation.modified_files:
        summary += "\n\n## Files\n"
        if preparation.read_files:
            summary += "\nRead: " + ", ".join(preparation.read_files)
        if preparation.modified_files:
            summary += "\nModified: " + ", ".join(preparation.modified_files)
    return CompactResult(
        summary=summary,
        tokens_before=preparation.tokens_before,
        retained_tail=preparation.retained_tail,
        usage=usage,
        details={
            "readFiles": list(preparation.read_files),
            "modifiedFiles": list(preparation.modified_files),
        },
    )
