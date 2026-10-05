"""Summary requests and result construction for Agent compaction."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from omh.agent.compaction.preparation import estimate_projection_tokens
from omh.agent.compaction.types import (
    _MODIFIED_FILES_KEY,
    _READ_FILES_KEY,
    CompactionFailure,
    CompactionPreparation,
    CompactionResult,
    SummaryRequest,
)
from omh.agent.conversation.messages import (
    AgentMessage,
    CompactionSummaryMessage,
    CustomAgentMessage,
)
from omh.llm.types import (
    AssistantMessage,
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

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Read the conversation and produce "
    "only the requested structured summary; do not continue the conversation."
)
_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this format: Goal, Constraints & Preferences, Progress, Key Decisions, Next Steps, and Critical Context. Be concise and preserve exact paths, names, and errors."""
_UPDATE_SUMMARIZATION_PROMPT = """Update the existing structured summary with the new conversation messages. Preserve prior goals, constraints, decisions, and critical context; update progress and next steps. Be concise and preserve exact paths, names, and errors."""
_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the prefix of a turn whose suffix is retained. Summarize the original request, early progress, and context needed to understand the retained suffix."""


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
        reported=(
            False if False in (left.reported, right.reported)
            else True if left.reported is True and right.reported is True else None
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
