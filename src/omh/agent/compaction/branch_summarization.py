from __future__ import annotations

import json
from dataclasses import dataclass

from omh.agent.compaction.compaction import (
    SUMMARIZATION_SYSTEM_PROMPT,
    SummaryRequest,
    estimate_tokens,
)
from omh.agent.context import Context
from omh.agent.messages import convert_to_llm
from omh.agent.session.types import Entry
from omh.agent.types import (
    AgentMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    ThinkingLevel,
)
from omh.llm.types import (
    AssistantMessage,
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


@dataclass(frozen=True, slots=True)
class BranchSummaryResult:
    summary: str
    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class BranchPreparation:
    messages: tuple[AgentMessage, ...]
    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    total_tokens: int


@dataclass(slots=True)
class BranchSummaryFailure(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def _message(entry: Entry) -> AgentMessage | None:
    if entry.type == "message":
        return None if isinstance(entry.message, ToolResultMessage) else entry.message
    if entry.type == "branch_summary":
        return BranchSummaryMessage(
            summary=entry.summary,
            from_id=entry.from_id,
            timestamp=entry.timestamp,
        )
    if entry.type == "compaction":
        return CompactionSummaryMessage(
            summary=entry.summary,
            tokens_before=entry.tokens_before,
            timestamp=entry.timestamp,
        )
    return None


def _record_file_operations(
    message: AgentMessage, read: set[str], modified: set[str]
) -> None:
    if not isinstance(message, AssistantMessage):
        return
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


def prepare_branch_entries(
    entries: list[Entry], token_budget: int = 0
) -> BranchPreparation:
    read: set[str] = set()
    modified: set[str] = set()
    for entry in entries:
        if entry.type != "branch_summary" or not isinstance(entry.details, dict):
            continue
        previous_read = entry.details.get("readFiles")
        previous_modified = entry.details.get("modifiedFiles")
        if isinstance(previous_read, list):
            read.update(path for path in previous_read if isinstance(path, str))
        if isinstance(previous_modified, list):
            modified.update(
                path for path in previous_modified if isinstance(path, str)
            )

    messages: list[AgentMessage] = []
    total_tokens = 0
    for entry in reversed(entries):
        message = _message(entry)
        if message is None:
            continue
        _record_file_operations(message, read, modified)
        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.type in {"compaction", "branch_summary"} and total_tokens < (
                token_budget * 0.9
            ):
                messages.insert(0, message)
                total_tokens += tokens
            break
        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(
        messages=tuple(messages),
        read_files=tuple(sorted(read - modified)),
        modified_files=tuple(sorted(modified)),
        total_tokens=total_tokens,
    )


BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\n"
    "Summary of that exploration:\n\n"
)
BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


def _text_content(message: AssistantMessage) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


def _serialize(messages: list[Message]) -> str:
    serialized: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = message.content
            text = content if isinstance(content, str) else "\n".join(
                block.text for block in content if isinstance(block, TextContent)
            )
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
            text = _text_content(message)
            if thinking:
                serialized.append("[Assistant thinking]: " + "\n".join(thinking))
            if text:
                serialized.append(f"[Assistant]: {text}")
            if calls:
                serialized.append("[Assistant tool calls]: " + "; ".join(calls))
        else:
            text = "\n".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            )
            serialized.append(f"[Tool result]: {text[:2_000]}")
    return "\n\n".join(serialized)


async def generate_branch_summary_with_request(
    preparation: BranchPreparation,
    model: Model,
    custom_instructions: str | None,
    thinking_level: ThinkingLevel,
    request: SummaryRequest,
    context: Context,
) -> BranchSummaryResult:
    if not preparation.messages:
        return BranchSummaryResult(
            summary="No content to summarize",
            read_files=(),
            modified_files=(),
        )
    instructions = BRANCH_SUMMARY_PROMPT
    if custom_instructions:
        instructions += f"\n\nAdditional focus: {custom_instructions}"
    prompt = (
        f"<conversation>\n{_serialize(convert_to_llm(list(preparation.messages)))}"
        f"\n</conversation>\n\n{instructions}"
    )
    max_tokens = min(2_048, model.max_tokens) if model.max_tokens > 0 else 2_048
    options = SimpleStreamOptions(max_tokens=max_tokens)
    if model.reasoning and thinking_level != "off":
        options.reasoning = thinking_level
    response = await request(
        LlmContext(
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            messages=[UserMessage(content=[TextContent(text=prompt)], timestamp=0)],
        ),
        options,
        context,
    )
    if response.stop_reason == "aborted":
        raise BranchSummaryFailure(
            "aborted", response.error_message or "Branch summary aborted"
        )
    if response.stop_reason == "error":
        raise BranchSummaryFailure(
            "summarization_failed",
            f"Branch summary failed: {response.error_message or 'Unknown error'}",
        )
    summary = BRANCH_SUMMARY_PREAMBLE + _text_content(response)
    if preparation.read_files or preparation.modified_files:
        summary += "\n\n## Files\n"
        if preparation.read_files:
            summary += "\nRead: " + ", ".join(preparation.read_files)
        if preparation.modified_files:
            summary += "\nModified: " + ", ".join(preparation.modified_files)
    return BranchSummaryResult(
        summary=summary or "No summary generated",
        usage=response.usage,
        read_files=preparation.read_files,
        modified_files=preparation.modified_files,
    )
