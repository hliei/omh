from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

from omh.llm.types import (
    Context,
    Message,
    SystemMessage,
    Tool,
    ToolReference,
    TranscriptContext,
)
from omh.llm.utils.text import content_text, get_system_message_text


def to_tool_declaration(tool: Tool) -> Tool:
    """Strip executable and display-only state from a tool for transcript comparison."""
    return Tool(name=tool.name, description=tool.description, parameters=copy.deepcopy(tool.parameters))


def declarations_equal(left: Tool, right: Tool) -> bool:
    """Whether two tools declare the same interface to the model."""
    return left.name == right.name and left.description == right.description and left.parameters == right.parameters


@dataclass(frozen=True, slots=True)
class ToolStateChanges:
    """Tool additions and removals between two complete tool states."""

    tools_added: list[Tool]
    tools_removed: list[ToolReference]


def get_tool_state_changes(previous: Sequence[Tool], current: Sequence[Tool]) -> ToolStateChanges:
    """Compare two tool states; a changed definition is a removal followed by an addition."""
    previous_by_name = {tool.name: tool for tool in previous}
    current_by_name = {tool.name: tool for tool in current}
    added = [
        tool
        for tool in current
        if tool.name not in previous_by_name or not declarations_equal(previous_by_name[tool.name], tool)
    ]
    removed = [
        ToolReference(name=tool.name)
        for tool in previous
        if tool.name not in current_by_name or not declarations_equal(tool, current_by_name[tool.name])
    ]
    return ToolStateChanges(
        tools_added=[to_tool_declaration(tool) for tool in added],
        tools_removed=removed,
    )


def create_initial_system_message(
    system_prompt: str | None,
    tools: list[Tool] | None,
) -> SystemMessage | None:
    """Build the leading system message for a prompt and tool set.

    Returns ``None`` when both are empty so an empty transcript stays empty.
    """
    has_system_prompt = system_prompt is not None and len(system_prompt) > 0
    has_tools = tools is not None and len(tools) > 0
    if not has_system_prompt and not has_tools:
        return None
    return SystemMessage(
        content=system_prompt or "",
        timestamp=0,
        tools_added=list(tools) if tools else None,
    )


def normalize_context(context: Context) -> TranscriptContext:
    """Fold ``Context.system_prompt`` and ``Context.tools`` into a leading system message."""
    initial_message = create_initial_system_message(context.system_prompt, context.tools)
    messages: list[Message] = [initial_message, *context.messages] if initial_message else list(context.messages)
    return TranscriptContext(messages=messages)


def get_initial_system_message(messages: Sequence[object]) -> SystemMessage | None:
    """Return the leading system message if the transcript starts with one."""
    first = messages[0] if messages else None
    return first if isinstance(first, SystemMessage) else None


def get_current_tools(messages: Sequence[object]) -> list[Tool]:
    """Resolve the tools available after applying every transcript delta in order."""
    tools: dict[str, Tool] = {}
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        for removed in message.tools_removed or []:
            tools.pop(removed.name, None)
        for added in message.tools_added or []:
            tools[added.name] = added
    return list(tools.values())


def get_current_system_message(messages: Sequence[object]) -> SystemMessage | None:
    """Replay every system message into one leading system message."""
    content: list[str] = []
    sections: dict[str, str] = {}
    timestamp: int | None = None
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        text = content_text(message.content)
        if text:
            content.append(text)
        for name, value in (message.sections or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and not tools:
        return None
    return SystemMessage(
        content="\n\n".join(content),
        timestamp=timestamp if timestamp is not None else 0,
        sections=dict(sections) if sections else None,
        tools_added=list(tools) if tools else None,
    )


def get_current_system_prompt(messages: Sequence[object]) -> str:
    """Render the current system prompt text after replaying every system message."""
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message is not None else ""


def collapse_system_messages(context: TranscriptContext) -> TranscriptContext:
    """Rebuild a transcript for APIs without mid-conversation system messages.

    The replayed system message leads and every later system message is dropped,
    so the prompt and tool declarations are expressed exactly once.
    """
    head = get_current_system_message(context.messages)
    messages: list[Message] = [message for message in context.messages if not isinstance(message, SystemMessage)]
    if head is not None:
        messages = [head, *messages]
    return TranscriptContext(messages=messages)


def resolve_transcript(
    context: TranscriptContext,
    supports_mid_convo_system_messages: bool | None,
) -> TranscriptContext:
    """Keep later system messages when the model accepts them; otherwise collapse them."""
    if supports_mid_convo_system_messages:
        return context
    return collapse_system_messages(context)
