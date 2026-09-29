from __future__ import annotations

from collections.abc import Sequence

from omh.llm.types import Context, Message, SystemMessage, Tool, TranscriptContext
from omh.llm.utils.text import content_text, get_system_message_text


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
