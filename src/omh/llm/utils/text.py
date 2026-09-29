from __future__ import annotations

from collections.abc import Sequence

from omh.llm.types import (
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
)

Content = TextContent | ImageContent | ThinkingContent | ToolCall


def content_text(content: str | Sequence[Content], separator: str = "\n") -> str:
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if block.type == "text")


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message as a complete prompt: its content followed by its sections."""
    parts = [content_text(message.content)]
    for text in (message.sections or {}).values():
        if text is not None:
            parts.append(text)
    return "\n\n".join(part for part in parts if part)


def render_system_message_update(message: SystemMessage) -> str:
    """Render a later system message for APIs that accept system messages mid-conversation.

    Section changes are framed by name so the model can relate them to the leading
    prompt. This framing is request-time only and may change between versions.
    """
    parts: list[str] = []
    text = content_text(message.content)
    if text:
        parts.append(text)
    for name, value in (message.sections or {}).items():
        if value is None:
            parts.append(f'Removed system prompt section "{name}".')
        else:
            parts.append(f'Updated system prompt section "{name}":\n\n{value}')
    return "\n\n".join(parts)
