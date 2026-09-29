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
