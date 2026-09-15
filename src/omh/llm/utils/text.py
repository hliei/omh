from __future__ import annotations

from omh.llm.types import ImageContent, TextContent, ThinkingContent, ToolCall

Content = TextContent | ImageContent | ThinkingContent | ToolCall


def content_text(content: str | list[Content], separator: str = "\n") -> str:
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if block.type == "text")
