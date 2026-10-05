from __future__ import annotations

from collections.abc import Sequence

from omh.llm.types import (
    AssistantMessage,
    ImageContent,
    Message,
    SystemMessage,
    TextContent,
    TranscriptContext,
    Usage,
)
from omh.llm.utils.text import get_system_message_text
from omh.llm.utils.transcript import get_current_system_message, get_current_tools

CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def estimate_text_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN if text else 0


def _content_chars(content: str | Sequence[TextContent | ImageContent]) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        chars += len(block.text) if block.type == "text" else ESTIMATED_IMAGE_CHARS
    return chars


def estimate_message_chars(message: Message) -> int:
    if message.role == "user":
        return _content_chars(message.content)
    if message.role == "assistant":
        chars = 0
        for block in message.content:
            if block.type == "text":
                chars += len(block.text)
            elif block.type == "thinking":
                chars += len(block.thinking)
            else:
                chars += len(block.name) + len(str(block.arguments))
        return chars
    return _content_chars(message.content)


def estimate_context_tokens(context: TranscriptContext) -> int:
    system = get_current_system_message(context.messages)
    chars = len(get_system_message_text(system)) if system is not None else 0
    for message in context.messages:
        if isinstance(message, SystemMessage):
            continue
        chars += estimate_message_chars(message)
    for tool in get_current_tools(context.messages):
        chars += len(tool.name) + len(tool.description) + len(str(tool.parameters))
    tokens = estimate_text_tokens("x" * chars) if chars else 0
    for message in reversed(context.messages):
        if isinstance(message, AssistantMessage) and message.usage.reported is not False and message.usage.total_tokens:
            return message.usage.total_tokens + estimate_text_tokens(
                "x" * sum(estimate_message_chars(item) for item in context.messages[context.messages.index(message) + 1 :])
            )
    return tokens
