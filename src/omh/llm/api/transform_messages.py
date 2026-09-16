from __future__ import annotations

from dataclasses import replace

from omh.llm.types import (
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"


def _replace_images(content: list[TextContent | ImageContent], placeholder: str) -> list[TextContent | ImageContent]:
    result: list[TextContent | ImageContent] = []
    previous_was_placeholder = False
    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue
        result.append(block)
        previous_was_placeholder = block.text == placeholder
    return result


def transform_messages(messages: list[Message], model: Model) -> list[Message]:
    normalized: list[Message] = []
    for message in messages:
        if message.role == "user" and message.content is None:
            normalized.append(UserMessage(content=[], timestamp=message.timestamp))
        else:
            normalized.append(message)
    if "image" in model.input:
        return normalized
    transformed: list[Message] = []
    for message in normalized:
        if message.role == "user" and isinstance(message.content, list):
            transformed.append(
                UserMessage(
                    content=_replace_images(message.content, NON_VISION_USER_IMAGE_PLACEHOLDER),
                    timestamp=message.timestamp,
                )
            )
        elif isinstance(message, ToolResultMessage):
            transformed.append(
                replace(
                    message,
                    content=_replace_images(message.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER),
                )
            )
        else:
            transformed.append(message)
    return transformed
