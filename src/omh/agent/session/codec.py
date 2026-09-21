"""Explicit JSON codec for durable session payloads.

Upstream TypeScript stores messages by serializing the plain object it already
holds. Python holds typed dataclasses, so the on-disk shape is written and read
here instead: keys keep the upstream names so a stored payload stays comparable
with pi's storage format. Decoding validates the payload it reads, because a
stored row is read back from a file rather than from live memory.
"""

from __future__ import annotations

from typing import cast

from omh.agent.types import AgentMessage, CompactionSummaryMessage
from omh.llm.types import (
    AssistantContent,
    AssistantMessage,
    ImageContent,
    JsonValue,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserContent,
    UserMessage,
)


def _record(value: JsonValue, where: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _required(record: dict[str, JsonValue], key: str, where: str) -> JsonValue:
    if key not in record:
        raise ValueError(f"{where}.{key} is missing")
    return record[key]


def _text(record: dict[str, JsonValue], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string")
    return value


def _integer(record: dict[str, JsonValue], key: str, where: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be an integer")
    return value


def _number(record: dict[str, JsonValue], key: str, where: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}.{key} must be a number")
    return float(value)


def _optional_text(record: dict[str, JsonValue], key: str, where: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string when present")
    return value


def _optional_integer(record: dict[str, JsonValue], key: str, where: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be an integer when present")
    return value


def _optional_boolean(record: dict[str, JsonValue], key: str, where: str) -> bool | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{where}.{key} must be a boolean when present")
    return value


def _put_optional(record: dict[str, JsonValue], key: str, value: JsonValue | None) -> None:
    if value is not None:
        record[key] = value


def encode_usage(usage: Usage) -> dict[str, JsonValue]:
    encoded: dict[str, JsonValue] = {
        "input": usage.input,
        "output": usage.output,
        "cacheRead": usage.cache_read,
        "cacheWrite": usage.cache_write,
        "totalTokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cacheRead": usage.cost.cache_read,
            "cacheWrite": usage.cost.cache_write,
            "total": usage.cost.total,
        },
    }
    _put_optional(encoded, "cacheWrite1h", usage.cache_write_1h)
    _put_optional(encoded, "reasoning", usage.reasoning)
    return encoded


def decode_usage(value: JsonValue) -> Usage:
    record = _record(value, "usage")
    cost = _record(_required(record, "cost", "usage"), "usage.cost")
    return Usage(
        input=_integer(record, "input", "usage"),
        output=_integer(record, "output", "usage"),
        cache_read=_integer(record, "cacheRead", "usage"),
        cache_write=_integer(record, "cacheWrite", "usage"),
        total_tokens=_integer(record, "totalTokens", "usage"),
        cost=UsageCost(
            input=_number(cost, "input", "usage.cost"),
            output=_number(cost, "output", "usage.cost"),
            cache_read=_number(cost, "cacheRead", "usage.cost"),
            cache_write=_number(cost, "cacheWrite", "usage.cost"),
            total=_number(cost, "total", "usage.cost"),
        ),
        cache_write_1h=_optional_integer(record, "cacheWrite1h", "usage"),
        reasoning=_optional_integer(record, "reasoning", "usage"),
    )


def _encode_content_block(block: AssistantContent | ImageContent) -> JsonValue:
    if isinstance(block, TextContent):
        encoded: dict[str, JsonValue] = {"type": "text", "text": block.text}
        _put_optional(encoded, "textSignature", block.text_signature)
        return encoded
    if isinstance(block, ThinkingContent):
        encoded = {"type": "thinking", "thinking": block.thinking}
        _put_optional(encoded, "thinkingSignature", block.thinking_signature)
        _put_optional(encoded, "redacted", block.redacted)
        return encoded
    if isinstance(block, ImageContent):
        return {"type": "image", "data": block.data, "mimeType": block.mime_type}
    encoded = {
        "type": "toolCall",
        "id": block.id,
        "name": block.name,
        "arguments": cast(JsonValue, block.arguments),
    }
    _put_optional(encoded, "thoughtSignature", block.thought_signature)
    _put_optional(encoded, "namespace", block.namespace)
    return encoded


def _decode_content_block(value: JsonValue, where: str) -> TextContent | ThinkingContent | ImageContent | ToolCall:
    record = _record(value, where)
    block_type = record.get("type")
    if block_type == "text":
        return TextContent(
            text=_text(record, "text", where),
            text_signature=_optional_text(record, "textSignature", where),
        )
    if block_type == "thinking":
        return ThinkingContent(
            thinking=_text(record, "thinking", where),
            thinking_signature=_optional_text(record, "thinkingSignature", where),
            redacted=_optional_boolean(record, "redacted", where),
        )
    if block_type == "image":
        return ImageContent(
            data=_text(record, "data", where),
            mime_type=_text(record, "mimeType", where),
        )
    if block_type == "toolCall":
        return ToolCall(
            id=_text(record, "id", where),
            name=_text(record, "name", where),
            arguments=cast(dict[str, object], _record(record.get("arguments"), f"{where}.arguments")),
            thought_signature=_optional_text(record, "thoughtSignature", where),
            namespace=_optional_text(record, "namespace", where),
        )
    raise ValueError(f"{where}.type is not a known content type")


def encode_tool_result_content(
    content: list[ToolResultContent],
) -> list[JsonValue]:
    return [_encode_content_block(block) for block in content]


def decode_tool_result_content(
    value: object, where: str
) -> list[ToolResultContent]:
    decoded: list[ToolResultContent] = []
    for index, block in enumerate(_record_list(cast(JsonValue, value), where)):
        item = _decode_content_block(block, f"{where}[{index}]")
        if isinstance(item, ThinkingContent | ToolCall):
            raise ValueError(f"{where}[{index}] is not tool result content")
        decoded.append(item)
    return decoded


def encode_message(message: AgentMessage) -> dict[str, JsonValue]:
    if isinstance(message, CompactionSummaryMessage):
        return {
            "role": "compactionSummary",
            "summary": message.summary,
            "tokensBefore": message.tokens_before,
            "timestamp": message.timestamp,
        }
    if isinstance(message, UserMessage):
        content: JsonValue = (
            message.content
            if isinstance(message.content, str)
            else [_encode_content_block(block) for block in message.content]
        )
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, AssistantMessage):
        encoded: dict[str, JsonValue] = {
            "role": "assistant",
            "content": [_encode_content_block(block) for block in message.content],
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": encode_usage(message.usage),
            "stopReason": message.stop_reason,
            "timestamp": message.timestamp,
        }
        _put_optional(encoded, "responseModel", message.response_model)
        _put_optional(encoded, "responseId", message.response_id)
        _put_optional(encoded, "providerThinkingLevel", message.provider_thinking_level)
        _put_optional(encoded, "errorMessage", message.error_message)
        _put_optional(encoded, "rawStopReason", message.raw_stop_reason)
        return encoded
    encoded = {
        "role": "toolResult",
        "toolCallId": message.tool_call_id,
        "toolName": message.tool_name,
        "content": [_encode_content_block(block) for block in message.content],
        "isError": message.is_error,
        "timestamp": message.timestamp,
    }
    _put_optional(encoded, "details", cast(JsonValue, message.details))
    if message.usage is not None:
        encoded["usage"] = encode_usage(message.usage)
    return encoded


def _decode_user_content(value: JsonValue, where: str) -> str | list[UserContent]:
    if isinstance(value, str):
        return value
    blocks: list[UserContent] = []
    for index, block in enumerate(_record_list(value, where)):
        decoded = _decode_content_block(block, f"{where}[{index}]")
        if isinstance(decoded, ThinkingContent | ToolCall):
            raise ValueError(f"{where}[{index}] is not user message content")
        blocks.append(decoded)
    return blocks


def _record_list(value: JsonValue, where: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    return value


_STOP_REASONS = frozenset({"pending", "stop", "length", "toolUse", "error", "aborted"})


def _stop_reason(record: dict[str, JsonValue], where: str) -> StopReason:
    value = _text(record, "stopReason", where)
    if value not in _STOP_REASONS:
        raise ValueError(f"{where}.stopReason is not a known stop reason: {value!r}")
    return cast(StopReason, value)


def decode_message(value: JsonValue) -> AgentMessage:
    record = _record(value, "message")
    role = record.get("role")
    if role == "compactionSummary":
        return CompactionSummaryMessage(
            summary=_text(record, "summary", "message"),
            tokens_before=_integer(record, "tokensBefore", "message"),
            timestamp=_integer(record, "timestamp", "message"),
        )
    if role == "user":
        return UserMessage(
            content=_decode_user_content(_required(record, "content", "message"), "message.content"),
            timestamp=_integer(record, "timestamp", "message"),
        )
    if role == "assistant":
        content: list[AssistantContent] = []
        for index, block in enumerate(_record_list(record.get("content"), "message.content")):
            decoded = _decode_content_block(block, f"message.content[{index}]")
            if isinstance(decoded, ImageContent):
                raise ValueError(f"message.content[{index}] is not assistant message content")
            content.append(decoded)
        return AssistantMessage(
            api=_text(record, "api", "message"),
            provider=_text(record, "provider", "message"),
            model=_text(record, "model", "message"),
            usage=decode_usage(_required(record, "usage", "message")),
            stop_reason=_stop_reason(record, "message"),
            timestamp=_integer(record, "timestamp", "message"),
            content=content,
            response_model=_optional_text(record, "responseModel", "message"),
            response_id=_optional_text(record, "responseId", "message"),
            provider_thinking_level=_optional_text(record, "providerThinkingLevel", "message"),
            error_message=_optional_text(record, "errorMessage", "message"),
            raw_stop_reason=_optional_text(record, "rawStopReason", "message"),
        )
    if role == "toolResult":
        tool_content: list[ToolResultContent] = []
        for index, block in enumerate(_record_list(record.get("content"), "message.content")):
            decoded = _decode_content_block(block, f"message.content[{index}]")
            if isinstance(decoded, ThinkingContent | ToolCall):
                raise ValueError(f"message.content[{index}] is not tool result content")
            tool_content.append(decoded)
        usage = record.get("usage")
        return ToolResultMessage(
            tool_call_id=_text(record, "toolCallId", "message"),
            tool_name=_text(record, "toolName", "message"),
            content=tool_content,
            timestamp=_integer(record, "timestamp", "message"),
            is_error=bool(_optional_boolean(record, "isError", "message")),
            details=record.get("details"),
            usage=None if usage is None else decode_usage(usage),
        )
    raise ValueError(f"message.role is not a known message role: {role!r}")
