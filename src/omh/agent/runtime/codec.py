from __future__ import annotations

from typing import cast

from omh.agent.session.codec import decode_message, encode_message
from omh.llm.types import (
    AssistantMessage,
    JsonValue,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from omh.llm.utils.assistant_message_frame import (
    AssistantMessageFrame,
    StartFrame,
    TextDeltaFrame,
    TextEndFrame,
    TextStartFrame,
    ThinkingDeltaFrame,
    ThinkingEndFrame,
    ThinkingStartFrame,
    ToolCallCheckpointFrame,
    ToolCallDeltaFrame,
    ToolCallEndFrame,
    ToolCallStartFrame,
)


def _optional(record: dict[str, JsonValue], key: str, value: JsonValue | None) -> None:
    if value is not None:
        record[key] = value


def _encode_text(content: TextContent) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"type": "text", "text": content.text}
    _optional(result, "textSignature", content.text_signature)
    return result


def _encode_thinking(content: ThinkingContent) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"type": "thinking", "thinking": content.thinking}
    _optional(result, "thinkingSignature", content.thinking_signature)
    _optional(result, "redacted", content.redacted)
    return result


def _encode_tool_call(tool_call: ToolCall) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "type": "toolCall",
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": cast(JsonValue, tool_call.arguments),
    }
    _optional(result, "thoughtSignature", tool_call.thought_signature)
    _optional(result, "namespace", tool_call.namespace)
    return result


def encode_assistant_frame(frame: AssistantMessageFrame) -> dict[str, JsonValue]:
    if isinstance(frame, StartFrame):
        return {"type": frame.type, "partial": encode_message(frame.partial)}
    result: dict[str, JsonValue] = {"type": frame.type, "contentIndex": frame.content_index}
    if isinstance(frame, TextStartFrame):
        result["content"] = _encode_text(frame.content)
    elif isinstance(frame, TextDeltaFrame | ThinkingDeltaFrame | ToolCallDeltaFrame):
        result["delta"] = frame.delta
    elif isinstance(frame, TextEndFrame):
        result["content"] = frame.content
        _optional(result, "textSignature", frame.text_signature)
    elif isinstance(frame, ThinkingStartFrame):
        result["content"] = _encode_thinking(frame.content)
    elif isinstance(frame, ThinkingEndFrame):
        result["content"] = frame.content
        _optional(result, "thinkingSignature", frame.thinking_signature)
        _optional(result, "redacted", frame.redacted)
    elif isinstance(frame, ToolCallStartFrame):
        result["toolCall"] = _encode_tool_call(frame.tool_call)
    elif isinstance(frame, ToolCallCheckpointFrame):
        result["json"] = frame.json
    elif isinstance(frame, ToolCallEndFrame):
        result.update(
            {
                "id": frame.id,
                "name": frame.name,
                "arguments": cast(JsonValue, frame.arguments),
            }
        )
        _optional(result, "thoughtSignature", frame.thought_signature)
        _optional(result, "namespace", frame.namespace)
    return result


def _record(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return cast(dict[str, object], value)


def _text(record: dict[str, object], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string")
    return value


def _index(record: dict[str, object], where: str) -> int:
    value = record.get("contentIndex")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.contentIndex must be an integer")
    return value


def _optional_text(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"assistant frame {key} must be a string")
    return value


def _decode_text(value: object) -> TextContent:
    record = _record(value, "assistant frame content")
    return TextContent(text=_text(record, "text", "assistant frame content"), text_signature=_optional_text(record, "textSignature"))


def _decode_thinking(value: object) -> ThinkingContent:
    record = _record(value, "assistant frame content")
    redacted = record.get("redacted")
    if redacted is not None and not isinstance(redacted, bool):
        raise ValueError("assistant frame redacted must be a boolean")
    return ThinkingContent(
        thinking=_text(record, "thinking", "assistant frame content"),
        thinking_signature=_optional_text(record, "thinkingSignature"),
        redacted=redacted,
    )


def _decode_tool_call(value: object) -> ToolCall:
    record = _record(value, "assistant frame toolCall")
    arguments = _record(record.get("arguments"), "assistant frame toolCall.arguments")
    return ToolCall(
        id=_text(record, "id", "assistant frame toolCall"),
        name=_text(record, "name", "assistant frame toolCall"),
        arguments=arguments,
        thought_signature=_optional_text(record, "thoughtSignature"),
        namespace=_optional_text(record, "namespace"),
    )


def decode_assistant_frame(value: object) -> AssistantMessageFrame:
    record = _record(value, "assistant frame")
    frame_type = record.get("type")
    if frame_type == "start":
        partial = decode_message(cast(JsonValue, record.get("partial")))
        if not isinstance(partial, AssistantMessage):
            raise ValueError("assistant start frame partial must be an assistant message")
        return StartFrame(partial=partial)
    content_index = _index(record, "assistant frame")
    if frame_type == "text_start":
        return TextStartFrame(content_index=content_index, content=_decode_text(record.get("content")))
    if frame_type == "text_delta":
        return TextDeltaFrame(content_index=content_index, delta=_text(record, "delta", "assistant frame"))
    if frame_type == "text_end":
        return TextEndFrame(
            content_index=content_index,
            content=_text(record, "content", "assistant frame"),
            text_signature=_optional_text(record, "textSignature"),
        )
    if frame_type == "thinking_start":
        return ThinkingStartFrame(content_index=content_index, content=_decode_thinking(record.get("content")))
    if frame_type == "thinking_delta":
        return ThinkingDeltaFrame(content_index=content_index, delta=_text(record, "delta", "assistant frame"))
    if frame_type == "thinking_end":
        redacted = record.get("redacted")
        if redacted is not None and not isinstance(redacted, bool):
            raise ValueError("assistant frame redacted must be a boolean")
        return ThinkingEndFrame(
            content_index=content_index,
            content=_text(record, "content", "assistant frame"),
            thinking_signature=_optional_text(record, "thinkingSignature"),
            redacted=redacted,
        )
    if frame_type == "toolcall_start":
        return ToolCallStartFrame(content_index=content_index, tool_call=_decode_tool_call(record.get("toolCall")))
    if frame_type == "toolcall_checkpoint":
        return ToolCallCheckpointFrame(content_index=content_index, json=_text(record, "json", "assistant frame"))
    if frame_type == "toolcall_delta":
        return ToolCallDeltaFrame(content_index=content_index, delta=_text(record, "delta", "assistant frame"))
    if frame_type == "toolcall_end":
        return ToolCallEndFrame(
            content_index=content_index,
            id=_text(record, "id", "assistant frame"),
            name=_text(record, "name", "assistant frame"),
            arguments=_record(record.get("arguments"), "assistant frame.arguments"),
            thought_signature=_optional_text(record, "thoughtSignature"),
            namespace=_optional_text(record, "namespace"),
        )
    raise ValueError(f"Unknown assistant frame type: {frame_type!r}")
