from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

from omh.llm.types import (
    AssistantMessage,
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from omh.llm.utils.json_parse import parse_streaming_json


@dataclass(slots=True)
class StartFrame:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True)
class TextStartFrame:
    content_index: int
    content: TextContent
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True)
class TextDeltaFrame:
    content_index: int
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEndFrame:
    content_index: int
    content: str
    type: Literal["text_end"] = "text_end"
    text_signature: str | None = None


@dataclass(slots=True)
class ThinkingStartFrame:
    content_index: int
    content: ThinkingContent
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(slots=True)
class ThinkingDeltaFrame:
    content_index: int
    delta: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True)
class ThinkingEndFrame:
    content_index: int
    content: str
    type: Literal["thinking_end"] = "thinking_end"
    thinking_signature: str | None = None
    redacted: bool | None = None


@dataclass(slots=True)
class ToolCallStartFrame:
    content_index: int
    tool_call: ToolCall
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True)
class ToolCallCheckpointFrame:
    content_index: int
    json: str
    type: Literal["toolcall_checkpoint"] = "toolcall_checkpoint"


@dataclass(slots=True)
class ToolCallDeltaFrame:
    content_index: int
    delta: str
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True)
class ToolCallEndFrame:
    content_index: int
    id: str
    name: str
    arguments: dict[str, object]
    type: Literal["toolcall_end"] = "toolcall_end"
    thought_signature: str | None = None
    namespace: str | None = None


AssistantMessageFrame = (
    StartFrame
    | TextStartFrame
    | TextDeltaFrame
    | TextEndFrame
    | ThinkingStartFrame
    | ThinkingDeltaFrame
    | ThinkingEndFrame
    | ToolCallStartFrame
    | ToolCallCheckpointFrame
    | ToolCallDeltaFrame
    | ToolCallEndFrame
)

EncoderTextState = tuple[Literal["text", "thinking"], int, int]
EncoderToolState = tuple[Literal["toolCall"], bool, str, str]
EncoderBlockState = EncoderTextState | EncoderToolState

ReducerTextState = tuple[Literal["text", "thinking"], bool]
ReducerToolState = tuple[Literal["toolCall"], bool, str]
ReducerBlockState = ReducerTextState | ReducerToolState


def _clone_text(content: TextContent) -> TextContent:
    return replace(content)


def _clone_thinking(content: ThinkingContent) -> ThinkingContent:
    return replace(content)


def _clone_tool_call(tool_call: ToolCall) -> ToolCall:
    return replace(tool_call, arguments=copy.deepcopy(tool_call.arguments))


def _clone_start_message(message: AssistantMessage) -> AssistantMessage:
    return replace(
        message,
        content=[],
        stop_reason="pending",
        usage=copy.deepcopy(message.usage),
        error_message=None,
        raw_stop_reason=None,
    )


def _assert_content_index(content_index: int) -> None:
    if content_index < 0:
        raise ValueError(f"Invalid assistant message frame contentIndex: {content_index}")


def _event_block(event: AssistantMessageEvent) -> TextContent | ThinkingContent | ToolCall:
    if isinstance(event, (StartEvent, DoneEvent, ErrorEvent)):
        raise ValueError(f"{event.type} event has no content block")
    _assert_content_index(event.content_index)
    try:
        return event.partial.content[event.content_index]
    except IndexError as error:
        raise ValueError(f"{event.type} event has no content block at index {event.content_index}") from error


def _serialized_arguments(arguments: dict[str, object]) -> str:
    import json

    return json.dumps(arguments, separators=(",", ":"), sort_keys=False)


_EMPTY_PARSED_TOOL_ARGUMENTS = _serialized_arguments(parse_streaming_json(""))


def _is_json_prefix(snapshot: object, current: object) -> bool:
    if isinstance(snapshot, str):
        return isinstance(current, str) and current.startswith(snapshot)
    if isinstance(snapshot, list):
        return (
            isinstance(current, list)
            and len(snapshot) <= len(current)
            and all(_is_json_prefix(value, current[index]) for index, value in enumerate(snapshot))
        )
    if not isinstance(snapshot, dict) or snapshot is None:
        return snapshot == current
    if not isinstance(current, dict):
        return False
    return all(key in current and _is_json_prefix(value, current[key]) for key, value in snapshot.items())


class AssistantMessageFrameEncoder:
    def __init__(self) -> None:
        self._started = False
        self._terminal = False
        self._blocks: dict[int, EncoderBlockState] = {}

    def encode(self, event: AssistantMessageEvent) -> AssistantMessageFrame | None:
        if self._terminal:
            raise ValueError(f"Assistant message event {event.type} follows a terminal event")
        if isinstance(event, StartEvent):
            if self._started:
                raise ValueError("Assistant message stream contains more than one start event")
            self._started = True
            return StartFrame(partial=_clone_start_message(event.partial))
        if isinstance(event, (DoneEvent, ErrorEvent)):
            if isinstance(event, DoneEvent) and not self._started:
                raise ValueError("Assistant message done event appears before start")
            self._terminal = True
            return None
        if not self._started:
            raise ValueError(f"Assistant message {event.type} event appears before start")
        if isinstance(event, TextStartEvent):
            content = _event_block(event)
            if content.type != "text":
                raise ValueError(f"text_start event points to {content.type} block at index {event.content_index}")
            self._start_block(event.content_index, ("text", len(content.text), 0))
            return TextStartFrame(content_index=event.content_index, content=_clone_text(content))
        if isinstance(event, TextDeltaEvent):
            return self._encode_text_delta(event.content_index, event.delta, "text")
        if isinstance(event, TextEndEvent):
            content = _event_block(event)
            if content.type != "text":
                raise ValueError(f"text_end event points to {content.type} block at index {event.content_index}")
            self._end_block(event.content_index, "text")
            return TextEndFrame(
                content_index=event.content_index,
                content=event.content,
                text_signature=content.text_signature,
            )
        if isinstance(event, ThinkingStartEvent):
            content = _event_block(event)
            if content.type != "thinking":
                raise ValueError(f"thinking_start event points to {content.type} block at index {event.content_index}")
            self._start_block(event.content_index, ("thinking", len(content.thinking), 0))
            return ThinkingStartFrame(content_index=event.content_index, content=_clone_thinking(content))
        if isinstance(event, ThinkingDeltaEvent):
            return self._encode_text_delta(event.content_index, event.delta, "thinking")
        if isinstance(event, ThinkingEndEvent):
            content = _event_block(event)
            if content.type != "thinking":
                raise ValueError(f"thinking_end event points to {content.type} block at index {event.content_index}")
            self._end_block(event.content_index, "thinking")
            return ThinkingEndFrame(
                content_index=event.content_index,
                content=event.content,
                thinking_signature=content.thinking_signature,
                redacted=content.redacted,
            )
        if isinstance(event, ToolCallStartEvent):
            content = _event_block(event)
            if content.type != "toolCall":
                raise ValueError(f"toolcall_start event points to {content.type} block at index {event.content_index}")
            snapshot_arguments = _serialized_arguments(content.arguments)
            caught_up = snapshot_arguments == _EMPTY_PARSED_TOOL_ARGUMENTS
            self._start_block(
                event.content_index,
                ("toolCall", caught_up, "", "" if caught_up else snapshot_arguments),
            )
            return ToolCallStartFrame(content_index=event.content_index, tool_call=_clone_tool_call(content))
        if isinstance(event, ToolCallDeltaEvent):
            state = self._block(event.content_index, "toolCall")
            if state[0] != "toolCall":
                raise ValueError("Unreachable tool-call encoder state")
            caught_up, catchup_json, snapshot_arguments = state[1], state[2], state[3]
            if caught_up:
                if len(event.delta) == 0:
                    return None
                return ToolCallDeltaFrame(content_index=event.content_index, delta=event.delta)
            catchup_json += event.delta
            arguments_value = parse_streaming_json(catchup_json)
            if _serialized_arguments(arguments_value) != snapshot_arguments:
                snapshot = parse_streaming_json(snapshot_arguments)
                if not _is_json_prefix(snapshot, arguments_value):
                    self._blocks[event.content_index] = ("toolCall", False, catchup_json, snapshot_arguments)
                    return None
            self._blocks[event.content_index] = ("toolCall", True, "", "")
            if len(catchup_json) == 0:
                return None
            return ToolCallCheckpointFrame(content_index=event.content_index, json=catchup_json)
        if isinstance(event, ToolCallEndEvent):
            content = _event_block(event)
            if content.type != "toolCall" or event.tool_call.type != "toolCall":
                raise ValueError(f"toolcall_end event has invalid tool call at index {event.content_index}")
            self._end_block(event.content_index, "toolCall")
            return ToolCallEndFrame(
                content_index=event.content_index,
                id=event.tool_call.id,
                name=event.tool_call.name,
                arguments=copy.deepcopy(event.tool_call.arguments),
                thought_signature=event.tool_call.thought_signature,
                namespace=event.tool_call.namespace,
            )
        raise ValueError(f"Unsupported assistant message event: {event.type}")

    def _start_block(self, content_index: int, state: EncoderBlockState) -> None:
        _assert_content_index(content_index)
        if content_index in self._blocks:
            raise ValueError(f"Assistant message block {content_index} starts more than once")
        self._blocks[content_index] = state

    def _block(self, content_index: int, kind: str) -> EncoderBlockState:
        _assert_content_index(content_index)
        state = self._blocks.get(content_index)
        if state is None:
            raise ValueError(f"Assistant message {kind} block {content_index} has not started")
        if state[0] != kind:
            raise ValueError(f"Assistant message block {content_index} is {state[0]}, not {kind}")
        return state

    def _end_block(self, content_index: int, kind: str) -> None:
        self._block(content_index, kind)
        del self._blocks[content_index]

    def _encode_text_delta(
        self,
        content_index: int,
        delta: str,
        kind: Literal["text", "thinking"],
    ) -> AssistantMessageFrame | None:
        state = self._block(content_index, kind)
        if state[0] == "toolCall":
            raise ValueError("Unreachable text encoder state")
        delta_start = state[2]
        delta_chars = delta_start + len(delta)
        self._blocks[content_index] = (kind, state[1], delta_chars)
        covered = max(0, state[1] - delta_start)
        if covered >= len(delta):
            return None
        uncovered = delta if covered == 0 else delta[covered:]
        if kind == "text":
            return TextDeltaFrame(content_index=content_index, delta=uncovered)
        return ThinkingDeltaFrame(content_index=content_index, delta=uncovered)


def _append_block(
    message: AssistantMessage,
    states: dict[int, ReducerBlockState],
    content_index: int,
    block: TextContent | ThinkingContent | ToolCall,
    state: ReducerBlockState,
) -> None:
    _assert_content_index(content_index)
    if content_index != len(message.content):
        reason = "already exists" if content_index < len(message.content) else "would leave a gap"
        raise ValueError(f"Cannot start assistant message block at index {content_index}: {reason}")
    message.content.append(copy.deepcopy(block))
    states[content_index] = state


def _active_block(
    message: AssistantMessage,
    states: dict[int, ReducerBlockState],
    content_index: int,
    expected_kind: str,
    frame_type: str,
) -> tuple[TextContent | ThinkingContent | ToolCall, ReducerBlockState]:
    _assert_content_index(content_index)
    state = states.get(content_index)
    block = message.content[content_index] if content_index < len(message.content) else None
    if state is None or block is None:
        raise ValueError(f"{frame_type} frame has no started block at index {content_index}")
    block_kind = "toolCall" if block.type == "toolCall" else block.type
    if state[0] != expected_kind or block_kind != expected_kind:
        raise ValueError(f"{frame_type} frame expected {expected_kind} block at index {content_index}, found {block.type}")
    if state[1]:
        raise ValueError(f"{frame_type} frame follows the end of block at index {content_index}")
    return block, state


def reduce_assistant_message_frames(frames: Iterable[AssistantMessageFrame]) -> AssistantMessage | None:
    message: AssistantMessage | None = None
    frame_before_start: str | None = None
    states: dict[int, ReducerBlockState] = {}

    for frame in frames:
        if isinstance(frame, StartFrame):
            if message is not None:
                raise ValueError("Assistant message frame sequence contains more than one start frame")
            if frame_before_start is not None:
                raise ValueError(f"{frame_before_start} frame appears before the start frame")
            message = copy.deepcopy(frame.partial)
            continue
        if message is None:
            if frame_before_start is None:
                frame_before_start = frame.type
            continue
        if isinstance(frame, TextStartFrame):
            if frame.content.type != "text":
                raise ValueError(f"text_start frame contains {frame.content.type} content")
            _append_block(message, states, frame.content_index, frame.content, ("text", False))
        elif isinstance(frame, TextDeltaFrame):
            block, _state = _active_block(message, states, frame.content_index, "text", frame.type)
            if block.type != "text":
                raise ValueError("Unreachable text frame state")
            block.text += frame.delta
        elif isinstance(frame, TextEndFrame):
            block, state = _active_block(message, states, frame.content_index, "text", frame.type)
            if block.type != "text":
                raise ValueError("Unreachable text frame state")
            block.text = frame.content
            block.text_signature = frame.text_signature
            states[frame.content_index] = ("text", True)
        elif isinstance(frame, ThinkingStartFrame):
            if frame.content.type != "thinking":
                raise ValueError(f"thinking_start frame contains {frame.content.type} content")
            _append_block(message, states, frame.content_index, frame.content, ("thinking", False))
        elif isinstance(frame, ThinkingDeltaFrame):
            block, _state = _active_block(message, states, frame.content_index, "thinking", frame.type)
            if block.type != "thinking":
                raise ValueError("Unreachable thinking frame state")
            block.thinking += frame.delta
        elif isinstance(frame, ThinkingEndFrame):
            block, state = _active_block(message, states, frame.content_index, "thinking", frame.type)
            if block.type != "thinking":
                raise ValueError("Unreachable thinking frame state")
            block.thinking = frame.content
            block.thinking_signature = frame.thinking_signature
            block.redacted = frame.redacted
            states[frame.content_index] = ("thinking", True)
        elif isinstance(frame, ToolCallStartFrame):
            if frame.tool_call.type != "toolCall":
                raise ValueError(f"toolcall_start frame contains {frame.tool_call.type} content")
            _append_block(message, states, frame.content_index, frame.tool_call, ("toolCall", False, ""))
        elif isinstance(frame, ToolCallCheckpointFrame):
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame.type)
            if block.type != "toolCall" or state[0] != "toolCall":
                raise ValueError("Unreachable tool-call checkpoint state")
            states[frame.content_index] = ("toolCall", False, frame.json)
            block.arguments = parse_streaming_json(frame.json)
        elif isinstance(frame, ToolCallDeltaFrame):
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame.type)
            if block.type != "toolCall" or state[0] != "toolCall":
                raise ValueError("Unreachable tool-call frame state")
            states[frame.content_index] = ("toolCall", False, state[2] + frame.delta)
        elif isinstance(frame, ToolCallEndFrame):
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame.type)
            if block.type != "toolCall":
                raise ValueError("Unreachable tool-call frame state")
            block.id = frame.id
            block.name = frame.name
            block.arguments = copy.deepcopy(frame.arguments)
            block.thought_signature = frame.thought_signature
            block.namespace = frame.namespace
            json_so_far = state[2] if state[0] == "toolCall" else ""
            states[frame.content_index] = ("toolCall", True, json_so_far)

    if message is None:
        return None
    for content_index, state in states.items():
        if state[0] != "toolCall" or state[1] or len(state[2]) == 0:
            continue
        block = message.content[content_index]
        if block.type != "toolCall":
            raise ValueError("Unreachable tool-call frame state")
        block.arguments = parse_streaming_json(state[2])
    return message
