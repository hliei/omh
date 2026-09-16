from __future__ import annotations

from omh.llm.types import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    empty_usage,
)
from omh.llm.types import (
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from omh.llm.utils.assistant_message_frame import (
    AssistantMessageFrameEncoder,
    reduce_assistant_message_frames,
)


def _seed() -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="deepseek",
        model="deepseek-flash",
        usage=empty_usage(),
        stop_reason="pending",
        timestamp=1,
    )


def test_text_frames_reduce_to_authoritative_end_content() -> None:
    partial = _seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [encoder.encode(StartEvent(partial=partial))]
    partial.content.append(TextContent(text="Hello "))
    frames.append(encoder.encode(TextStartEvent(content_index=0, partial=partial)))
    partial.content[0] = TextContent(text="Hello world", text_signature="sig-text")
    frames.append(
        encoder.encode(TextDeltaEvent(content_index=0, delta="incorrect", partial=partial))
    )
    frames.append(
        encoder.encode(TextEndEvent(content_index=0, content="Hello world", partial=partial))
    )
    recorded = [frame for frame in frames if frame is not None]
    reduced = reduce_assistant_message_frames(recorded)
    assert reduced is not None
    assert reduced.content == [TextContent(text="Hello world", text_signature="sig-text")]


def test_thinking_and_tool_call_frames_round_trip() -> None:
    partial = _seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [encoder.encode(StartEvent(partial=partial))]
    partial.content.append(ThinkingContent(thinking=""))
    frames.append(encoder.encode(ThinkingStartEvent(content_index=0, partial=partial)))
    frames.append(encoder.encode(ThinkingDeltaEvent(content_index=0, delta="plan", partial=partial)))
    partial.content[0] = ThinkingContent(thinking="plan")
    frames.append(encoder.encode(ThinkingEndEvent(content_index=0, content="plan", partial=partial)))
    tool = ToolCall(id="call_1", name="math_operation", arguments={})
    partial.content.append(tool)
    frames.append(encoder.encode(ToolCallStartEvent(content_index=1, partial=partial)))
    frames.append(
        encoder.encode(ToolCallDeltaEvent(content_index=1, delta='{"a":1}', partial=partial))
    )
    tool.arguments = {"a": 1}
    frames.append(encoder.encode(ToolCallEndEvent(content_index=1, tool_call=tool, partial=partial)))
    recorded = [frame for frame in frames if frame is not None]
    reduced = reduce_assistant_message_frames(recorded)
    assert reduced is not None
    assert reduced.content[0] == ThinkingContent(thinking="plan")
    assert reduced.content[1].type == "toolCall"
    assert reduced.content[1].arguments == {"a": 1}
