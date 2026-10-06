

async def test_assistant_stream_preserves_partial_at_emission() -> None:
    from omh.llm.types import (
        AssistantMessage,
        DoneEvent,
        StartEvent,
        TextContent,
        TextDeltaEvent,
        TextStartEvent,
        empty_usage,
    )
    from omh.llm.utils.event_stream import create_assistant_message_event_stream

    message = AssistantMessage(api="openai-completions", provider="offline", model="offline",
                               usage=empty_usage(), stop_reason="pending", timestamp=1000)
    stream = create_assistant_message_event_stream()
    stream.push(StartEvent(partial=message))
    message.content.append(TextContent(text=""))
    stream.push(TextStartEvent(content_index=0, partial=message))
    message.content[0].text = "hello"
    stream.push(TextDeltaEvent(content_index=0, delta="hello", partial=message))
    message.stop_reason = "stop"
    stream.push(DoneEvent(reason="stop", message=message))
    events = []
    async for event in stream:
        events.append(event)
        if event.type == "done":
            break
    assert events[0].partial.content == []
    assert events[0].partial.stop_reason == "pending"
    assert events[1].partial.content[0].text == ""
    assert events[2].partial.content[0].text == "hello"
    assert (await stream.result()).content[0].text == "hello"
