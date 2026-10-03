"""Offline bounded recovery and history restore: python examples/overflow_recovery.py."""

import asyncio
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from compaction import make_model

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CompactionEndEvent,
    CompactionSettings,
    ContextEditHistoryEntry,
)
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    SystemMessage,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


async def main() -> None:
    model = make_model()
    dialogue_calls = 0

    def reply(text: str) -> AssistantMessage:
        return AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, timestamp=0,
            usage=empty_usage(), content=[TextContent(text=text)], stop_reason="stop",
        )

    def stream_fn(model, context, options):
        nonlocal dialogue_calls
        first = context.messages[0]
        if isinstance(first, SystemMessage) and "summarization assistant" in first.content:
            message = reply("Goal: write once. Progress: the file was written; finish without repeating it.")
        else:
            dialogue_calls += 1
            message = reply("Completed without repeating the earlier write.")
            if dialogue_calls == 1:
                message = replace(message, stop_reason="toolUse", content=[
                    ToolCall(id="completed-write", name="write", arguments={}),
                ])
            elif dialogue_calls == 2:
                # The original model limit is much greater than these 12 tokens.
                # This truncated call is never executed; its result is omitted
                # together with the failed assistant before recovery compaction.
                message = replace(message, stop_reason="length", content=[
                    ToolCall(id="truncated-write", name="write", arguments={}),
                ], usage=Usage(input=50, output=12, cache_read=0, cache_write=0,
                               total_tokens=62, cost=UsageCost()))
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
        return stream

    with TemporaryDirectory() as directory:
        path = Path(directory) / "written.txt"

        async def write(call_id, arguments, signal, on_update):
            with path.open("a") as file:
                file.write("written once\n")
            return AgentToolResult(content=[TextContent(text="The file was written.")])

        agent = Agent(AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(model=model, messages=[
                UserMessage(content="Earlier question.", timestamp=0),
                reply("Earlier answer."),
                UserMessage(content="Recent context.", timestamp=0),
            ], tools=[AgentTool(name="write", label="Write", description="Write the fixture",
                               parameters={"type": "object"}, execute=write)]),
            compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
        ))

        def listener(event, signal):
            if isinstance(event, CompactionEndEvent):
                assert event.reason == "overflow" and event.will_retry
                assert event.result is not None
                print(f"{event.reason}: compacted; will_retry={event.will_retry}")

        agent.subscribe(listener)
        await agent.prompt("Write once, then finish.")
        assert dialogue_calls == 3
        assert path.read_text() == "written once\n"
        assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 2
        restored = Agent.from_history(agent.history, AgentOptions(
            stream_fn=stream_fn, initial_state=AgentInitialState(model=model),
        ))
        assert restored.history == agent.history
        assert restored.state.messages == agent.state.messages
        assert dialogue_calls == 3  # Restoration starts no request or tool work.
        print(f"Preserved {len(agent.history.entries)} records; restored context; one file write.")


if __name__ == "__main__":
    asyncio.run(main())
