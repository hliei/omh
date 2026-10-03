"""Offline threshold compaction: python examples/auto_compaction.py."""

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
    CompactionHistoryEntry,
    CompactionSettings,
    CompactionStartEvent,
    CustomAgentMessage,
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
    model = replace(make_model(), context_window=1000)
    dialogue_calls = 0

    def reply(text: str, tokens: int = 0) -> AssistantMessage:
        return AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, timestamp=0,
            usage=Usage(input=tokens, output=0, cache_read=0, cache_write=0,
                        total_tokens=tokens, cost=UsageCost()),
            content=[TextContent(text=text)], stop_reason="stop",
        )

    def stream_fn(model, context, options):
        nonlocal dialogue_calls
        first = context.messages[0]
        if isinstance(first, SystemMessage) and "summarization assistant" in first.content:
            message = reply("Goal: inspect the local fixture. Progress: previous context summarized.")
        else:
            dialogue_calls += 1
            message = reply("Fixture inspected.", 20)
            if dialogue_calls == 1:
                message.content = [ToolCall(id="read-1", name="read", arguments={})]
                message.stop_reason = "toolUse"
                message.usage = empty_usage()
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
        return stream

    with TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.txt"
        path.write_text("x" * 4000)

        async def read(call_id, arguments, signal, on_update):
            return AgentToolResult(content=[TextContent(text=path.read_text())])

        agent = Agent(AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(model=model, messages=[
                UserMessage(content="Earlier question.", timestamp=0),
                reply("Earlier answer.", 900),
                UserMessage(content="tail", timestamp=0),
            ], tools=[AgentTool(name="read", label="Read", description="Read fixture",
                               parameters={"type": "object"}, execute=read)]),
            compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
        ))
        queued_custom = False

        async def listener(event, signal):
            nonlocal queued_custom
            if isinstance(event, CompactionStartEvent):
                assert agent.state.is_busy
                if not queued_custom:
                    queued_custom = True
                    await agent.submit_custom_message(CustomAgentMessage(
                        custom_type="note", content="Keep the fixture path in mind.",
                    ))
            elif isinstance(event, CompactionEndEvent):
                assert event.result is not None
                print(f"{event.reason}: {event.result.tokens_before} -> {event.result.estimated_tokens_after}")

        agent.subscribe(listener)
        await agent.prompt("Read the fixture.")
        compactions = [entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]
        assert len(compactions) == 2 and dialogue_calls == 2
        assert not agent.state.is_busy
        restored = Agent.from_history(agent.history, AgentOptions(
            stream_fn=stream_fn, initial_state=AgentInitialState(model=model),
        ))
        assert restored.state.messages == agent.state.messages
        print(f"Preserved {len(agent.history.entries)} history records; restore made no request.")


if __name__ == "__main__":
    asyncio.run(main())
