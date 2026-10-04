"""Offline prepared switching, cancelled waiting and retained unsaved history."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.agent import AgentSettledEvent, AgentStartEvent
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream

from coding_agent import AgentSessionRuntime, CodingAgentOptions, decode_history


def offline_stream(model, context, options):
    message = AssistantMessage(
        api=model.api, provider=model.provider, model=model.id, timestamp=1000,
        content=[TextContent(text="done")], usage=empty_usage(), stop_reason="stop",
    )
    stream = create_assistant_message_event_stream()
    stream.push(DoneEvent(reason="stop", message=message))
    return stream


async def main():
    model = Model(
        id="offline", name="offline", api="openai-completions", provider="test",
        base_url="", reasoning=False, input=("text",), context_window=100000,
        max_tokens=1024, cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    with TemporaryDirectory() as directory:
        cwd = Path(directory)
        runtime = AgentSessionRuntime(CodingAgentOptions(
            cwd=cwd, model=model, stream_fn=offline_stream, tools=(),
        ))
        starts = []
        runtime.subscribe(lambda event, signal: starts.append(event) if isinstance(event, AgentStartEvent) else None)
        old = await runtime.new_session(display_name="old conversation")
        settled, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def hold_terminal(event, signal):
            if isinstance(event, AgentSettledEvent):
                settled.set()
                signal.add_callback(closing.set)
                await release.wait()

        old.agent.subscribe(hold_terminal)
        prompt = asyncio.create_task(old.prompt("complete this turn"))
        await settled.wait()
        old.follow_up("retained queued input")
        waiter = asyncio.create_task(runtime.new_session(display_name="new conversation"))
        await closing.wait()
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            print("Stopped waiting; the runtime still owns the handoff.")
        release.set()
        await prompt
        new = await runtime.wait_for_switch()
        assert runtime.current_session is new and old.agent.state.is_closed
        assert old.agent.history.conversation_id != new.agent.history.conversation_id
        assert old.queued_messages.follow_up[0].content[0].text == "retained queued input"
        assert not new.agent.has_queued_messages()
        await runtime.prompt("work in the new conversation")
        assert len(starts) == 2  # The runtime subscription followed the replacement.
        await old.save("old.jsonl")
        restored = await runtime.switch_session("old.jsonl")
        assert restored.agent.history == old.agent.history
        assert not restored.agent.has_queued_messages()

        # A failed exclusive save leaves committed memory available after switching.
        occupied = cwd / "occupied.jsonl"
        occupied.write_text("existing file")
        runtime.options.session_file = occupied
        unsaved = await runtime.new_session()
        unsaved.follow_up("also retained")
        try:
            await unsaved.prompt("unsaved input")
        except FileExistsError:
            assert unsaved.save_state == "unsaved"
        runtime.options.session_file = None
        await runtime.new_session()
        assert unsaved in runtime.retained_sessions
        exported = await unsaved.export("backup.jsonl")
        assert decode_history(exported).history == unsaved.agent.history
        assert unsaved.save_state == "unsaved"
        await unsaved.save("repaired.jsonl")
        assert unsaved.save_state == "saved"
        print("Switched, rebound observers, and exported/repaired retained history.")


if __name__ == "__main__":
    asyncio.run(main())
