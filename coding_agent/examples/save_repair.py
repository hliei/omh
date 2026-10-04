"""Run offline: python coding_agent/examples/save_repair.py."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

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


async def main() -> None:
    model = Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",), context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    requests = 0

    def stream_fn(model, context, options):
        nonlocal requests
        requests += 1
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            content=[TextContent(text="Saved work can continue.")],
            stop_reason="stop", usage=empty_usage(), timestamp=requests,
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason="stop", message=message))
        return stream

    with TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        path = cwd / "conversation.jsonl"
        path.write_text("Existing file protected from automatic creation.\n")
        options = CodingAgentOptions(
            cwd=cwd, model=model, stream_fn=stream_fn, tools=(), session_file=path,
        )
        runtime = AgentSessionRuntime(options)
        session = await runtime.new_session()
        try:
            await runtime.prompt("Keep this input even if saving fails.")
        except FileExistsError:
            print(f"Save failed: {session.save_error}")
        assert session.save_state == "unsaved"
        retained = session.agent.history
        backup = await runtime.export_session("backup.jsonl")
        assert decode_history(backup).history == retained
        assert session.save_state == "unsaved"
        try:
            await runtime.prompt("This must wait for repair.")
        except RuntimeError as error:
            print(f"Paused: {error}")
        assert session.agent.history == retained
        assert requests == 0
        # Explicit save authorizes overwriting this chosen destination in full.
        await runtime.save_session()
        assert session.save_state == "saved"
        reopened = await AgentSessionRuntime(options).open_session(path)
        assert reopened.agent.history == retained
        await runtime.continue_()
        assert requests == 1
        assert decode_history(path.read_bytes()).history == session.agent.history
        await reopened.agent.close()
        await session.agent.close()
        print(f"Repaired {len(retained.entries)} records; resumed with {requests} request.")


asyncio.run(main())
