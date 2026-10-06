"""Run offline: python coding_agent/examples/history_roundtrip.py."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream

from coding_agent import AgentSessionRuntime, CodingAgentOptions


async def main() -> None:
    model = Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",), context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    responses = [
        [ToolCall(id="write", name="write", arguments={"path": "demo.py", "content": "answer = 41\n"})],
        [ToolCall(id="edit", name="edit", arguments={"path": "demo.py", "edits": [{"oldText": "41", "newText": "42"}]})],
        [TextContent(text="Saved the corrected answer.")],
    ]
    requests = 0

    def stream_fn(model, context, options):
        nonlocal requests
        requests += 1
        content = responses.pop(0) if responses else [TextContent(text="The answer is 42.")]
        reason = "toolUse" if content[0].type == "toolCall" else "stop"
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, content=content,
            stop_reason=reason, usage=empty_usage(), timestamp=requests,
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason=reason, message=message))
        return stream

    with TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        runtime = AgentSessionRuntime(CodingAgentOptions(
            cwd=cwd, model=model, stream_fn=stream_fn, session_file="conversation.jsonl",
        ))
        session = await runtime.new_session(display_name="Offline coding")
        assert session.save_state == "pending"
        await runtime.prompt("Correct the answer in demo.py.")
        assert (cwd / "demo.py").read_text() == "answer = 42\n"
        original = session.agent.history
        await runtime.export_session("copy.jsonl")
        await session.close()
        restored = await runtime.open_session("conversation.jsonl")
        assert restored.agent.history == original
        assert requests == 3
        await restored.prompt("Explain the answer.")
        print(f"Reopened {len(original.entries)} records with identity {original.conversation_id}")
        print(f"File: {(cwd / 'demo.py').read_text().strip()}; requests: {requests}; save state: {restored.save_state}")
        await restored.close()


asyncio.run(main())
