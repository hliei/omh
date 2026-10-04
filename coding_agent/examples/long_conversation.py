"""Offline installed-package acceptance: run this file from any working directory."""

import asyncio
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from omh.agent import (
    AgentEvent,
    AgentOptions,
    CompactionEndEvent,
    CompactionHistoryEntry,
    CompactionSettings,
    ContextEditHistoryEntry,
    HistoryCommitEvent,
    RetryPolicy,
    RetryStartEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantContent,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    TranscriptContext,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

from coding_agent import AgentSessionRuntime, CodingAgentOptions, decode_history


class OfflineCodingStream:
    """A finite dialogue script with one transient failure in each retry scope."""

    def __init__(self) -> None:
        self.dialogue: deque[list[AssistantContent] | str] = deque([
            "503 temporarily unavailable",
            [ToolCall(id="write", name="write", arguments={"path": "demo.py", "content": "answer = 41\n"})],
            [ToolCall(id="edit", name="edit", arguments={"path": "demo.py", "edits": [{"oldText": "41", "newText": "42"}]})],
            [ToolCall(id="read", name="read", arguments={"path": "reference.txt"})],
            [TextContent(text="Corrected the answer to 42.")],
            "prompt is too long",
            [TextContent(text="Recovered the next coding turn.")],
            "prompt is too long",
            "prompt is too long",
            [TextContent(text="A new question starts a new recovery allowance.")],
            [TextContent(text="Reopened and ready.")],
        ])
        self.dialogue_requests: list[TranscriptContext] = []
        self.request_ids: list[str | None] = []
        self.summary_attempts = 0

    def __call__(
        self, model: Model, context: TranscriptContext, options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        first = context.messages[0] if context.messages else None
        summary = first is not None and first.role == "system" and "summarization assistant" in first.content
        self.request_ids.append(options.session_id if options else None)
        if summary:
            self.summary_attempts += 1
            response: list[AssistantContent] | str = (
                "503 temporarily unavailable" if self.summary_attempts == 1
                else [TextContent(text="Goal: correct demo.py. Progress: answer changed to 42. Read reference.txt.")]
            )
        else:
            self.dialogue_requests.append(context)
            response = self.dialogue.popleft()  # An unexpected request fails the example.
        reason: Literal["error", "toolUse", "stop"]
        if isinstance(response, str):
            content: list[AssistantContent] = []
            error_message, reason = response, "error"
        else:
            content = response
            error_message = None
            reason = "toolUse" if any(isinstance(block, ToolCall) for block in content) else "stop"
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            content=content, error_message=error_message, stop_reason=reason,
            usage=empty_usage(), timestamp=len(self.request_ids),
        )
        stream = create_assistant_message_event_stream()
        if reason == "error":
            stream.push(ErrorEvent(reason="error", error=message))
        else:
            stream.push(DoneEvent(reason=reason, message=message))
        return stream


async def main() -> None:
    model = Model(
        id="offline", name="Offline coding", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",), context_window=4096, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    stream = OfflineCodingStream()
    with TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        (cwd / "reference.txt").write_text("reference " * 3200)
        options = CodingAgentOptions(
            cwd=cwd, model=model, stream_fn=stream, session_file="conversation.jsonl",
            agent_options=AgentOptions(
                compaction=CompactionSettings(reserve_tokens=512, keep_recent_tokens=0),
                retry=RetryPolicy(max_retries=1, base_delay_ms=0),
            ),
        )
        runtime = AgentSessionRuntime(options)
        session = await runtime.new_session(display_name="Long coding conversation")
        events: list[AgentEvent] = []

        def observe(event: AgentEvent, signal: AbortSignal) -> None:
            events.append(event)
            if isinstance(event, HistoryCommitEvent):
                current = runtime.current_session
                assert current is not None
                if current.save_state == "saved":
                    assert current.path is not None
                    assert decode_history(current.path.read_bytes()).history == current.agent.history

        runtime.subscribe(observe)
        await runtime.prompt("Correct demo.py and inspect reference.txt.")
        assert (cwd / "demo.py").read_text() == "answer = 42\n"
        retries = [event.scope for event in events if isinstance(event, RetryStartEvent)]
        assert retries == ["dialogue", "summary"]
        ends = [event for event in events if isinstance(event, CompactionEndEvent)]
        assert any(event.reason == "threshold" and event.result is not None for event in ends)
        assert len(stream.dialogue_requests) == 5
        await runtime.prompt("Continue the coding conversation after the checkpoint.")
        ends = [event for event in events if isinstance(event, CompactionEndEvent)]
        assert any(event.reason == "overflow" and event.will_retry for event in ends)
        assert len(stream.dialogue_requests) == 7
        assert session.agent.state.error_message is None
        records = session.agent.history.entries
        assert any(isinstance(entry, ContextEditHistoryEntry) for entry in records)
        assert any(isinstance(entry, CompactionHistoryEntry) for entry in records)
        assert "503 temporarily unavailable" in await runtime.export_session()
        assert "Corrected the answer to 42." in await runtime.export_session()

        # A second overflow on the same unresolved chain consumes no further recovery.
        before = len([event for event in events if isinstance(event, CompactionEndEvent) and event.reason == "overflow" and event.result is not None])
        await runtime.prompt("Demonstrate bounded overflow recovery.")
        after = len([event for event in events if isinstance(event, CompactionEndEvent) and event.reason == "overflow" and event.result is not None])
        assert after - before == 1
        assert len(stream.dialogue_requests) == 9
        assert "after one compact-and-retry attempt" in (session.agent.state.error_message or "")
        await runtime.prompt("Start a fresh question.")
        assert session.agent.state.error_message is None
        await runtime.save_session()
        history, context = session.agent.history, session.agent.state.messages
        restored = await runtime.open_session("conversation.jsonl")
        assert session.agent.state.is_closed
        assert runtime.retained_sessions == (session,)
        assert restored.agent.history == history
        assert restored.agent.state.messages == context
        assert not restored.agent.state.is_busy
        assert len(stream.dialogue_requests) == 10  # Opening itself made no request.
        assert all(identity == history.conversation_id for identity in stream.request_ids)
        await runtime.prompt("Continue after reopening.")
        assert not stream.dialogue
        assert restored.save_state == "saved"
        assert decode_history(await runtime.export_session()).history == restored.agent.history
        await restored.agent.close()
        print(f"Long conversation verified: {len(history.entries)} records, {stream.summary_attempts} summary attempts, identity preserved.")


if __name__ == "__main__":
    asyncio.run(main())
