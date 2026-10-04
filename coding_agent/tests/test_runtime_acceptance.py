"""Cross-capability acceptance through application sessions and real file I/O."""

import asyncio
from dataclasses import replace

import pytest
from omh.agent import (
    AgentInitialState,
    AgentOptions,
    CompactionEndEvent,
    CompactionHistoryEntry,
    CompactionSettings,
    RetryPolicy,
)
from omh.llm.types import AssistantMessage, TextContent, UserMessage, empty_usage
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions, decode_history


@pytest.mark.parametrize("automatic", [False, True])
async def test_committed_summary_save_failure_preserves_result_and_retired_history(tmp_path, automatic):
    path = tmp_path / "history.jsonl"
    stream = OfflineStream()
    options = CodingAgentOptions(
        cwd=tmp_path, model=replace(model(), context_window=2000), stream_fn=stream,
        tools=(), session_file=path,
        agent_options=AgentOptions(
            initial_state=AgentInitialState(messages=[
                UserMessage(content="old question " * 700, timestamp=1000),
                AssistantMessage(
                    api="openai-completions", provider="test", model="offline",
                    content=[TextContent(text="old answer")], timestamp=1500,
                    stop_reason="stop", usage=replace(empty_usage(), input=2100, total_tokens=2100),
                ),
                UserMessage(content="recent question", timestamp=2000),
            ]),
            compaction=CompactionSettings(enabled=automatic, reserve_tokens=100, keep_recent_tokens=0),
            retry=RetryPolicy(base_delay_ms=0),
        ),
    )
    runtime = AgentSessionRuntime(options)
    old = await runtime.new_session()
    await old.save()
    old.follow_up("keep unconsumed input")
    events = []

    def break_destination(event, signal):
        events.append(event)
        if event.type == "compaction_start":
            # Fail only the real summary's subsequent append, after setup saving.
            path.rename(tmp_path / "before-summary.jsonl")
            path.mkdir()

    old.agent.subscribe(break_destination)
    with pytest.raises(IsADirectoryError) as failed:
        if automatic:
            await runtime.prompt("must not reach dialogue after save failure")
        else:
            await runtime.compact()
    assert old.save_state == "unsaved"
    assert old.save_error is failed.value
    assert not old.agent.state.is_busy
    assert all("summarization assistant" in request[1].messages[0].content for request in stream.requests)
    assert not any(event.type == "retry_start" for event in events)
    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    records = [entry for entry in old.agent.history.entries if isinstance(entry, CompactionHistoryEntry)]
    assert len(records) == 1
    assert records[0].summary == "done"
    if automatic:
        # Saving is an ordinary notification failure: stop rather than report
        # a recoverable summary-model failure and continue the original prompt.
        assert not ends
        assert events[-1].type == "agent_settled"
        assert events[-1].error_message == str(failed.value)
    else:
        assert len(ends) == 1
        end = ends[0]
        assert end.reason == "manual"
        assert end.result is not None and end.result.summary == "done"
        assert not end.aborted and end.error_message == str(failed.value)
        assert records[0].first_kept_entry_id == end.result.first_kept_entry_id
        assert records[0].usage == end.result.usage
    assert old.agent.state.messages[0].role == "compactionSummary"
    assert decode_history(await old.export()).history == old.agent.history
    with pytest.raises(RuntimeError, match="unsaved"):
        await runtime.prompt("paused")

    # Switch the unsaved, genuinely compacted session; retain and repair it.
    history, messages = old.agent.history, old.agent.state.messages
    options.session_file = "new.jsonl"
    options.agent_options = AgentOptions(compaction=CompactionSettings(enabled=False))
    new = await runtime.new_session()
    assert new.agent.history.conversation_id != history.conversation_id
    assert old.agent.state.is_closed
    assert runtime.retained_sessions == (old,)
    assert old.save_error is failed.value
    assert old.queued_messages.follow_up[0].content[0].text == "keep unconsumed input"
    assert not new.queued_messages.follow_up
    assert old.agent.history == history and old.agent.state.messages == messages
    await old.export("unsaved-copy.jsonl")
    assert old.save_state == "unsaved"
    await old.save("repaired.jsonl")
    restored = await AgentSessionRuntime(replace(options, agent_options=AgentOptions())).open_session("repaired.jsonl")
    assert restored.agent.history == history
    assert restored.agent.state.messages == messages
    assert old.save_state == "saved" and old.save_error is None
    await runtime.prompt("new session still works")
    await new.agent.close()
    await restored.agent.close()


@pytest.mark.parametrize("terminal", ["agent_settled", "compaction_end"])
async def test_terminal_application_prompt_has_fresh_signal_and_is_saved(tmp_path, terminal):
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file="history.jsonl",
        agent_options=AgentOptions(
            initial_state=AgentInitialState(messages=[
                UserMessage(content="older question", timestamp=1000),
                UserMessage(content="recent question", timestamp=2000),
            ]),
            compaction=CompactionSettings(enabled=False, keep_recent_tokens=0),
        ),
    ))
    session = await runtime.new_session()
    signals = []
    accepted = False

    async def follow_on(event, signal):
        nonlocal accepted
        if event.type == terminal and not accepted:
            accepted = True
            await runtime.prompt("new activity from terminal callback")
            session.agent.abort()  # Still addresses the old activity during notification.
        if event.type == "agent_start":
            signals.append(signal)
            assert not signal.aborted

    runtime.subscribe(follow_on)
    if terminal == "compaction_end":
        result = await runtime.compact()
        assert result.summary == "done"
    else:
        await runtime.prompt("first activity")
    await asyncio.wait_for(session.agent.wait_for_idle(), 2)
    assert accepted
    assert len(signals) == (2 if terminal == "agent_settled" else 1)
    assert not signals[-1].aborted
    assert session.save_state == "saved"
    assert session.agent.state.messages[-1].content[0].text == "done"
    history, messages = session.agent.history, session.agent.state.messages
    runtime.options.agent_options = replace(runtime.options.agent_options, initial_state=None)
    restored = await runtime.open_session("history.jsonl")
    assert restored.agent.history == history
    assert restored.agent.state.messages == messages
    assert session.agent.state.is_closed
    await restored.agent.close()


def test_long_coding_example_runs_outside_repository(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "long_conversation.py"
    subprocess.run([sys.executable, str(example)], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=30)


async def test_custom_tool_boundary_and_explicit_end_survive_application_roundtrip(tmp_path):
    from omh.agent import CustomAgentMessage, MessageHistoryEntry
    from omh.llm.types import ToolCall

    (tmp_path / "large.txt").write_text("reference " * 3200)
    stream = OfflineStream([[
        ToolCall(id="write", name="write", arguments={"path": "effect.txt", "content": "completed"}),
        ToolCall(id="read", name="read", arguments={"path": "large.txt"}),
    ]])
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=replace(model(), context_window=4096), stream_fn=stream,
        session_file="history.jsonl", tools=("write", "read"),
        agent_options=AgentOptions(
            compaction=CompactionSettings(reserve_tokens=512, keep_recent_tokens=0),
            finish_turn=lambda context, signal: "end",
        ),
    ))
    session = await runtime.new_session()
    submitted = False
    events = []

    async def during_tools(event, signal):
        nonlocal submitted
        events.append(event.type)
        if event.type == "tool_execution_start" and not submitted:
            submitted = True
            await session.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="after all results"))
            runtime.steer("preserved steering")
            runtime.follow_up("preserved follow-up")

    runtime.subscribe(during_tools)
    await runtime.prompt("Write and inspect, then stop the activity.")
    assert submitted
    assert (tmp_path / "effect.txt").read_text() == "completed"
    assert len(stream.requests) == 1
    assert "compaction_start" not in events and "retry_start" not in events
    assert events.count("agent_settled") == 1
    history, context = session.agent.history, session.agent.state.messages
    assert [message.role for message in context if message.role != "system"] == [
        "user", "assistant", "toolResult", "toolResult", "custom",
    ]
    results = [entry.message for entry in history.entries
               if isinstance(entry, MessageHistoryEntry) and entry.message.role == "toolResult"]
    assert [result.tool_call_id for result in results] == ["write", "read"]
    assert all(not result.is_error for result in results)
    assert session.save_state == "saved"
    restored = await runtime.open_session("history.jsonl")
    assert restored.agent.history == history
    assert restored.agent.state.messages == context
    assert len(stream.requests) == 1
    assert len(session.queued_messages.steering) == len(session.queued_messages.follow_up) == 1
    assert not restored.queued_messages.steering and not restored.queued_messages.follow_up
    await restored.agent.close()
