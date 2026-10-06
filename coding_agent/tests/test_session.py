from omh.llm.types import ToolCall, UserMessage
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions


async def test_default_tools_prompt_save_reopen_and_continue(tmp_path):
    stream = OfflineStream([
        [ToolCall(id="write", name="write", arguments={"path": "demo.txt", "content": "before\n"})],
        [ToolCall(id="edit", name="edit", arguments={"path": "demo.txt", "oldText": "before", "newText": "after"})],
        [ToolCall(id="read", name="read", arguments={"path": "demo.txt"}),
         ToolCall(id="bash", name="bash", arguments={"command": "printf shell"})],
    ])
    path = tmp_path / "history.jsonl"
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, session_file=path)
    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session(display_name="Demo")
    assert [tool.name for tool in session.agent.state.tools] == ["read", "bash", "edit", "write"]
    assert session.save_state == "pending"
    assert session.save_error is None
    assert not path.exists()
    await runtime.prompt("Change the file and inspect it")
    assert (tmp_path / "demo.txt").read_text() == "after\n"
    assert session.save_state == "saved"
    assert len(stream.requests) == 4
    saved = session.agent.history
    await session.close()
    reopened = await AgentSessionRuntime(options).open_session(path)
    assert reopened.agent.history == saved
    assert reopened.agent.state.messages == session.agent.state.messages
    assert reopened.display_name == "Demo"
    assert not reopened.agent.state.is_busy
    assert len(stream.requests) == 4
    reopened.agent.steer(UserMessage(content="continue", timestamp=2000))
    await reopened.continue_()
    await reopened.close()
    again = await AgentSessionRuntime(options).open_session(path)
    assert again.agent.history == reopened.agent.history
    assert len(stream.requests) == 5
    assert stream.requests[-1][2].session_id == saved.conversation_id


async def test_history_commit_observers_see_saved_records_in_order(tmp_path):
    from omh.agent import CustomAgentMessage, HistoryCommitEvent

    from coding_agent import decode_history

    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, session_file="ordered.jsonl", tools=()))
    session = await runtime.new_session()
    observed = []

    async def observe(event, signal):
        if isinstance(event, HistoryCommitEvent):
            if session.save_state == "pending":
                # Resource section synchronization precedes the first user;
                # automatic saving still waits for a user/assistant commit.
                assert not session.path.exists()
                assert all(entry.type != "message" or entry.message.role not in ("user", "assistant")
                           for entry in session.agent.history.entries)
            else:
                on_disk = decode_history(session.path.read_bytes()).history
                assert on_disk == session.agent.history
                assert on_disk.leaf_id == event.leaf_id
            observed.extend(entry.id for entry in event.entries)

    session.agent.subscribe(observe)
    initial = session.agent.history
    await runtime.prompt("first")
    await session.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="note"))
    await runtime.prompt("second")
    assert observed == [entry.id for entry in session.agent.history.entries[len(initial.entries):]]


async def test_tools_subset_and_current_host_credentials_are_used(tmp_path):
    from omh.agent import AgentOptions

    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("read", "write"),
        agent_options=AgentOptions(api_key="host-secret", session_id="host-request-id"),
    ))
    session = await runtime.new_session()
    await runtime.prompt("test")
    assert [tool.name for tool in session.agent.state.tools] == ["read", "write"]
    assert stream.requests[0][2].api_key == "host-secret"
    assert stream.requests[0][2].session_id == "host-request-id"
    text = await session.export()
    assert "host-secret" not in text
    assert "host-request-id" not in text


async def test_save_failure_retains_observable_state_and_memory(tmp_path):
    import pytest

    stream = OfflineStream()
    path = tmp_path / "occupied.jsonl"
    path.write_text("original file")
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file=path,
    )).new_session()
    with pytest.raises(FileExistsError):
        await session.prompt("retained input")
    assert session.save_state == "unsaved"
    assert isinstance(session.save_error, FileExistsError)
    assert "retained input" in await session.export()
    assert path.read_text() == "original file"
    assert not stream.requests


async def test_failed_save_is_not_reported_saved_by_a_later_append(tmp_path):
    import pytest
    from omh.agent import CustomAgentMessage

    path = tmp_path / "occupied.jsonl"
    path.write_text("original file")
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file=path,
    )).new_session()
    with pytest.raises(FileExistsError):
        await session.prompt("retained input")
    try:
        await session.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="also retained"))
    except FileExistsError:
        pass
    assert session.save_state == "unsaved"
    assert path.read_text() == "original file"


async def test_cancelled_waiter_does_not_drop_tool_settlement_or_saved_history(tmp_path):
    import asyncio

    import pytest
    from omh.agent import ToolExecutionUpdateEvent

    stream = OfflineStream([[ToolCall(id="bash", name="bash", arguments={"command": "printf ready; exec sleep 30"})]])
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, session_file="cancelled.jsonl", tools=("bash",))
    session = await AgentSessionRuntime(options).new_session()
    ready = asyncio.Event()

    def observe(event, signal):
        if isinstance(event, ToolExecutionUpdateEvent):
            ready.set()

    session.agent.subscribe(observe)
    waiter = asyncio.create_task(session.prompt("start shell"))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert session.agent.state.is_busy
    finally:
        session.agent.abort()
        await session.agent.wait_for_idle()
    await session.close()
    saved = await AgentSessionRuntime(options).open_session(session.path)
    assert saved.agent.history == session.agent.history
    assert saved.save_state == "saved"
    assert not saved.agent.state.is_busy
    results = [entry.message for entry in saved.agent.history.entries
               if entry.type == "message" and entry.message.role == "toolResult"]
    assert len(results) == 1
    assert results[0].is_error
    assert len(stream.requests) == 1
