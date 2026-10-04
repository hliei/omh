import asyncio
import json
from pathlib import Path

import pytest
from omh.agent import (
    AgentSettledEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    CustomAgentMessage,
)
from omh.llm.types import TextContent, ToolCall
from support import OfflineStream, model

from coding_agent import CodingAgentOptions, CodingAgentRuntime, decode_history


async def test_new_and_open_retire_old_instances_and_preserve_identity(tmp_path):
    stream = OfflineStream()
    runtime = CodingAgentRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=()))
    first = await runtime.new_session()
    await first.prompt("first conversation")
    await first.save("first.jsonl")
    history = first.agent.history
    second = await runtime.new_session(display_name="second")
    assert runtime.current_session is second
    assert first.agent.state.is_closed
    assert second.agent.history.conversation_id != history.conversation_id
    assert not second.agent.state.is_busy
    assert len(stream.requests) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await first.prompt("stale input")
    restored = await runtime.open_session("first.jsonl")
    assert second.agent.state.is_closed
    assert runtime.current_session is restored
    assert restored.agent.history == history
    await restored.prompt("continue restored conversation")
    assert stream.requests[-1][2].session_id == history.conversation_id


@pytest.mark.parametrize("entry", ["new_session", "open_session", "switch_session"])
async def test_cancelled_switch_waiter_still_publishes_after_tool_finishes(tmp_path, entry):
    stream = OfflineStream([[ToolCall(id="hold", name="hold", arguments={})]])
    runtime = CodingAgentRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=()))
    old = await runtime.new_session()
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(call_id, arguments, signal, on_update):
        signal.add_callback(closing.set)
        started.set()
        await release.wait()
        (tmp_path / "effect.txt").write_text("completed once")
        return AgentToolResult(content=[TextContent(text="completed")])

    await old.agent.set_tools([AgentTool(name="hold", description="hold", label="hold",
                                       parameters={"type": "object"}, execute=execute)])
    target = tmp_path / "target.jsonl"
    target.write_text(await old.export())
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("context prepared before close")
    observed = []
    unsubscribe = runtime.subscribe(lambda event, signal: observed.append(event))
    prompt = asyncio.create_task(old.prompt("run tool"))
    await asyncio.wait_for(started.wait(), 2)
    switching = asyncio.create_task(runtime.new_session() if entry == "new_session"
                                    else getattr(runtime, entry)(target))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        instructions.write_text("context changed during close")
        observers = [asyncio.create_task(runtime.wait_for_switch()) for _ in range(2)]
        switching.cancel()
        with pytest.raises(asyncio.CancelledError):
            await switching
        observers[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await observers[0]
        assert runtime.current_session is old
        assert not old.agent.state.is_closed
        assert len(stream.requests) == 1
        with pytest.raises(RuntimeError, match="closed"):
            await old.prompt("stale work")
        with pytest.raises(RuntimeError, match="switch"):
            await runtime.new_session()
    finally:
        release.set()
        await asyncio.wait_for(prompt, 2)
    new = await asyncio.wait_for(observers[1], 2)
    assert runtime.current_session is new
    assert old.agent.state.is_closed
    assert (tmp_path / "effect.txt").read_text() == "completed once"
    assert len(stream.requests) == 1
    assert not new.agent.state.is_busy
    assert "context prepared before close" in new.agent.state.system_sections["project_context"]
    assert "context changed during close" not in new.agent.state.system_sections["project_context"]
    before = len(observed)
    await new.prompt("new work")
    assert any(isinstance(event, AgentStartEvent) for event in observed[before:])
    unsubscribe()
    before = len(observed)
    await runtime.prompt("unobserved work")
    assert len(observed) == before


@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_close_notification_error_is_reported_after_publication(tmp_path, cancel_waiter):
    stream = OfflineStream()
    runtime = CodingAgentRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=()))
    old = await runtime.new_session()
    settled, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    error = LookupError("old terminal notification failed")

    async def fail(event, signal):
        if isinstance(event, AgentSettledEvent):
            settled.set()
            await release.wait()
            raise error

    old.agent.subscribe(fail)
    prompt = asyncio.create_task(old.prompt("old work"))
    await asyncio.wait_for(settled.wait(), 2)
    old.agent.signal.add_callback(closing.set)
    waiter = asyncio.create_task(runtime.new_session())
    await asyncio.wait_for(closing.wait(), 2)
    if cancel_waiter:
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    release.set()
    with pytest.raises(LookupError) as failed_prompt:
        await asyncio.wait_for(prompt, 2)
    assert failed_prompt.value is error
    observer = runtime.wait_for_switch() if cancel_waiter else waiter
    with pytest.raises(LookupError) as failed_switch:
        await asyncio.wait_for(observer, 2)
    assert failed_switch.value is error
    assert old.agent.state.is_closed
    new = runtime.current_session
    assert new is not old
    assert not new.agent.state.is_closed
    assert not new.agent.state.is_busy
    await new.prompt("new work")
    assert len(stream.requests) == 2


async def test_switch_retains_unsaved_history_and_complete_separate_queues(tmp_path):
    occupied = tmp_path / "occupied.jsonl"
    occupied.write_text("existing file")
    stream = OfflineStream()
    runtime = CodingAgentRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file=occupied,
    ))
    old = await runtime.new_session()
    for text in ("steer one", "steer two"):
        old.steer(text)
    for text in ("follow one", "follow two"):
        old.follow_up(text)
    with pytest.raises(FileExistsError):
        await old.prompt("unsaved history")
    history, error = old.agent.history, old.save_error
    runtime.options.session_file = "new.jsonl"
    new = await runtime.new_session()
    assert runtime.retained_sessions == (old,)
    assert old.agent.state.is_closed
    assert old.agent.history == history
    assert old.save_state == "unsaved"
    assert old.save_error is error
    queues = old.queued_messages
    assert [message.content[0].text for message in queues.steering] == ["steer one", "steer two"]
    assert [message.content[0].text for message in queues.follow_up] == ["follow one", "follow two"]
    queues.follow_up[0].content[0].text = "mutated copy"
    old.agent.clear_all_queues()
    assert old.queued_messages.follow_up[0].content[0].text == "follow one"
    assert not new.agent.has_queued_messages()
    assert decode_history(await old.export("backup.jsonl")).history == history
    assert old.save_state == "unsaved"
    await old.save("repaired.jsonl")
    assert old.save_state == "saved"
    restored = await CodingAgentRuntime(runtime.options).open_session("repaired.jsonl")
    assert restored.agent.history == history
    assert not restored.agent.has_queued_messages()
    await runtime.prompt("new input")
    assert len(stream.requests) == 1
    assert all("follow one" not in str(message) for message in stream.requests[0][1].messages)
    third = await runtime.new_session()
    assert runtime.retained_sessions == (old, new)
    assert runtime.current_session is third


@pytest.mark.parametrize("failure", ["read", "decode", "validate", "model", "tools", "resources", "construction"])
async def test_target_preparation_failure_keeps_current_usable(tmp_path, failure):
    stream = OfflineStream()
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=())
    runtime = CodingAgentRuntime(options)
    old = await runtime.new_session()
    await old.prompt("saved source")
    target = tmp_path / "target.jsonl"
    target.write_text(await old.export())
    if failure == "read":
        target.unlink()
    elif failure == "decode":
        target.write_text('{"format":"unsupported","version":1}\n')
    elif failure == "validate":
        lines = [json.loads(line) for line in target.read_text().splitlines()]
        lines[1]["parentId"] = "missing-parent"
        target.write_text("\n".join(json.dumps(line) for line in lines))
    elif failure == "model":
        options.model = None
    elif failure == "tools":
        options.tools = ("read", "read")
    elif failure == "resources":
        options.context_dirs = ("invalid\0directory",)
    else:
        from omh.agent import AgentInitialState

        options.agent_options.initial_state = AgentInitialState(messages=[])
    before = target.read_bytes() if target.exists() else None
    snapshot = old.agent.history
    observed = []
    runtime.subscribe(lambda event, signal: observed.append(event))
    with pytest.raises((ValueError, OSError)):
        await runtime.switch_session(target)
    assert runtime.current_session is old
    assert runtime.retained_sessions == ()
    assert old.agent.history == snapshot
    assert not old.agent.state.is_closed
    assert not observed
    assert len(stream.requests) == 1
    if before is not None:
        assert target.read_bytes() == before
    await runtime.prompt("old session continues")
    assert len(stream.requests) == 2
    assert any(isinstance(event, AgentStartEvent) for event in observed)


@pytest.mark.parametrize("entry", ["new_session", "open_session", "switch_session"])
async def test_cancelled_preparation_keeps_old_current_and_no_standby_work(tmp_path, monkeypatch, entry):
    stream = OfflineStream()
    runtime = CodingAgentRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=()))
    old = await runtime.new_session()
    target = tmp_path / "target.jsonl"
    target.write_text(await old.export())
    read_bytes = Path.read_bytes

    def cancel_read(path):
        raw = read_bytes(path)
        if path == target:
            asyncio.current_task().cancel()
        return raw

    monkeypatch.setattr(Path, "read_bytes", cancel_read)
    call = runtime.new_session() if entry == "new_session" else getattr(runtime, entry)(target)
    waiter = asyncio.create_task(call)
    if entry == "new_session":
        # Cancel at the initial preparation checkpoint, before any close starts.
        await asyncio.sleep(0)
        waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert runtime.current_session is old
    assert not old.agent.state.is_closed
    assert runtime.retained_sessions == ()
    assert not stream.requests
    await runtime.prompt("still usable")
    assert len(stream.requests) == 1


async def test_runtime_subscriptions_rebind_and_callback_switch_rejects_self_wait(tmp_path):
    stream = OfflineStream()
    runtime = CodingAgentRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=()))
    starts = []
    unsubscribe = runtime.subscribe(lambda event, signal: starts.append(event) if isinstance(event, AgentStartEvent) else None)
    old = await runtime.new_session()

    async def replace(event, signal):
        if isinstance(event, AgentSettledEvent):
            with pytest.raises(RuntimeError, match="own activity callback"):
                await runtime.new_session()

    stop_replacing = old.agent.subscribe(replace)
    await runtime.prompt("old input")
    assert runtime.current_session is old
    assert not old.agent.state.is_closed
    stop_replacing()
    new = await runtime.new_session()
    await new.prompt("new input")
    assert len(starts) == 2
    unsubscribe()
    await runtime.prompt("unsubscribed")
    assert len(starts) == 2


async def test_reopening_busy_current_file_rejects_a_stale_prepared_history(tmp_path):
    stream = OfflineStream()
    runtime = CodingAgentRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file="current.jsonl",
    ))
    old = await runtime.new_session()
    settled, release = asyncio.Event(), asyncio.Event()

    async def hold(event, signal):
        if isinstance(event, AgentSettledEvent):
            settled.set()
            await release.wait()

    old.agent.subscribe(hold)
    prompt = asyncio.create_task(old.prompt("current work"))
    await asyncio.wait_for(settled.wait(), 2)
    try:
        with pytest.raises(RuntimeError, match="current session file"):
            await asyncio.wait_for(runtime.switch_session(old.path), 2)
        assert runtime.current_session is old
        assert not old.agent.signal.aborted
        assert runtime.retained_sessions == ()
    finally:
        release.set()
        await prompt
    restored = await runtime.open_session(old.path)
    assert restored.agent.history == old.agent.history


async def test_terminal_listener_cannot_await_its_own_pending_handoff(tmp_path):
    runtime = CodingAgentRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=()))
    old = await runtime.new_session()
    settled, closing = asyncio.Event(), asyncio.Event()

    async def observe(event, signal):
        if isinstance(event, AgentSettledEvent):
            signal.add_callback(closing.set)
            settled.set()
            await closing.wait()
            with pytest.raises(RuntimeError, match="own activity callback"):
                await asyncio.wait_for(runtime.wait_for_switch(), 0.1)

    old.agent.subscribe(observe)
    prompt = asyncio.create_task(old.prompt("old input"))
    await asyncio.wait_for(settled.wait(), 2)
    switch = asyncio.create_task(runtime.new_session())
    await asyncio.wait_for(prompt, 2)
    new = await asyncio.wait_for(switch, 2)
    assert runtime.current_session is new
    assert old.agent.state.is_closed


async def test_same_file_handoff_rejects_history_changed_during_preparation(tmp_path, monkeypatch):
    runtime = CodingAgentRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="current.jsonl",
    ))
    old = await runtime.new_session()
    settled, release = asyncio.Event(), asyncio.Event()

    async def append_last_record(event, signal):
        if isinstance(event, AgentSettledEvent):
            settled.set()
            await release.wait()
            await old.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="late history"))

    old.agent.subscribe(append_last_record)
    prompt = asyncio.create_task(old.prompt("old work"))
    await asyncio.wait_for(settled.wait(), 2)
    read_bytes = Path.read_bytes

    def release_after_read(path):
        raw = read_bytes(path)
        if path == old.path:
            release.set()
        return raw

    monkeypatch.setattr(Path, "read_bytes", release_after_read)
    try:
        with pytest.raises(RuntimeError, match="current session file"):
            await asyncio.wait_for(runtime.open_session(old.path), 2)
    finally:
        release.set()
        await prompt
    assert runtime.current_session is old
    assert not old.agent.state.is_closed
    assert "late history" in await old.export()
    assert runtime.retained_sessions == ()
    restored = await runtime.open_session(old.path)
    assert restored.agent.history == old.agent.history
