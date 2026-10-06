import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    CustomAgentMessage,
    RetryPolicy,
)
from omh.llm.types import ToolCall
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions, decode_history


@pytest.fixture
def fail_writes(monkeypatch):
    """Fail at the file-system boundary after writing a real byte prefix."""
    original = Path.open

    def install(path, mode, *, contains="", attempts=None):
        error = OSError("503 temporary save failure")

        @contextmanager
        def open_file(destination, requested_mode="r", *args, **kwargs):
            nonlocal attempts
            requested_mode = kwargs.pop("mode", requested_mode)
            with original(destination, requested_mode, *args, **kwargs) as file:
                if requested_mode == mode and (
                    destination == path or (mode == "w" and destination.parent == path.parent
                                            and destination.name.startswith(f".{path.name}."))
                ):
                    write = file.write

                    def partial_write(text):
                        nonlocal attempts
                        if contains in text and (attempts is None or attempts > 0):
                            if attempts is not None:
                                attempts -= 1
                            write(text[:len(text) // 2])
                            file.flush()
                            raise error
                        return write(text)

                    file.write = partial_write
                yield file

        monkeypatch.setattr(Path, "open", open_file)
        return error

    return install


@pytest.mark.parametrize("entry", ["session", "runtime"])
@pytest.mark.parametrize("operation", ["prompt", "continue_", "compact", "steer", "follow_up", "submit_custom_message", "ensure_can_accept_work"])
async def test_unsaved_rejects_application_work_without_changing_history(tmp_path, entry, operation):
    path = tmp_path / "occupied.jsonl"
    path.write_text("existing file")
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file=path,
    ))
    session = await runtime.new_session()
    runtime.follow_up("previously accepted input")
    with pytest.raises(FileExistsError):
        await runtime.prompt("retained input")
    history = session.agent.history
    messages = session.agent.state.messages
    queued = session.agent.peek_queued_messages()
    assert queued[0].content[0].text == "previously accepted input"
    error = session.save_error
    target = session if entry == "session" else runtime
    with pytest.raises(RuntimeError, match="unsaved") as rejected:
        if operation == "prompt":
            await target.prompt("rejected input")
        elif operation == "continue_":
            await target.continue_()
        elif operation == "compact":
            await target.compact()
        elif operation == "submit_custom_message":
            await target.submit_custom_message(CustomAgentMessage(custom_type="bashExecution", content="! echo hello"))
        elif operation == "ensure_can_accept_work":
            target.ensure_can_accept_work()
        else:
            getattr(target, operation)("rejected queued input")
    assert rejected.value.__cause__ is error
    assert session.agent.history == history
    assert session.agent.state.messages == messages
    assert session.agent.peek_queued_messages() == queued
    assert session.save_state == "unsaved"
    assert session.save_error is error
    assert not stream.requests
    session.agent.abort()
    await asyncio.wait_for(session.agent.close(), timeout=2)
    assert session.agent.state.is_closed
    assert session.agent.history == history
    assert session.agent.peek_queued_messages() == queued
    assert "retained input" in await session.export()


@pytest.mark.parametrize("phase", ["initial", "append", "repair"])
@pytest.mark.parametrize("repair", ["save", "export_bound"])
async def test_partial_write_requires_full_save_before_resuming(tmp_path, fail_writes, phase, repair):
    path = tmp_path / "history.jsonl"
    stream = OfflineStream()
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file=path)
    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session()
    if phase != "initial":
        await session.prompt("first")
    error = fail_writes(path, {"initial": "x", "append": "a", "repair": "w"}[phase], attempts=1)
    with pytest.raises(OSError) as failed:
        if phase == "repair":
            await runtime.save_session()
        else:
            await session.prompt("retained input")
    assert failed.value is error
    history, messages = session.agent.history, session.agent.state.messages
    assert session.save_state == "unsaved"
    damaged = path.read_bytes()
    exported = await runtime.export_session("copy.jsonl")
    copy = await AgentSessionRuntime(options).open_session(tmp_path / "copy.jsonl")
    assert copy.agent.history == history
    assert copy.agent.state.messages == messages
    assert decode_history(exported).history == history
    assert session.path == path
    assert session.save_state == "unsaved"
    assert session.save_error is error
    assert path.read_bytes() == damaged
    with pytest.raises(RuntimeError, match="unsaved"):
        await runtime.prompt("not repaired by export")
    if repair == "save":
        await runtime.save_session()
    else:
        assert decode_history(await runtime.export_session(path)).history == history
    assert session.save_state == "saved"
    assert session.save_error is None
    session = await runtime.open_session(path)
    assert session.agent.history == history
    assert session.agent.state.messages == messages
    await session.prompt("after repair")
    reopened = await runtime.open_session(path)
    assert reopened.agent.history == session.agent.history
    assert len({entry.id for entry in reopened.agent.history.entries}) == len(reopened.agent.history.entries)


async def test_persistent_failure_can_export_and_rebind_complete_history(tmp_path, fail_writes):
    path = tmp_path / "history.jsonl"
    stream = OfflineStream()
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file=path)
    session = await AgentSessionRuntime(options).new_session()
    await session.prompt("first")
    error = fail_writes(path, "w")
    history = session.agent.history
    for _ in range(2):
        with pytest.raises(OSError) as failed:
            await session.save()
        assert failed.value is error
        assert session.save_state == "unsaved"
        assert session.agent.history == history
    await session.export("copy.jsonl")
    assert session.save_state == "unsaved"
    damaged = path.read_bytes()
    await session.save("replacement.jsonl")
    assert session.path == tmp_path / "replacement.jsonl"
    assert session.save_state == "saved"
    assert session.save_error is None
    assert path.read_bytes() == damaged
    await session.prompt("continue on replacement")
    await session.close()
    reopened = await AgentSessionRuntime(options).open_session(session.path)
    assert reopened.agent.history == session.agent.history


async def test_serialization_failure_preserves_committed_assistant_without_retry(tmp_path, monkeypatch):
    original = json.dumps
    error = ValueError("503 temporary serialization failure")

    def dumps(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("message", {}).get("role") == "assistant":
            raise error
        return original(value, *args, **kwargs)

    stream = OfflineStream()
    options = CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file="history.jsonl",
        agent_options=AgentOptions(retry=RetryPolicy(base_delay_ms=0)),
    )
    session = await AgentSessionRuntime(options).new_session()
    events = []
    session.agent.subscribe(lambda event, signal: events.append(event.type))
    with monkeypatch.context() as patch:
        patch.setattr(json, "dumps", dumps)
        with pytest.raises(ValueError) as failed:
            await session.prompt("retain response")
    assert failed.value is error
    assert session.save_state == "unsaved"
    assert session.save_error is error
    assert len(stream.requests) == 1
    assert "retry_start" not in events
    assistants = [message for message in session.agent.state.messages if message.role == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].stop_reason == "stop"
    assert assistants[0].content[0].text == "done"
    assert events[-1] == "agent_settled"
    history = session.agent.history
    # The SDK can independently restore and run the retained history.
    independent = Agent.from_history(history, AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=model()),
    ))
    await independent.prompt("SDK remains usable")
    assert len(stream.requests) == 2
    await session.save()
    await session.close()
    reopened = await AgentSessionRuntime(options).open_session(session.path)
    assert reopened.agent.history == history


async def test_save_and_terminal_failures_release_prompt_idle_and_close_waiters(tmp_path, fail_writes):
    path = tmp_path / "history.jsonl"
    stream = OfflineStream([[ToolCall(id="write", name="write", arguments={"path": "demo.txt", "content": "completed"})]])
    options = CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("write",), session_file=path,
        agent_options=AgentOptions(retry=RetryPolicy(base_delay_ms=0)),
    )
    session = await AgentSessionRuntime(options).new_session()
    error = fail_writes(path, "a", contains='"role": "toolResult"')
    settling, release = asyncio.Event(), asyncio.Event()
    events = []

    async def terminal_failure(event, signal):
        events.append(event.type)
        if event.type == "agent_end":
            raise RuntimeError("end notification failed")
        if event.type == "agent_settled":
            settling.set()
            await release.wait()
            # A final callback cannot manufacture more work while unsaved.
            await session.prompt("must not run")

    session.agent.subscribe(terminal_failure)
    prompt = asyncio.create_task(session.prompt("write a file"))
    try:
        await asyncio.wait_for(settling.wait(), timeout=2)
        idle = asyncio.create_task(session.agent.wait_for_idle())
        close = asyncio.create_task(session.agent.close())
        release.set()
        results = await asyncio.wait_for(asyncio.gather(prompt, idle, close, return_exceptions=True), timeout=2)
    finally:
        release.set()
        session.agent.abort()
        await asyncio.gather(prompt, return_exceptions=True)
    assert results == [error, None, error]
    assert session.agent.state.is_closed
    assert not session.agent.state.is_busy
    assert session.save_state == "unsaved"
    assert len(stream.requests) == 1
    assert events.count("agent_end") == events.count("agent_settled") == 1
    assert "retry_start" not in events
    assert (tmp_path / "demo.txt").read_text() == "completed"
    messages = session.agent.state.messages
    assert [message.role for message in messages if message.role != "system"] == ["user", "assistant", "toolResult"]
    assert messages[-1].is_error is False
    history = session.agent.history
    assert decode_history(await session.export()).history == history
    with pytest.raises(OSError):
        await session.close()
    await session.save()
    reopened = await AgentSessionRuntime(options).open_session(path)
    assert reopened.agent.history == history
    assert reopened.agent.state.messages == messages
