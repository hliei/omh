import subprocess
import sys

import pytest
from omh.agent import Agent, AgentOptions

from coding_agent import encode_history

WRITER = '''
import asyncio
import sys
from pathlib import Path
from omh.agent import Agent, AgentOptions
from coding_agent import SessionManager

async def main():
    path = Path(sys.argv[1])
    manager = SessionManager.load(path)[0] if path.exists() else SessionManager(cwd=path.parent, path=path)
    print("bound", flush=True)
    for line in sys.stdin:
        command, _, target = line.strip().partition(" ")
        if command == "save":
            await manager.save(Agent(AgentOptions(stream_fn=lambda *args: None)).history, target or None)
        elif command == "close":
            await manager.close()
            print("closed", flush=True)
            return
        print("saved", flush=True)

asyncio.run(main())
'''


def writer(path):
    return subprocess.Popen(
        [sys.executable, "-u", "-c", WRITER, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def stop(process):
    if process.poll() is None:
        process.terminate()
    process.communicate(timeout=5)


def assert_locked(path):
    contender = writer(path)
    try:
        output, error = contender.communicate("", timeout=5)
        assert contender.returncode != 0, (output, error)
        assert "SessionWriterError" in error
    finally:
        stop(contender)


@pytest.mark.parametrize("existing", [False, True])
def test_two_writer_processes_exclude_then_release(tmp_path, existing):
    path = tmp_path / "history.jsonl"
    if existing:
        path.write_text(encode_history(Agent(AgentOptions(stream_fn=lambda *args: None)).history, cwd=str(tmp_path)))
    first = writer(path)
    try:
        assert first.stdout.readline().strip() == "bound"
        assert_locked(path)
        first.stdin.write("save\n")
        first.stdin.flush()
        assert first.stdout.readline().strip() == "saved"
        assert_locked(path)
        # A read-only snapshot does not need a writer lock.
        if existing:
            from coding_agent import decode_history
            assert decode_history(path.read_bytes()).cwd == str(tmp_path)
        first.stdin.write("close\n")
        first.stdin.flush()
        assert first.stdout.readline().strip() == "closed"
        assert first.wait(timeout=5) == 0
        next_writer = writer(path)
        try:
            assert next_writer.stdout.readline().strip() == "bound"
            output, error = next_writer.communicate("close\n", timeout=5)
            assert next_writer.returncode == 0, error
            assert output.strip() == "closed"
        finally:
            stop(next_writer)
    finally:
        stop(first)


async def test_same_file_handoff_transfers_writer_and_blocks_retained_overwrite(tmp_path):
    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions, SessionWriterError

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="history.jsonl",
    ))
    old = await runtime.new_session()
    await old.prompt("saved input")
    assert_locked(old.path)
    current = await runtime.open_session(old.path)
    assert current.agent.history == old.agent.history
    assert old.agent.state.is_closed
    assert_locked(current.path)
    before = current.path.read_bytes()
    with pytest.raises(SessionWriterError):
        await old.save()
    assert current.path.read_bytes() == before
    assert old.save_state == "unsaved"
    await old.export("retained-backup.jsonl")
    assert old.save_state == "unsaved"
    await runtime.close()
    await old.save()
    # Retained repair releases its temporary writer immediately.
    next_writer = writer(old.path)
    try:
        assert next_writer.stdout.readline().strip() == "bound"
    finally:
        stop(next_writer)


async def test_failed_candidate_preparation_releases_its_writer(tmp_path):
    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions

    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=()))
    old = await runtime.new_session()
    await old.export("candidate.jsonl")
    runtime.options.tools = ("read", "read")
    with pytest.raises(ValueError):
        await runtime.open_session("candidate.jsonl")
    assert runtime.current_session is old
    assert not old.agent.state.is_closed
    next_writer = writer(tmp_path / "candidate.jsonl")
    try:
        assert next_writer.stdout.readline().strip() == "bound"
    finally:
        stop(next_writer)


@pytest.mark.parametrize("operation", ["close", "switch"])
async def test_cancelled_waiter_keeps_writers_until_owned_cleanup(tmp_path, operation):
    import asyncio

    from omh.agent import AgentTool, AgentToolResult
    from omh.llm.types import TextContent, ToolCall
    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream([[ToolCall(id="hold", name="hold", arguments={})]]),
        tools=(), session_file="old.jsonl",
    ))
    old = await runtime.new_session()
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(call_id, arguments, signal, on_update):
        signal.add_callback(closing.set)
        started.set()
        await release.wait()
        return AgentToolResult(content=[TextContent(text="finished")])

    await old.agent.set_tools([AgentTool(
        name="hold", description="hold", label="hold", parameters={"type": "object"}, execute=execute,
    )])
    prompt = asyncio.create_task(old.prompt("hold tool"))
    await asyncio.wait_for(started.wait(), 2)
    target = tmp_path / "candidate.jsonl"
    await old.export(target)
    waiter = asyncio.create_task(old.close() if operation == "close" else runtime.open_session(target))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not old.agent.state.is_closed
        assert_locked(old.path)
        if operation == "switch":
            assert_locked(target)
    finally:
        release.set()
        await asyncio.wait_for(prompt, 2)
    if operation == "close":
        await old.close()
    else:
        current = await runtime.wait_for_switch()
        assert current is runtime.current_session
        assert_locked(target)
        await runtime.close()
    next_writer = writer(old.path)
    try:
        assert next_writer.stdout.readline().strip() == "bound"
    finally:
        stop(next_writer)


async def test_cancelled_preparation_releases_candidate_writer(tmp_path, monkeypatch):
    import asyncio
    from pathlib import Path

    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions

    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=()))
    old = await runtime.new_session()
    target = tmp_path / "candidate.jsonl"
    await old.export(target)
    read_bytes = Path.read_bytes

    def cancel_read(path):
        raw = read_bytes(path)
        if path == target:
            asyncio.current_task().cancel()
        return raw

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", cancel_read)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(runtime.open_session(target))
    assert runtime.current_session is old
    contender = writer(target)
    try:
        assert contender.stdout.readline().strip() == "bound"
    finally:
        stop(contender)


async def test_export_refuses_to_overwrite_another_writer(tmp_path):
    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions, SessionWriterError

    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=()))
    session = await runtime.new_session()
    path = tmp_path / "backup.jsonl"
    await session.export(path)
    before = path.read_bytes()
    contender = writer(path)
    try:
        assert contender.stdout.readline().strip() == "bound"
        with pytest.raises(SessionWriterError):
            await session.export(path)
        assert path.read_bytes() == before
        assert session.path is None
        assert session.save_state == "pending"
        assert session.save_error is None
    finally:
        stop(contender)


async def test_runtime_close_releases_published_writer_after_handoff_error(tmp_path):
    import asyncio

    from omh.agent import AgentTool, AgentToolResult
    from omh.llm.types import TextContent, ToolCall
    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream([[ToolCall(id="hold", name="hold", arguments={})]]),
        tools=(), session_file="old.jsonl",
    ))
    old = await runtime.new_session()
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    error = OSError("terminal notification failed")

    async def execute(call_id, arguments, signal, on_update):
        signal.add_callback(closing.set)
        started.set()
        await release.wait()
        return AgentToolResult(content=[TextContent(text="finished")])

    def fail(event, signal):
        if event.type == "agent_settled":
            raise error

    await old.agent.set_tools([AgentTool(
        name="hold", description="hold", label="hold", parameters={"type": "object"}, execute=execute,
    )])
    old.agent.subscribe(fail)
    prompt = asyncio.create_task(old.prompt("hold tool"))
    await asyncio.wait_for(started.wait(), 2)
    target = tmp_path / "candidate.jsonl"
    await old.export(target)
    switching = asyncio.create_task(runtime.open_session(target))
    await asyncio.wait_for(closing.wait(), 2)
    close = asyncio.create_task(runtime.close())
    # Let close begin waiting at the public handoff boundary.
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.wait_for(asyncio.gather(prompt, switching, close, return_exceptions=True), 2)
    assert results == [error, error, error]
    assert runtime.current_session is not old
    assert runtime.current_session.agent.state.is_closed
    contender = writer(target)
    try:
        assert contender.stdout.readline().strip() == "bound"
    finally:
        stop(contender)
