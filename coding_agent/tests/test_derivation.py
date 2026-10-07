"""Complete-history derivation through public application and SDK contracts."""

from pathlib import Path

from omh.agent import Agent, AgentInitialState, AgentOptions, validate_history
from support import OfflineStream, model
from test_history import complete_history

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentOptions,
    decode_history,
    encode_history,
)


async def test_clone_copies_only_active_records_and_reopens_with_new_identity(tmp_path: Path) -> None:
    original = complete_history()
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(encode_history(original, cwd=str(tmp_path)))
    original_bytes = source_path.read_bytes()
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        stream_fn=stream, available_models=(model("physical"), model()), tools=(),
    ))
    source = await runtime.open_session(source_path)
    target = await runtime.clone_session()
    try:
        assert runtime.current_session is target
        assert runtime.retained_sessions == (source,)
        assert target.agent.history.entries == original.entries[:-1]
        assert target.agent.history.leaf_id == "last"
        assert target.agent.history.conversation_id != original.conversation_id
        assert target.agent.history.created_at > original.created_at
        assert target.agent.state.model.id == "physical"
        assert target.agent.state.thinking_level == "high"
        assert target.agent.state.messages == source.agent.state.messages
        assert target.cwd == source.cwd
        assert target.path != source.path
        assert target.save_state == "saved"
        assert target.source.conversation_id == "rich"
        assert target.source.leaf_id == "last"
        assert target.source.entry_id is None
        assert target.source.path == str(source_path)
        validate_history(target.agent.history)
        decoded = decode_history(target.path.read_bytes())
        assert decoded.source == target.source
        restored = Agent.from_history(decoded.history, AgentOptions(
            stream_fn=stream, initial_state=AgentInitialState(model=model("physical")),
        ))
        assert restored.state.messages == target.agent.state.messages
        assert source_path.read_bytes() == original_bytes
        assert not stream.requests
        identity = target.agent.history.conversation_id
        await target.prompt("independent continuation")
        assert stream.requests[-1][2].session_id == identity
        target_path = target.path
        await runtime.close()
        reopened = await runtime.open_session(target_path)
        assert reopened.agent.history.conversation_id == identity
        assert reopened.source == target.source
        assert source_path.read_bytes() == original_bytes
    finally:
        await runtime.close()


async def test_fork_preserves_ancestors_but_leaves_selected_user_uncommitted(tmp_path: Path) -> None:
    original = complete_history()
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(encode_history(original, cwd=str(tmp_path)))
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        stream_fn=stream, available_models=(model("physical"), model()), tools=(),
    ))
    source = await runtime.open_session(source_path)
    source.steer("source queued input")
    target = await runtime.fork_session("last")
    try:
        assert target.agent.history.entries == original.entries[:-2]
        assert target.agent.history.leaf_id == "delta"
        assert target.source.kind == "fork"
        assert target.source.entry_id == "last"
        assert target.source.leaf_id == "delta"
        assert target.agent.state.messages[-1].role == "system"
        assert target.agent.state.thinking_level == "high"
        assert target.agent.state.model.id == "physical"
        assert not target.agent.has_queued_messages()
        assert source.queued_messages.steering[0].content[0].text == "source queued input"
        assert source.agent.history == original
        assert decode_history(source_path.read_bytes()).history == original
        saved = decode_history(target.path.read_bytes())
        assert saved.history == target.agent.history
        assert saved.source == target.source
        assert not stream.requests
    finally:
        await runtime.close()


async def test_go_requests_use_derived_identity_after_save_and_reopen(tmp_path: Path) -> None:
    from support import RecordingFetch

    from coding_agent import CodingAgentHost

    fetch = RecordingFetch()
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent",
                           api_key="offline-key", fetch=fetch, no_context_files=True)
    runtime = AgentSessionRuntime(host.build_options(host.select_new()))
    source = await runtime.new_session()
    await source.prompt("original")
    old_id = source.agent.history.conversation_id
    derived = await runtime.clone_session()
    new_id = derived.agent.history.conversation_id
    await derived.prompt("derived")
    assert [request.headers["x-opencode-session"] for request in fetch.requests] == [old_id, new_id]
    assert "offline-key" not in derived.path.read_text()
    path = derived.path
    await runtime.close()
    runtime = AgentSessionRuntime(host.build_options(host.select_open(path)))
    try:
        reopened = await runtime.open_session(path)
        await reopened.prompt("continued")
        assert fetch.requests[-1].headers["x-opencode-session"] == new_id
        assert reopened.source.conversation_id == old_id
    finally:
        await runtime.close()


async def test_derived_save_failure_keeps_published_memory_and_retained_source(tmp_path: Path) -> None:
    import pytest

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="source.jsonl",
    ))
    source = await runtime.new_session()
    await source.prompt("keep full history")
    original = source.path.read_bytes()
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    with pytest.raises(OSError):
        await runtime.clone_session(session_dir=blocked)
    try:
        target = runtime.current_session
        assert target is not source
        assert runtime.retained_sessions == (source,)
        assert target.agent.history.entries == source.agent.history.entries
        assert target.save_state == "unsaved"
        assert target.save_error is not None
        with pytest.raises(RuntimeError, match="unsaved history"):
            await target.prompt("must not run")
        error = target.save_error
        await target.export(tmp_path / "backup.jsonl")
        assert target.save_error is error
        await target.save(tmp_path / "repaired.jsonl")
        assert decode_history(target.path.read_bytes()).history == target.agent.history
        assert decode_history(target.path.read_bytes()).source == target.source
        assert source.path.read_bytes() == original
    finally:
        await runtime.close()


async def test_target_environment_reassembles_resources_and_explicit_storage(tmp_path: Path) -> None:
    source_cwd, target_cwd = tmp_path / "source", tmp_path / "target"
    source_cwd.mkdir()
    target_cwd.mkdir()
    (source_cwd / "AGENTS.md").write_text("source instructions")
    (target_cwd / "AGENTS.md").write_text("target instructions")
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=source_cwd, model=model(), thinking_level="high", stream_fn=OfflineStream(), tools=(),
    ))
    source = await runtime.new_session()
    await source.prompt("original")
    target = await runtime.clone_session(cwd=target_cwd)
    try:
        assert source.save_mode == target.save_mode == "memory"
        assert target.cwd == target_cwd
        assert "target instructions" in target.agent.state.system_sections["project_context"]
        assert "source instructions" not in target.agent.state.system_sections["project_context"]
        assert target.agent.state.thinking_level == "high"
        assert target.agent.history.entries == source.agent.history.entries
        saved = await runtime.clone_session(save_mode="auto", session_dir="derived")
        assert saved.path.parent == target_cwd / "derived"
        assert saved.save_state == "saved"
        memory = await runtime.clone_session(save_mode="memory")
        assert memory.path is None
        assert memory.save_mode == "memory"
        assert memory.agent.history.entries == saved.agent.history.entries
    finally:
        await runtime.close()


async def test_reopened_derived_json_header_carries_source_file(tmp_path: Path) -> None:
    import io
    import json

    from support import RecordingFetch

    from coding_agent import CodingAgentHost, PrintTask
    from coding_agent.print_runner import run_print_json

    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent",
                           api_key="offline-key", fetch=RecordingFetch())
    runtime = AgentSessionRuntime(host.build_options(host.select_new()))
    source = await runtime.new_session()
    await source.prompt("source")
    target = await runtime.clone_session()
    target_path = target.path
    target_id = target.agent.history.conversation_id
    await runtime.close()
    stdout, stderr = io.StringIO(), io.StringIO()
    result = await run_print_json(host, host.select_open(target_path), [PrintTask("continue")],
                                  stdout=stdout, stderr=stderr)
    assert result == 0, stderr.getvalue()
    header = json.loads(stdout.getvalue().splitlines()[0])
    assert header["id"] == target_id
    assert header["parentSession"] == str(source.path)


async def test_fork_before_first_user_inherits_earlier_selection_and_preserves_source(tmp_path: Path) -> None:
    original = complete_history()
    path = tmp_path / "source.jsonl"
    path.write_text(encode_history(original, cwd=str(tmp_path)))
    runtime = AgentSessionRuntime(CodingAgentOptions(
        stream_fn=OfflineStream(), available_models=(model(), model("physical")), tools=(),
    ))
    source = await runtime.open_session(path)
    target = await runtime.fork_session("original")
    try:
        assert [entry.id for entry in target.agent.history.entries] == ["model", "thinking", "system"]
        assert target.agent.state.model.id == "offline"
        assert target.agent.state.thinking_level == "high"
        assert target.save_state == "pending"
        assert not target.path.exists()
        assert source.agent.history == original
        await target.save()
        assert decode_history(target.path.read_bytes()).history == target.agent.history
    finally:
        await runtime.close()


async def test_invalid_positions_and_failed_preparation_leave_source_usable(tmp_path: Path) -> None:
    import pytest

    path = tmp_path / "source.jsonl"
    path.write_text(encode_history(complete_history(), cwd=str(tmp_path)))
    runtime = AgentSessionRuntime(CodingAgentOptions(
        stream_fn=OfflineStream(), available_models=(model(), model("physical")), tools=(),
    ))
    source = await runtime.open_session(path)
    original = source.agent.history
    try:
        for entry in ("inactive", "compact2", "missing"):
            with pytest.raises(ValueError, match="active path"):
                await runtime.fork_session(entry)
            assert runtime.current_session is source
            assert source.agent.history == original
        with pytest.raises(ValueError, match="session directory"):
            memory = AgentSessionRuntime(CodingAgentOptions(
                model=model(), stream_fn=OfflineStream(), tools=(), cwd=tmp_path,
            ))
            await memory.new_session()
            try:
                await memory.clone_session(save_mode="auto")
            finally:
                await memory.close()
        with pytest.raises(ValueError, match="cannot specify"):
            await runtime.clone_session(save_mode="memory", session_dir=tmp_path)
        runtime.options.tools = ("read", "read")
        with pytest.raises(ValueError, match="distinct subset"):
            await runtime.clone_session()
        runtime.options.tools = ()
        assert runtime.current_session is source
        assert not source.agent.state.is_closed
        assert runtime.retained_sessions == ()
        await source.prompt("source still continues")
        assert "source still continues" in path.read_text()
    finally:
        await runtime.close()


async def test_busy_agent_rejects_derivation_without_abort_or_history_changes(tmp_path: Path) -> None:
    import asyncio

    import pytest

    started, release = asyncio.Event(), asyncio.Event()
    stream = OfflineStream()

    async def held(model, context, options):
        started.set()
        await release.wait()
        return stream(model, context, options)

    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=held, tools=()))
    source = await runtime.new_session()
    prompt = asyncio.create_task(source.prompt("original"))
    await asyncio.wait_for(started.wait(), 2)
    original = source.agent.history
    try:
        with pytest.raises(RuntimeError, match="idle"):
            await runtime.clone_session()
        with pytest.raises(RuntimeError, match="idle"):
            await runtime.fork_session(original.leaf_id)
        assert runtime.current_session is source
        assert source.agent.history == original
        assert source.agent.state.is_busy
        assert not source.agent.signal.aborted
        assert runtime.retained_sessions == ()
    finally:
        release.set()
        await asyncio.wait_for(prompt, 2)
        await runtime.close()
