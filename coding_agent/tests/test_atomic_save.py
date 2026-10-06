import os

import pytest
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions, decode_history


async def test_failed_full_replacement_and_rename_keep_original_then_repair(tmp_path, monkeypatch):
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="history.jsonl",
    ))
    session = await runtime.new_session(display_name="Original")
    await session.prompt("retained conversation")
    original = session.path.read_bytes()
    history = session.agent.history
    session.display_name = "Renamed"
    replace = os.replace
    error = OSError("replacement denied")

    def fail_replace(source, destination):
        if destination == session.path:
            assert source.parent == destination.parent
            assert decode_history(source.read_bytes()).history == history
            raise error
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError) as failed:
            await session.save()
        assert failed.value is error
        assert session.path.read_bytes() == original
        assert session.agent.history == history
        assert session.save_state == "unsaved"
        assert session.save_error is error
        await session.export("backup.jsonl")
        assert session.save_error is error
        assert decode_history((tmp_path / "backup.jsonl").read_bytes()).history == history
        with pytest.raises(RuntimeError, match="unsaved"):
            await runtime.prompt("blocked until repair")
    await session.save()
    assert session.save_state == "saved"
    assert session.save_error is None
    saved = decode_history(session.path.read_bytes())
    assert saved.history == history
    assert saved.display_name == "Renamed"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.jsonl", "history.jsonl"]
    await runtime.close()


async def test_failed_rebind_keeps_old_lock_and_releases_new_target(tmp_path, monkeypatch):
    from test_writer_locks import assert_locked, stop, writer

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="old.jsonl",
    ))
    session = await runtime.new_session()
    await session.prompt("complete memory")
    old_path = session.path
    old_bytes = old_path.read_bytes()
    target = tmp_path / "new.jsonl"
    await session.export(target)
    target_bytes = target.read_bytes()
    error = OSError("replace failed")
    replace = os.replace

    def fail_replace(source, destination):
        if destination == target:
            assert_locked(old_path)
            assert_locked(target)
            raise error
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError):
            await session.save(target)
    assert session.path == old_path
    assert session.save_error is error
    assert old_path.read_bytes() == old_bytes
    assert target.read_bytes() == target_bytes
    assert_locked(old_path)
    contender = writer(target)
    try:
        assert contender.stdout.readline().strip() == "bound"
    finally:
        stop(contender)
    await session.save(target)
    assert session.path == target
    assert session.save_state == "saved"
    assert_locked(target)
    contender = writer(old_path)
    try:
        assert contender.stdout.readline().strip() == "bound"
    finally:
        stop(contender)
    await runtime.close()
