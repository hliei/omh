import pytest
from omh.agent import Agent, AgentInitialState, AgentOptions
from support import OfflineStream, model

from coding_agent import AgentSession, SessionManager, decode_history, encode_history


async def test_manager_saves_supplied_snapshots_and_appends_only_new_records(tmp_path):
    agent = Agent(AgentOptions(initial_state=AgentInitialState(model=model()), stream_fn=OfflineStream()))
    path = tmp_path / "history.jsonl"
    manager = SessionManager(cwd=tmp_path, path=path, display_name="Snapshot consumer")
    initial = agent.history
    await manager.commit(initial, initial.entries)
    assert manager.save_state == "pending"
    assert not path.exists()

    await agent.prompt("first")
    first = agent.history
    await manager.commit(first, first.entries)
    first_bytes = path.read_bytes()
    assert decode_history(first_bytes).history == first

    await agent.prompt("second")
    second = agent.history
    # A full save or loaded snapshot can already include some notified records.
    await manager.commit(second, second.entries)
    assert path.read_bytes().startswith(first_bytes)
    assert decode_history(path.read_bytes()).history == second
    assert manager.save_state == "saved"
    assert manager.save_error is None
    assert decode_history(path.read_bytes()).display_name == "Snapshot consumer"


async def test_manager_load_defers_tail_repair_until_host_preparation(tmp_path):
    agent = Agent(AgentOptions(initial_state=AgentInitialState(model=model()), stream_fn=OfflineStream()))
    await agent.prompt("saved conversation")
    path = tmp_path / "history.jsonl"
    raw = encode_history(agent.history, cwd=str(tmp_path), display_name="Loaded").encode() + b"{broken"
    path.write_bytes(raw)

    manager, decoded = SessionManager.load(path)
    assert path.read_bytes() == raw
    assert decoded.history == agent.history
    assert manager.cwd == tmp_path
    assert manager.path == path
    assert manager.display_name == "Loaded"
    assert manager.save_state == "saved"
    await manager.prepare_append()
    await manager.prepare_append()
    assert path.read_bytes() == raw + b"\n"

    restored = Agent.from_history(decoded.history, AgentOptions(
        initial_state=AgentInitialState(model=model()), stream_fn=OfflineStream(),
    ))
    await restored.prompt("next")
    await manager.commit(restored.history, restored.history.entries)
    assert path.read_bytes().startswith(raw + b"\n")
    assert decode_history(path.read_bytes()).history == restored.history


async def test_manager_separate_export_preserves_failure_and_full_save_repairs(tmp_path):
    agent = Agent(AgentOptions(initial_state=AgentInitialState(model=model()), stream_fn=OfflineStream()))
    await agent.prompt("retained history")
    history = agent.history
    occupied = tmp_path / "occupied.jsonl"
    occupied.write_text("existing file")
    manager = SessionManager(cwd=tmp_path, path=occupied)
    with pytest.raises(FileExistsError) as failure:
        await manager.commit(history, history.entries)
    assert manager.save_state == "unsaved"
    assert manager.save_error is failure.value
    assert occupied.read_text() == "existing file"

    exported = await manager.export(history, "copy.jsonl")
    assert decode_history(exported).history == history
    assert manager.path == occupied
    assert manager.save_state == "unsaved"
    assert manager.save_error is failure.value
    with pytest.raises(FileExistsError) as repeated:
        await manager.commit(history, history.entries)
    assert repeated.value is failure.value

    await manager.export(history, occupied)
    assert manager.save_state == "saved"
    assert manager.save_error is None
    assert decode_history(occupied.read_bytes()).history == history
    await agent.prompt("after repair")
    await manager.commit(agent.history, agent.history.entries)
    assert decode_history(occupied.read_bytes()).history == agent.history


async def test_direct_session_composition_delegates_metadata_and_saves_after_close(tmp_path):
    agent = Agent(AgentOptions(initial_state=AgentInitialState(model=model()), stream_fn=OfflineStream()))
    manager = SessionManager(cwd=tmp_path)
    session = AgentSession(agent, session_manager=manager)
    session.display_name = "Direct composition"
    session.cwd = tmp_path / "work"
    session.cwd.mkdir()
    session.path = session.cwd / "bound.jsonl"
    assert manager.display_name == session.display_name
    assert manager.cwd == session.cwd
    assert manager.path == session.path
    await session.prompt("first")
    assert session.save_state == manager.save_state == "saved"
    await session.close()
    replacement = await session.save("replacement.jsonl")
    assert replacement == manager.path == session.cwd / "replacement.jsonl"
    assert session.save_error is manager.save_error is None
    decoded = decode_history(await session.export())
    assert decoded.history == agent.history
    assert decoded.cwd == str(session.cwd)
    assert decoded.display_name == session.display_name
