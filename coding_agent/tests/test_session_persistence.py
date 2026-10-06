"""Default auto-save activation, cwd grouping, memory mode and first activity."""

from __future__ import annotations

from pathlib import Path

import pytest
from omh.agent import AgentInitialState, AgentOptions, CustomAgentMessage
from omh.llm.types import UserMessage
from support import OfflineStream, model

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentHost,
    CodingAgentOptions,
    ConfigError,
    decode_history,
    encode_history,
)
from coding_agent.session_paths import (
    cwd_session_directory,
    session_file_name,
    session_file_path,
    sessions_root,
)


def make_host(startup_dir: Path, **kwargs: object) -> CodingAgentHost:
    return CodingAgentHost(
        startup_dir=startup_dir, agent_dir=startup_dir / "agent",
        api_key="temporary", **kwargs,
    )


def test_session_layout_keeps_cwd_groups_distinct() -> None:
    root = Path("/tmp/sessions")
    assert sessions_root("/tmp/agent") == Path("/tmp/agent/sessions")
    group = cwd_session_directory(root, "/work/my-project")
    assert group.parent == root
    assert group.name.startswith("--work-my-project--")
    # Different working directories never share one group directory, including
    # paths whose sanitized names would otherwise collide.
    assert cwd_session_directory(root, "/work/a") != cwd_session_directory(root, "/work/b")
    assert cwd_session_directory(root, "/work/a-b") != cwd_session_directory(root, "/work/a/b")


async def test_default_root_groups_by_cwd_and_uses_conversation_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    root = project / "agent" / "sessions"
    host = make_host(project)
    selection = host.select_new()
    assert host.session_root == root
    options = host.build_options(selection)
    assert options.session_dir == cwd_session_directory(root, project)
    assert options.session_file is None

    runtime = AgentSessionRuntime(options)
    first = await runtime.new_session()
    assert first.path is not None
    assert first.path.parent == cwd_session_directory(root, project)
    assert first.path.name.endswith(f"_{first.agent.history.conversation_id}.jsonl")
    assert first.save_mode == "auto"
    assert first.save_state == "pending"
    # An empty conversation leaves no file and no storage directory behind.
    assert not first.path.exists()
    assert not root.exists()

    await first.prompt("first user activity")
    assert first.path.exists()
    assert first.save_state == "saved"
    assert decode_history(first.path.read_bytes()).history == first.agent.history

    # Each conversation gets its own identity and file.
    second = await runtime.new_session()
    assert second.path is not None and second.path != first.path
    assert second.save_mode == "auto"


async def test_explicit_cwd_selects_its_own_group(tmp_path: Path) -> None:
    root = tmp_path / "agent" / "sessions"
    other = tmp_path / "other"
    other.mkdir()
    host = make_host(tmp_path)
    options = host.build_options(host.select_new(cwd=other))
    session = await AgentSessionRuntime(options).new_session()
    assert session.path is not None
    assert session.path.parent == cwd_session_directory(root, other)


async def test_session_dir_flag_overrides_the_default_root(tmp_path: Path) -> None:
    custom = tmp_path / "custom sessions"
    host = make_host(tmp_path, session_dir=custom)
    assert host.session_root == custom.resolve()
    session = await AgentSessionRuntime(host.build_options(host.select_new())).new_session()
    assert session.path is not None
    # The flag replaces the root and holds conversation files directly.
    assert session.path.parent == custom.resolve()
    await session.prompt("activity")
    assert session.path.exists()


async def test_no_session_is_observable_memory_mode_and_never_writes(tmp_path: Path) -> None:
    host = make_host(tmp_path, no_session=True)
    assert host.session_root is None
    options = host.build_options(host.select_new())
    assert options.session_dir is None and options.session_file is None

    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session()
    assert session.path is None
    assert session.save_mode == "memory"
    await session.prompt("memory conversation")
    assert session.save_state == "pending"
    assert not (tmp_path / "agent" / "sessions").exists()

    exported = await runtime.export_session()
    assert decode_history(exported).history == session.agent.history
    with pytest.raises(ValueError, match="path"):
        await session.save()
    # An explicit save before exit binds a new file and becomes automatic.
    bound = await session.save(tmp_path / "rescue" / "explicit.jsonl")
    assert session.save_mode == "auto"
    assert bound == tmp_path / "rescue" / "explicit.jsonl"
    assert decode_history(bound.read_bytes()).history == session.agent.history


async def test_no_session_does_not_modify_a_reopened_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    options = CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(),
        session_file=source,
    )
    original = await AgentSessionRuntime(options).new_session()
    await original.prompt("saved conversation")
    before = source.read_bytes()

    memory = make_host(tmp_path, no_session=True)
    memory_runtime = AgentSessionRuntime(memory.build_options(memory.select_new()))
    session = await memory_runtime.new_session()
    await session.prompt("memory only")
    assert session.path is None
    assert source.read_bytes() == before
    assert session.save_state == "pending"


async def test_custom_only_activity_creates_a_saveable_file(tmp_path: Path) -> None:
    stream = OfflineStream()
    root = tmp_path / "sessions"
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_dir=root)
    session = await AgentSessionRuntime(options).new_session()
    assert session.save_state == "pending"
    assert session.path is not None and not session.path.exists()

    await session.submit_custom_message(
        CustomAgentMessage(custom_type="bashExecution", content="! ls", display=True),
    )
    assert stream.requests == []
    assert session.save_state == "saved"
    assert session.path.exists()
    decoded = decode_history(session.path.read_bytes())
    assert decoded.history == session.agent.history
    assert decoded.history.entries[-1].type == "custom_message"

    await session.close()
    reopened = await AgentSessionRuntime(options).open_session(session.path)
    assert reopened.agent.history == session.agent.history
    assert not stream.requests


async def test_seed_and_setup_records_wait_for_activity_or_explicit_save(tmp_path: Path) -> None:
    options = CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(),
        session_dir=tmp_path / "sessions",
        agent_options=AgentOptions(initial_state=AgentInitialState(
            messages=[UserMessage(content="seed", timestamp=1)],
        )),
    )
    session = await AgentSessionRuntime(options).new_session()
    assert session.save_state == "pending"
    assert session.path is not None and not session.path.exists()

    saved = await session.save()
    assert saved == session.path
    assert session.save_state == "saved"
    assert decode_history(saved.read_bytes()).history == session.agent.history


async def test_default_path_is_stable_for_a_conversation(tmp_path: Path) -> None:
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(),
                                 session_dir="sessions")
    session = await AgentSessionRuntime(options).new_session()
    history = session.agent.history
    assert session.path == session_file_path(
        tmp_path / "sessions", history.conversation_id, history.created_at,
    )


async def test_default_root_reopens_complete_history(tmp_path: Path) -> None:
    from test_history import complete_history

    project = tmp_path / "project"
    project.mkdir()
    directory = cwd_session_directory(sessions_root(tmp_path / "agent"), project)
    history = complete_history()
    path = directory / session_file_name(history.conversation_id, history.created_at)
    path.parent.mkdir(parents=True)
    path.write_text(encode_history(history, cwd=str(project), display_name="all records"))

    stream = OfflineStream()
    options = CodingAgentOptions(cwd=project, model=model(), stream_fn=stream, tools=())
    session = await AgentSessionRuntime(options).open_session(path)
    assert session.agent.history == history
    assert session.save_mode == "auto"
    assert not stream.requests


async def test_default_root_open_repairs_an_unterminated_tail(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    host = make_host(project)
    options = host.build_options(host.select_new())
    session = await AgentSessionRuntime(options).new_session()
    await session.prompt("first")
    assert session.path is not None
    raw = session.path.read_bytes().rstrip(b"\n") + b"\n{broken"
    session.path.write_bytes(raw)

    await session.close()
    reopened = await AgentSessionRuntime(options).open_session(session.path)
    assert session.path.read_bytes() == raw + b"\n"
    assert reopened.agent.history == session.agent.history


def test_no_session_refuses_to_bind_a_session_file(tmp_path: Path) -> None:
    host = make_host(tmp_path, no_session=True)
    with pytest.raises(ConfigError, match="no-session"):
        host.build_options(host.select_new(), session_file=tmp_path / "bound.jsonl")
