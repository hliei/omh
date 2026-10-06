import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from omh.agent import AgentHistory, MessageHistoryEntry
from omh.llm.types import UserMessage

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentHost,
    SessionDirectory,
    SessionLookupError,
    encode_history,
)


def write_history(path, cwd, identity, name, text, *, timestamp=None):
    stamp = timestamp or datetime(2026, 10, 6, tzinfo=UTC)
    history = AgentHistory(
        conversation_id=identity, created_at=stamp, leaf_id="user",
        entries=(MessageHistoryEntry(
            id="user", parent_id=None, timestamp=stamp,
            message=UserMessage(content=text, timestamp=1),
        ),),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode_history(history, cwd=str(cwd), display_name=name))
    return history


def test_discovery_uses_header_cwd_and_searches_metadata_and_messages(tmp_path: Path):
    root = tmp_path / "sessions"
    project = tmp_path / "project"
    other = tmp_path / "other"
    first = root / "arbitrary-group" / "a.jsonl"
    second = root / "b.jsonl"
    write_history(first, project, "abc-one", "Fix parser", "needle in conversation",
                  timestamp=datetime(2024, 2, 3, tzinfo=UTC))
    write_history(second, other, "def-two", None, "other message")
    (root / "foreign.jsonl").write_text('{"type":"session","version":3}\n')
    directory = SessionDirectory(root)
    entries = directory.list(cwd=project)
    assert [(item.path, item.conversation_id, item.display_name, item.cwd)
            for item in entries] == [(first.resolve(), "abc-one", "Fix parser", project)]
    assert len(directory.list()) == 2
    for search in ("PARSER", "abc-", str(project), "2024-02-03", "needle"):
        assert [item.path for item in directory.list(search=search)] == [first.resolve()]
    assert directory.list(search="absent") == ()
    assert any("foreign.jsonl" in diagnostic for diagnostic in directory.diagnostics)


def test_recent_sorting_unique_ids_and_external_paths(tmp_path):
    root = tmp_path / "sessions"
    project = tmp_path / "project"
    a, b, c = (root / name for name in ("a.jsonl", "b.jsonl", "c.jsonl"))
    write_history(a, project, "abc-one", "Zulu", "older")
    write_history(b, project, "abc-two", "Alpha", "newer")
    write_history(c, tmp_path / "other", "def-one", "Beta", "other")
    for path, mtime in ((a, 10), (b, 20), (c, 30)):
        os.utime(path, (mtime, mtime))
    directory = SessionDirectory(root)
    assert directory.recent(project) == b.resolve()
    assert directory.recent(tmp_path / "missing") is None
    assert [item.path for item in directory.list(sort="mtime")] == [c, b, a]
    assert [item.path for item in directory.list(sort="mtime", reverse=False)] == [a, b, c]
    assert [item.display_name for item in directory.list(sort="name")] == ["Alpha", "Beta", "Zulu"]
    assert directory.resolve("abc-one") == a
    assert directory.resolve("abc-t") == b
    with pytest.raises(SessionLookupError, match="Ambiguous") as ambiguous:
        directory.resolve("abc")
    assert [item.path for item in ambiguous.value.candidates] == [b, a]
    assert str(a) in str(ambiguous.value) and str(b) in str(ambiguous.value)
    with pytest.raises(SessionLookupError, match="No session"):
        directory.resolve("absent")
    external = tmp_path / "external.jsonl"
    write_history(external, project, "outside", None, "external")
    assert directory.resolve(external) == external
    assert directory.resolve("external.jsonl", cwd=tmp_path) == external
    external.write_text('{"type":"session","version":3}\n')
    with pytest.raises(ValueError, match="format"):
        directory.resolve(external)
    assert not (tmp_path / "missing-root").exists()
    assert SessionDirectory(tmp_path / "missing-root").list() == ()


async def test_common_host_continues_and_reopens_without_forking(tmp_path):
    project, other = tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    root = tmp_path / "sessions"
    a, b = root / "a.jsonl", root / "b.jsonl"
    original = write_history(a, project, "abc-one", "Original", "project history")
    write_history(b, other, "abc-two", None, "other history")
    os.utime(a, (10, 10))
    os.utime(b, (20, 20))
    host = CodingAgentHost(startup_dir=other, agent_dir=tmp_path / "agent", session_dir=root)
    selected = host.select_continue(cwd=project)
    assert selected.ready and selected.session_path == a
    selected = host.select_open("abc-o")
    assert selected.ready and selected.cwd == project
    assert selected.history.history == original
    assert host.build_options(selected).session_file == a
    runtime = AgentSessionRuntime(host.build_options(selected))
    reopened = await runtime.open_session(selected.session_path)
    assert reopened.cwd == project and reopened.path == a
    assert reopened.agent.history.conversation_id == "abc-one"
    await runtime.close()
    override = host.select_open("abc-o", cwd=other)
    assert override.ready and override.cwd == other and override.session_path == a
    ambiguous = host.select_open("abc")
    assert not ambiguous.ready
    assert any(str(a) in d.message and str(b) in d.message for d in ambiguous.diagnostics)
    empty = host.select_continue(cwd=tmp_path)
    assert empty.ready and empty.history is None and empty.session_path is None
    assert any("new session" in d.message for d in empty.diagnostics)
    assert not (tmp_path / "agent").exists()
