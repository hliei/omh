import os
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from omh.agent import (
    Agent,
    AgentHistory,
    AgentInitialState,
    AgentOptions,
    MessageHistoryEntry,
    validate_history,
)
from omh.llm.types import UserMessage
from support import OfflineStream, model

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentOptions,
    SessionDirectory,
    SessionManager,
    decode_history,
)


async def test_jsonl_reopens_all_branches_and_html_contains_only_selected_path(tmp_path):
    stamp = datetime(2026, 10, 6, tzinfo=UTC)
    root = MessageHistoryEntry(id="root", parent_id=None, timestamp=stamp,
                               message=UserMessage(content="root <script>alert(1)</script>", timestamp=1))
    active = replace(root, id="active", parent_id="root",
                     message=UserMessage(content="selected conversation", timestamp=2))
    inactive = replace(root, id="inactive", parent_id="root",
                       message=UserMessage(content="inactive branch secret", timestamp=3))
    history = AgentHistory(conversation_id="tree", created_at=stamp,
                           entries=(root, active, inactive), leaf_id="active")
    manager = SessionManager(cwd=tmp_path, path=tmp_path / "bound.jsonl", display_name="<Title>")
    await manager.save(history)
    await manager.export(history, "backup.jsonl")
    decoded = decode_history((tmp_path / "backup.jsonl").read_bytes())
    validate_history(decoded.history)
    restored = Agent.from_history(decoded.history, AgentOptions(
        initial_state=AgentInitialState(model=model()), stream_fn=OfflineStream(),
    ))
    assert restored.history == history
    await restored.close()
    text = await manager.export(history, "conversation.html", format="html")
    assert (tmp_path / "conversation.html").read_text() == text
    assert "selected conversation" in text and "inactive branch secret" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "&lt;Title&gt;" in text and "<script" not in text
    assert 'src="http' not in text and 'href="http' not in text
    assert manager.path == tmp_path / "bound.jsonl" and manager.save_state == "saved"
    await manager.close()


async def test_persistent_name_clear_and_failed_rename_preserve_recovery(tmp_path, monkeypatch):
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="bound.jsonl",
    ))
    session = await runtime.new_session(display_name="Original")
    await session.prompt("retained conversation")
    await runtime.set_session_name("Renamed")
    assert decode_history(session.path.read_bytes()).display_name == "Renamed"
    await runtime.open_session(session.path)
    session = runtime.current_session
    assert session.display_name == "Renamed"
    await session.set_name(None)
    assert session.display_name is None
    assert SessionDirectory(tmp_path).list()[0].display_name is None
    original, history = session.path.read_bytes(), session.agent.history
    error = OSError("replacement denied")
    replace_file = os.replace

    def fail_replace(source, destination):
        if destination == session.path:
            raise error
        return replace_file(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError) as failed:
            await runtime.set_session_name("Recover this name")
        assert failed.value is error
        assert session.display_name == "Recover this name"
        assert session.path.read_bytes() == original and session.agent.history == history
        assert session.save_state == "unsaved" and session.save_error is error
        await runtime.export_session("recovery.html", format="html")
        await runtime.export_session("recovery.jsonl")
        assert session.save_state == "unsaved" and session.save_error is error
        assert session.path == tmp_path / "bound.jsonl"
        assert decode_history((tmp_path / "recovery.jsonl").read_bytes()).display_name == "Recover this name"
        with pytest.raises(OSError):
            await runtime.export_session(session.path, format="html")
        assert session.agent.history == history and session.path.read_bytes() == original
    # Even an HTML request to the bound destination is a complete JSONL save.
    repaired = await runtime.export_session(session.path, format="html")
    assert decode_history(repaired).display_name == "Recover this name"
    assert session.save_state == "saved" and session.save_error is None
    assert session.agent.history == history
    assert not list(tmp_path.glob(".bound.jsonl.*"))
    await runtime.close()


async def test_offline_html_renders_images_tools_and_structural_records(tmp_path):
    from html.parser import HTMLParser

    from test_history import complete_history

    from coding_agent import encode_history_html

    class Resources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sources = []
            self.links = []

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key == "src":
                    self.sources.append(value)
                if key == "href":
                    self.links.append(value)

    history = complete_history()
    manager = SessionManager(cwd=tmp_path)
    await manager.set_name(history, "Memory name")
    assert manager.save_state == "pending" and manager.path is None
    html = await manager.export(history, "rich.html", format="html")
    assert html == encode_history_html(history, cwd=str(tmp_path), display_name="Memory name")
    parser = Resources()
    parser.feed((tmp_path / "rich.html").read_text())
    assert parser.sources == ["data:image/png;base64,abc", "data:image/jpeg;base64,def", "data:image/png;base64,ghi"]
    assert parser.links == []
    for text in ("summarized original", "Thinking", "reasoning", "Tool: read", "snake_key",
                 "toolResult: read (error)", "old summary", "latest summary", "context_edit", "note"):
        assert text in html
    assert "inactive branch" not in html
    assert manager.save_state == "pending" and manager.path is None
    await manager.close()
