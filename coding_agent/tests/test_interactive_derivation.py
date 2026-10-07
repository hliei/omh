"""Fork and clone through the installed command, PTY and saved history."""

from pathlib import Path

from test_interactive import sends
from test_interactive_sessions import start

from coding_agent import decode_history


def test_clone_keeps_source_draft_and_starts_an_empty_target_editor(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    try:
        ui.wait_for("phase input")
        ui.send(b"/name Source\r")
        ui.wait_for("name Source")
        source_path = next((home / "sessions").glob("*.jsonl"))
        ui.send(b"original\r")
        ui.wait_for("reply:original")
        ui.wait_for("phase input", after=ui.visible().index("reply:original"))
        original = source_path.read_bytes()
        source_id = decode_history(original).history.conversation_id
        ui.send(b"unsent source\x03/clone\r")
        ui.wait_for("cloned current")
        target_path = next(path for path in (home / "sessions").glob("*.jsonl") if path != source_path)
        target = decode_history(target_path.read_bytes())
        assert target.history.conversation_id != source_id
        assert target.source.conversation_id == source_id
        assert target.source.kind == "clone"
        assert target.history.entries == decode_history(original).history.entries
        assert source_path.read_bytes() == original
        assert "unsent source" not in target_path.read_text()
        ui.send(b"independent\r")
        ui.wait_for("reply:independent")
        ui.wait_for("phase input", after=ui.visible().index("reply:independent"))
        ui.send(f"/resume {source_id}\r".encode())
        ui.wait_for(f"resumed current {source_id}")
        ui.wait_for("> unsent source", after=ui.visible().index("resumed current"))
        ui.send(b"\x03/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert ui.restored()
        assert sends(home) == ["dialogue", "dialogue"]
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_fork_selector_refills_only_selected_user_text_and_images(tmp_path: Path) -> None:
    from support import png_bytes

    ui, home = start(tmp_path)
    image = tmp_path / "project" / "selected.png"
    image.write_bytes(png_bytes())
    try:
        ui.wait_for("phase input")
        ui.send(b"first turn\r")
        ui.wait_for("reply:first turn")
        ui.wait_for("phase input", after=ui.visible().index("reply:first turn"))
        ui.send(b"/attach selected.png\r")
        ui.wait_for("attached #1 selected.png")
        ui.send(b"change this\r")
        ui.wait_for("reply:change this")
        ui.wait_for("phase input", after=ui.visible().index("reply:change this"))
        image.unlink()  # The fork must use history bytes, not the original path.
        source_path = next((home / "sessions").glob("*.jsonl"))
        original = source_path.read_bytes()
        source = decode_history(original)
        ui.send(b"source draft\x03/fork\r")
        ui.wait_for("Select fork")
        ui.send(b"\x1b")
        ui.wait_for("fork selector canceled")
        ui.wait_for("> source draft")
        ui.send(b"\x03/fork\r")
        ui.wait_for("Select fork", after=ui.visible().index("fork selector canceled"))
        ui.send(b"\r")
        ui.wait_for("forked current")
        ui.wait_for("> change this", after=ui.visible().index("forked current"))
        ui.send(b"\x03/attach\r")
        ui.wait_for("origin=history")
        ui.send(b"changed input\r")
        ui.wait_for("reply:changed input")
        ui.wait_for("phase input", after=ui.visible().index("reply:changed input"))
        target_path = next(path for path in (home / "sessions").glob("*.jsonl") if path != source_path)
        target = decode_history(target_path.read_bytes())
        selected = next(entry for entry in source.history.entries
                        if getattr(getattr(entry, "message", None), "content", None)
                        and "change this" in str(entry))
        assert target.source.entry_id == selected.id
        assert target.source.leaf_id == selected.parent_id
        target_users = [entry.message for entry in target.history.entries
                        if getattr(getattr(entry, "message", None), "role", None) == "user"]
        assert "first turn" in str(target_users[0])
        assert "changed input" in str(target_users[1])
        assert "change this" not in str(target_users)
        assert target_users[1].content[1] == selected.message.content[1]
        assert source_path.read_bytes() == original
        ui.send(f"/resume {source.history.conversation_id}\r".encode())
        ui.wait_for("resumed current")
        ui.wait_for("> source draft", after=ui.visible().index("resumed current"))
        ui.send(b"\x03/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert sends(home) == ["dialogue", "dialogue", "dialogue"]
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_double_escape_opens_fork_and_cancel_keeps_pending_image(tmp_path: Path) -> None:
    from support import png_bytes

    ui, home = start(tmp_path)
    (tmp_path / "project" / "pending.png").write_bytes(png_bytes())
    try:
        ui.wait_for("phase input")
        ui.send(b"recorded user\r")
        ui.wait_for("reply:recorded user")
        ui.wait_for("phase input", after=ui.visible().index("reply:recorded user"))
        source = next((home / "sessions").glob("*.jsonl"))
        original = source.read_bytes()
        ui.send(b"/attach pending.png\r")
        ui.wait_for("attached #1 pending.png")
        ui.send(b"\x1b\x1b")
        ui.wait_for("Select fork")
        ui.send(b"\x1b")
        ui.wait_for("fork selector canceled")
        ui.send(b"/attach\r")
        ui.wait_for("#1 pending.png 6x4 image/png ready origin=file")
        assert source.read_bytes() == original
        ui.send(b"/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert sends(home) == ["dialogue"]
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_busy_derivation_refuses_and_idle_clone_recalls_source_queues(tmp_path: Path) -> None:
    from support import png_bytes

    ui, home = start(tmp_path, scenario="queued-end")
    (tmp_path / "project" / "queue.png").write_bytes(png_bytes())
    try:
        ui.wait_for("phase input")
        ui.send(b"go\r")
        ui.wait_for("phase tool bash")
        for command in (b"/clone\r", b"/fork\r"):
            offset = len(ui.visible())
            ui.send(command)
            ui.wait_for("is busy; wait or cancel", after=offset)
        ui.send(b"/attach queue.png\r")
        ui.wait_for("attached #1 queue.png")
        offset = len(ui.visible())
        ui.send(b"steering text\rfollow text\x1b\r")
        ui.wait_for("queued follow-up")
        ui.wait_for("phase input", after=offset)
        source_path = next((home / "sessions").glob("*.jsonl"))
        source = decode_history(source_path.read_bytes())
        original = source_path.read_bytes()
        ui.send(b"current draft\x03/clone\r")
        ui.wait_for("cloned current")
        ui.send(b"/attach\r")
        ui.wait_for("attachments none pending")
        target = next(path for path in (home / "sessions").glob("*.jsonl") if path != source_path)
        assert "steering text" not in target.read_text()
        assert "follow text" not in target.read_text()
        assert "image/png" not in target.read_text()
        ui.send(f"/resume {source.history.conversation_id}\r".encode())
        ui.wait_for("resumed current")
        offset = ui.visible().index("resumed current")
        for text in ("steering text", "follow text", "current draft"):
            ui.wait_for(text, after=offset)
        ui.send(b"\x03/attach\r")
        ui.wait_for("#1 queue.png 6x4 image/png ready origin=file")
        assert source_path.read_bytes() == original
        assert (tmp_path / "project" / "first.txt").exists()
        assert (tmp_path / "project" / "second.txt").exists()
        ui.send(b"/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert sends(home) == ["dialogue"]
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_shell_busy_rejects_fork_and_clone_without_implicit_cancel(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    try:
        ui.wait_for("phase input")
        ui.send(b"!printf shell-ready; while [ ! -f release-shell ]; do sleep 0.02; done; printf done > effect.txt\r")
        ui.wait_for("shell-ready")
        ui.wait_for("| shell running")
        for command in (b"/clone\r", b"/fork\r"):
            offset = len(ui.visible())
            ui.send(command)
            ui.wait_for("is busy; wait or cancel", after=offset)
        assert "shell cancel" not in ui.visible()
        assert "Select fork" not in ui.visible()
        (tmp_path / "project" / "release-shell").touch()
        ui.wait_for("shell completed")
        ui.wait_for("save saved", after=ui.visible().index("shell completed"))
        assert (tmp_path / "project" / "effect.txt").read_text() == "done"
        source = next((home / "sessions").glob("*.jsonl"))
        original = source.read_bytes()
        ui.send(b"/clone\r")
        ui.wait_for("cloned current")
        target = next(path for path in (home / "sessions").glob("*.jsonl") if path != source)
        assert "shell-ready" in target.read_text()
        assert source.read_bytes() == original
        assert (tmp_path / "project" / "effect.txt").read_text() == "done"
        ui.send(b"/quit\r")
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_cancelled_derivation_preparation_restores_source_editor_and_image(tmp_path: Path) -> None:
    from support import png_bytes

    ui, home = start(tmp_path, scenario="switch-prepare")
    (tmp_path / "project" / "draft.png").write_bytes(png_bytes())
    try:
        ui.wait_for("phase input")
        ui.send(b"original\r")
        ui.wait_for("reply:original")
        ui.wait_for("phase input", after=ui.visible().index("reply:original"))
        source = next((home / "sessions").glob("*.jsonl"))
        original = source.read_bytes()
        identity = decode_history(original).history.conversation_id
        ui.send(b"/attach draft.png\r")
        ui.wait_for("attached #1 draft.png")
        ui.send(b"source draft\x03/clone\r")
        ui.wait_for("selection preparation")
        ui.send(b"\x1b")
        ui.wait_for("session switch wait canceled")
        ui.wait_for("> source draft", after=ui.visible().index("session switch wait canceled"))
        ui.send(b"\x03/session\r/attach\r")
        ui.wait_for(f"current {identity}")
        ui.wait_for("#1 draft.png 6x4 image/png ready origin=file")
        assert source.read_bytes() == original
        assert len(list((home / "sessions").glob("*.jsonl"))) == 1
        ui.send(b"/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert sends(home) == ["dialogue"]
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_failed_target_save_displays_actual_current_and_allows_repair(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    blocked = tmp_path / "project" / "blocked"
    blocked.write_text("not a directory")
    try:
        ui.wait_for("phase input")
        ui.send(b"original\r")
        ui.wait_for("reply:original")
        ui.wait_for("phase input", after=ui.visible().index("reply:original"))
        source = next((home / "sessions").glob("*.jsonl"))
        original = source.read_bytes()
        ui.send(b"source draft\x03/clone --session-dir blocked\r")
        ui.wait_for("cloned current")
        ui.wait_for("save unsaved", after=ui.visible().index("cloned current"))
        ui.send(b"/session\r")
        ui.wait_for("source clone")
        ui.wait_for("retained 1")
        ui.send(b"must not send\r")
        ui.wait_for("New work is paused")
        assert sends(home) == ["dialogue"]
        ui.send(b"\x03/export jsonl rescue.jsonl\r")
        ui.wait_for("exported jsonl current")
        ui.send(b"/save repaired.jsonl\r")
        ui.wait_for("saved current")
        repaired = tmp_path / "project" / "repaired.jsonl"
        target = decode_history(repaired.read_bytes())
        assert target.history.entries == decode_history(original).history.entries
        assert target.source.path == str(source)
        assert source.read_bytes() == original
        ui.send(b"/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert ui.restored()
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_fork_from_saved_compacted_edited_path_preserves_complete_ancestors(tmp_path: Path) -> None:
    from dataclasses import replace

    from omh.agent import (
        Agent,
        AgentInitialState,
        AgentOptions,
        ModelChangeHistoryEntry,
        validate_history,
    )
    from omh.llm.types import AssistantMessage
    from support import OfflineStream, model
    from test_history import complete_history

    from coding_agent import encode_history

    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "rich.jsonl"
    history = complete_history()
    records = []
    for entry in history.entries:
        if isinstance(entry, ModelChangeHistoryEntry):
            entry = replace(entry, provider="opencode-go", model_id="deepseek-v4.1-flash")
        elif isinstance(getattr(entry, "message", None), AssistantMessage):
            entry = replace(entry, message=replace(entry.message, provider="opencode-go", model="deepseek-v4.1-flash"))
        records.append(entry)
    history = replace(history, entries=tuple(records))
    source.write_text(encode_history(history, cwd=str(project)))
    original = source.read_bytes()
    ui, home = start(tmp_path, "--session", str(source))
    try:
        ui.wait_for("phase input")
        ui.send(b"/fork last\r")
        ui.wait_for("forked current")
        ui.wait_for("> after checkpoint", after=ui.visible().index("forked current"))
        # Inherit the actual source directory, rather than the startup default.
        target_path = next(path for path in tmp_path.glob("*.jsonl") if path != source)
        target = decode_history(target_path.read_bytes())
        assert target.history.entries == history.entries[:-2]
        assert target.history.leaf_id == "delta"
        validate_history(target.history)
        restored = Agent.from_history(target.history, AgentOptions(
            stream_fn=OfflineStream(), initial_state=AgentInitialState(model=model()),
        ))
        projection = str(restored.state.messages)
        assert "latest summary" in projection
        assert "replacement" in projection
        assert "inactive branch" not in projection
        assert "summarized original" not in projection
        assert source.read_bytes() == original
        ui.send(b"\x03/quit\r")
        ui.wait_for("unsubmitted content remains")
        ui.send(b"\r")
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_explicit_target_modes_and_cwd_keep_history_and_reopenable_origin(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    target_cwd = tmp_path / "experiment"
    target_cwd.mkdir()
    try:
        ui.wait_for("phase input")
        ui.send(b"original\r")
        ui.wait_for("reply:original")
        ui.wait_for("phase input", after=ui.visible().index("reply:original"))
        source_path = next((home / "sessions").glob("*.jsonl"))
        original = source_path.read_bytes()
        ui.send(f"/clone --cwd {target_cwd} --save-mode memory\r".encode())
        ui.wait_for("cloned current")
        ui.wait_for("mode memory", after=ui.visible().index("cloned current"))
        ui.wait_for(f"cwd {target_cwd.resolve()}", after=ui.visible().index("cloned current"))
        assert len(list((home / "sessions").glob("*.jsonl"))) == 1
        offset = len(ui.visible())
        ui.send(b"/clone --save-mode auto --session-dir derived\r")
        ui.wait_for("cloned current", after=offset)
        ui.wait_for(f"path {target_cwd.resolve() / 'derived'}", after=offset)
        target = next((target_cwd / "derived").glob("*.jsonl"))
        decoded = decode_history(target.read_bytes())
        assert decoded.history.entries == decode_history(original).history.entries
        assert decoded.cwd == str(target_cwd.resolve())
        assert decoded.source.path is None
        assert decoded.source.kind == "clone"
        assert source_path.read_bytes() == original
        offset = len(ui.visible())
        ui.send(b"/clone --save-mode memory --session-dir forbidden\r")
        ui.wait_for("Memory mode cannot specify --session-dir", after=offset)
        assert not (target_cwd / "forbidden").exists()
        ui.send(b"/quit\r")
        assert ui.finish() == 0
        assert ui.restored()
        assert sends(home) == ["dialogue"]
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()
