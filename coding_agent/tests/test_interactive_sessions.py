"""Session management and recovery through the installed command and PTY."""

import signal
from pathlib import Path

import pytest
from support import png_bytes
from test_cli import run_cli
from test_interactive import InteractiveSession, provider_env, sends
from test_session_directory import write_history

from coding_agent.history import decode_history


def start(tmp_path: Path, *args: str, scenario: str = "echo") -> tuple[InteractiveSession, Path]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), *args,
        home=home, cwd=project, env=provider_env(home, scenario),
    ), home


def test_names_save_export_new_and_resume_keep_identity(tmp_path: Path) -> None:
    session, home = start(tmp_path)
    try:
        session.wait_for("phase input")
        session.send(b"/name First conversation\r")
        session.wait_for("name First conversation")
        saved = next((home / "sessions").glob("*.jsonl"))
        identity = decode_history(saved.read_bytes()).history.conversation_id
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        session.wait_for("phase input", after=session.visible().index("reply:hello"))
        session.send(b"/export html reading.html\r")
        session.wait_for("exported html")
        assert "reply:hello" in (tmp_path / "project" / "reading.html").read_text()
        session.send(b"/new\r")
        session.wait_for("new current")
        session.send(f"/resume {identity[:12]}\r".encode())
        session.wait_for(f"resumed current {identity}")
        session.send(b"/name --clear\r")
        session.wait_for("name (unnamed)")
        assert decode_history(saved.read_bytes()).display_name is None
        session.send(b"/save\r")
        session.wait_for("saved current")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert session.restored()
        assert sends(home) == ["dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


@pytest.mark.parametrize("scenario", ["handoff", "handoff-error"])
def test_cancelled_handoff_shows_published_current_and_rebinds_events(tmp_path: Path, scenario: str) -> None:
    session, home = start(tmp_path, scenario=scenario)
    try:
        session.wait_for("phase input")
        session.send(b"/name Old\r")
        session.wait_for("name Old")
        identity = decode_history(next((home / "sessions").glob("*.jsonl")).read_bytes()).history.conversation_id
        session.send(b"/new\r")
        session.wait_for("handoff closing")
        session.send(b"\x1b")
        session.wait_for("session switch wait canceled")
        # A second Escape cannot cancel the observer that synchronizes the
        # screen with Runtime's already-owned publication.
        session.send(b"\x1b")
        session.wait_for("handoff continues")
        session.send(b"/session\r")
        session.wait_for("Recovery:")
        (home / "release-handoff").touch()
        session.wait_for("new current")
        offset = len(session.visible())
        session.send(b"/session\r")
        session.wait_for(f"retained 1 {identity}", after=offset)
        if scenario == "handoff-error":
            session.wait_for("terminal notification failed")
        session.send(b"after switch\r")
        session.wait_for("reply:after switch")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert session.restored()
        assert sends(home) == ["dialogue", "dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_selector_cancels_and_restores_source_text_and_image_after_resume(tmp_path: Path) -> None:
    session, home = start(tmp_path)
    (tmp_path / "project" / "shot.png").write_bytes(png_bytes())
    try:
        session.wait_for("phase input")
        session.send(b"/name Source\r")
        session.wait_for("name Source")
        saved = next((home / "sessions").glob("*.jsonl"))
        identity = decode_history(saved.read_bytes()).history.conversation_id
        session.send(b"/attach shot.png\r")
        session.wait_for("attached #1 shot.png")
        session.send(b"unsent text\x03/resume\r")
        session.wait_for("Select session")
        session.send(b"\x1b")
        session.wait_for("session selector canceled")
        session.wait_for("> unsent text")
        session.send(b"\x03/new\r")
        session.wait_for("new current")
        session.send(b"/attach\r")
        session.wait_for("attachments none pending")
        session.send("/resume --search Source --sort name\r".encode())
        session.wait_for("Select session", after=session.visible().index("new current"))
        session.send(b"\r")
        session.wait_for(f"resumed current {identity}")
        session.wait_for("> unsent text", after=session.visible().index("resumed current"))
        # The image is still a draft, not in the source file or a target queue.
        assert "unsent text" not in saved.read_text()
        assert "image/png" not in saved.read_text()
        session.send(b"\x03/attach\r")
        session.wait_for("#1 shot.png 6x4 image/png ready origin=file")
        session.send(b"/quit\r")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
        assert sends(home) == []
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_retained_save_failure_stays_recoverable_and_blocks_exit(tmp_path: Path) -> None:
    # A real disk obstacle fails first saving, without altering the provider or
    # patching the saving implementation. The in-memory user entry is retained.
    home = tmp_path / "home"
    home.mkdir()
    (home / "sessions").write_text("not a directory")
    session, home = start(tmp_path)
    try:
        session.wait_for("phase input")
        session.send(b"keep this history\r")
        session.wait_for("save unsaved")
        session.send(b"!printf must-not-run > forbidden.txt\r")
        session.wait_for("New work is paused")
        session.send(b"\x03/new\r")
        session.wait_for("new current")
        session.send(b"/session\r")
        session.wait_for("retained 1")
        session.send(b"/quit\r")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        session.wait_for("exit blocked")
        assert session.process.poll() is None
        assert not (tmp_path / "project" / "forbidden.txt").exists()
        session.send(b"/export --retained 1 jsonl backup.jsonl\r")
        session.wait_for("exported jsonl retained 1")
        backup = tmp_path / "project" / "backup.jsonl"
        decoded = decode_history(backup.read_bytes())
        assert "keep this history" in backup.read_text()
        assert decoded.history.entries
        # Backup does not repair the original file or its error.
        offset = len(session.visible())
        session.send(b"/session\r")
        session.wait_for("save unsaved", after=offset)
        session.send(b"/save --retained 1 repaired.jsonl\r")
        session.wait_for("saved retained 1")
        repaired = tmp_path / "project" / "repaired.jsonl"
        assert decode_history(repaired.read_bytes()).history == decoded.history
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert session.restored()
        assert sends(home) == []
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_prepare_cancel_keeps_source_current_and_usable(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="switch-prepare")
    try:
        session.wait_for("phase input")
        session.send(b"/name Source\r")
        session.wait_for("name Source")
        saved = next((home / "sessions").glob("*.jsonl"))
        identity = decode_history(saved.read_bytes()).history.conversation_id
        session.send(b"/new\r")
        session.wait_for("selection preparation")
        session.send(b"\x1b")
        session.wait_for("session switch wait canceled")
        offset = len(session.visible())
        session.send(b"/session\r")
        session.wait_for(f"current {identity}", after=offset)
        session.send(b"still here\r")
        session.wait_for("reply:still here")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert sends(home) == ["dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_readonly_lists_and_selector_choose_all_projects_without_cwd_override(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    write_history(home / "sessions" / "a.jsonl", project, "abc-one", "Zulu", "local needle")
    target = home / "sessions" / "b.jsonl"
    write_history(target, other, "abc-two", "Alpha", "remote needle")
    session, _ = start(tmp_path, "--resume")
    try:
        session.wait_for("Select session (current project")
        session.wait_for("Zulu abc-one")
        session.send(b"\t")
        session.wait_for("Select session (all projects")
        session.wait_for("Alpha abc-two")
        session.send(b"remote needle")
        # Search matches message contents, not just the name shown in the row.
        offset = len(session.visible())
        session.send(b"\r")
        session.wait_for("resumed current abc-two", after=offset)
        session.wait_for(f"cwd {other.resolve()}", after=offset)
        session.send(b"/resume --list --all-projects --sort name\r")
        session.wait_for("Sessions (all projects, sort name)")
        session.send(b"/resume abc\r")
        session.wait_for("Ambiguous session")
        session.wait_for("abc-one")
        session.wait_for("abc-two")
        session.send(b"continued\r")
        session.wait_for("reply:continued")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert "continued" in target.read_text()
        assert sends(home) == ["dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_busy_management_refuses_without_abort_and_switch_recalls_complete_queues(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="queued-end")
    (tmp_path / "project" / "shot.png").write_bytes(png_bytes())
    try:
        session.wait_for("phase input")
        session.send(b"/name Source\r")
        session.wait_for("name Source")
        saved = next((home / "sessions").glob("*.jsonl"))
        identity = decode_history(saved.read_bytes()).history.conversation_id
        session.send(b"go\r")
        session.wait_for("phase tool bash")
        session.send(b"/new\r")
        session.wait_for("is busy; wait or cancel")
        session.send(b"/resume --list\r")
        session.wait_for("Sessions (current project")
        session.send(b"/attach shot.png\r")
        session.wait_for("attached #1 shot.png")
        session.send(b"steer one\rsteer two\rfollow one\x1b\rfollow two\x1b\r")
        session.wait_for("queued follow-up (2 waiting): follow two")
        session.wait_for("phase input", after=session.visible().index("queued follow-up (2 waiting)"))
        session.send(b"source remainder\x03/new\r")
        session.wait_for("new current")
        assert (tmp_path / "project" / "first.txt").read_text() == "a"
        assert (tmp_path / "project" / "second.txt").read_text() == "b"
        session.send(f"/resume {identity}\r".encode())
        session.wait_for(f"resumed current {identity}")
        offset = session.visible().index("resumed current")
        for text in ("> steer one", "  steer two", "  follow one", "  follow two", "  source remainder"):
            session.wait_for(text, after=offset)
        # Every still-unconsumed input belongs to the source editor, not the new
        # Agent, and drafts/images never become committed history by switching.
        assert "steer one" not in saved.read_text()
        assert "follow two" not in saved.read_text()
        assert "image/png" not in saved.read_text()
        assert sends(home) == ["dialogue"]
        session.send(b"\x03/attach\r")
        session.wait_for("#1 shot.png")
        session.send(b"/quit\r")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_running_shell_refuses_replacement_and_readonly_info_stays_available(tmp_path: Path) -> None:
    session, home = start(tmp_path)
    try:
        session.wait_for("phase input")
        session.send(b"!printf shell-gate; while [ ! -f release ]; do sleep 0.02; done; printf done\r")
        session.wait_for("| shell running")
        session.send(b"/new\r")
        session.wait_for("is busy; wait or cancel")
        session.send(b"/resume --list\r/session\r")
        session.wait_for("Sessions (current project")
        session.wait_for("Recovery:")
        (tmp_path / "project" / "release").touch()
        session.wait_for("shell settled")
        session.send(b"/new\r")
        session.wait_for("new current")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert sends(home) == []
        assert "shell-gate" in next((home / "sessions").glob("*.jsonl")).read_text()
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


@pytest.mark.parametrize("exit_signal", [None, signal.SIGTERM])
def test_complete_backup_allows_exit_without_pretending_to_repair(tmp_path: Path, exit_signal: int | None) -> None:
    session, home = start(tmp_path)
    try:
        session.wait_for("phase input")
        (home / "sessions").write_text("blocks saving")
        session.send(b"!printf preserved-output\r")
        session.wait_for("save unsaved")
        if exit_signal is None:
            session.send(b"/quit\r")
        else:
            session.process.send_signal(exit_signal)
        session.wait_for("exit blocked")
        session.send(b"/export html partial.html\r")
        session.wait_for("exported html")
        offset = len(session.visible())
        session.send(b"/quit\r")
        session.wait_for("exit blocked", after=offset)
        session.send(b"/export jsonl full.jsonl\r")
        session.wait_for("exported jsonl current")
        assert "preserved-output" in (tmp_path / "project" / "full.jsonl").read_text()
        session.send(b"/quit\r")
        assert session.finish() == (0 if exit_signal is None else 128 + exit_signal)
        assert session.restored()
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_current_repair_restores_work_after_failed_repair_and_export(tmp_path: Path) -> None:
    session, home = start(tmp_path)
    try:
        session.wait_for("phase input")
        blocked = home / "sessions"
        blocked.write_text("blocks saving")
        session.send(b"preserved input\r")
        session.wait_for("save unsaved")
        session.send(b"/save\r")
        session.wait_for("notice save current")
        session.send(f"/export jsonl {blocked}/backup.jsonl\r".encode())
        session.wait_for("notice export current")
        session.send(b"cannot run yet\r")
        session.wait_for("> cannot run yet")
        assert sends(home) == []
        blocked.unlink()
        session.send(b"\x03/save\r")
        session.wait_for("saved current")
        session.send(b"after repair\r")
        session.wait_for("reply:after repair")
        assert "preserved input" in next(blocked.glob("*.jsonl")).read_text()
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert sends(home) == ["dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_exit_checks_every_retained_error_and_cancel_revokes_discard(tmp_path: Path) -> None:
    session, home = start(tmp_path)
    (tmp_path / "project" / "shot.png").write_bytes(png_bytes())
    try:
        session.wait_for("phase input")
        (home / "sessions").write_text("blocks saving")
        session.send(b"first retained input\r")
        session.wait_for("save unsaved")
        session.send(b"/new\r")
        session.wait_for("new current")
        offset = len(session.visible())
        session.send(b"second retained input\r")
        session.wait_for("save unsaved", after=offset)
        session.send(b"/new\r")
        session.wait_for("retained unsaved 2")
        session.send(b"/export --retained 1 jsonl one.jsonl\r")
        session.wait_for("exported jsonl retained 1")
        session.send(b"/attach shot.png\r")
        session.wait_for("attached #1 shot.png")
        session.send(b"/quit --discard-unsaved\r")
        session.wait_for("unsubmitted content remains")
        session.send(b"x")
        session.wait_for("exit canceled")
        # Cancelling this decision also revokes the history discard. A normal
        # quit must still be blocked by the second unbacked retained history.
        offset = len(session.visible())
        session.send(b"/quit\r")
        session.wait_for("unsubmitted content remains", after=offset)
        session.send(b"\r")
        session.wait_for("exit blocked: unsaved history in retained 2")
        assert session.process.poll() is None
        session.send(b"/export --retained 2 jsonl two.jsonl\r")
        session.wait_for("exported jsonl retained 2")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert "first retained input" in (tmp_path / "project" / "one.jsonl").read_text()
        assert "second retained input" in (tmp_path / "project" / "two.jsonl").read_text()
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_quit_stops_admission_before_processing_more_terminal_bytes(tmp_path: Path) -> None:
    session, home = start(tmp_path)
    try:
        session.wait_for("phase input")
        session.send(b"/quit\rnew prompt\r!touch forbidden.txt\r")
        assert session.finish() == 0
        assert sends(home) == []
        assert not (tmp_path / "project" / "forbidden.txt").exists()
        assert session.restored()
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_print_and_interactive_continue_the_same_saved_conversation(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    result = run_cli(
        "--print", "--no-approve", "--no-context-files", "--api-key", "offline",
        "--cwd", str(project), "--session-dir", str(home / "sessions"), "before",
        home=home, env=provider_env(home, "echo"),
    )
    assert result.returncode == 0 and "reply:before" in result.stdout
    saved = next((home / "sessions").glob("*.jsonl"))
    original = decode_history(saved.read_bytes()).history
    session, _ = start(tmp_path, "--session", str(saved))
    try:
        session.wait_for("reply:before")
        session.send(b"during\r")
        session.wait_for("reply:during")
        session.send(b"/quit\r")
        assert session.finish() == 0
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()
    result = run_cli(
        "--print", "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(saved), "after", home=home, env=provider_env(home, "echo"),
    )
    assert result.returncode == 0 and "reply:after" in result.stdout
    restored = decode_history(saved.read_bytes()).history
    assert restored.conversation_id == original.conversation_id
    assert restored.entries[:len(original.entries)] == original.entries
    assert all(text in saved.read_text() for text in ("reply:before", "reply:during", "reply:after"))
    assert sends(home) == ["dialogue", "dialogue", "dialogue"]


def test_handoff_refuses_async_draft_edits(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="handoff")
    (tmp_path / "project" / "shot.png").write_bytes(png_bytes())
    try:
        session.wait_for("phase input")
        session.send(b"/name Source\r")
        session.wait_for("name Source")
        saved = next((home / "sessions").glob("*.jsonl"))
        identity = decode_history(saved.read_bytes()).history.conversation_id
        session.send(b"/new\r")
        session.wait_for("handoff closing")
        for key in (b"/attach shot.png\r", b"\x16", b"\x07"):
            offset = len(session.visible())
            session.send(key)
            session.wait_for("session switch in progress; wait before editing the draft", after=offset)
        (home / "release-handoff").touch()
        session.wait_for("new current")
        session.send(b"/attach\r")
        session.wait_for("attachments none pending")
        session.send(f"/resume {identity}\r".encode())
        session.wait_for(f"resumed current {identity}")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert session.restored()
        assert sends(home) == ["dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_handoff_keeps_source_editor_isolated_from_commands_and_selector(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="handoff")
    try:
        session.wait_for("phase input")
        session.send(b"/name Source\r")
        session.wait_for("name Source")
        saved = next((home / "sessions").glob("*.jsonl"))
        identity = decode_history(saved.read_bytes()).history.conversation_id
        session.send(b"source draft\x03/new\r")
        session.wait_for("handoff closing")
        session.send(b"\x03/resume\r")
        session.wait_for("session management is in progress; selector was not opened")
        session.send(b"/resume --list\r/session\r")
        session.wait_for("Sessions (current project")
        session.wait_for("Recovery:")
        (home / "release-handoff").touch()
        session.wait_for("new current")
        session.send(b"/quit\r")
        session.wait_for("unsubmitted content remains")
        # Cancel exit to continue inspecting the protected source draft.
        session.send(b"\x1b")
        session.wait_for("exit canceled")
        session.send(f"/resume {identity}\r".encode())
        session.wait_for(f"resumed current {identity}")
        session.wait_for("> source draft", after=session.visible().index("resumed current"))
        session.send(b"\x03/quit\r")
        session.wait_for("unsubmitted content remains", after=session.visible().index("resumed current"))
        session.send(b"\r")
        assert session.finish() == 0
        assert session.restored()
        assert sends(home) == ["dialogue"]
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()
