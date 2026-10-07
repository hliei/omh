"""Manual compaction, idle admission and usage display through the terminal UI."""

from pathlib import Path

from test_interactive import sends, wait_sends
from test_interactive_sessions import start


def _phase_input_after(session, marker: str) -> None:  # type: ignore[no-untyped-def]
    session.wait_for("phase input", after=session.visible().index(marker))


def test_manual_compact_summarizes_idle_history_and_keeps_originals(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="manual-compact")
    try:
        session.wait_for("phase input")
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        _phase_input_after(session, "reply:hello")

        session.send(b"/compact focus on the retry budget\r")
        session.wait_for("phase compact manual")
        session.wait_for("older context summarized")
        session.wait_for("phase input", after=session.visible().index("older context summarized"))
        assert sends(home) == ["dialogue", "summary"]

        # The compaction appends a record; the original exchange stays saved.
        saved = next((home / "sessions").glob("*.jsonl"))
        records = saved.read_text().replace(" ", "")
        assert '"type":"compaction"' in records
        assert "hello" in records

        # A later /session recomputes the estimate from the rebuilt history.
        session.send(b"/session\r")
        session.wait_for("context ~")
        session.send(b"/quit\r")
        assert session.finish() == 0
        assert session.restored()
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_manual_compact_rejects_a_busy_model_without_aborting_it(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="hang")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("partial answer")
        session.send(b"/compact\r")
        session.wait_for("busy; wait or cancel with Escape")
        assert "summary" not in sends(home)
        # The rejected command left the running activity alone; Escape cancels it.
        session.send(b"\x1b")
        session.wait_for("phase cancel")
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_manual_compact_rejects_an_unsaved_session(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="echo")
    try:
        session.wait_for("phase input")
        session.send(b"one\r")
        session.wait_for("reply:one")
        session.wait_for("save saved")
        saved = next((home / "sessions").glob("*.jsonl"))
        saved.unlink()
        saved.mkdir()
        session.send(b"two\r")
        session.wait_for("save unsaved")
        session.send(b"\x7f\x7f\x7f")  # clear the restored rejected draft
        offset = len(session.visible())
        session.send(b"/compact\r")
        session.wait_for("In-memory history is retained", after=offset)
        assert "summary" not in sends(home)
        session.send(b"/quit --discard-unsaved\r")
        assert session.finish() == 0
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_manual_compact_cancels_and_refuses_a_new_user_shell(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="manual-compact-cancel")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("reply:go")
        _phase_input_after(session, "reply:go")

        session.send(b"/compact\r")
        session.wait_for("phase compact manual")
        session.send(b"!echo hi\r")
        session.wait_for("no shell was started")
        session.send(b"\x7f" * len("!echo hi"))  # clear the refused draft
        session.send(b"\x1b")
        session.wait_for("compact canceled; original history kept")
        wait_sends(home, 3, session)
        assert sends(home) == ["dialogue", "summary", "summary-aborted"]
        saved = next((home / "sessions").glob("*.jsonl"))
        assert '"type":"compaction"' not in saved.read_text().replace(" ", "")

        _phase_input_after(session, "compact canceled")
        session.send(b"after\r")
        session.wait_for("reply:after")
        session.send(b"/quit\r")
        assert session.finish() == 0
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_session_displays_context_recorded_usage_and_effective_policy(tmp_path: Path) -> None:
    session, home = start(tmp_path, scenario="usage")
    try:
        session.wait_for("phase input")
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        session.wait_for("| context ~")
        session.send(b"/session\r")
        session.wait_for("context ~1250 / 1000000 tokens")
        session.wait_for("policy auto-compact on (reserve 16384, keep 20000)")
        session.wait_for(
            "recorded usage input 1000 output 200 cache-read 50 cache-write 0 summary 0"
        )
        session.wait_for("reasoning 25 not added to output")
        session.wait_for("estimated cost ~$0.0006 from catalog rates; not a bill")
        session.wait_for("usage reported for 1 record(s)")
        # A short history has no compactable prefix; the failure is explicit.
        session.send(b"/compact\r")
        session.wait_for("notice compact: Nothing to compact")
        session.send(b"/quit\r")
        assert session.finish() == 0
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()
