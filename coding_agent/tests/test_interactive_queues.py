"""Installed interactive steering, follow-up, recall, cancel and exit guards.

The tests drive the installed ``omh`` through a PTY against the controlled
provider. Queued inputs are observed through the public screen and the requests
the provider records: queue notes, status counts, recalled editor text and
request counts. No internal queue state is asserted and no random delays are
used; the slow controlled tools hold the turn open long enough to queue input
while the model is demonstrably running.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support import png_bytes
from test_interactive import InteractiveSession, provider_env, sends, settle, wait_sends


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def start(home: Path, tmp_path: Path, scenario: str) -> tuple[InteractiveSession, Path]:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    env = provider_env(home, scenario)
    env["INTERACTIVE_REQUESTS"] = str(home / "requests.txt")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=env,
    )
    return session, project


def requests(home: Path) -> list[str]:
    """The last user text of every model request the provider received."""
    path = home / "requests.txt"
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line]


def test_busy_enter_queues_steering_consumed_after_the_tool_batch(
    home: Path, tmp_path: Path,
) -> None:
    session, project = start(home, tmp_path, "steer")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("phase tool bash")
        session.send(b"redirect the plan\r")
        session.wait_for("queued steering (1 waiting): redirect the plan")
        session.wait_for("steering 1 (redirect the plan)")
        session.wait_for("steered:redirect the plan")
        # The started tool batch was not preempted: both tools ran and the
        # steering request followed their results, exactly once.
        assert (project / "first.txt").read_text() == "a"
        assert (project / "second.txt").read_text() == "b"
        session.wait_for("you redirect the plan")
        assert requests(home) == ["go", "redirect the plan"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_busy_alt_enter_queues_a_follow_up_consumed_at_the_stop_boundary(
    home: Path, tmp_path: Path,
) -> None:
    session, project = start(home, tmp_path, "follow")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("phase tool bash")
        session.send(b"and then check\x1b\r")
        session.wait_for("queued follow-up (1 waiting): and then check")
        session.wait_for("follow-up 1 (and then check)")
        session.wait_for("turn-done")
        session.wait_for("followed:and then check")
        session.wait_for("you and then check")
        assert (project / "marker.txt").read_text() == "mark"
        assert requests(home) == ["go", "go", "and then check"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_queued_input_expands_resources_at_acceptance(
    home: Path, tmp_path: Path,
) -> None:
    prompts = home / ".omh" / "agent" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "notes.md").write_text(
        "---\ndescription: Notes\n---\nTEMPLATE BODY $ARGUMENTS\n"
    )
    session, _project = start(home, tmp_path, "steer")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("phase tool bash")
        session.send(b"/notes extra\r")
        session.wait_for("queued steering (1 waiting): /notes extra")
        session.wait_for("steered:TEMPLATE BODY")
        # Acceptance expanded the template once; retry or consumption reuses it.
        assert "TEMPLATE BODY" in requests(home)[1]
        assert "extra" in requests(home)[1]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_idle_alt_enter_submits_an_ordinary_prompt(home: Path, tmp_path: Path) -> None:
    session, _project = start(home, tmp_path, "echo")
    try:
        session.wait_for("phase input")
        session.send(b"plain prompt\x1b\r")
        session.wait_for("reply:plain prompt")
        assert "queued" not in session.visible()
        assert requests(home) == ["plain prompt"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_alt_up_recalls_both_queues_with_images_and_the_original_draft(
    home: Path, tmp_path: Path,
) -> None:
    session, project = start(home, tmp_path, "steer")
    image = project / "shot.png"
    image.write_bytes(png_bytes())
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("phase tool bash")
        session.send(b"/attach shot.png\r")
        session.wait_for("attached #1 shot.png")
        session.send(b"look at this\r")
        session.wait_for("queued steering (1 waiting): look at this +1 image")
        # The waiting text and its image stay visible in the status, not only in
        # the one-shot acceptance note.
        session.wait_for("steering 1 (look at this +1 image)")
        session.send(b"and also\x1b\r")
        session.wait_for("queued follow-up (1 waiting): and also")
        session.wait_for("follow-up 1 (and also)")
        session.send(b"fresh draft")
        session.wait_for("fresh draft")
        session.send(b"\x1b[1;3A")
        session.wait_for(
            "recalled 2 queued input(s) into the editor"
            " and 1 image(s) back to the pending draft"
        )
        # All steering first, then all follow-ups, then the untouched draft,
        # merged with blank lines into the editor.
        session.wait_for("> look at this")
        session.wait_for("  and also")
        session.wait_for("  fresh draft")
        visible = session.visible()
        entry = visible[visible.index("> look at this"):]
        assert entry.index("and also") < entry.index("fresh draft")
        # The recalled queues are not consumed afterwards: the turn that was
        # already running finishes on its own input and nothing follows it.
        session.wait_for("batch-done")
        assert requests(home) == ["go", "go"]
        # The recalled image submits with the next prompt in the same session.
        session.send(b"\x03")
        session.send(b"see this\r")
        session.wait_for("steered:see this")
        session.wait_for("image shot.png 6x4 image/png")
        assert requests(home) == ["go", "go", "see this"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_escape_recalls_the_queue_cancels_and_keeps_tool_side_effects(
    home: Path, tmp_path: Path,
) -> None:
    session, project = start(home, tmp_path, "cancel-side")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("partial answer")
        session.send(b"redirect\r")
        session.wait_for("queued steering (1 waiting): redirect")
        session.wait_for("steering 1 (redirect)")
        session.send(b"\x1b")
        session.wait_for("phase cancel")
        session.wait_for("recalled 1 queued input(s) into the editor")
        session.wait_for("> redirect")
        # Cancel does not roll back the executed tool's side effect.
        assert (project / "kept.txt").read_text() == "kept"
        # The same session continues after the cancel.
        session.wait_for("phase input")
        session.send(b"\x03")
        session.send(b"continue now\r")
        session.wait_for("reply:continue now")
        # The recalled steering never reached the provider.
        assert requests(home) == ["go", "go", "continue now"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_ctrl_d_with_pending_content_returns_to_editing_or_discards(
    home: Path, tmp_path: Path,
) -> None:
    session, project = start(home, tmp_path, "echo")
    image = project / "shot.png"
    image.write_bytes(png_bytes())
    try:
        session.wait_for("phase input")
        session.send(b"/attach shot.png\r")
        session.wait_for("attached #1 shot.png")
        session.send(b"\x04")
        session.wait_for("unsubmitted content remains")
        session.wait_for("any other key returns to editing")
        session.send(b"x")
        session.wait_for("exit canceled")
        # The screen is still usable: an ordinary prompt still works.
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        session.send(b"/attach clear\r")
        session.wait_for("attachments cleared")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_double_ctrl_c_discards_a_queued_input_only_explicitly(
    home: Path, tmp_path: Path,
) -> None:
    session, _project = start(home, tmp_path, "cancel-side")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("partial answer")
        session.send(b"redirect\r")
        session.wait_for("queued steering (1 waiting): redirect")
        session.wait_for("steering 1 (redirect)")
        session.send(b"\x03\x03")
        session.wait_for("unsubmitted content remains")
        session.wait_for("Enter discards it and exits")
        session.send(b"\r")
        assert session.finish() == 0
        assert session.restored()
        saved = next((home / "sessions").glob("*.jsonl"))
        assert "redirect" not in saved.read_text()
    finally:
        session.close()


def test_idle_escape_and_alt_up_are_inert_and_ctrl_d_needs_an_empty_editor(
    home: Path, tmp_path: Path,
) -> None:
    session, _project = start(home, tmp_path, "echo")
    try:
        session.wait_for("phase input")
        session.send(b"kept text")
        session.wait_for("> kept text")
        # Ctrl+D does nothing while the editor still has text.
        session.send(b"\x04")
        session.send(b"\x1b[1;3A")
        session.wait_for("no queued inputs to recall")
        assert session.process.poll() is None
        assert "unsubmitted content remains" not in session.visible()
        assert "> kept text" in session.visible()
        # A lone Escape with no completion and no queue is inert.
        session.send(b"\x1b")
        settle(session)
        assert session.process.poll() is None
        assert "> kept text" in session.visible()
        session.send(b"\x03\x03")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_ctrl_c_then_ctrl_d_still_protects_the_cleared_line(
    home: Path, tmp_path: Path,
) -> None:
    session, _project = start(home, tmp_path, "echo")
    try:
        session.wait_for("phase input")
        session.send(b"unsent draft")
        session.wait_for("unsent draft")
        session.send(b"\x03")
        settle(session)
        session.send(b"\x04")
        session.wait_for("unsubmitted content remains")
        session.wait_for("cleared editor text")
        session.send(b"x")
        session.wait_for("exit canceled")
        session.wait_for("> unsent draft")
        session.send(b"\x03\x03")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_retry_reuses_the_accepted_input(home: Path, tmp_path: Path) -> None:
    session, _project = start(home, tmp_path, "retry")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("retried")
        # The retry reuses the accepted user message instead of expanding or
        # duplicating the input.
        assert requests(home) == ["go", "go"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_escape_cancels_a_pending_retry_and_keeps_the_session(
    home: Path, tmp_path: Path,
) -> None:
    session, _project = start(home, tmp_path, "retry-cancel")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("phase retry")
        session.send(b"\x1b")
        session.wait_for("phase cancel")
        session.wait_for("phase input")
        session.send(b"after\r")
        session.wait_for("reply:after")
        # The cancelled retry never reached the provider again.
        assert requests(home) == ["go", "after"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_double_ctrl_c_offers_the_cleared_text_back(home: Path, tmp_path: Path) -> None:
    session, _project = start(home, tmp_path, "echo")
    try:
        session.wait_for("phase input")
        session.send(b"unsent draft")
        session.wait_for("unsent draft")
        session.send(b"\x03\x03")
        session.wait_for("unsubmitted content remains")
        session.wait_for("cleared editor text")
        session.send(b"q")
        session.wait_for("exit canceled")
        session.wait_for("> unsent draft")
        session.send(b"\x03\x03")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_escape_cancels_automatic_compaction_recalls_and_keeps_the_session(
    home: Path, tmp_path: Path,
) -> None:
    session, _project = start(home, tmp_path, "compact-cancel")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("phase compact threshold")
        session.send(b"redirect\r")
        session.wait_for("queued steering (1 waiting): redirect")
        session.send(b"\x1b")
        session.wait_for("recalled 1 queued input(s) into the editor")
        session.wait_for("phase cancel")
        session.wait_for("> redirect")
        # The summary provider saw cooperative cancellation, and the recalled
        # input never became a request or a committed compaction record.
        wait_sends(home, 3, session)
        assert sends(home) == ["dialogue", "summary", "summary-aborted"]
        assert requests(home) == ["go"]
        saved = next((home / "sessions").glob("*.jsonl"))
        assert '"type":"compaction"' not in saved.read_text().replace(" ", "")
        session.send(b"\x03")
        session.send(b"after\r")
        session.wait_for("reply:after")
        assert requests(home) == ["go", "after"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()
