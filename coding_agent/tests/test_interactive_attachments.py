"""Installed interactive image attachments through a PTY and controlled provider.

The tests use real temporary image files and the real installed command. The
clipboard fixture only proves acceptance and the missing-backend fallback; a
real desktop screenshot paste is recorded separately by the manual two-platform
acceptance.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from test_interactive import InteractiveSession, provider_env, sends

from coding_agent import decode_history


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def write_image(path: Path, size: tuple[int, int] = (6, 4), fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format=fmt)
    path.write_bytes(buffer.getvalue())
    return path.read_bytes()


def images_env(home: Path, scenario: str, **extra: str) -> dict[str, str]:
    env = provider_env(home, scenario)
    env["INTERACTIVE_IMAGES"] = str(home / "images.txt")
    env.update(extra)
    return env


def recorded_images(home: Path) -> list[str]:
    path = home / "images.txt"
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line]


def fake_clipboard_env(home: Path, tmp_path: Path, payload: bytes) -> dict[str, str]:
    """A PATH-injected backend that returns one real PNG from the clipboard."""
    bin_dir = tmp_path / "clipboard-bin"
    bin_dir.mkdir(exist_ok=True)
    fixture = tmp_path / "clipboard-source.png"
    fixture.write_bytes(payload)
    common = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "CLIPBOARD_FIXTURE": str(fixture),
    }
    if sys.platform == "darwin":
        script = bin_dir / "osascript"
        script.write_text(
            "#!/bin/sh\n"
            "for arg in \"$@\"; do\n"
            "  case \"$arg\" in\n"
            "    *'POSIX file \"'*)\n"
            "      target=$(printf '%s' \"$arg\" | sed -n 's/.*POSIX file \"\\([^\"]*\\)\".*/\\1/p')\n"
            "      [ -n \"$target\" ] && cp \"$CLIPBOARD_FIXTURE\" \"$target\" && exit 0;;\n"
            "  esac\n"
            "done\n"
            "exit 1\n"
        )
        script.chmod(0o755)
        return common
    if sys.platform.startswith("linux"):
        script = bin_dir / "wl-paste"
        script.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  --list-types) printf 'image/png\\n';;\n"
            "  --no-newline) cat \"$CLIPBOARD_FIXTURE\";;\n"
            "esac\n"
        )
        script.chmod(0o755)
        return {**common, "WAYLAND_DISPLAY": "wayland-0"}
    pytest.skip("no controlled clipboard backend for this platform")


def no_clipboard_env(tmp_path: Path) -> dict[str, str]:
    """Hide every desktop clipboard dependency to prove the file fallback."""
    empty = tmp_path / "empty-bin"
    empty.mkdir(exist_ok=True)
    return {"PATH": str(empty), "WAYLAND_DISPLAY": "", "DISPLAY": ""}


def test_attach_manages_the_pending_draft_without_touching_the_editor(
    home: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    write_image(project / "picture.png")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project, env=images_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/attach picture.png\r")
        session.wait_for("attached #1 picture.png 6x4 image/png ready")
        session.wait_for("pending 1")
        session.send(b"/attach\r")
        session.wait_for("attachments 1 pending")
        session.wait_for("#1 picture.png 6x4 image/png ready")
        session.wait_for("origin=file")
        session.send(b"/attach remove 1\r")
        session.wait_for("removed #1 picture.png")
        session.send(b"/attach\r")
        session.wait_for("attachments none pending")
        session.send(b"plain text\r")
        session.wait_for("reply:plain text")
        assert sends(home) == ["dialogue"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_attach_reports_a_corrupt_or_unsupported_file_and_adds_nothing(
    home: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + b"\x00" * 8)
    (project / "notes.txt").write_text("not an image", encoding="utf-8")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project, env=images_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/attach broken.png\r")
        session.wait_for("broken.png")
        session.wait_for("could not be decoded")
        session.send(b"/attach notes.txt\r")
        session.wait_for("notes.txt")
        session.wait_for("not a supported image format")
        session.send(b"/attach\r")
        session.wait_for("attachments none pending")
        assert "pending 0" in session.visible()
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_attach_reports_dimension_resize_status(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    write_image(project / "wide.png", (2001, 10))
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project, env=images_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/attach wide.png\r")
        session.wait_for("attached #1 wide.png 2000x10 image/png ready")
        session.wait_for("resized-from=2001x10")
        session.send(b"\x03")
        session.send(b"/attach clear\r")
        session.wait_for("attachments cleared")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_ctrl_v_without_a_backend_explains_and_keeps_the_file_path(
    home: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project,
        env=images_env(home, "echo", **no_clipboard_env(tmp_path)),
    )
    try:
        session.wait_for("phase input")
        session.send(b"still here")
        session.send(b"\x16")
        session.wait_for("no clipboard image backend")
        session.wait_for("/attach <image-path>")
        assert "> still here" in session.visible()
        assert sends(home) == []
        session.send(b"\r")
        session.wait_for("reply:still here")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_ctrl_v_adds_a_manageable_attachment_without_submitting(
    home: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    payload = write_image(tmp_path / "shot.png", (8, 5))
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project,
        env=images_env(home, "vision", **fake_clipboard_env(home, tmp_path, payload)),
    )
    try:
        session.wait_for("phase input")
        session.send(b"keep me")
        session.send(b"\x16")
        session.wait_for("attached #1 clipboard.png 8x5 image/png ready")
        session.wait_for("pending 1")
        assert sends(home) == []
        assert "reply:" not in session.visible()
        assert "> keep me" in session.visible()
        session.send(b"\r")
        session.wait_for("seen-image:1")
        session.wait_for("image clipboard.png 8x5 image/png")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()
    recorded = recorded_images(home)
    assert recorded == [f"image/png {base64.b64encode(payload).decode('ascii')}"]


def test_text_and_image_submit_enter_history_and_reopen_after_the_source_disappears(
    home: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    payload = write_image(project / "picture.png", (6, 4))
    sessions = home / "sessions"
    first = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(sessions), "--cwd", str(project),
        home=home, cwd=project, env=images_env(home, "vision"),
    )
    try:
        first.wait_for("phase input")
        first.send(b"/attach picture.png\r")
        first.wait_for("attached #1 picture.png 6x4 image/png ready")
        first.send(b"describe it\r")
        first.wait_for("seen-image:1")
        first.wait_for("image picture.png 6x4 image/png")
        first.wait_for("save saved")
        first.send(b"\x04")
        assert first.finish() == 0
        assert first.restored()
    finally:
        first.close()
    assert image_payloads(sessions) == [base64.b64encode(payload).decode("ascii")]
    source = project / "picture.png"
    saved = next(sessions.glob("*.jsonl"))
    source.unlink()

    degraded = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(saved), "--model", "opencode-go/glm-5.3",
        home=home, cwd=project, env=images_env(home, "echo"),
    )
    try:
        degraded.wait_for("you describe it")
        degraded.wait_for("image image/png")
        degraded.wait_for("placeholder")
        degraded.wait_for("original history is kept")
        degraded.send(b"\x04")
        assert degraded.finish() == 0
    finally:
        degraded.close()

    reopened = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(saved), home=home, cwd=project, env=images_env(home, "vision"),
    )
    try:
        reopened.wait_for("you describe it")
        reopened.wait_for("image image/png")
        reopened.send(b"continue\r")
        reopened.wait_for("seen-image:0")
        assert image_payloads(sessions) == [base64.b64encode(payload).decode("ascii")]
        reopened.send(b"\x04")
        assert reopened.finish() == 0
        assert reopened.restored()
    finally:
        reopened.close()


def test_a_new_image_is_rejected_before_a_text_only_model_and_the_draft_survives(
    home: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    write_image(project / "picture.png")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--model", "opencode-go/glm-5.3", "--cwd", str(project),
        home=home, cwd=project, env=images_env(home, "vision"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/attach picture.png\r")
        session.wait_for("attached #1 picture.png 6x4 image/png ready")
        session.wait_for("does not accept image input")
        session.wait_for("select a vision-capable model manually")
        session.wait_for("pending 1")
        session.send(b"describe it\r")
        session.wait_for("No request was sent.")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if "> describe it" in session.visible():
                break
            session._pump(0.1)
        assert "> describe it" in session.visible()
        assert "you describe it" not in session.visible()
        assert "phase model" not in session.visible()
        assert sends(home) == []
        assert recorded_images(home) == []
        session.send(b"\x03")
        session.send(b"/attach clear\r")
        session.wait_for("attachments cleared")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def image_payloads(sessions: Path) -> list[str]:
    """Base64 image data saved in the newest history file, in record order."""
    saved = next(sessions.glob("*.jsonl"))
    history = decode_history(saved.read_bytes()).history
    payloads: list[str] = []
    for entry in history.entries:
        message = getattr(entry, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            payloads.extend(
                block.data for block in content if getattr(block, "type", None) == "image"
            )
    return payloads
