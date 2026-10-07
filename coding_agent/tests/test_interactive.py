"""Installed interactive sessions through a PTY and a controlled provider."""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest
from test_cli import clean_env, cli_command, run_cli

PROVIDER = Path(__file__).with_name("fixtures") / "interactive_provider.py"
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SGR = re.compile(r"\x1b\[[0-9;]*m")


class InteractiveSession:
    """One installed ``omh`` process attached to a PTY."""

    def __init__(
        self, *args: str, home: Path, cwd: Path, env: dict[str, str] | None = None,
        rows: int = 24, cols: int = 80,
    ) -> None:
        self.master, self.slave = pty.openpty()
        fcntl_size = struct.pack("HHHH", rows, cols, 0, 0)
        import fcntl
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, fcntl_size)
        self.before = termios.tcgetattr(self.slave)
        self.output = b""
        command_env = clean_env(home)
        if env:
            command_env.update(env)
        self.process = subprocess.Popen(
            [*cli_command(), *args],
            stdin=self.slave, stdout=self.slave, stderr=self.slave,
            cwd=cwd, env=command_env,
        )

    def send(self, data: bytes) -> None:
        os.write(self.master, data)

    def resize(self, rows: int, cols: int) -> None:
        import fcntl
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        assert self.process.pid is not None
        os.kill(self.process.pid, signal.SIGWINCH)

    def wait_for(self, text: str, timeout: float = 8, *, after: int = 0) -> None:
        """Wait for visible text after an optional captured output position."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if text in self.visible()[after:]:
                return
            self._pump(0.1)
            if self.process.poll() is not None and text not in self.visible()[after:]:
                break
        raise AssertionError(f"missing {text!r} in {self.visible()!r}")

    def visible(self) -> str:
        return _ANSI.sub("", self.output.decode("utf-8", "replace")).replace("\r", "")

    def finish(self, timeout: float = 8) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.process.poll() is None:
            self._pump(0.1)
        if self.process.poll() is None:
            self.process.kill()
        code = self.process.wait(timeout=2)
        self._pump(0.05)
        return code

    def restored(self) -> bool:
        after = termios.tcgetattr(self.slave)
        mask = termios.ECHO | termios.ICANON | termios.ISIG
        return (after[3] & mask) == (self.before[3] & mask)

    def close(self) -> None:
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def _pump(self, delay: float) -> None:
        ready, _, _ = select.select([self.master], [], [], delay)
        if not ready:
            return
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            return
        if chunk:
            self.output += chunk


def provider_env(home: Path, scenario: str, **extra: str) -> dict[str, str]:
    site = home / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "sitecustomize.py").write_text(PROVIDER.read_text())
    env = {
        "PYTHONPATH": str(site),
        "INTERACTIVE_SCENARIO": scenario,
        "INTERACTIVE_SENDS": str(home / "sends.txt"),
        "TERM": "xterm-256color",
    }
    env.update(extra)
    return env


def sends(home: Path) -> list[str]:
    path = home / "sends.txt"
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line]


def settle(session: "InteractiveSession", delay: float = 0.3) -> None:
    """Let a just-finished turn return the screen to the input phase."""
    time.sleep(delay)
    session._pump(0.1)


def wait_sends(home: Path, count: int, session: "InteractiveSession", timeout: float = 8) -> None:
    """Wait until the controlled provider has received ``count`` requests."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(sends(home)) >= count:
            return
        session._pump(0.1)
    raise AssertionError(f"expected {count} sends, saw {sends(home)!r}")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def test_unknown_theme_is_rejected_before_a_terminal_session(home: Path) -> None:
    result = run_cli("--use-theme", "neon", home=home)
    assert result.returncode == 2
    assert "unknown theme" in result.stderr
    assert result.stdout == ""


def test_missing_key_keeps_the_screen_and_sends_nothing(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--cwd", str(project), home=home, cwd=project,
        env={"TERM": "xterm-256color"},
    )
    try:
        session.wait_for("No API key")
        session.wait_for("No request was sent.")
        session.wait_for("theme dark")
        session.wait_for("phase input")
        assert "\x1b[38;5;117m" in session.output.decode()
        session.send(b"hello\r")
        session.wait_for("you hello")
        assert session.visible().count("No request was sent.") >= 2
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
        assert list(project.rglob("*.jsonl")) == []
    finally:
        session.close()


def test_invalid_configuration_stays_usable_without_a_request(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent = home / ".omh" / "agent"
    agent.mkdir(parents=True)
    (agent / "settings.json").write_text('{"defaultModel": "not-a-model"}')
    session = InteractiveSession(
        "--no-approve", "--api-key", "offline", "--cwd", str(project),
        home=home, cwd=project, env={"TERM": "xterm-256color"},
    )
    try:
        session.wait_for("not-a-model")
        session.wait_for("No request was sent.")
        session.send(b"hello\r")
        session.wait_for("The session is not ready.")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_themes_and_plain_fallback(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agent = home / ".omh" / "agent"
    agent.mkdir(parents=True)
    (agent / "settings.json").write_text('{"theme": "light"}')
    light = InteractiveSession(
        "--no-approve", "--cwd", str(project), home=home, cwd=project,
        env={"TERM": "xterm-256color"},
    )
    try:
        light.wait_for("theme light")
        assert "\x1b[38;5;25m" in light.output.decode()
        light.send(b"\x04")
        assert light.finish() == 0
    finally:
        light.close()

    forced = InteractiveSession(
        "--no-approve", "--use-theme", "dark", "--cwd", str(project),
        home=home, cwd=project, env={"TERM": "xterm-256color"},
    )
    try:
        forced.wait_for("theme dark")
        forced.send(b"\x04")
        assert forced.finish() == 0
    finally:
        forced.close()

    plain = InteractiveSession(
        "--no-approve", "--use-theme", "light", "--cwd", str(project),
        home=home, cwd=project, env={"NO_COLOR": "1", "TERM": "xterm-256color"},
    )
    try:
        plain.wait_for("theme plain")
        plain.wait_for("phase input")
        assert _SGR.search(plain.output.decode()) is None
        plain.send(b"\x04")
        assert plain.finish() == 0
        assert plain.restored()
    finally:
        plain.close()

    dumb = InteractiveSession(
        "--no-approve", "--cwd", str(project), home=home, cwd=project,
        env={"TERM": "dumb"}, rows=8, cols=12,
    )
    try:
        dumb.wait_for("theme plain")
        dumb.send("中文".encode())
        dumb.wait_for("中文")
        # The line cleared by Ctrl+C is protected: Ctrl+D asks, and Enter then
        # discards it explicitly. (The wrapped notice is unreadable at 12 cols.)
        dumb.send(b"\x03\x04\r")
        assert dumb.finish() == 0
        assert "中文" in dumb.visible()
        assert _SGR.search(dumb.output.decode()) is None
    finally:
        dumb.close()


def test_narrow_window_resize_and_chinese(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"), rows=8, cols=12,
    )
    try:
        session.wait_for("phase input")
        session.send("中文".encode())
        session.resize(6, 20)
        session.send(b"\r")
        session.wait_for("reply:中文")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_ctrl_c_clears_input_and_a_signal_exits_differently(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"hello\x03world\r")
        session.wait_for("reply:world")
        assert "reply:hello" not in session.visible()
        session.send(b"\x03\x03")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()

    hanging = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "signals"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "hang"),
    )
    try:
        hanging.wait_for("phase input")
        hanging.send(b"wait\r")
        hanging.wait_for("partial answer")
        assert hanging.process.pid is not None
        os.kill(hanging.process.pid, signal.SIGINT)
        assert hanging.finish() == 130
        assert "phase cancel" in hanging.visible()
        assert "partial answer" in hanging.visible()
        assert hanging.restored()
    finally:
        hanging.close()


def test_controlled_tools_save_reopen_and_folds(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("value = 1\n")
    sessions = home / "sessions"
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(sessions), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "tools"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"fix the test\r")
        session.wait_for("tool read running", timeout=15)
        session.wait_for("tool edit ok", timeout=15)
        session.wait_for("tool write ok", timeout=15)
        session.wait_for("tool bash ok", timeout=15)
        session.wait_for("Fixed the value", timeout=15)
        session.wait_for("thinking collapsed")
        session.wait_for("code python")
        session.wait_for("value = 2")
        session.wait_for("save saved")
        assert "checked the assertion" not in session.visible()
        assert "Successfully replaced" not in session.visible()
        session.send(b"\x14")
        session.wait_for("thinking expanded")
        session.wait_for("checked the assertion")
        session.send(b"\x0f")
        session.wait_for("output expanded recorded")
        session.wait_for("Successfully replaced")
        session.send(b"continue\r")
        session.wait_for("reply:continue")
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()
    assert (project / "app.py").read_text() == "value = 2\n"
    assert (project / "note.txt").read_text() == "noted"
    saved = list(sessions.glob("*.jsonl"))
    assert len(saved) == 1

    reopened = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(saved[0]), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "tools"),
    )
    try:
        reopened.wait_for("Fixed the value")
        reopened.wait_for("path=app.py")
        reopened.wait_for("you fix the test")
        reopened.wait_for("thinking collapsed")
        assert "checked the assertion" not in reopened.visible()
        reopened.send(b"continue\r")
        reopened.wait_for("reply:continue")
        reopened.send(b"\x04")
        assert reopened.finish() == 0
        assert reopened.restored()
    finally:
        reopened.close()
    text = saved[0].read_text()
    assert "fix the test" in text
    assert text.count("continue") >= 2


def test_model_and_tool_errors_can_continue(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    failed = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "errors"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "error"),
    )
    try:
        failed.wait_for("phase input")
        failed.send(b"first\r")
        failed.wait_for("kept visible")
        failed.wait_for("model error provider rejected the turn")
        failed.wait_for("phase input")
        failed.send(b"next\r")
        failed.wait_for("reply:next")
        assert "kept visible" in failed.visible()
        failed.send(b"\x04")
        assert failed.finish() == 0
    finally:
        failed.close()

    tools = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "tool-errors"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "tool-error"),
    )
    try:
        tools.wait_for("phase input")
        tools.send(b"read missing\r")
        tools.wait_for("tool read error")
        tools.wait_for("still here")
        tools.send(b"again\r")
        tools.wait_for("reply:again")
        tools.send(b"\x04")
        assert tools.finish() == 0
    finally:
        tools.close()


def test_retry_and_compact_phases_are_labeled(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    retry = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "retry"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "retry"),
    )
    try:
        retry.wait_for("phase input")
        retry.send(b"try\r")
        retry.wait_for("phase retry dialogue 1/")
        retry.wait_for("retried")
        retry.wait_for("phase input")
        retry.send(b"\x04")
        assert retry.finish() == 0
    finally:
        retry.close()

    compact = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "compact"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "compact"),
    )
    try:
        compact.wait_for("phase input")
        compact.send(b"shrink\r")
        compact.wait_for("phase compact", timeout=15)
        compact.wait_for("compacted", timeout=15)
        compact.send(b"\x04")
        assert compact.finish() == 0
    finally:
        compact.close()


def test_bounded_tool_output_stays_the_recorded_text(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "bounded"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "bounded"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"run\r")
        session.wait_for("output collapsed recorded bounded", timeout=15)
        session.wait_for("bounded done", timeout=15)
        assert "Full output:" not in session.visible()
        session.send(b"\x0f")
        session.wait_for("output expanded recorded bounded")
        session.wait_for("Full output:")
        assert "complete output follows" not in session.visible()
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_unsaved_history_refuses_new_work_and_keeps_the_screen(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    sessions = home / "unsaved"
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(sessions), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"one\r")
        session.wait_for("reply:one")
        session.wait_for("save saved")
        saved = next(sessions.glob("*.jsonl"))
        saved.unlink()
        saved.mkdir()
        session.send(b"two\r")
        session.wait_for("In-memory history is retained")
        session.wait_for("save unsaved")
        assert "reply:one" in session.visible()
        before = len(sends(home))
        session.send(b"three\r")
        session.wait_for("New work is paused")
        time.sleep(0.4)
        session._pump(0.1)
        assert len(sends(home)) == before
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_missing_history_cwd_asks_for_a_replacement(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    sessions = home / "sessions"
    first = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(sessions), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        first.wait_for("phase input")
        first.send(b"first\r")
        first.wait_for("reply:first")
        first.send(b"\x04")
        assert first.finish() == 0
    finally:
        first.close()
    saved = next(sessions.glob("*.jsonl"))
    project.rename(tmp_path / "gone")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    second = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(saved), home=home, cwd=tmp_path, env=provider_env(home, "echo"),
    )
    try:
        second.wait_for("no longer exists")
        second.wait_for("directory>")
        second.send(str(replacement).encode() + b"\r")
        second.wait_for("you first")
        second.wait_for("save saved")
        second.send(b"mark\r")
        second.wait_for("marked")
        assert (replacement / "marker.txt").read_text() == "marked"
        second.send(b"\x04")
        assert second.finish() == 0
        assert second.restored()
    finally:
        second.close()


def test_resume_without_a_session_does_not_start_one(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--resume", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("Pass --session")
        session.send(b"hello\r")
        session.wait_for("No request was sent")
        assert "reply:hello" not in session.visible()
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()
    assert list(home.rglob("*.jsonl")) == []


def test_memory_mode_is_labeled_and_print_does_not_use_the_screen(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--no-session", "--api-key", "offline",
        "--cwd", str(project), home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("mode memory")
        session.wait_for("save pending")
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()
    assert list(home.rglob("*.jsonl")) == []

    import subprocess
    result = subprocess.run(
        [*cli_command(), "--no-approve", "--no-context-files", "--print", "--api-key", "offline",
         "--cwd", str(project), "--session-dir", str(home / "print"), "hello"],
        cwd=project, capture_output=True, text=True, env={**clean_env(home), **provider_env(home, "echo")},
        timeout=15,
    )
    assert result.returncode == 0
    assert result.stdout == "reply:hello"
    assert "phase input" not in result.stdout
    assert "theme " not in result.stdout


@pytest.mark.parametrize("theme", ["dark", "plain"])
def test_terminal_controls_are_visible_text_and_history_stays_original(
    home: Path, tmp_path: Path, theme: str,
) -> None:
    from omh.llm.types import AssistantMessage, TextContent

    from coding_agent import decode_history

    project = tmp_path / "project"
    project.mkdir()
    sessions = home / "sessions"
    env = provider_env(home, "controls", **({"NO_COLOR": "1"} if theme == "plain" else {}))
    first = InteractiveSession(
        "--no-approve", "--api-key", "offline", "--session-dir", str(sessions),
        home=home, cwd=project, env=env, cols=200,
    )
    try:
        first.wait_for("phase input")
        first.send(b"hello\r")
        first.wait_for(r"before\x1b]52;c;dGVzdA==\x07after")
        first.wait_for("save saved")
        assert b"\x1b]52" not in first.output
        assert b"\x1b[2J" not in first.output
        first.send(b"\x04")
        assert first.finish() == 0
        assert first.restored()
    finally:
        first.close()
    saved = next(sessions.glob("*.jsonl"))
    history = decode_history(saved.read_bytes()).history
    answer = next(
        e.message for e in reversed(history.entries)
        if hasattr(e, "message") and isinstance(e.message, AssistantMessage)
    )
    assert any(isinstance(block, TextContent) and "\x1b]52" in block.text for block in answer.content)
    second = InteractiveSession(
        "--no-approve", "--api-key", "offline", "--session", str(saved),
        home=home, cwd=project, env=env, cols=200,
    )
    try:
        second.wait_for(r"before\x1b]52;c;dGVzdA==\x07after")
        assert b"\x1b]52" not in second.output
        assert b"\x1b[2J" not in second.output
        second.send(b"\x04")
        assert second.finish() == 0
        assert second.restored()
    finally:
        second.close()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_during_startup_restores_terminal(home: Path, tmp_path: Path, signum: int) -> None:
    ready = home / "startup-ready"
    session = InteractiveSession(
        "--no-approve", "--api-key", "offline", home=home, cwd=tmp_path,
        env=provider_env(home, "startup", INTERACTIVE_READY=str(ready)),
    )
    try:
        deadline = time.monotonic() + 8
        while not ready.exists() and time.monotonic() < deadline:
            session._pump(0.05)
        assert ready.exists()
        session.process.send_signal(signum)
        assert session.finish() == 128 + signum
        assert session.restored()
        assert sends(home) == []
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()


def test_failed_startup_rename_keeps_runtime_owned_and_closes_it(home: Path, tmp_path: Path) -> None:
    sessions = home / "sessions"
    first = InteractiveSession(
        "--no-approve", "--api-key", "offline", "--session-dir", str(sessions),
        home=home, cwd=tmp_path, env=provider_env(home, "echo"),
    )
    try:
        first.wait_for("phase input")
        first.send(b"first\r")
        first.wait_for("reply:first")
        first.wait_for("save saved")
        first.send(b"\x04")
        assert first.finish() == 0
    finally:
        first.close()
    saved = next(sessions.glob("*.jsonl"))
    second = InteractiveSession(
        "--no-approve", "--api-key", "offline", "--session", str(saved), "--name", "new name",
        home=home, cwd=tmp_path, env=provider_env(home, "rename-failure"),
    )
    try:
        second.wait_for("controlled rename failure")
        second.wait_for("save unsaved")
        second.wait_for("reply:first")
        second.wait_for("phase input")
        second.send(b"\x04")
        assert second.finish() == 0
        assert second.restored()
    finally:
        second.close()
    reopened = InteractiveSession(
        "--no-approve", "--api-key", "offline", "--session", str(saved),
        home=home, cwd=tmp_path, env=provider_env(home, "echo"),
    )
    try:
        reopened.wait_for("reply:first")
        reopened.send(b"again\r")
        reopened.wait_for("reply:again")
        reopened.send(b"\x04")
        assert reopened.finish() == 0
    finally:
        reopened.close()


def write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_multiline_editing_and_paste_do_not_autosubmit(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"first")
        session.send(b"\x0a")
        session.send("中文".encode())
        session.send(b"\x1b[13;2u")
        session.send(b"third")
        session.send(b"\x0a")
        session.send(b"\x1b[200~pasted\nlines\x1b[201~")
        session.wait_for("pasted")
        session.wait_for("中文")
        assert sends(home) == []
        assert list((home / "sessions").rglob("*.jsonl")) == []
        session.send(b"\r")
        session.wait_for("reply:first")
        assert "中文" in session.visible()
        assert "pasted" in session.visible()
        assert "lines" in session.visible()
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_trailing_backslash_enter_inserts_a_newline(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"one\\")
        session.send(b"\r")
        session.send(b"two\r")
        session.wait_for("reply:one")
        assert "two" in session.visible()
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_up_and_down_use_session_history_and_restore_the_draft(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"alpha\r")
        wait_sends(home, 1, session)
        settle(session)
        session.send(b"beta\r")
        wait_sends(home, 2, session)
        settle(session)
        session.send(b"\x1b[A")  # beta
        session.send(b"\r")
        wait_sends(home, 3, session)
        settle(session)
        session.send(b"\x1b[A")  # beta again (deduplicated)
        session.send(b"\x1b[A")  # alpha
        session.send(b"\r")
        wait_sends(home, 4, session)
        settle(session)
        session.send(b"my draft")
        session.send(b"\x1b[A")  # line start
        session.send(b"\x1b[A")  # history
        session.send(b"\x1b[B")  # restore the draft
        session.send(b"\r")
        wait_sends(home, 5, session)
        assert len(sends(home)) == 5
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_reopened_session_seeds_editor_history(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    sessions = home / "sessions"
    first = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(sessions), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        first.wait_for("phase input")
        first.send(b"saved prompt\r")
        first.wait_for("reply:saved prompt")
        first.send(b"\x04")
        assert first.finish() == 0
    finally:
        first.close()
    saved = next(sessions.glob("*.jsonl"))
    second = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(saved), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        second.wait_for("you saved prompt")
        second.send(b"\x1b[A")
        second.send(b"\r")
        wait_sends(home, 2, second)
        assert len(sends(home)) == 2
        second.send(b"\x04")
        assert second.finish() == 0
    finally:
        second.close()


def test_escape_closes_completion_and_keeps_the_edited_text(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/he")
        session.send(b"\t")
        session.wait_for("/help")
        session.send(b"\x1b")
        time.sleep(0.2)
        session._pump(0.1)
        session.send(b"\r")
        session.wait_for("Unknown command /he")
        session.wait_for("reply:/he")
        session.send(b"/he")
        session.send(b"\t")
        session.send(b"\t")
        session.send(b"\r")
        session.wait_for("Commands:")
        session.wait_for("/hotkeys")
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_skill_template_and_path_completion_expand_real_input(home: Path, tmp_path: Path) -> None:
    agent = home / ".omh" / "agent"
    skill = agent / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review the change\n---\nReview body text.\n"
    )
    prompts = agent / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "notes.md").write_text("---\ndescription: Notes\n---\nTEMPLATE BODY\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("x")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/skill:re")
        session.send(b"\t")
        session.send(b"\t")
        session.send(b"\r")
        session.wait_for("Review body text.")
        session.send(b"/not")
        session.send(b"\t")
        session.send(b"\t")
        session.send(b"\r")
        session.wait_for("TEMPLATE BODY")
        session.send(b"read @ap")
        session.send(b"\t")
        session.send(b"\t")
        session.send(b"\r")
        session.wait_for("reply:read @app.py")
        assert len(sends(home)) == 3
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_ctrl_g_external_editor_refills_without_submitting(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    editor = write_script(tmp_path / "fake-editor", "printf 'edited by script\\n' > \"$1\"")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project,
        env=provider_env(home, "echo", EDITOR=str(editor), VISUAL=""),
    )
    try:
        session.wait_for("phase input")
        session.send(b"draft text")
        session.send(b"\x07")
        session.wait_for("External editor returned")
        time.sleep(0.3)
        session._pump(0.1)
        assert sends(home) == []
        session.send(b"\r")
        session.wait_for("reply:edited by script")
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_ctrl_g_missing_editor_diagnoses_and_keeps_the_draft(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project,
        env=provider_env(home, "echo", EDITOR="/definitely/missing/editor", VISUAL=""),
    )
    try:
        session.wait_for("phase input")
        session.send(b"keep me")
        session.send(b"\x07")
        session.wait_for("command not found")
        session.send(b"\r")
        session.wait_for("reply:keep me")
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_copy_without_a_backend_diagnoses_and_keeps_the_ui(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    env = provider_env(home, "echo")
    env["PATH"] = str(empty)
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=env,
    )
    try:
        session.wait_for("phase input")
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        session.send(b"\x18")
        session.wait_for("No clipboard backend")
        session.send(b"next\r")
        session.wait_for("reply:next")
        assert len(sends(home)) == 2
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_copy_and_slash_copy_use_an_available_backend(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured = tmp_path / "clipboard.txt"
    backend = "pbcopy" if sys.platform == "darwin" else "wl-copy"
    write_script(bin_dir / backend, f"/bin/cat >> {captured}")
    env = provider_env(home, "echo")
    env["PATH"] = str(bin_dir)
    if sys.platform.startswith("linux"):
        env["WAYLAND_DISPLAY"] = "controlled-display"
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=env,
    )
    try:
        session.wait_for("phase input")
        session.send(b"hello\r")
        session.wait_for("reply:hello")
        session.send(b"\x18")
        session.wait_for("Copied the last assistant answer")
        assert captured.read_text().count("reply:hello") == 1
        session.send(b"/copy\r")
        session.wait_for("Copied the last assistant answer")
        deadline = time.monotonic() + 5
        while captured.read_text().count("reply:hello") < 2 and time.monotonic() < deadline:
            session._pump(0.1)
        assert captured.read_text().count("reply:hello") == 2
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_builtin_arguments_reserved_names_and_unknown_slash(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/copy extra\r")
        session.wait_for("takes no arguments")
        session.send(b"/new\r")
        session.wait_for("reserved for a later delivery")
        time.sleep(0.3)
        session._pump(0.1)
        assert sends(home) == []
        session.send(b"/mystery\r")
        session.wait_for("Unknown command /mystery")
        session.wait_for("reply:/mystery")
        assert len(sends(home)) == 1
        session.send(b"/help\r")
        session.wait_for("Commands:")
        session.wait_for("/hotkeys")
        session.send(b"/hotkeys\r")
        session.wait_for("Ctrl+G")
        time.sleep(0.2)
        session._pump(0.1)
        assert len(sends(home)) == 1
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_help_argument_completion_through_the_screen(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/help c")
        session.send(b"\t")
        session.wait_for("Copy the last assistant answer")
        session.send(b"\t")
        session.send(b"\r")
        session.wait_for("/copy: Copy the last assistant answer")
        assert sends(home) == []
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


def test_ctrl_c_latch_resets_after_new_typing(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"discarded")
        session.send(b"\x03")
        session.send(b"kept")
        session.send(b"\x03")
        session.send(b"sent\r")
        session.wait_for("reply:sent")
        assert "reply:kept" not in session.visible()
        assert session.process.poll() is None
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()
