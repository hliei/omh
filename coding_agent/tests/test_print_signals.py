"""Cooperative print signals and complete-history rescue at the process boundary.

The installed ``omh`` process runs a controlled provider. The provider, tool or
close path writes a ``ready_<name>`` marker at a public synchronization point;
the test sends a real signal only after that marker, then releases a ``go_<name>``
file. Timing therefore never depends on guessed sleeps.
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from test_cli import clean_env, cli_command

from coding_agent import decode_history

PROVIDER = Path(__file__).with_name("fixtures") / "json_provider.py"

RESCUE_PATH = re.compile(r"rescued complete history to (\S+)")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def environment(home: Path, scenario: str, gate: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    site = home / "site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(PROVIDER.read_text())
    env = clean_env(home)
    env.update(
        PYTHONPATH=str(site),
        JSON_SCENARIO=scenario,
        JSON_SENDS=str(home / "sends"),
        JSON_GATE=str(gate),
        JSON_SESSIONS=str(home / "sessions"),
    )
    if extra:
        env.update(extra)
    return env


def launch(
    home: Path, tmp_path: Path, scenario: str, gate: Path, *prompts: str,
    mode: str = "--mode=json", extra: tuple[str, ...] = (), env_extra: dict[str, str] | None = None,
    session_dir: bool = True, stderr_path: Path | None = None,
) -> subprocess.Popen[str]:
    argv = [
        *cli_command(), mode, "--no-approve", "--no-context-files", "--cwd", str(tmp_path),
        "--api-key", "offline",
    ]
    if session_dir:
        argv += ["--session-dir", str(home / "sessions")]
    argv += [*extra, *prompts]
    gate.mkdir(parents=True, exist_ok=True)
    stderr_target: Any = subprocess.PIPE
    if stderr_path is not None:
        stderr_target = open(stderr_path, "w")
    try:
        return subprocess.Popen(
            argv, cwd=tmp_path, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=stderr_target, text=True,
            env=environment(home, scenario, gate, env_extra),
        )
    finally:
        if stderr_path is not None:
            stderr_target.close()


def wait_for(path: Path, process: subprocess.Popen[str], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            _, stderr = process.communicate()
            pytest.fail(f"process exited {process.returncode} before {path.name}\n{stderr}")
        time.sleep(0.02)
    process.kill()
    _, stderr = process.communicate()
    pytest.fail(f"timed out waiting for {path.name}\n{stderr}")


def finish(process: subprocess.Popen[str], timeout: float = 20.0) -> tuple[str, str]:
    stdout, stderr = process.communicate(timeout=timeout)
    return stdout, stderr


def records(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines()]


def wait_for_file_text(path: Path, needle: str, timeout: float = 15.0) -> str:
    """Poll a redirected stderr file until the process acknowledges the signal."""
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        text = path.read_text() if path.exists() else ""
        if needle in text:
            return text
        time.sleep(0.02)
    return text


def test_first_sigint_stops_model_work_and_exits_130(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    process = launch(home, tmp_path, "gate_model", gate, "first", "second", mode="--mode=text")
    wait_for(gate / "ready_model", process)
    process.send_signal(signal.SIGINT)

    stdout, stderr = finish(process)
    gate_go = gate / "go_model"
    gate_go.write_text("")
    assert process.returncode == 130, stderr
    assert stdout == ""
    assert "SIGINT" in stderr
    history = decode_history(next((home / "sessions").glob("*.jsonl")).read_bytes()).history
    users = [
        entry.message.content[0].text
        for entry in history.entries
        if hasattr(entry, "message") and entry.message.role == "user"
    ]
    assert users == ["first"]
    assistant = next(
        entry.message for entry in reversed(history.entries)
        if hasattr(entry, "message") and entry.message.role == "assistant"
    )
    assert assistant.stop_reason == "aborted"


def test_first_sigterm_cancels_tool_and_exits_143(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    process = launch(home, tmp_path, "gate_tool", gate, "use a tool")
    wait_for(gate / "ready_tool", process)
    process.send_signal(signal.SIGTERM)

    stdout, stderr = finish(process)
    (gate / "go_tool").write_text("")
    assert process.returncode == 143, stderr
    values = records(stdout)
    assert values[0]["type"] == "session"
    assert all(value["type"] not in {"result", "cancelled"} for value in values)
    lines = (home / "sends").read_text().splitlines()
    assert lines == ["dialogue"]


def test_first_sighup_cancels_retry_backoff_and_exits_129(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    process = launch(home, tmp_path, "signal_retry", gate, "first", "second")
    wait_for(gate / "ready_retry", process)
    process.send_signal(signal.SIGHUP)

    stdout, stderr = finish(process)
    assert process.returncode == 129, stderr
    assert all(value["type"] not in {"result", "cancelled"} for value in records(stdout))
    assert (home / "sends").read_text().splitlines() == ["dialogue"]


def test_first_signal_cancels_summary_and_exits_143(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    process = launch(home, tmp_path, "signal_summary", gate, "first", "second")
    wait_for(gate / "ready_summary", process)
    process.send_signal(signal.SIGTERM)

    stdout, stderr = finish(process)
    (gate / "go_summary").write_text("")
    assert process.returncode == 143, stderr
    assert all(value["type"] not in {"result", "cancelled"} for value in records(stdout))
    assert (home / "sends").read_text().splitlines() == ["dialogue"]


def test_first_signal_during_history_save_waits_for_the_save(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    stderr_path = tmp_path / "stderr.log"
    process = launch(home, tmp_path, "signal_save", gate, "first", stderr_path=stderr_path)
    wait_for(gate / "ready_save", process)
    process.send_signal(signal.SIGTERM)
    acknowledged = wait_for_file_text(stderr_path, "SIGTERM")
    assert "SIGTERM" in acknowledged, acknowledged

    # The gated save is owned work: it must finish before the cooperative exit.
    (gate / "go_save").write_text("")
    process.communicate(timeout=20)
    stderr = stderr_path.read_text()
    assert process.returncode == 143, stderr
    history = decode_history(next((home / "sessions").glob("*.jsonl")).read_bytes()).history
    users = [
        entry.message.content[0].text
        for entry in history.entries
        if hasattr(entry, "message") and entry.message.role == "user"
    ]
    assert users == ["first"]


def test_second_signal_forces_exit_and_warns_saving_may_be_incomplete(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    process = launch(home, tmp_path, "signal_close", gate, "use a tool")
    wait_for(gate / "ready_tool", process)
    process.send_signal(signal.SIGINT)
    wait_for(gate / "ready_close", process)
    process.send_signal(signal.SIGINT)

    _, stderr = finish(process)
    (gate / "go_tool").write_text("")
    (gate / "go_close").write_text("")
    assert process.returncode == 130, stderr
    assert "saving may be incomplete" in stderr


def test_auto_save_failure_rescues_complete_history_and_reopens(home: Path, tmp_path: Path) -> None:
    from omh.llm.types import ImageContent
    from PIL import Image

    gate = tmp_path / "gate"
    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    process = launch(
        home, tmp_path, "gate_tool", gate, "first", "second", mode="--mode=text",
        extra=("@image.png",),
    )
    wait_for(gate / "ready_tool", process)
    target = next((home / "sessions").glob("*.jsonl"))
    target.unlink()
    target.mkdir()
    (gate / "go_tool").write_text("")

    _, stderr = finish(process)
    assert process.returncode == 1, stderr
    assert (home / "sends").read_text().splitlines() == ["dialogue"]
    match = RESCUE_PATH.search(stderr)
    assert match is not None, stderr
    rescued = Path(match.group(1))
    assert rescued != target and rescued.is_file()
    history = decode_history(rescued.read_bytes()).history
    users = [
        entry.message
        for entry in history.entries
        if hasattr(entry, "message") and entry.message.role == "user"
    ]
    text = "".join(block.text for block in users[0].content if hasattr(block, "text"))
    assert text.endswith("first") and "second" not in text
    assert any(isinstance(block, ImageContent) for block in users[0].content)

    reopened = launch(
        home, tmp_path, "rich", gate, "continue", extra=("--session", str(rescued)), mode="--mode=text",
    )
    stdout, reopened_stderr = finish(reopened)
    assert reopened.returncode == 0, reopened_stderr
    assert "answer" in stdout


def test_failed_rescue_reports_no_complete_history(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        process = launch(
            home, tmp_path, "gate_tool", gate, "first", mode="--mode=text",
            env_extra={"JSON_TMPDIR": str(readonly)},
        )
        wait_for(gate / "ready_tool", process)
        target = next((home / "sessions").glob("*.jsonl"))
        target.unlink()
        target.mkdir()
        (gate / "go_tool").write_text("")

        _, stderr = finish(process)
        assert process.returncode == 1, stderr
        assert "no complete history" in stderr.lower()
    finally:
        readonly.chmod(0o700)


def test_no_session_does_not_rescue(home: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    process = launch(
        home, tmp_path, "rich", gate, "first", mode="--mode=text",
        extra=("--no-session",), session_dir=False,
    )
    stdout, stderr = finish(process)
    assert process.returncode == 0, stderr
    assert "answer" in stdout
    assert "rescue" not in stderr.lower()
    assert list(home.rglob("*.jsonl")) == []


@pytest.mark.parametrize("mode", ["--mode=json", "--mode=text"])
def test_full_stdout_pipe_still_handles_first_and_second_signals(
    home: Path, tmp_path: Path, mode: str,
) -> None:
    gate = tmp_path / "gate"
    stderr_path = tmp_path / "stderr"
    prompts = ("first", "second") if mode == "--mode=json" else ("first",)
    process = launch(home, tmp_path, "stdout_block", gate, *prompts,
                     mode=mode, stderr_path=stderr_path)
    try:
        wait_for(gate / "ready_stdout", process)
        process.send_signal(signal.SIGINT)
        acknowledged = wait_for_file_text(stderr_path, "received SIGINT")
        assert "received SIGINT" in acknowledged
        assert process.poll() is None
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 143
        assert "saving may be incomplete" in stderr_path.read_text()
        assert (home / "sends").read_text().splitlines() == ["dialogue"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_while_waiting_for_stdin_eof_exits_without_a_request(
    home: Path, tmp_path: Path, signum: int,
) -> None:
    site = home / "stdin-site"
    site.mkdir()
    ready = home / "stdin-ready"
    (site / "sitecustomize.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "class Input:\n"
        "    def isatty(self): return False\n"
        "    def read(self):\n"
        f"        Path({str(ready)!r}).write_text('ready')\n"
        "        return sys.__stdin__.read()\n"
        "sys.stdin = Input()\n"
    )
    env = clean_env(home)
    env["PYTHONPATH"] = str(site)
    process = subprocess.Popen(
        [*cli_command(), "--mode=text", "--no-approve", "--no-session", "--api-key", "offline"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmp_path, env=env, text=True,
    )
    try:
        wait_for(ready, process)
        process.send_signal(signum)
        assert process.wait(timeout=8) == 128 + signum
        stdout, stderr = process.communicate(timeout=2)
        assert stdout == ""
        assert "received" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("stderr_open", [True, False])
def test_unwritable_stderr_cannot_block_signal_abort_or_force_exit(
    home: Path, tmp_path: Path, stderr_open: bool,
) -> None:
    import os

    gate = tmp_path / "gate"
    site = home / "site"
    site.mkdir()
    ready = gate / "ready_stderr"
    script = PROVIDER.read_text() + """
import errno
_real_stderr_write = os.write
os.set_blocking(2, False)
while True:
    try:
        _real_stderr_write(2, b'x' * 4096)
    except OSError as error:
        if error.errno in {errno.EAGAIN, errno.EWOULDBLOCK, errno.EPIPE}:
            break
        raise
os.set_blocking(2, True)
Path(os.environ['JSON_GATE'], 'ready_stderr').write_text('ready')
"""
    (site / "sitecustomize.py").write_text(script)
    env = clean_env(home)
    env.update(PYTHONPATH=str(site), JSON_SCENARIO="signal_close", JSON_GATE=str(gate),
               JSON_SENDS=str(home / "sends"), JSON_SESSIONS=str(home / "sessions"))
    gate.mkdir()
    read_fd, write_fd = os.pipe()
    if not stderr_open:
        os.close(read_fd)
    process = subprocess.Popen(
        [*cli_command(), "--mode=json", "--no-approve", "--api-key", "offline",
         "--session-dir", str(home / "sessions"), "use a tool"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=write_fd, env=env,
        cwd=tmp_path, text=True,
    )
    os.close(write_fd)
    try:
        wait_for(ready, process)
        wait_for(gate / "ready_tool", process)
        process.send_signal(signal.SIGINT)
        wait_for(gate / "ready_close", process)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if stderr_open:
            os.close(read_fd)
