"""Print stdout backpressure, temporary retries and permanent write failures.

The installed command runs against real pipes with a controlled provider. A
sitecustomize wrapper injects ``EAGAIN`` at the ``os.write`` file-descriptor
boundary or records the bytes that actually reached stdout, so the tests
observe the public process output and exit code instead of private internals.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
from test_cli import clean_env, cli_command


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path

#: Controlled provider plus an ``os.write`` boundary wrapper. ``OMH_STDOUT_INJECT``
#: rejects that many fd-1 writes with EAGAIN before succeeding; ``OMH_STDOUT_LOG``
#: records every accepted write; ``OMH_RELEASE`` makes the model wait until the
#: test has closed stdout.
STDOUT_SCRIPT = '''\
import errno
import os
import time

from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream

import coding_agent.cli as cli
import coding_agent.print_runner as runner
from coding_agent.agent_session_runtime import AgentSessionRuntime

_INJECT = int(os.environ.get("OMH_STDOUT_INJECT", "0"))
_FAIL_AFTER = int(os.environ.get("OMH_STDOUT_FAIL_AFTER", "-1"))
_LOG = os.environ.get("OMH_STDOUT_LOG")
_INJECT_LOG = os.environ.get("OMH_STDOUT_INJECT_LOG")
_DELTAS = int(os.environ.get("OMH_PROVIDER_DELTAS", "1"))
_RELEASE = os.environ.get("OMH_RELEASE")
_SLOW_CLOSE = float(os.environ.get("OMH_SLOW_CLOSE", "0"))
_SEND_LOG = os.environ.get("OMH_SEND_LOG")
_real_write = os.write
_injections_left = _INJECT
_injections = 0
_accepted = 0


def _write(fd, data):
    global _injections_left, _injections, _accepted
    if fd == 1:
        if _FAIL_AFTER >= 0 and _accepted >= _FAIL_AFTER:
            raise BrokenPipeError(errno.EPIPE, "controlled permanent stdout failure")
        if _injections_left > 0:
            _injections_left -= 1
            _injections += 1
            if _INJECT_LOG:
                with open(_INJECT_LOG, "a") as handle:
                    handle.write("injected\\n")
            raise BlockingIOError(errno.EAGAIN, "controlled temporary stdout rejection")
        _accepted += 1
    written = _real_write(fd, data)
    if fd == 1 and _LOG:
        with open(_LOG, "ab") as handle:
            handle.write(bytes(data))
    return written


os.write = _write


def stream_fn(model, context, options):
    if _SEND_LOG:
        with open(_SEND_LOG, "a") as handle:
            handle.write("dialogue\\n")
    if _RELEASE:
        deadline = time.monotonic() + 20
        while not os.path.exists(_RELEASE) and time.monotonic() < deadline:
            time.sleep(0.01)
    output = AssistantMessage(api=model.api, provider=model.provider, model=model.id,
                              usage=empty_usage(), stop_reason="pending", timestamp=1000)
    stream = create_assistant_message_event_stream()
    stream.push(StartEvent(partial=output))
    text = TextContent(text="")
    output.content.append(text)
    stream.push(TextStartEvent(content_index=0, partial=output))
    for _ in range(_DELTAS):
        text.text += "x"
        stream.push(TextDeltaEvent(content_index=0, delta="x", partial=output))
    stream.push(TextEndEvent(content_index=0, content=text.text, partial=output))
    output.stop_reason = "stop"
    stream.push(DoneEvent(reason="stop", message=output))
    return stream


class ControlledHost(cli.CodingAgentHost):
    def build_options(self, selection, **kwargs):
        options = super().build_options(selection, **kwargs)
        options.stream_fn = stream_fn
        return options


cli.CodingAgentHost = ControlledHost

if _SLOW_CLOSE:
    class SlowCloseRuntime(AgentSessionRuntime):
        async def close(self):
            time.sleep(_SLOW_CLOSE)

    runner.AgentSessionRuntime = SlowCloseRuntime
'''


def installed_command(home: Path, tmp_path: Path, *args: str) -> tuple[list[str], dict[str, str]]:
    site = home / "site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(STDOUT_SCRIPT)
    env = clean_env(home)
    env["PYTHONPATH"] = str(site)
    command = [
        *cli_command(), "--no-approve", "--no-context-files", "--cwd", str(tmp_path),
        "--session-dir", str(home / "sessions"), "--api-key", "offline", *args,
    ]
    return command, env


def run_installed(
    home: Path, tmp_path: Path, *args: str, extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command, env = installed_command(home, tmp_path, *args)
    env.update(extra_env or {})
    return subprocess.run(
        command, cwd=tmp_path, input=b"", capture_output=True, env=env, timeout=40,
    )


def final_assistant_text(stdout: bytes) -> str:
    values = [json.loads(line) for line in stdout.splitlines()]
    finals = [
        event["message"] for event in values
        if event["type"] == "message_end" and event["message"]["role"] == "assistant"
    ]
    assert finals, "no authoritative final assistant message"
    return "".join(
        block["text"] for block in finals[-1]["content"] if block["type"] == "text"
    )


def test_json_retries_temporary_stdout_rejections(tmp_path: Path, home: Path) -> None:
    write_log = tmp_path / "writes.bin"
    injection_log = tmp_path / "injections.txt"
    send_log = tmp_path / "sends.txt"
    result = run_installed(
        home, tmp_path, "--mode=json", "hello",
        extra_env={
            "OMH_STDOUT_INJECT": "3", "OMH_STDOUT_LOG": str(write_log),
            "OMH_STDOUT_INJECT_LOG": str(injection_log), "OMH_PROVIDER_DELTAS": "5",
            "OMH_SEND_LOG": str(send_log),
        },
    )
    assert result.returncode == 0, result.stderr
    assert b"Traceback" not in result.stderr
    assert injection_log.read_text().splitlines() == ["injected"] * 3
    # Temporary stdout retries stay inside one model response.
    assert send_log.read_text().splitlines() == ["dialogue"]
    # Every accepted write is one piece of the final stream: retries neither
    # dropped nor duplicated bytes.
    assert write_log.read_bytes() == result.stdout
    assert final_assistant_text(result.stdout) == "xxxxx"


def test_json_slow_reader_sees_complete_ordered_stream(tmp_path: Path, home: Path) -> None:
    command, env = installed_command(home, tmp_path, "--mode=json", "hello")
    env["OMH_PROVIDER_DELTAS"] = "1500"
    process = subprocess.Popen(
        command, cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env,
    )
    assert process.stdout is not None
    lines: list[bytes] = []
    with process.stdout:
        for index, line in enumerate(process.stdout):
            lines.append(line)
            if index % 100 == 0:
                time.sleep(0.005)
    code = process.wait(timeout=40)
    assert code == 0, process.stderr.read() if process.stderr else b""
    values = [json.loads(line) for line in lines]
    assert values[0]["type"] == "session"
    assert values[-1] == {"type": "agent_settled"}
    assert final_assistant_text(b"".join(lines)) == "x" * 1500


def test_json_closed_stdout_pipe_exits_one_without_slow_cleanup(
    tmp_path: Path, home: Path,
) -> None:
    release = tmp_path / "release"
    command, env = installed_command(home, tmp_path, "--mode=json", "hello")
    env.update({
        "OMH_RELEASE": str(release), "OMH_PROVIDER_DELTAS": "50", "OMH_SLOW_CLOSE": "30",
    })
    process = subprocess.Popen(
        command, cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env,
    )
    assert process.stdout is not None
    header = json.loads(process.stdout.readline())
    assert header["type"] == "session"
    process.stdout.close()
    release.write_text("go")
    started = time.monotonic()
    code = process.wait(timeout=40)
    elapsed = time.monotonic() - started
    assert code == 1, process.stderr.read() if process.stderr else b""
    assert elapsed < 30, "permanent stdout failure waited for cooperative close"
    assert b"Traceback" not in (process.stderr.read() if process.stderr else b"")


def test_json_write_failure_after_send_does_not_retry_the_model(
    tmp_path: Path, home: Path,
) -> None:
    send_log = tmp_path / "sends.txt"
    command, env = installed_command(home, tmp_path, "--mode=json", "hello")
    env.update({
        "OMH_STDOUT_FAIL_AFTER": "12", "OMH_PROVIDER_DELTAS": "50",
        "OMH_SLOW_CLOSE": "30", "OMH_SEND_LOG": str(send_log),
    })
    started = time.monotonic()
    process = subprocess.run(
        command, cwd=tmp_path, input=b"", capture_output=True, env=env, timeout=40,
    )
    elapsed = time.monotonic() - started
    assert process.returncode == 1, process.stderr
    assert elapsed < 30, "permanent stdout failure waited for cooperative close"
    assert b"Traceback" not in process.stderr
    # The request that was already in flight stays a single send; the write
    # failure is not treated as a retryable model response failure.
    assert send_log.read_text().splitlines() == ["dialogue"]
    prefix = [json.loads(line) for line in process.stdout.splitlines()]
    assert prefix[0]["type"] == "session"
    assert not any(event["type"] in {"result", "cancelled"} for event in prefix)


def test_text_closed_stdout_pipe_exits_one(tmp_path: Path, home: Path) -> None:
    release = tmp_path / "release"
    command, env = installed_command(home, tmp_path, "--mode=text", "hello")
    env["OMH_RELEASE"] = str(release)
    process = subprocess.Popen(
        command, cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env,
    )
    assert process.stdout is not None
    process.stdout.close()
    release.write_text("go")
    code = process.wait(timeout=40)
    assert code == 1, process.stderr.read() if process.stderr else b""
    assert code != 141
    assert b"Traceback" not in (process.stderr.read() if process.stderr else b"")


def test_text_normal_finish_flushes_the_final_answer(tmp_path: Path, home: Path) -> None:
    result = run_installed(home, tmp_path, "--mode=text", "hello")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"x"
    assert result.stderr == b""


def test_json_normal_finish_flushes_every_line(tmp_path: Path, home: Path) -> None:
    result = run_installed(
        home, tmp_path, "--mode=json", "hello", extra_env={"OMH_PROVIDER_DELTAS": "25"},
    )
    assert result.returncode == 0, result.stderr
    values = [json.loads(line) for line in result.stdout.splitlines()]
    assert values[-1] == {"type": "agent_settled"}
    assert final_assistant_text(result.stdout) == "x" * 25
