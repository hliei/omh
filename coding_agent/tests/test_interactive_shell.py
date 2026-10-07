"""User shell behavior through the installed command and real PTY."""

import json
import signal
from dataclasses import asdict
from pathlib import Path

import pytest
from omh.agent import (
    AgentOptions,
    CompactionSettings,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
    validate_history,
)
from omh.llm.types import ToolResultMessage
from support import OfflineStream, model
from test_interactive import InteractiveSession, provider_env, sends

from coding_agent import AgentSessionRuntime, CodingAgentOptions
from coding_agent.history import decode_history


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def start(home: Path, tmp_path: Path, *args: str, scenario: str = "echo") -> InteractiveSession:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"), "--cwd", str(project), *args,
        home=home, cwd=project, env=provider_env(
            home, scenario, INTERACTIVE_PAYLOADS=str(home / "payloads.jsonl"),
        ),
    )


@pytest.mark.parametrize("prefix", ["!", "!!"])
def test_shell_only_session_saves_and_reopens_without_execution(
    home: Path, tmp_path: Path, prefix: str,
) -> None:
    session = start(home, tmp_path, "--no-tools")
    command = "printf shell-output; printf x >> marker.txt"
    try:
        session.wait_for("phase input")
        session.send(f"{prefix}{command}\r".encode())
        session.wait_for("shell completed")
        session.wait_for("shell-output")
        session.wait_for("save saved")
        assert sends(home) == []
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()

    saved = next((home / "sessions").glob("*.jsonl"))
    assert "shell-output" in saved.read_text()
    assert (tmp_path / "project" / "marker.txt").read_text() == "x"
    reopened = start(home, tmp_path, "--session", str(saved), "--no-tools")
    try:
        reopened.wait_for("shell completed")
        reopened.wait_for("shell-output")
        assert sends(home) == []
        assert (tmp_path / "project" / "marker.txt").read_text() == "x"
        reopened.send(b"\x04")
        assert reopened.finish() == 0
    finally:
        reopened.close()


def test_shell_and_model_tools_are_parallel_and_custom_commits_after_results(
    home: Path, tmp_path: Path,
) -> None:
    session = start(home, tmp_path, scenario="shell-tools")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("progress model-ready")
        session.send(b"!printf user-output\r")
        session.wait_for("shell completed !printf user-output")
        assert "parallel done" not in session.visible()
        (tmp_path / "project" / "release-model").touch()
        session.wait_for("parallel done")
        session.send(b"\x04")
        assert session.finish() == 0
        saved = next((home / "sessions").glob("*.jsonl"))
        history = decode_history(saved.read_text()).history
        validate_history(history)
        entries = history.entries
        result = next(i for i, entry in enumerate(entries) if isinstance(entry, MessageHistoryEntry)
                      and isinstance(entry.message, ToolResultMessage))
        custom = next(i for i, entry in enumerate(entries) if isinstance(entry, CustomMessageHistoryEntry))
        assert result < custom
        payloads = (home / "payloads.jsonl").read_text().splitlines()
        assert len(payloads) == 2
        assert "user-output" in payloads[1]
    finally:
        session.close()


def test_second_shell_is_refused_and_escape_cancels_model_before_shell(
    home: Path, tmp_path: Path,
) -> None:
    session = start(home, tmp_path, scenario="hang")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("partial answer")
        session.send(b"!printf user-; printf ready; sleep 60\r")
        session.wait_for("user-ready")
        session.wait_for("| shell running")
        session.send(b"redirect\r")
        session.wait_for("queued steering (1 waiting): redirect")
        session.send(b"!touch forbidden.txt\r")
        session.wait_for("a user shell is already running")
        after = len(session.visible())
        session.send(b"\x1b")
        session.wait_for("recalled 1 queued input(s) into the editor")
        session.wait_for("phase input", after=after)
        session.wait_for("| shell running", after=after)
        assert "shell cancelled" not in session.visible()
        assert not (tmp_path / "project" / "forbidden.txt").exists()
        # The refused command remains editable. Clearing it does not stop the shell.
        session.send(b"\x03")
        session.send(b"\x1b")
        session.wait_for("shell cancelled")
        session.wait_for("Command aborted")
        session.send(b"\x04")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
        assert session.restored()
        saved = next((home / "sessions").glob("*.jsonl"))
        history = decode_history(saved.read_text()).history
        assert "redirect" not in saved.read_text()
        shell = next(entry for entry in history.entries if isinstance(entry, CustomMessageHistoryEntry))
        assert shell.details["status"] == "cancelled"
    finally:
        session.close()


@pytest.mark.parametrize("exit_signal, expected", [(None, 0), (signal.SIGINT, 130),
                                                  (signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_exit_joins_both_activities_and_saves_final_shell(
    home: Path, tmp_path: Path, exit_signal: int | None, expected: int,
) -> None:
    session = start(home, tmp_path, scenario="hang")
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("partial answer")
        session.send(b"!!printf exit-; printf ready; sleep 60\r")
        session.wait_for("exit-ready")
        session.wait_for("| shell running")
        if exit_signal is None:
            session.send(b"\x04")
        else:
            session.process.send_signal(exit_signal)
        assert session.finish() == expected
        assert session.restored()
        saved = next((home / "sessions").glob("*.jsonl"))
        history = decode_history(saved.read_text()).history
        validate_history(history)
        shell = next(entry for entry in history.entries if isinstance(entry, CustomMessageHistoryEntry))
        assert shell.content == ""
        assert shell.details["status"] == "cancelled"
        assert "exit-ready" in shell.details["output"]
        assert "Command aborted" in shell.details["output"]
    finally:
        session.close()


@pytest.mark.parametrize("failure", [False, True])
def test_bounded_output_and_spill_path_survive_success_or_failure(
    home: Path, tmp_path: Path, failure: bool,
) -> None:
    session = start(home, tmp_path)
    try:
        session.wait_for("phase input")
        command = "i=0; while [ $i -lt 2500 ]; do echo line-$i; i=$((i+1)); done"
        if failure:
            command += "; exit 7"
        session.send(f"!{command}\r".encode())
        session.wait_for("shell failed" if failure else "shell completed")
        session.wait_for("Full output:")
        session.send(b"\x04")
        assert session.finish() == 0
        saved = next((home / "sessions").glob("*.jsonl"))
        history = decode_history(saved.read_text()).history
        shell = next(entry for entry in history.entries if isinstance(entry, CustomMessageHistoryEntry))
        details = shell.details
        assert details["result"]["truncation"]["truncated"] is True
        assert "line-0\n" not in details["output"]
        assert "line-2499" in details["output"]
        path = Path(details["result"]["fullOutputPath"])
        assert len(path.read_text().splitlines()) == 2500
        if failure:
            assert "Command exited with code 7" in details["output"]
        path.unlink()
    finally:
        session.close()


def test_shell_context_inclusion_and_exclusion(home: Path, tmp_path: Path) -> None:
    session = start(home, tmp_path)
    try:
        session.wait_for("phase input")
        session.send(b"!printf included-output\r")
        session.wait_for("shell completed !printf included-output")
        session.wait_for("save saved")
        after = len(session.visible())
        session.send(b"!!printf hidden-output\r")
        session.wait_for("shell completed !!printf hidden-output", after=after)
        session.send(b"check\r")
        session.wait_for("reply:check")
        payloads = (home / "payloads.jsonl").read_text()
        assert "included-output" in payloads
        assert "hidden-output" not in payloads
        assert all(message["content"] for message in json.loads(payloads) if message["role"] == "user")
        session.send(b"\x04")
        assert session.finish() == 0
    finally:
        session.close()


async def test_hidden_shell_is_absent_from_reopened_requests_and_split_iterative_summaries(
    home: Path, tmp_path: Path,
) -> None:
    terminal = start(home, tmp_path)
    try:
        terminal.wait_for("phase input")
        terminal.send(b"!printf public-output\r")
        terminal.wait_for("shell completed !printf public-output")
        terminal.wait_for("save saved")
        terminal.send(b"!!printf secret-output\r")
        terminal.wait_for("shell completed !!printf secret-output")
        terminal.send(b"\x04")
        assert terminal.finish() == 0
    finally:
        terminal.close()
    saved = next((home / "sessions").glob("*.jsonl"))
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path / "project", model=model(), tools=(), stream_fn=stream,
        load_context_files=False,
        agent_options=AgentOptions(compaction=CompactionSettings(enabled=False, keep_recent_tokens=1)),
    ))
    session = await runtime.open_session(saved)
    try:
        hidden = next(message for message in session.agent.state.messages
                      if getattr(message, "custom_type", "") == "user_shell_hidden")
        assert hidden.content == ""
        assert "secret-output" in await session.export(format="html")
        await runtime.prompt("recent question " + "x" * 4000)
        first = await runtime.compact()
        assert "Turn Context (split turn)" in first.summary
        assert len(stream.requests) == 3  # Dialogue plus history and turn-prefix summaries.
        first_payloads = json.dumps([[asdict(message) for message in context.messages]
                                     for _, context, _ in stream.requests])
        assert "public-output" in first_payloads
        assert "secret-output" not in first_payloads
        await runtime.prompt("next question " + "y" * 4000)
        await runtime.compact()
        payloads = json.dumps([[asdict(message) for message in context.messages]
                              for _, context, _ in stream.requests])
        assert "Update the existing structured summary" in payloads
        assert "secret-output" not in payloads
    finally:
        await runtime.close()


def test_unsaved_session_refuses_a_second_shell_without_side_effects(
    home: Path, tmp_path: Path,
) -> None:
    session = start(home, tmp_path)
    try:
        session.wait_for("phase input")
        (home / "sessions").write_text("blocks the saving directory")
        session.send(b"!printf initial-output\r")
        session.wait_for("shell completed")
        session.wait_for("save unsaved")
        session.send(b"!touch forbidden.txt\r")
        session.wait_for("Session has unsaved history")
        assert not (tmp_path / "project" / "forbidden.txt").exists()
        assert sends(home) == []
        session.send(b"\x03\x03")
        session.wait_for("unsubmitted content remains")
        session.send(b"\r")
        assert session.finish() == 0
    finally:
        session.close()
