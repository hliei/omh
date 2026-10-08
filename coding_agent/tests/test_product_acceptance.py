"""Installed coding journeys across resource reload, derivation and modes.

These integration checks join previously separate public CLI/PTY flows. They
use the existing controlled providers and real files; actual desktop and live
provider acceptance remain separate requirements.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest
from omh.agent import MessageHistoryEntry
from test_cli import run_cli
from test_interactive import InteractiveSession, provider_env, sends
from test_json_print import controlled, records
from test_print_signals import finish, launch, wait_for

from coding_agent import decode_history


def test_coding_clone_reload_and_print_preserve_work_and_history(tmp_path: Path) -> None:
    home, project = tmp_path / "home", tmp_path / "project"
    project.mkdir()
    prompts = home / ".omh" / "agent" / "prompts"
    prompts.mkdir(parents=True)
    template = prompts / "notes.md"
    template.write_text("FIRST $ARGUMENTS")
    (project / "app.py").write_text("value = 1\n")
    requests = home / "requests.txt"
    env = provider_env(home, "tools", INTERACTIVE_REQUESTS=str(requests))
    ui = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session-dir", str(home / "sessions"),
        home=home, cwd=project, env=env,
    )
    try:
        ui.wait_for("phase input")
        ui.send(b"fix the test\r")
        for tool in ("read", "edit", "write", "bash"):
            ui.wait_for(f"tool {tool} ok")
        ui.wait_for("Fixed the value")
        ui.wait_for("thinking collapsed")
        ui.wait_for("phase input", after=ui.visible().index("Fixed the value"))
        assert (project / "app.py").read_text() == "value = 2\n"
        assert (project / "note.txt").read_text() == "noted"
        ui.send(b"!python3 -c \"assert open('app.py').read() == 'value = 2\\n'\" && printf passed > checks.txt\r")
        ui.wait_for("shell settled")
        assert (project / "checks.txt").read_text() == "passed"
        source_path = next((home / "sessions").glob("*.jsonl"))
        source_bytes = source_path.read_bytes()
        source = decode_history(source_bytes).history

        ui.send(b"/clone\r")
        ui.wait_for("cloned current")
        target_path = next(path for path in (home / "sessions").glob("*.jsonl")
                           if path != source_path)
        target = decode_history(target_path.read_bytes())
        assert target.history.conversation_id != source.conversation_id
        assert target.history.entries == source.entries
        assert target.source is not None and target.source.conversation_id == source.conversation_id

        template.write_text("SECOND $ARGUMENTS")
        before_reload = sends(home)
        ui.send(b"/reload\r")
        ui.wait_for("resources reloaded; next new prompt; accepted inputs unchanged")
        assert sends(home) == before_reload
        offset = len(ui.visible())
        ui.send(b"/notes marker\r")
        ui.wait_for("Fixed the value", after=offset)
        ui.wait_for("phase input", after=offset)
        assert requests.read_text().splitlines()[-1] == "SECOND marker"
        ui.send(b"/quit\r")
        assert ui.finish() == 0
        assert ui.restored()
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()

    saved = decode_history(target_path.read_bytes()).history
    file_times = [(project / name).stat().st_mtime_ns for name in ("app.py", "note.txt")]
    result = run_cli(
        "--print", "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(target_path), "continue", home=home, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "reply:continue"
    continued = decode_history(target_path.read_bytes()).history
    assert continued.conversation_id == saved.conversation_id
    assert continued.entries[:len(saved.entries)] == saved.entries
    assert not any(isinstance(entry, MessageHistoryEntry) and entry.message.role == "toolResult"
                   for entry in continued.entries[len(saved.entries):])
    assert [(project / name).stat().st_mtime_ns for name in ("app.py", "note.txt")] == file_times
    assert source_path.read_bytes() == source_bytes


@pytest.mark.parametrize("outcome", ["error", "aborted", "signal"])
def test_failed_print_history_continues_through_interactive_and_json(
    tmp_path: Path, outcome: str,
) -> None:
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    if outcome == "signal":
        gate = project / "gate"
        process = launch(home, project, "gate_model", gate, "first", "must not run")
        try:
            wait_for(gate / "ready_model", process)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = finish(process)
            assert process.returncode == 143, stderr
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    else:
        result = controlled(home, project, outcome, "first", "must not run")
        assert result.returncode == 0, result.stderr
        stdout = result.stdout

    assert json.loads(stdout.splitlines()[0])["type"] == "session"
    path = next((home / "sessions").glob("*.jsonl"))
    original = decode_history(path.read_bytes()).history
    users = [entry.message for entry in original.entries
             if isinstance(entry, MessageHistoryEntry) and entry.message.role == "user"]
    assert len(users) == 1 and "first" in str(users[0].content)
    assert "must not run" not in path.read_text()
    finals = [entry.message for entry in original.entries
              if isinstance(entry, MessageHistoryEntry) and entry.message.role == "assistant"]
    assert finals[-1].stop_reason == ("aborted" if outcome == "signal" else outcome)
    env = provider_env(home, "echo")
    ui = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(path), home=home, cwd=project, env=env,
    )
    try:
        ui.wait_for("phase input")
        ui.send(b"recovered\r")
        ui.wait_for("reply:recovered")
        ui.wait_for("phase input", after=ui.visible().index("reply:recovered"))
        ui.send(b"/quit\r")
        assert ui.finish() == 0
        assert ui.restored()
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()
    result = run_cli(
        "--mode=json", "--no-approve", "--no-context-files", "--api-key", "offline",
        "--session", str(path), "final check", home=home, env=env,
    )
    assert result.returncode == 0, result.stderr
    events = records(result)
    assert events[0]["id"] == original.conversation_id
    final = [event["message"] for event in events
             if event["type"] == "message_end" and event["message"]["role"] == "assistant"]
    assert final[-1]["stopReason"] == "stop"
    assert final[-1]["content"] == [{"type": "text", "text": "reply:final check"}]
    restored = decode_history(path.read_bytes()).history
    assert restored.conversation_id == original.conversation_id
    assert restored.entries[:len(original.entries)] == original.entries
    assert sends(home) == ["dialogue", "dialogue"]
