"""Strict print wire through the installed command and controlled providers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from test_cli import run_cli_controlled


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


SCHEMA = Path(__file__).parents[1] / "docs" / "print-json.schema.json"
VALIDATOR = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())


def records(result):
    assert "Traceback" not in result.stderr
    assert "\x1b" not in result.stdout
    values = [json.loads(line) for line in result.stdout.splitlines()]
    for value in values:
        VALIDATOR.validate(value)
    return values


def test_installed_json_initial_delta_and_final_message(home: Path, tmp_path: Path) -> None:
    result = run_cli_controlled(
        "--mode=json", "--no-approve", "--cwd", str(tmp_path), "--api-key", "secret-test-key",
        "first", "second", home=home, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    values = records(result)
    header = values[0]
    assert header.keys() == {"type", "version", "id", "timestamp", "cwd"}
    assert header["type"] == "session" and header["version"] == 3
    assert header["cwd"] == str(tmp_path)
    assert datetime.fromisoformat(header["timestamp"]).tzinfo is not None
    assert "secret-test-key" not in result.stdout + result.stderr
    rebuilt = None
    finals = []
    for event in values[1:]:
        assert "version" not in event
        assert event["type"] not in {"result", "history_commit", "model_change", "thinking_level_change"}
        if event["type"] == "message_start" and event["message"]["role"] == "assistant":
            rebuilt = event["message"]["content"]
            assert rebuilt == []
        elif event["type"] == "message_update":
            assert event.keys() == {"type", "usage", "assistantMessageEvent"}
            delta = event["assistantMessageEvent"]
            assert "partial" not in delta
            if delta["type"] == "text_start":
                rebuilt.append({"type": "text", "text": ""})
            elif delta["type"] == "text_delta":
                rebuilt[delta["contentIndex"]]["text"] += delta["delta"]
        elif event["type"] == "message_end" and event["message"]["role"] == "assistant":
            assert rebuilt == event["message"]["content"]
            finals.append(event["message"])
    assert [message["content"][0]["text"] for message in finals] == ["answer-1", "answer-2"]
    assert all(message["stopReason"] == "stop" for message in finals)
    assert values[-1] == {"type": "agent_settled"}


PROVIDER = Path(__file__).with_name("fixtures") / "json_provider.py"


def controlled_env(home: Path, scenario: str):
    from test_cli import clean_env

    site = home / "site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(PROVIDER.read_text())
    env = clean_env(home)
    env.update(PYTHONPATH=str(site), JSON_SCENARIO=scenario, JSON_SENDS=str(home / "sends"),
               JSON_SESSIONS=str(home / "sessions"))
    return env


def controlled(home: Path, tmp_path: Path, scenario: str, *prompts: str, extra=()):
    import subprocess

    from test_cli import cli_command

    env = controlled_env(home, scenario)
    return subprocess.run(
        [*cli_command(), "--mode=json", "--no-approve", "--no-context-files", "--cwd", str(tmp_path),
         "--session-dir", str(home / "sessions"), "--api-key", "offline", *extra, *prompts],
        cwd=tmp_path, input="", capture_output=True, text=True, env=env, timeout=15,
    )


def test_rich_messages_signatures_usage_and_opaque_keys(home: Path, tmp_path: Path) -> None:
    result = controlled(home, tmp_path, "rich", "use a tool")
    assert result.returncode == 0, result.stderr
    values = records(result)
    final = [event["message"] for event in values if event["type"] == "message_end"]
    assistant = next(message for message in final if message["role"] == "assistant")
    assert assistant["content"] == [
        {"type": "thinking", "thinking": "consider", "thinkingSignature": "replay-token", "redacted": False},
        {"type": "toolCall", "id": "call-1", "name": "bash",
         "arguments": {"command": "printf progress", "snake_key": {"inner_key": 1}},
         "thoughtSignature": "thought-token", "namespace": "shell"},
    ]
    assert assistant["usage"] == {
        "input": 7, "output": 5, "cacheRead": 2, "cacheWrite": 3, "cacheWrite1h": 1, "reasoning": 4,
        "totalTokens": 17, "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.3, "cacheWrite": 0.4, "total": 1},
    }
    assert assistant["responseModel"] == "concrete-model" and assistant["responseId"] == "response-1"
    assert assistant["providerThinkingLevel"] == "high"
    tool = next(message for message in final if message["role"] == "toolResult")
    assert tool["details"] == {"snake_key": {"inner_key": "opaque"}, "null_key": None}
    assert tool["content"] == [
        {"type": "text", "text": "tool answer", "textSignature": "tool-text"},
        {"type": "image", "data": "AQID", "mimeType": "image/png"},
    ]
    start = next(event["assistantMessageEvent"] for event in values
                 if event["type"] == "message_update" and event["assistantMessageEvent"]["type"] == "toolcall_start")
    assert start == {"type": "toolcall_start", "contentIndex": 1, "id": "call-1", "toolName": "bash"}
    assert any(event["type"] == "tool_execution_update" for event in values)
    assert all("terminate" not in event.get("result", {}) for event in values if event["type"] == "tool_execution_end")


@pytest.mark.parametrize("scenario", ["error", "aborted"])
def test_message_failure_returns_zero_and_stops_prompts(home: Path, tmp_path: Path, scenario: str) -> None:
    result = controlled(home, tmp_path, scenario, "first", "must not run")
    assert result.returncode == 0, result.stderr
    values = records(result)
    last = [event["message"] for event in values if event["type"] == "message_end"][-1]
    assert last["stopReason"] == scenario and last["errorMessage"]
    assert (home / "sends").read_text().splitlines() == ["dialogue"]
    assert "must not run" not in next((home / "sessions").glob("*.jsonl")).read_text()


@pytest.mark.parametrize("scenario,code", [("retry", 0), ("intent_abort", 0), ("intent_listener", 1), ("intent_save", 1)])
def test_retry_intent_does_not_wait_for_actual_retry(home: Path, tmp_path: Path, scenario: str, code: int) -> None:
    result = controlled(home, tmp_path, scenario, "first", "must not run" if scenario != "retry" else "next")
    assert result.returncode == code, result.stderr
    values = records(result)
    ends = [event for event in values if event["type"] == "agent_end"]
    assert ends[0]["willRetry"] is True
    assert ends[0]["messages"][-1]["stopReason"] == "error"
    sends = (home / "sends").read_text().splitlines()
    if scenario == "retry":
        assert sends == ["dialogue"] * 3
        retry_start = next(event for event in values if event["type"] == "auto_retry_start")
        assert retry_start == {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3,
                               "delayMs": 0, "errorMessage": "503 service unavailable"}
        assert next(event for event in values if event["type"] == "auto_retry_end") == {
            "type": "auto_retry_end", "success": True, "attempt": 1,
        }
    else:
        assert sends == ["dialogue"]
        if scenario != "intent_save":
            assert not any(event["type"] == "auto_retry_start" for event in values)


@pytest.mark.parametrize("scenario,code", [("compact", 0), ("compact_error", 0)])
def test_compaction_and_summary_retry_use_fixed_events(home: Path, tmp_path: Path, scenario: str, code: int) -> None:
    result = controlled(home, tmp_path, scenario, "first", "second")
    assert result.returncode == code, result.stderr
    values = records(result)
    start = next(event for event in values if event["type"] == "compaction_start")
    assert start == {"type": "compaction_start", "reason": "threshold"}
    scheduled = next(event for event in values if event["type"] == "summarization_retry_scheduled")
    assert scheduled == {"type": "summarization_retry_scheduled", "attempt": 1, "maxAttempts": 3,
                         "delayMs": 0, "errorMessage": "503 service unavailable"}
    assert next(event for event in values if event["type"] == "summarization_retry_finished") == {
        "type": "summarization_retry_finished",
    }
    end = next(event for event in values if event["type"] == "compaction_end")
    assert end["aborted"] is False and end["willRetry"] is False
    if scenario == "compact":
        assert end["result"]["summary"] == "No prior history.\n\n---\n\n**Turn Context (split turn):**\n\nsummary"
        assert end["result"]["usage"]["totalTokens"] == 17
        assert "errorMessage" not in end
    else:
        assert "result" not in end and end["errorMessage"]


@pytest.mark.parametrize("scenario", ["prompt_failure", "notification_failure", "close_failure"])
def test_thrown_failures_exit_one_with_a_valid_prefix(home: Path, tmp_path: Path, scenario: str) -> None:
    result = controlled(home, tmp_path, scenario, "first", "must not run")
    assert result.returncode == 1
    assert result.stderr.startswith("omh: ")
    values = records(result)
    assert values[0]["type"] == "session"
    assert not any(value["type"] == "result" for value in values)
    if scenario != "prompt_failure":
        assert (home / "sends").read_text().splitlines() == ["dialogue"] * (2 if scenario == "close_failure" else 1)


def test_json_reopen_does_not_replay_history(home: Path, tmp_path: Path) -> None:
    first = controlled(home, tmp_path, "retry", "old input")
    assert first.returncode == 0, first.stderr
    path = next((home / "sessions").glob("*.jsonl"))
    result = controlled(home, tmp_path, "retry", "new input", extra=("--session", str(path)))
    assert result.returncode == 0, result.stderr
    values = records(result)
    assert values[0] == records(first)[0]
    users = [event["message"]["content"] for event in values
             if event["type"] == "message_end" and event["message"]["role"] == "user"]
    assert users == [[{"type": "text", "text": "new input"}]]
    assert "new input" in path.read_text() and "old input" in path.read_text()
    from coding_agent import decode_history
    with pytest.raises(ValueError, match="unrecognized history format"):
        decode_history(result.stdout)


GOLDEN = Path(__file__).parents[1] / "docs" / "print-json.examples.jsonl"


def test_fixed_golden_messages_and_deltas_match_installed_wire(home: Path, tmp_path: Path) -> None:
    Draft202012Validator.check_schema(VALIDATOR.schema)
    examples = [json.loads(line) for line in GOLDEN.read_text().splitlines()]
    for example in examples:
        VALIDATOR.validate(example)
    values = records(controlled(home, tmp_path, "rich", "first"))
    expected_assistants = [event for event in examples
                           if event["type"] == "message_end" and event["message"]["role"] == "assistant"
                           and event["message"]["stopReason"] in ("toolUse", "stop")]
    actual_assistants = [event for event in values if event["type"] == "message_end"
                         and event["message"]["role"] == "assistant"]
    assert actual_assistants == expected_assistants
    # Compare each shape to independent literals, without fixing the full trace
    # or progress count (SDK awaits listeners and tool progress may coalesce).
    golden_updates = [event for event in examples if event["type"] == "message_update"]
    actual_updates = [event for event in values if event["type"] == "message_update"]
    assert len(actual_updates) == len(golden_updates)
    for update in actual_updates:
        assert update in golden_updates
    assert next(event for event in values if event["type"] == "message_start"
                and event["message"]["role"] == "assistant") in examples


def test_no_session_header_and_stderr_are_separate(home: Path, tmp_path: Path) -> None:
    result = run_cli_controlled(
        "--mode=json", "--no-approve", "--cwd", str(tmp_path), "--no-session", "--api-key", "offline",
        "first", home=home, cwd=tmp_path,
    )
    assert result.returncode == 0
    assert "in-memory" in result.stderr
    assert records(result)[0].keys() == {"type", "version", "id", "timestamp", "cwd"}
    assert list(home.rglob("*.jsonl")) == []


def test_input_failure_after_header_is_a_valid_prefix(home: Path, tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    result = controlled(home, tmp_path, "rich", "@image.png", "look",
                        extra=("--model", "deepseek/deepseek-v4-pro"))
    assert result.returncode == 2 and "does not accept image" in result.stderr
    assert [event["type"] for event in records(result)] == ["session"]
    assert not (home / "sends").exists()


@pytest.mark.parametrize("extra", [("--unknown",), ("--thinking", "bogus"), ("@missing",)])
def test_preflight_failure_has_only_stderr(home: Path, tmp_path: Path, extra: tuple[str, ...]) -> None:
    result = controlled(home, tmp_path, "rich", *extra)
    assert result.returncode == 2
    assert records(result) == [] and result.stderr.startswith("omh: ")
    assert not (home / "sends").exists()


def test_provider_exception_encoded_by_sdk_is_a_zero_exit_message_failure(home: Path, tmp_path: Path) -> None:
    result = controlled(home, tmp_path, "runtime_failure", "first", "must not run")
    assert result.returncode == 0
    values = records(result)
    last = [event["message"] for event in values if event["type"] == "message_end"][-1]
    assert last["stopReason"] == "error" and last["errorMessage"] == "controlled runtime failure"
    assert (home / "sends").read_text().splitlines() == ["dialogue"]


@pytest.mark.parametrize("scenario,code,output", [("rich", 0, "answer"), ("error", 1, ""), ("prompt_failure", 1, "")])
def test_documented_consumer_checks_message_and_process(home: Path, tmp_path: Path, scenario: str, code: int, output: str) -> None:
    import os
    import subprocess
    import sys

    from test_cli import cli_command

    env = controlled_env(home, scenario)
    env["PATH"] = str(Path(cli_command()[0]).parent) + os.pathsep + env["PATH"]
    example = Path(__file__).parents[1] / "examples" / "consume_print_json.py"
    result = subprocess.run(
        [sys.executable, str(example), "--no-approve", "--cwd", str(tmp_path),
         "--session-dir", str(home / "sessions"), "--api-key", "offline", "first"],
        input="", capture_output=True, text=True, env=env, timeout=15,
    )
    assert result.returncode == code, result.stderr
    assert result.stdout == output


@pytest.mark.parametrize("extra", [
    {"message": {}}, {"partial": {}}, {"version": 3}, {"taskNumber": 1},
])
def test_schema_rejects_extra_update_envelopes(extra: dict[str, object]) -> None:
    example = next(json.loads(line) for line in GOLDEN.read_text().splitlines()
                   if json.loads(line)["type"] == "message_update")
    assert list(VALIDATOR.iter_errors({**example, **extra}))


def test_installed_json_user_image_and_saved_history(home: Path, tmp_path: Path) -> None:
    from PIL import Image

    from coding_agent import decode_history

    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    result = controlled(home, tmp_path, "rich", "@image.png", "look")
    assert result.returncode == 0, result.stderr
    values = records(result)
    user = next(event["message"] for event in values
                if event["type"] == "message_end" and event["message"]["role"] == "user")
    image = next(block for block in user["content"] if block["type"] == "image")
    assert image.keys() == {"type", "data", "mimeType"}
    assert image["mimeType"] == "image/png" and image["data"]
    history = decode_history(next((home / "sessions").glob("*.jsonl")).read_bytes()).history
    assert any(entry.message.role == "user" for entry in history.entries if hasattr(entry, "message"))


def test_close_exception_overrides_input_failure_exit(home: Path, tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    result = controlled(home, tmp_path, "close_failure", "@image.png", "look",
                        extra=("--model", "deepseek/deepseek-v4-pro"))
    assert result.returncode == 1
    assert "does not accept image" in result.stderr and "controlled close failure" in result.stderr
    assert [event["type"] for event in records(result)] == ["session"]
