"""Installed ``omh`` command behavior at the subprocess boundary."""

from __future__ import annotations

import json
import os
import pty
import select
import shutil
import subprocess
import sys
import time
from importlib.metadata import distribution, entry_points
from pathlib import Path

import pytest
from test_session_directory import write_history

EXIT_FAILURE = 1
EXIT_USAGE = 2


def cli_command() -> list[str]:
    script = Path(sys.executable).with_name("omh")
    if script.exists():
        return [str(script)]
    found = shutil.which("omh")
    if found is not None:
        return [found]
    raise RuntimeError("the installed omh command is required for this test")


def clean_env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(home)
    env.pop("DEEPSEEK_API_KEY", None)
    env.pop("OPENCODE_API_KEY", None)
    env.pop("NO_COLOR", None)
    return env


def run_cli(
    *args: str, home: Path, stdin: str | None = None, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command_env = clean_env(home)
    if env:
        command_env.update(env)
    return subprocess.run(
        [*cli_command(), *args],
        input=stdin,
        stdin=subprocess.DEVNULL if stdin is None else None,
        capture_output=True,
        text=True,
        env=command_env,
    )


#: Injected into the installed process through ``sitecustomize`` so the real
#: console script runs offline against the host's public HTTP boundary.
CONTROLLED_FETCH = '''\
import json

_count = 0

async def _fetch(request):
    global _count
    from omh.llm.types import FetchResponse
    _count += 1
    chunk = {"choices": [{"delta": {"content": "answer-%d" % _count}, "finish_reason": "stop"}]}
    body = "data: " + json.dumps(chunk) + chr(10) + chr(10) + "data: [DONE]" + chr(10)
    return FetchResponse(status=200, headers={"content-type": "text/event-stream"}, text=body)

import coding_agent.cli as cli
_real = cli.CodingAgentHost

def _host(*args, **kwargs):
    kwargs["fetch"] = _fetch
    return _real(*args, **kwargs)

cli.CodingAgentHost = _host
'''

#: A controlled provider that always fails, for the process-boundary exit code.
CONTROLLED_ERROR_FETCH = '''\
import json

async def _fetch(request):
    from omh.llm.types import FetchResponse
    body = json.dumps({"error": {"message": "bad key"}})
    return FetchResponse(status=401, headers={"content-type": "application/json"}, text=body)

import coding_agent.cli as cli
_real = cli.CodingAgentHost

def _host(*args, **kwargs):
    kwargs["fetch"] = _fetch
    return _real(*args, **kwargs)

cli.CodingAgentHost = _host
'''


def run_cli_controlled(
    *args: str, home: Path, cwd: Path, stdin: str = "", script: str = CONTROLLED_FETCH,
) -> subprocess.CompletedProcess[str]:
    """Run the installed command in a real process with a controlled provider."""
    site = home / "site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(script)
    env = clean_env(home)
    env["PYTHONPATH"] = str(site)
    return subprocess.run(
        [*cli_command(), *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
    )


def run_cli_tty(
    *args: str, home: Path, stdin_tty: bool = True, stdout_tty: bool = True,
    keys: bytes | None = None, timeout: float | None = None,
) -> tuple[int, str, str]:
    master, slave = pty.openpty()
    try:
        process = subprocess.Popen(
            [*cli_command(), *args],
            stdin=slave if stdin_tty else subprocess.DEVNULL,
            stdout=slave if stdout_tty else subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env(home),
        )
    finally:
        os.close(slave)
    if keys and stdin_tty:
        os.write(master, keys)
    captured = b""
    if stdout_tty:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None and time.monotonic() > deadline:
                process.kill()
                break
            if deadline is None:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
            else:
                ready, _, _ = select.select([master], [], [], 0.1)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
            if not chunk:
                break
            captured += chunk
    os.close(master)
    if not stdout_tty:
        assert process.stdout is not None
        captured = process.stdout.read()
        process.stdout.close()
    assert process.stderr is not None
    errors = process.stderr.read()
    process.stderr.close()
    code = process.wait()
    return code, captured.decode(errors="replace").replace("\r\n", "\n"), errors.decode(errors="replace")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


def assert_no_terminal_noise(text: str) -> None:
    assert "\x1b[" not in text
    assert "Traceback" not in text


def test_version_is_read_only(home: Path) -> None:
    result = run_cli("--version", home=home)
    assert result.returncode == 0
    assert result.stdout.strip() != ""
    assert result.stderr == ""
    assert list(home.iterdir()) == []
    assert_no_terminal_noise(result.stdout)


def test_help_lists_delivered_entry_points(home: Path) -> None:
    result = run_cli("--help", home=home)
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    for option in ("--list-models", "--mode", "--session-dir", "--cwd", "--no-approve"):
        assert option in result.stdout
    assert result.stderr == ""
    assert_no_terminal_noise(result.stdout)


def test_list_models_succeeds_without_key(home: Path) -> None:
    result = run_cli("--list-models", home=home)
    assert result.returncode == 0
    assert "deepseek-flash" in result.stdout
    assert "deepseek-v4-pro" in result.stdout
    assert "deepseek-v4.1-flash" in result.stdout
    assert "openai-completions" in result.stdout
    assert "builtin" in result.stdout
    assert "2026-10-05" in result.stdout
    assert result.stderr == ""
    assert list(home.iterdir()) == []
    assert_no_terminal_noise(result.stdout)


def test_session_listing_and_print_resume_are_read_only(home, tmp_path):
    root = home / ".omh" / "agent" / "sessions"
    project, other = tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    a, b = root / "group" / "a.jsonl", root / "b.jsonl"
    write_history(a, project, "id-one", "Zulu", "find this needle")
    write_history(b, other, "id-two", "Alpha", "another conversation")
    # Invalid configuration must not prevent a history-only query.
    config = root.parent / "settings.json"
    config.write_text("invalid settings")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in home.rglob("*") if path.is_file()}
    result = run_cli("--list-sessions", "--cwd", str(project), home=home)
    assert result.returncode == 0 and result.stderr == ""
    assert "id-one" in result.stdout and "id-two" not in result.stdout
    result = run_cli("--list-sessions", "--all-projects", "--sort", "name", home=home)
    assert result.returncode == 0 and result.stderr == ""
    assert result.stdout.index("id-two") < result.stdout.index("id-one")
    result = run_cli("--list-sessions", "needle", "--all-projects", home=home)
    assert "id-one" in result.stdout and "id-two" not in result.stdout
    result = run_cli("-p", "-r", "--cwd", str(project), home=home)
    assert result.returncode == 0 and "id-one" in result.stdout
    assert "selector" not in result.stdout.lower() and result.stderr == ""
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in home.rglob("*") if path.is_file()} == before


def test_continue_and_ambiguous_session_diagnostics_without_writes(home, tmp_path):
    root = tmp_path / "history"
    project = tmp_path / "project"
    project.mkdir()
    result = run_cli("-c", "--cwd", str(project), "--session-dir", str(root), home=home)
    assert "No saved session" in result.stderr and "new session" in result.stderr
    assert not root.exists() and list(home.iterdir()) == []
    a, b = root / "a.jsonl", root / "b.jsonl"
    write_history(a, project, "abc-one", "one\n\x1b[31m", "older")
    write_history(b, project, "abc-two", "two", "newer")
    os.utime(a, (10, 10))
    os.utime(b, (20, 20))
    result = run_cli("-c", "--cwd", str(project), "--session-dir", str(root), home=home)
    assert str(b) in result.stderr and str(a) not in result.stderr
    result = run_cli("--session", "abc", "--session-dir", str(root), home=home)
    assert result.returncode == EXIT_USAGE
    assert "Ambiguous" in result.stderr and str(a) in result.stderr and str(b) in result.stderr
    assert len(result.stderr.splitlines()) == 1
    assert "one\\n\\x1b[31m" in result.stderr
    assert_no_terminal_noise(result.stderr)
    listed = run_cli("--list-sessions", "--all-projects", "--session-dir", str(root), home=home)
    assert len(listed.stdout.splitlines()) == 3
    assert "one\\n\\x1b[31m" in listed.stdout
    assert_no_terminal_noise(listed.stdout)


@pytest.mark.parametrize("args", [
    ("--sort", "mtime"), ("--all-projects",), ("--reverse",),
    ("--list-sessions", "--sort", "unknown"),
    ("--list-sessions", "--cwd", "/does-not-exist/omh"),
    ("--list-sessions", "--sort", "name", "--sort", "id"),
])
def test_session_listing_rejects_invalid_options_before_writes(home, args):
    result = run_cli(*args, home=home)
    assert result.returncode == EXIT_USAGE
    assert result.stdout == "" and "omh:" in result.stderr
    assert list(home.iterdir()) == []


def test_list_models_reads_global_models_json_without_writing(home: Path, tmp_path: Path) -> None:
    agent_dir = tmp_path / "omh-agent"
    agent_dir.mkdir()
    (agent_dir / "models.json").write_text(json.dumps({
        "providers": {
            "deepseek": {
                "modelOverrides": {"deepseek-flash": {"cost": {"input": 7.5, "output": 9.0}}},
            },
        },
    }))
    result = run_cli(
        "--list-models", "deepseek-flash",
        home=home, env={"OMH_CODING_AGENT_DIR": str(agent_dir)},
    )
    assert result.returncode == 0
    assert "7.5/9" in result.stdout
    assert "user" in result.stdout
    assert_no_terminal_noise(result.stdout)


def test_list_models_surfaces_models_json_diagnostics(home: Path, tmp_path: Path) -> None:
    agent_dir = tmp_path / "omh-agent"
    agent_dir.mkdir()
    (agent_dir / "models.json").write_text(json.dumps({
        "providers": {"deepseek": {"models": [{"id": "incomplete"}]}},
    }))
    result = run_cli("--list-models", home=home, env={"OMH_CODING_AGENT_DIR": str(agent_dir)})
    assert result.returncode == 0
    assert "incomplete" in result.stderr
    assert "deepseek-flash" in result.stdout
    assert_no_terminal_noise(result.stderr)


def test_list_models_covers_the_eight_registered_combinations(home: Path) -> None:
    result = run_cli("--list-models", home=home)
    assert result.returncode == 0
    rows = {
        tuple(line.split()[:2]): line.split()[2:]
        for line in result.stdout.splitlines()[1:]
    }
    assert set(rows) == {
        ("deepseek", "deepseek-flash"),
        ("deepseek", "deepseek-v4-pro"),
        ("opencode-go", "deepseek-v4.1-flash"),
        ("opencode-go", "deepseek-v4-pro"),
        ("opencode-go", "glm-5.3"),
        ("opencode-go", "glm-5.3-flash"),
        ("opencode-go", "kimi-k3"),
        ("opencode-go", "kimi-k2.7-code"),
    }
    for columns in rows.values():
        api, _input, _thinking, _context, _output, _cost, source, date = columns
        assert api == "openai-completions"
        assert source == "builtin"
        assert date == "2026-10-05"

    def thinking_and_shape(provider: str, model: str) -> tuple[str, str, str, str]:
        api, _input, thinking, context, output, _cost, _source, _date = rows[(provider, model)]
        return (_input, thinking, context, output)

    assert thinking_and_shape("opencode-go", "glm-5.3") == ("text", "low,high,max", "1000000", "131072")
    assert thinking_and_shape("opencode-go", "glm-5.3-flash") == (
        "text,image", "low,high,max", "1000000", "131072",
    )
    assert thinking_and_shape("opencode-go", "kimi-k3") == ("text,image", "max", "1048576", "131072")
    # Fixed-on: no adjustable level and no fabricated ``off``.
    assert thinking_and_shape("opencode-go", "kimi-k2.7-code") == ("text,image", "-", "262144", "262144")


def test_list_models_search_filters_the_directory(home: Path) -> None:
    matched = run_cli("--list-models", "flash", home=home)
    assert matched.returncode == 0
    assert "deepseek-flash" in matched.stdout
    assert "deepseek-v4-pro" not in matched.stdout

    unmatched = run_cli("--list-models", "no-such-model", home=home)
    assert unmatched.returncode == 0
    assert unmatched.stdout == ""


def test_empty_session_listing_is_read_only(home: Path) -> None:
    result = run_cli("--list-sessions", home=home)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(home.iterdir()) == []
    assert_no_terminal_noise(result.stderr)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--unknown-option",),
        ("-z",),
        ("--model",),
        ("--model", "--provider"),
        ("--session", "--cwd"),
        ("--session", "--cwd=/tmp"),
        ("--name", "--unknown-option"),
        ("--session=", "prompt"),
        ("--provider",),
        ("--thinking",),
        ("--thinking", "--model"),
        ("--skill", "/no/such/explicit-skill-path"),
        ("--prompt-template", "/no/such/explicit-template-path"),
        ("--system-prompt-file", "/no/such/system-prompt-file"),
        ("--append-system-prompt-file", "/no/such/append-prompt-file"),
        ("--help=yes",),
        ("--list-models=flash", "--list-models"),
        ("--mode",),
        ("--mode", "bogus"),
        ("--thinking", "bogus"),
        ("--model", "one", "--model", "two"),
        ("--provider", "deepseek", "--model", "opencode-go/glm-5.3"),
        ("--model", "deepseek/does-not-exist"),
        ("--provider", "does-not-exist"),
        ("--model", "deepseek/"),
        ("--tools", "grep"),
        ("--tools", "read,read"),
        ("--tools", "read", "--no-tools"),
        ("--approve", "--no-approve"),
        ("--session", "one", "--continue"),
        ("--session", "one", "--resume"),
        ("--continue", "--resume"),
        ("--no-session", "--session", "one"),
        ("--no-session", "--continue"),
        ("--no-session", "--session-dir", "somewhere"),
        ("--print", "--mode", "interactive"),
        ("--system-prompt", "text", "--system-prompt-file", "file"),
        ("--help", "--tools", "read", "--no-tools"),
        ("--version", "-p", "--mode", "interactive"),
    ],
)
def test_pre_request_rejections_exit_two_without_stdout(home: Path, arguments: tuple[str, ...]) -> None:
    result = run_cli(*arguments, home=home)
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""
    assert result.stderr.startswith("omh: ")
    assert_no_terminal_noise(result.stderr)


def test_repeatable_options_are_accepted(home: Path, tmp_path: Path) -> None:
    resources = []
    for name in ("one", "two", "three"):
        path = tmp_path / f"{name}.md"
        path.write_text(f"---\nname: {name}\n---\nbody\n")
        resources.append(path)
    appended = tmp_path / "appended.md"
    appended.write_text("literal append")
    result = run_cli(
        "--no-approve",
        "--append-system-prompt", "a",
        "--append-system-prompt-file", str(appended),
        "--skill", str(resources[0]),
        "--skill", str(resources[1]),
        "--prompt-template", str(resources[2]),
        home=home,
    )
    # Every repeatable option was accepted; print then stops before any request
    # because no credential is configured.
    assert result.returncode == EXIT_USAGE
    assert "No API key" in result.stderr
    assert "cannot be repeated" not in result.stderr


def test_equals_form_and_existing_resource_paths_are_accepted(home: Path, tmp_path: Path) -> None:
    resource = tmp_path / "skill.md"
    resource.write_text("---\nname: example\n---\nbody\n")
    result = run_cli(
        "--mode=json",
        "--model=deepseek/deepseek-flash",
        "--thinking=high",
        "--skill",
        str(resource),
        home=home,
    )
    assert result.returncode == EXIT_USAGE
    assert "No API key" in result.stderr

    listed = run_cli("--list-sessions=work", home=home)
    assert listed.returncode == 0
    assert listed.stdout == listed.stderr == ""


def test_missing_value_before_another_option_is_rejected(home: Path) -> None:
    result = run_cli("--session", "--cwd", "/tmp", home=home)
    assert result.returncode == EXIT_USAGE
    assert "requires a value" in result.stderr


def test_double_dash_ends_options_and_file_arguments(home: Path) -> None:
    for arguments in (("--unknown-option",), ("-c",), ("@not-a-file",)):
        result = run_cli("--no-approve", "--", *arguments, home=home)
        # After -- every token is literal task text, so no flag, option or
        # attachment interpretation error appears; print stops at the missing key.
        assert result.returncode == EXIT_USAGE, arguments
        assert "No API key" in result.stderr, arguments
        assert "unknown option" not in result.stderr, arguments
        assert "attachment" not in result.stderr, arguments


def test_thinking_is_validated_against_the_selected_model(home: Path) -> None:
    supported = run_cli("--no-approve", "--model", "deepseek/deepseek-v4-pro", "--thinking", "off", home=home)
    assert supported.returncode == EXIT_USAGE
    assert "not supported" not in supported.stderr
    unsupported = run_cli("--model", "deepseek/deepseek-v4-pro", "--thinking", "xhigh", home=home)
    assert unsupported.returncode == EXIT_USAGE
    assert "not supported" in unsupported.stderr


@pytest.mark.parametrize(
    ("model", "level"),
    [
        ("opencode-go/glm-5.3", "medium"),
        ("opencode-go/glm-5.3-flash", "minimal"),
        ("opencode-go/kimi-k3", "high"),
        ("opencode-go/kimi-k2.7-code", "high"),
        ("opencode-go/kimi-k2.7-code", "off"),
    ],
)
def test_go_thinking_is_validated_against_each_models_real_levels(home: Path, model: str, level: str) -> None:
    result = run_cli("--model", model, "--thinking", level, "prompt", home=home)
    assert result.returncode == EXIT_USAGE
    assert "not supported" in result.stderr or "fixed thinking mode" in result.stderr


@pytest.mark.parametrize(
    ("model", "level"),
    [
        ("opencode-go/glm-5.3", "low"),
        ("opencode-go/glm-5.3-flash", "max"),
        ("opencode-go/kimi-k3", "max"),
    ],
)
def test_go_supported_thinking_passes_selection(home: Path, model: str, level: str) -> None:
    result = run_cli("--no-approve", "--model", model, "--thinking", level, "prompt", home=home)
    assert result.returncode == EXIT_USAGE
    assert "No API key" in result.stderr
    assert "not supported" not in result.stderr


@pytest.mark.parametrize("provider_args", [(), ("--provider", "deepseek")])
def test_bare_model_id_is_accepted(home: Path, provider_args: tuple[str, ...]) -> None:
    result = run_cli("--no-approve", *provider_args, "--model", "deepseek-flash", "prompt", home=home)
    assert result.returncode == EXIT_USAGE
    assert "No API key" in result.stderr


def test_default_mode_needs_both_streams_to_be_ttys(home: Path) -> None:
    code, output, errors = run_cli_tty(home=home, keys=b"\x04", timeout=8)
    assert code == 0
    assert "No API key" in output
    assert "not available" not in output
    assert "not available" not in errors
    assert "No API key" in run_cli("--no-approve", home=home).stderr
    assert "No API key" in run_cli_tty("--no-approve", home=home, stdout_tty=False)[2]
    assert "No API key" in run_cli_tty("--no-approve", home=home, stdin_tty=False)[2]


def test_explicit_modes_override_tty_inference(home: Path) -> None:
    assert "No API key" in run_cli_tty("--no-approve", "--mode", "text", home=home)[2]
    assert "No API key" in run_cli_tty("--no-approve", "--print", home=home)[2]
    assert "No API key" in run_cli_tty("--mode", "json", "--no-approve", home=home)[2]
    code, output, errors = run_cli_tty("--mode", "interactive", home=home, keys=b"\x04", timeout=8)
    assert code == 0
    assert "No API key" in output
    assert "not available" not in errors


def test_explicit_interactive_without_a_terminal_fails_clearly(home: Path) -> None:
    result = run_cli("--mode", "interactive", home=home)
    assert result.returncode == EXIT_USAGE
    assert "terminal" in result.stderr
    assert result.stdout == ""


def test_print_whitespace_stdin_is_no_task_without_a_request(home: Path) -> None:
    result = run_cli(
        "--no-approve", "-p", home=home, stdin="  \n\t\n", env={"OPENCODE_API_KEY": "test-key"},
    )
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""
    assert "no task" in result.stderr


def test_print_missing_key_is_rejected_before_a_request(home: Path) -> None:
    result = run_cli("--no-approve", "-p", "hello", home=home)
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""
    assert "No API key" in result.stderr


def test_installed_print_outputs_only_the_final_answer(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    result = run_cli_controlled(
        "--no-approve", "-p", "--cwd", str(project), "--api-key", "k", "first task", "second task",
        home=home, cwd=project,
    )
    assert result.returncode == 0
    assert result.stdout == "answer-2"
    assert result.stderr == ""
    assert_no_terminal_noise(result.stdout)
    saved = list((home / ".omh" / "agent" / "sessions").rglob("*.jsonl"))
    assert len(saved) == 1
    content = saved[0].read_text()
    assert "first task" in content and "second task" in content


def test_installed_print_model_error_exits_one_without_stdout(home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    result = run_cli_controlled(
        "--no-approve", "-p", "--cwd", str(project), "--api-key", "k", "hello",
        home=home, cwd=project, script=CONTROLLED_ERROR_FETCH,
    )
    assert result.returncode == EXIT_FAILURE
    assert result.stdout == ""
    assert "401" in result.stderr
    assert_no_terminal_noise(result.stderr)


def test_credentials_never_appear_in_diagnostics(home: Path) -> None:
    result = run_cli("--api-key", "super-secret-value", "--unknown-option", home=home)
    assert result.returncode == EXIT_USAGE
    assert "super-secret-value" not in result.stderr
    assert "super-secret-value" not in result.stdout

    inline = run_cli("--no-approve", "--api-key=super-secret-value", home=home)
    assert inline.returncode == EXIT_USAGE
    assert "super-secret-value" not in inline.stderr
    assert "super-secret-value" not in inline.stdout
    assert "no task" in inline.stderr


def test_product_distribution_installs_the_omh_script() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}
    assert scripts.get("omh") == "coding_agent.cli:main"


def test_sdk_distribution_has_no_product_script() -> None:
    sdk = distribution("omh")
    console_scripts = [entry for entry in sdk.entry_points if entry.group == "console_scripts"]
    assert all(entry.name != "omh" for entry in console_scripts)
