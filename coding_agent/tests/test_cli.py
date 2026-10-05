"""Installed ``omh`` command behavior at the subprocess boundary."""

from __future__ import annotations

import json
import os
import pty
import shutil
import subprocess
import sys
from importlib.metadata import distribution, entry_points
from pathlib import Path

import pytest

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


def run_cli_tty(
    *args: str, home: Path, stdin_tty: bool = True, stdout_tty: bool = True,
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
    captured = b""
    if stdout_tty:
        while True:
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


def test_list_models_search_filters_the_directory(home: Path) -> None:
    matched = run_cli("--list-models", "flash", home=home)
    assert matched.returncode == 0
    assert "deepseek-flash" in matched.stdout
    assert "deepseek-v4-pro" not in matched.stdout

    unmatched = run_cli("--list-models", "no-such-model", home=home)
    assert unmatched.returncode == 0
    assert unmatched.stdout == ""


def test_list_sessions_is_read_only_but_undelivered(home: Path) -> None:
    result = run_cli("--list-sessions", home=home)
    assert result.returncode == EXIT_FAILURE
    assert result.stdout == ""
    assert "session listing" in result.stderr
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
    result = run_cli(
        "--append-system-prompt", "a",
        "--append-system-prompt-file", "b",
        "--skill", str(resources[0]),
        "--skill", str(resources[1]),
        "--prompt-template", str(resources[2]),
        home=home,
    )
    assert result.returncode == EXIT_FAILURE
    assert "text mode" in result.stderr


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
    assert result.returncode == EXIT_FAILURE
    assert "json mode" in result.stderr

    listed = run_cli("--list-sessions=work", home=home)
    assert listed.returncode == EXIT_FAILURE
    assert "session listing" in listed.stderr


def test_missing_value_before_another_option_is_rejected(home: Path) -> None:
    result = run_cli("--session", "--cwd", "/tmp", home=home)
    assert result.returncode == EXIT_USAGE
    assert "requires a value" in result.stderr


def test_double_dash_ends_options_and_file_arguments(home: Path) -> None:
    for arguments in (("--", "--unknown-option"), ("--", "-c"), ("--", "@not-a-file")):
        result = run_cli(*arguments, home=home)
        assert result.returncode == EXIT_FAILURE, arguments
        assert "text mode" in result.stderr, arguments


def test_thinking_is_validated_against_the_selected_model(home: Path) -> None:
    supported = run_cli("--model", "deepseek/deepseek-v4-pro", "--thinking", "off", home=home)
    assert supported.returncode == EXIT_FAILURE
    unsupported = run_cli("--model", "deepseek/deepseek-v4-pro", "--thinking", "xhigh", home=home)
    assert unsupported.returncode == EXIT_USAGE
    assert "not supported" in unsupported.stderr


@pytest.mark.parametrize("provider_args", [(), ("--provider", "deepseek")])
def test_bare_model_id_is_accepted(home: Path, provider_args: tuple[str, ...]) -> None:
    result = run_cli(*provider_args, "--model", "deepseek-flash", "prompt", home=home)
    assert result.returncode == EXIT_FAILURE
    assert "text mode" in result.stderr


def test_default_mode_needs_both_streams_to_be_ttys(home: Path) -> None:
    assert "interactive mode" in run_cli_tty(home=home)[2]
    assert "text mode" in run_cli(home=home).stderr
    assert "text mode" in run_cli_tty(home=home, stdout_tty=False)[2]
    assert "text mode" in run_cli_tty(home=home, stdin_tty=False)[2]


def test_explicit_modes_override_tty_inference(home: Path) -> None:
    assert "text mode" in run_cli_tty("--mode", "text", home=home)[2]
    assert "text mode" in run_cli_tty("--print", home=home)[2]
    assert "json mode" in run_cli_tty("--mode", "json", home=home)[2]
    code, _, errors = run_cli_tty("--mode", "interactive", home=home)
    assert code == EXIT_FAILURE
    assert "interactive mode" in errors


def test_explicit_interactive_without_a_terminal_fails_clearly(home: Path) -> None:
    result = run_cli("--mode", "interactive", home=home)
    assert result.returncode == EXIT_USAGE
    assert "terminal" in result.stderr
    assert result.stdout == ""


def test_credentials_never_appear_in_diagnostics(home: Path) -> None:
    result = run_cli("--api-key", "super-secret-value", "--unknown-option", home=home)
    assert result.returncode == EXIT_USAGE
    assert "super-secret-value" not in result.stderr
    assert "super-secret-value" not in result.stdout

    inline = run_cli("--api-key=super-secret-value", home=home)
    assert inline.returncode == EXIT_FAILURE
    assert "super-secret-value" not in inline.stderr
    assert "super-secret-value" not in inline.stdout


def test_product_distribution_installs_the_omh_script() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}
    assert scripts.get("omh") == "coding_agent.cli:main"


def test_sdk_distribution_has_no_product_script() -> None:
    sdk = distribution("omh")
    console_scripts = [entry for entry in sdk.entry_points if entry.group == "console_scripts"]
    assert all(entry.name != "omh" for entry in console_scripts)
