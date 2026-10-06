"""Print text behavior through the public print entry and installed command.

Model-dependent behavior is observed against the common host's injectable HTTP
boundary with the real runtime, session files and tools. Command-level input,
configuration and read-only behavior is observed through ``coding_agent.cli``
with a temporary HOME; the installed subprocess boundary re-checks the same
pre-request contract in ``test_cli.py``.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from omh.agent import AgentHistory, MessageHistoryEntry
from omh.llm.types import FetchFunction, TextContent
from PIL import Image
from support import (
    RecordingFetch,
    SequencedFetch,
    json_error_response,
    text_stream,
    tool_call_stream,
)

import coding_agent.cli as cli
from coding_agent import CodingAgentHost, decode_history
from coding_agent.attachments import read_file_attachment
from coding_agent.print_runner import PrintInputError, compose_tasks, run_print_text
from coding_agent.session_directory import SessionDirectory

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def make_host(tmp_path: Path, fetch: FetchFunction, **kwargs: object) -> CodingAgentHost:
    agent_dir = kwargs.pop("agent_dir", tmp_path / "agent")
    return CodingAgentHost(
        startup_dir=tmp_path, agent_dir=agent_dir, api_key="offline-key", fetch=fetch, **kwargs,  # type: ignore[arg-type]
    )


async def run_task_chain(
    host: CodingAgentHost, tasks, **selection_kwargs: object,
) -> tuple[int, str, str]:
    selection = host.select_new(**selection_kwargs)  # type: ignore[arg-type]
    stdout, stderr = StringIO(), StringIO()
    code = await run_print_text(host, selection, tasks, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def user_texts(history: AgentHistory) -> list[str]:
    """Return every user message text in saved history order."""
    texts: list[str] = []
    for entry in history.entries:
        if not isinstance(entry, MessageHistoryEntry) or entry.message.role != "user":
            continue
        content = entry.message.content
        texts.append(content if isinstance(content, str) else "".join(
            block.text for block in content if isinstance(block, TextContent)
        ))
    return texts


# --------------------------------------------------------------------------- #
# Input composition
# --------------------------------------------------------------------------- #


def test_compose_tasks_orders_stdin_attachments_and_first_prompt(tmp_path: Path) -> None:
    attachment = tmp_path / "code.py"
    attachment.write_text("print(1)\n")
    tasks = compose_tasks(
        stdin_text="piped line\n",
        attachments=[read_file_attachment("code.py", cwd=tmp_path)],
        prompts=("fix it", "then explain"),
    )
    assert len(tasks) == 2
    first = tasks[0].text
    assert first.index("piped line") < first.index('<file name="') < first.index("print(1)") < first.index("fix it")
    assert "\n\n" in first
    assert tasks[1].text == "then explain" and tasks[1].images == ()


def test_compose_tasks_treats_whitespace_as_no_task() -> None:
    with pytest.raises(PrintInputError):
        compose_tasks(stdin_text="  \n\t\n", prompts=("", "   "))
    assert compose_tasks(stdin_text="  ", prompts=("real",))[0].text == "real"


# --------------------------------------------------------------------------- #
# Serial task execution and stdout
# --------------------------------------------------------------------------- #


async def test_serial_prompts_output_only_the_last_answer(tmp_path: Path) -> None:
    fetch = SequencedFetch(text_stream("first answer"), text_stream("second answer"))
    code, stdout, stderr = await run_task_chain(
        make_host(tmp_path, fetch), compose_tasks(prompts=("first task", "second task")),
    )
    assert code == EXIT_OK
    assert stdout == "second answer"
    assert stderr == ""
    assert len(fetch.requests) == 2
    second = json.dumps(fetch.bodies[1])
    assert "first task" in second and "first answer" in second and "second task" in second


async def test_first_task_combines_piped_input_and_attachment(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("attached body")
    fetch = RecordingFetch(text_stream("done"))
    tasks = compose_tasks(
        stdin_text="piped body\n",
        attachments=[read_file_attachment("notes.txt", cwd=tmp_path)],
        prompts=("explain",),
    )
    code, stdout, _ = await run_task_chain(make_host(tmp_path, fetch), tasks)
    assert code == EXIT_OK and stdout == "done"
    body = fetch.bodies[0]["messages"][-1]["content"]
    text = body if isinstance(body, str) else "".join(
        block.get("text", "") for block in body if isinstance(block, dict)
    )
    assert text.index("piped body") < text.index("attached body") < text.index("explain")


async def test_image_attachment_is_sent_as_real_image_content(tmp_path: Path) -> None:
    Image.new("RGB", (4, 2), (1, 2, 3)).save(tmp_path / "picture.png")
    fetch = RecordingFetch(text_stream("seen"))
    tasks = compose_tasks(
        attachments=[read_file_attachment("picture.png", cwd=tmp_path)], prompts=("what is this",),
    )
    code, stdout, _ = await run_task_chain(make_host(tmp_path, fetch), tasks)
    assert code == EXIT_OK and stdout == "seen"
    sent = json.dumps(fetch.bodies[0])
    assert "image_url" in sent and "data:image/png;base64," in sent


async def test_image_for_text_only_model_is_rejected_before_request(tmp_path: Path) -> None:
    Image.new("RGB", (4, 2)).save(tmp_path / "picture.png")
    fetch = RecordingFetch(text_stream("unused"))
    tasks = compose_tasks(
        attachments=[read_file_attachment("picture.png", cwd=tmp_path)], prompts=("look",),
    )
    code, stdout, stderr = await run_task_chain(
        make_host(tmp_path, fetch), tasks, model="deepseek/deepseek-v4-pro",
    )
    assert code == EXIT_USAGE and stdout == ""
    assert "does not accept image input" in stderr
    assert fetch.requests == []


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #


async def test_transient_error_is_retried_before_judging_the_task(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"retry": {"baseDelayMs": 0}}))
    fetch = SequencedFetch(
        json_error_response(500, {"error": {"message": "internal server error"}}),
        text_stream("recovered"),
    )
    code, stdout, stderr = await run_task_chain(
        make_host(tmp_path, fetch, agent_dir=agent_dir), compose_tasks(prompts=("go",)),
    )
    assert code == EXIT_OK and stdout == "recovered" and stderr == ""
    assert len(fetch.requests) == 2


async def test_final_error_stops_remaining_prompts_without_history(tmp_path: Path) -> None:
    fetch = SequencedFetch(json_error_response(401, {"error": {"message": "bad key"}}))
    host = make_host(tmp_path, fetch)
    code, stdout, stderr = await run_task_chain(
        host, compose_tasks(prompts=("first", "second")),
    )
    assert code == EXIT_FAILURE and stdout == ""
    assert "401" in stderr
    assert len(fetch.requests) == 1
    # The unconsumed second prompt never entered history.
    saved = next((tmp_path / "agent" / "sessions").rglob("*.jsonl"))
    assert user_texts(decode_history(saved.read_bytes()).history) == ["first"]


async def test_mid_chain_failure_stops_later_prompts_and_hides_earlier_answer(
    tmp_path: Path,
) -> None:
    fetch = SequencedFetch(
        text_stream("first answer"),
        json_error_response(401, {"error": {"message": "bad key"}}),
    )
    host = make_host(tmp_path, fetch)
    code, stdout, stderr = await run_task_chain(
        host, compose_tasks(prompts=("first", "second", "third")),
    )
    assert code == EXIT_FAILURE and stdout == ""
    assert "401" in stderr
    assert len(fetch.requests) == 2
    saved = next((tmp_path / "agent" / "sessions").rglob("*.jsonl"))
    assert user_texts(decode_history(saved.read_bytes()).history) == ["first", "second"]


async def test_tool_failure_is_model_feedback_not_process_failure(tmp_path: Path) -> None:
    fetch = SequencedFetch(
        tool_call_stream("call-1", "read", {"path": "missing.txt"}),
        text_stream("the file is missing"),
    )
    code, stdout, stderr = await run_task_chain(
        make_host(tmp_path, fetch), compose_tasks(prompts=("read missing.txt",)),
    )
    assert code == EXIT_OK and stdout == "the file is missing"
    assert stderr == ""
    assert len(fetch.requests) == 2


async def test_read_and_write_tools_execute_against_the_session_cwd(tmp_path: Path) -> None:
    fetch = SequencedFetch(
        tool_call_stream("call-1", "write", {"path": "made.txt", "content": "written"}),
        text_stream("created it"),
    )
    code, stdout, _ = await run_task_chain(
        make_host(tmp_path, fetch), compose_tasks(prompts=("create made.txt",)),
    )
    assert code == EXIT_OK and stdout == "created it"
    assert (tmp_path / "made.txt").read_text() == "written"


# --------------------------------------------------------------------------- #
# Expansion boundary
# --------------------------------------------------------------------------- #


async def test_print_expands_skill_but_keeps_unknown_slash_and_bang_literal(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    skill = agent_dir / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: review\ndescription: Review\n---\nReview carefully")
    fetched: list[RecordingFetch] = []

    def fetch_factory() -> RecordingFetch:
        fetch = RecordingFetch(text_stream("ok"))
        fetched.append(fetch)
        return fetch

    fetch = fetch_factory()
    code, _, _ = await run_task_chain(
        make_host(tmp_path, fetch, agent_dir=agent_dir),
        compose_tasks(prompts=("/skill:review please",)),
    )
    assert code == EXIT_OK
    sent = json.dumps(fetch.bodies[0])
    assert "Review carefully" in sent and "please" in sent

    literal = fetch_factory()
    code, _, _ = await run_task_chain(
        make_host(tmp_path, literal, agent_dir=agent_dir),
        compose_tasks(prompts=("!ls -la",)),
    )
    assert code == EXIT_OK
    assert "!ls -la" in json.dumps(literal.bodies[0])


# --------------------------------------------------------------------------- #
# Saving modes through the installed command
# --------------------------------------------------------------------------- #


class CliHarness:
    """Run the installed command in-process with a controlled provider."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
        self.home = home
        self.fetch: FetchFunction | None = None
        real = CodingAgentHost

        def factory(*args: object, **kwargs: object) -> CodingAgentHost:
            assert self.fetch is not None, "set a controlled provider before running"
            kwargs["fetch"] = self.fetch
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(cli, "CodingAgentHost", factory)

    def run(
        self, argv: list[str], *, stdin_text: str = "", fetch: FetchFunction | None = None,
    ) -> tuple[int, str, str]:
        if fetch is not None:
            self.fetch = fetch
        stdout, stderr = StringIO(), StringIO()
        code = cli.run(argv, stdin=StringIO(stdin_text), stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def cli_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CliHarness:
    home = tmp_path / "home"
    home.mkdir()
    return CliHarness(monkeypatch, home)


def test_whitespace_stdin_is_a_usage_error_without_a_request(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("unused"))
    code, stdout, stderr = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--api-key", "k"], stdin_text="  \n\t", fetch=fetch,
    )
    assert code == EXIT_USAGE and stdout == ""
    assert "no task" in stderr
    assert fetch.requests == []


def test_cli_reads_non_tty_stdin_with_attachment_and_prompt(
    tmp_path: Path, cli_harness: CliHarness,
) -> None:
    (tmp_path / "context.txt").write_text("file body")
    fetch = RecordingFetch(text_stream("ok"))
    code, stdout, _ = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--api-key", "k", "@context.txt", "explain"],
        stdin_text="piped body\n", fetch=fetch,
    )
    assert code == EXIT_OK and stdout == "ok"
    body = json.dumps(fetch.bodies[0])
    assert body.index("piped body") < body.index("file body") < body.index("explain")


def test_missing_key_is_rejected_before_any_request(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("unused"))
    code, stdout, stderr = cli_harness.run(["-p", "--cwd", str(tmp_path), "hello"], fetch=fetch)
    assert code == EXIT_USAGE and stdout == ""
    assert "No API key" in stderr
    assert fetch.requests == []


def test_missing_attachment_is_rejected_before_any_request(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("unused"))
    code, stdout, stderr = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--api-key", "k", "@missing.txt"], fetch=fetch,
    )
    assert code == EXIT_USAGE and stdout == ""
    assert "attachment" in stderr
    assert fetch.requests == []


def test_double_dash_keeps_flags_and_files_literal(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("ok"))
    code, stdout, stderr = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--api-key", "k", "--", "--unknown-option", "@not-a-file"],
        fetch=fetch,
    )
    assert code == EXIT_OK and stdout == "ok"
    assert "attachment" not in stderr
    assert len(fetch.requests) == 2


def test_print_saves_and_continues_the_same_identity(tmp_path: Path, cli_harness: CliHarness) -> None:
    project = tmp_path / "project"
    project.mkdir()
    first = RecordingFetch(text_stream("first answer"))
    code, stdout, _ = cli_harness.run(
        ["-p", "--cwd", str(project), "--api-key", "k", "first question"], fetch=first,
    )
    assert code == EXIT_OK and stdout == "first answer"
    root = cli_harness.home / ".omh" / "agent" / "sessions"
    saved = list(root.rglob("*.jsonl"))
    assert len(saved) == 1
    conversation_id = decode_history(saved[0].read_bytes()).history.conversation_id

    second = RecordingFetch(text_stream("second answer"))
    code, stdout, stderr = cli_harness.run(
        ["-p", "--cwd", str(project), "--api-key", "k", "-c", "second question"], fetch=second,
    )
    assert code == EXIT_OK and stdout == "second answer"
    assert str(saved[0]) in stderr
    assert decode_history(saved[0].read_bytes()).history.conversation_id == conversation_id
    body = json.dumps(second.bodies[0])
    assert "first question" in body and "first answer" in body and "second question" in body


def test_print_name_and_session_dir_are_applied(tmp_path: Path, cli_harness: CliHarness) -> None:
    directory = tmp_path / "history"
    fetch = RecordingFetch(text_stream("ok"))
    code, stdout, _ = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--session-dir", str(directory),
         "--name", "My chat", "--api-key", "k", "hi"],
        fetch=fetch,
    )
    assert code == EXIT_OK and stdout == "ok"
    entries = SessionDirectory(directory).list()
    assert len(entries) == 1
    assert entries[0].display_name == "My chat"
    assert entries[0].path.parent == directory


def test_no_session_runs_in_memory(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("ok"))
    code, stdout, _ = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--no-session", "--api-key", "k", "hi"], fetch=fetch,
    )
    assert code == EXIT_OK and stdout == "ok"
    assert list(cli_harness.home.rglob("*.jsonl")) == []


def test_explicit_session_path_reopens_and_appends(tmp_path: Path, cli_harness: CliHarness) -> None:
    directory = tmp_path / "history"
    first = RecordingFetch(text_stream("one"))
    assert cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--session-dir", str(directory), "--api-key", "k", "start"],
        fetch=first,
    )[0] == EXIT_OK
    path = next(directory.glob("*.jsonl"))
    before = len(decode_history(path.read_bytes()).history.entries)

    second = RecordingFetch(text_stream("two"))
    code, stdout, _ = cli_harness.run(
        ["-p", "--session", str(path), "--api-key", "k", "next"], fetch=second,
    )
    assert code == EXIT_OK and stdout == "two"
    history = decode_history(path.read_bytes()).history
    assert len(history.entries) > before
    assert "start" in json.dumps(second.bodies[0])


def test_stderr_diagnostics_never_reach_stdout(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("answer"))
    code, stdout, stderr = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--api-key", "k", "question"], fetch=fetch,
    )
    assert code == EXIT_OK
    assert stdout == "answer"
    assert "omh:" not in stdout


def test_run_records_usage_and_saves_complete_history(tmp_path: Path, cli_harness: CliHarness) -> None:
    fetch = RecordingFetch(text_stream("answer"))
    code, _, _ = cli_harness.run(
        ["-p", "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "history"),
         "--api-key", "k", "question"],
        fetch=fetch,
    )
    assert code == EXIT_OK
    path = next((tmp_path / "history").glob("*.jsonl"))
    history = decode_history(path.read_bytes()).history
    roles = [
        entry.message.role
        for entry in history.entries
        if hasattr(entry, "message")
    ]
    assert "user" in roles and "assistant" in roles
