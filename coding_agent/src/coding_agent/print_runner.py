"""Print text and JSON execution over the common host.

Print text resolves the same host, selection and resources as interactive use,
then runs the composed first task and every later prompt serially in one
session. Only the final executed task's assistant text reaches stdout;
diagnostics stay on stderr. A task whose final assistant is an error or an
abort stops the remaining prompts (JSON may still exit zero), while a recoverable tool
failure is fed back to the model instead of failing the process.

The runner never builds a second Agent loop, retry policy or history state
machine; it consumes ``AgentSessionRuntime`` and the SDK's public settled
outcome. ``compose_tasks`` is the pure input boundary and
:func:`run_print_text` is the observable execution boundary.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from omh.agent import AgentEvent
from omh.llm.types import AbortSignal, AssistantMessage, ImageContent, TextContent

from coding_agent.agent_session import UnsupportedImageModelError
from coding_agent.agent_session_runtime import AgentSessionRuntime
from coding_agent.attachments import (
    FileAttachment,
    attachment_images,
    compose_first_task,
)
from coding_agent.host import CodingAgentHost, SessionSelection
from coding_agent.json_wire import project_event
from coding_agent.terminal import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, write_diagnostic


class PrintInputError(ValueError):
    """The invocation produced no actual task to run."""


@dataclass(frozen=True, slots=True)
class PrintTask:
    """One composed task: its exact text and any first-task images."""

    text: str
    images: tuple[ImageContent, ...] = ()


@dataclass(frozen=True, slots=True)
class _RunOutcome:
    """The final executed task's assistant text; ``None`` means the run failed."""

    text: str | None = None


def _has_task_content(text: str | None) -> bool:
    """Report whether text holds a real task instead of whitespace."""
    return bool(text and text.strip())


def compose_tasks(
    *,
    stdin_text: str | None = None,
    attachments: Sequence[FileAttachment] = (),
    prompts: Sequence[str] = (),
) -> tuple[PrintTask, ...]:
    """Compose the serial task chain from stdin, attachments and prompts.

    Piped stdin, the attachment boundaries and the first positional prompt form
    the first task in that order, separated so no part runs into the next.
    Whitespace-only input contributes nothing. Every later prompt becomes its
    own task, and accepted images belong to the first task only. An invocation
    with no real task raises :class:`PrintInputError` before any request.
    """
    first_prompt = prompts[0] if prompts and _has_task_content(prompts[0]) else None
    first_text = compose_first_task(
        stdin_text=stdin_text if _has_task_content(stdin_text) else None,
        attachments=list(attachments),
        first_prompt=first_prompt,
    )
    images = tuple(attachment_images(list(attachments)))
    tasks: list[PrintTask] = []
    if _has_task_content(first_text) or images:
        tasks.append(PrintTask(text=first_text, images=images))
    tasks.extend(PrintTask(text=prompt) for prompt in prompts[1:] if _has_task_content(prompt))
    if not tasks:
        raise PrintInputError("no task to run; provide a prompt, piped input or an @file attachment")
    return tuple(tasks)


async def run_print_text(
    host: CodingAgentHost,
    selection: SessionSelection,
    tasks: Sequence[PrintTask],
    *,
    display_name: str | None = None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Run the task chain and return the process outcome.

    The final executed task's assistant text is written to ``stdout`` only when
    the whole activity and its saving finish normally. A new image attachment
    for a text-only model is a pre-request input error, reported through the
    session's own modality message. Request, notification, saving and close
    failures are reported on ``stderr`` and fail the process.
    """
    return await _run_print(host, selection, tasks, display_name=display_name,
                            stdout=stdout, stderr=stderr, json_mode=False)


async def run_print_json(
    host: CodingAgentHost, selection: SessionSelection, tasks: Sequence[PrintTask], *,
    display_name: str | None = None, stdout: TextIO, stderr: TextIO,
) -> int:
    """Stream the session header and strict events; model failures can exit 0."""
    return await _run_print(host, selection, tasks, display_name=display_name,
                            stdout=stdout, stderr=stderr, json_mode=True)


async def _run_print(
    host: CodingAgentHost, selection: SessionSelection, tasks: Sequence[PrintTask], *,
    display_name: str | None, stdout: TextIO, stderr: TextIO, json_mode: bool,
) -> int:
    try:
        options = host.build_options(selection)
    except Exception as error:
        write_diagnostic(stderr, error)
        return EXIT_USAGE

    runtime = AgentSessionRuntime(options)
    outcome = _RunOutcome()
    input_rejected = False
    close_failed = False
    try:
        try:
            await _start(runtime, selection, display_name)
            if json_mode:
                session = runtime.current_session
                assert session is not None
                history = session.agent.history
                _write_json(stdout, {"type": "session", "version": 3,
                                     "id": history.conversation_id,
                                     "timestamp": history.created_at.isoformat(), "cwd": str(session.cwd)})
                if session.save_mode == "memory":
                    write_diagnostic(stderr, "in-memory session; history will not be saved")

                def on_event(event: AgentEvent, signal: AbortSignal) -> None:
                    projected = project_event(event)
                    if projected is not None:
                        _write_json(stdout, projected)

                runtime.subscribe(on_event)
        except Exception as error:
            write_diagnostic(stderr, error)
            return EXIT_FAILURE
        try:
            outcome = await _run_tasks(runtime, tasks, stderr, json_mode=json_mode)
        except UnsupportedImageModelError as error:
            write_diagnostic(stderr, error)
            input_rejected = True
        except Exception as error:
            write_diagnostic(stderr, error)
            outcome = _RunOutcome()
    finally:
        try:
            await runtime.close()
        except Exception as error:
            write_diagnostic(stderr, error)
            outcome = _RunOutcome()
            close_failed = True

    if close_failed:
        return EXIT_FAILURE
    if input_rejected:
        return EXIT_USAGE
    if outcome.text is not None:
        if not json_mode:
            stdout.write(outcome.text)
        return EXIT_OK
    return EXIT_FAILURE


async def _start(
    runtime: AgentSessionRuntime, selection: SessionSelection, display_name: str | None,
) -> None:
    """Create or reopen the session, applying an explicit display name."""
    if selection.session_path is None:
        await runtime.new_session(display_name=display_name)
        return
    await runtime.open_session(selection.session_path)
    if display_name is not None:
        await runtime.set_session_name(display_name)


async def _run_tasks(
    runtime: AgentSessionRuntime, tasks: Sequence[PrintTask], stderr: TextIO, *, json_mode: bool = False,
) -> _RunOutcome:
    """Run tasks serially until completion, reporting the last executed outcome."""
    outcome = _RunOutcome()
    reported = 0
    for task in tasks:
        await runtime.prompt(task.text, images=list(task.images) or None)
        session = runtime.current_session
        assert session is not None
        for diagnostic in session.input_diagnostics[reported:]:
            write_diagnostic(stderr, diagnostic.message)
        reported = len(session.input_diagnostics)
        text, stop_reason, error_message = _final_assistant(session.agent.state.messages)
        if stop_reason in ("error", "aborted"):
            write_diagnostic(stderr, error_message or f"the model response ended with {stop_reason}")
            return _RunOutcome(text="" if json_mode else None)
        outcome = _RunOutcome(text=text)
    return outcome


def _final_assistant(messages: Sequence[object]) -> tuple[str, str | None, str | None]:
    """Return the last assistant message's text, stop reason and error message."""
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            text = "".join(
                block.text for block in message.content if isinstance(block, TextContent)
            )
            return text, message.stop_reason, message.error_message
    return "", None, None


def _write_json(stdout: TextIO, value: dict[str, object]) -> None:
    stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stdout.flush()
