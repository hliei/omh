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
:func:`run_print_text` is the observable execution boundary. Both modes write
through one :class:`~coding_agent.stdout_writer.StdoutWriter`, so a slow
consumer applies backpressure to the producer and a permanent stdout failure
ends print with exit 1 without waiting for cooperative cleanup or saving.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
from coding_agent.cancellation import PrintCancellation
from coding_agent.host import CodingAgentHost, SessionSelection
from coding_agent.json_wire import project_event
from coding_agent.stdout_writer import StdoutWriteError, StdoutWriter
from coding_agent.terminal import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    write_best_effort_diagnostic,
    write_diagnostic,
)


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
    handle_signals: bool = False,
) -> int:
    """Run the task chain and return the process outcome.

    The final executed task's assistant text is written to ``stdout`` only when
    the whole activity and its saving finish normally. A new image attachment
    for a text-only model is a pre-request input error, reported through the
    session's own modality message. Request, notification, saving and close
    failures are reported on ``stderr`` and fail the process.

    With ``handle_signals`` the installed command cooperatively handles the
    first SIGINT/SIGTERM/SIGHUP; a failed automatic save is rescued to an
    independent temporary copy in either case.
    """
    return await _run_print(host, selection, tasks, display_name=display_name,
                            stdout=stdout, stderr=stderr, json_mode=False,
                            handle_signals=handle_signals)


async def run_print_json(
    host: CodingAgentHost, selection: SessionSelection, tasks: Sequence[PrintTask], *,
    display_name: str | None = None, stdout: TextIO, stderr: TextIO,
    handle_signals: bool = False,
) -> int:
    """Stream the session header and strict events; model failures can exit 0."""
    return await _run_print(host, selection, tasks, display_name=display_name,
                            stdout=stdout, stderr=stderr, json_mode=True,
                            handle_signals=handle_signals)


async def _run_print(
    host: CodingAgentHost, selection: SessionSelection, tasks: Sequence[PrintTask], *,
    display_name: str | None, stdout: TextIO, stderr: TextIO, json_mode: bool,
    handle_signals: bool = False,
) -> int:
    try:
        options = host.build_options(selection)
    except Exception as error:
        write_diagnostic(stderr, error)
        return EXIT_USAGE

    runtime = AgentSessionRuntime(options)
    cancellation = PrintCancellation(stderr) if handle_signals else None
    if cancellation is not None:
        cancellation.install(lambda: _abort_current(runtime), loop=asyncio.get_running_loop())
    writer = StdoutWriter(stdout)
    try:
        return await _execute_print(
            runtime, selection, tasks, display_name=display_name, writer=writer,
            stderr=stderr, json_mode=json_mode, cancellation=cancellation,
        )
    finally:
        if cancellation is not None:
            cancellation.uninstall()


async def _execute_print(
    runtime: AgentSessionRuntime, selection: SessionSelection, tasks: Sequence[PrintTask], *,
    display_name: str | None, writer: StdoutWriter, stderr: TextIO, json_mode: bool,
    cancellation: PrintCancellation | None,
) -> int:
    outcome = _RunOutcome()
    input_rejected = False
    close_failed = False
    start_failed = False
    try:
        try:
            await _start(runtime, selection, display_name)
            if json_mode:
                session = runtime.current_session
                assert session is not None
                history = session.agent.history
                header: dict[str, object] = {"type": "session", "version": 3,
                                             "id": history.conversation_id,
                                             "timestamp": history.created_at.isoformat(),
                                             "cwd": str(session.cwd)}
                if session.source is not None and session.source.path is not None:
                    header["parentSession"] = session.source.path
                writer.write(_json_line(header))
                if session.save_mode == "memory":
                    write_diagnostic(stderr, "in-memory session; history will not be saved")

                def on_event(event: AgentEvent, signal: AbortSignal) -> None:
                    projected = project_event(event)
                    if projected is not None:
                        writer.write(_json_line(projected))

                runtime.subscribe(on_event)
        except Exception as error:
            (write_best_effort_diagnostic if writer.failed else write_diagnostic)(stderr, error)
            start_failed = True
        if not start_failed:
            try:
                outcome = await _run_tasks(
                    runtime, tasks, stderr, json_mode=json_mode, cancellation=cancellation,
                )
            except UnsupportedImageModelError as error:
                write_diagnostic(stderr, error)
                input_rejected = True
            except Exception as error:
                (write_best_effort_diagnostic if writer.failed else write_diagnostic)(stderr, error)
                outcome = _RunOutcome()
    finally:
        # Permanent stdout errors take precedence over cooperative cancellation.
        if not writer.failed:
            try:
                await runtime.close()
            except Exception as error:
                (write_best_effort_diagnostic if writer.failed else write_diagnostic)(stderr, error)
                outcome = _RunOutcome()
                close_failed = True
            if not writer.failed:
                await _rescue_unsaved(runtime, stderr)

    if writer.failed:
        return EXIT_FAILURE
    if _cancel_requested(cancellation):
        assert cancellation is not None
        return cancellation.exit_code
    if start_failed or close_failed:
        return EXIT_FAILURE
    if input_rejected:
        return EXIT_USAGE
    if outcome.text is not None:
        try:
            if not json_mode:
                writer.write(outcome.text)
            writer.flush()
        except StdoutWriteError as error:
            write_best_effort_diagnostic(stderr, error)
            return EXIT_FAILURE
        return cancellation.exit_code if _cancel_requested(cancellation) and cancellation is not None else EXIT_OK
    return EXIT_FAILURE


def _abort_current(runtime: AgentSessionRuntime) -> None:
    """Cooperatively abort the current Agent activity through its public API."""
    session = runtime.current_session
    if session is not None and not session.agent.state.is_closed:
        session.agent.abort()


def _cancel_requested(cancellation: PrintCancellation | None) -> bool:
    """Report whether a termination signal asked this run to stop."""
    return cancellation is not None and cancellation.requested


async def _rescue_unsaved(runtime: AgentSessionRuntime, stderr: TextIO) -> None:
    """Write an independent complete-history copy after a failed auto save.

    This is the print runner's host policy for a failed automatic save. It
    never rebinds the session, repairs the original target or turns the failed
    call into a success; it only preserves the complete in-memory history in a
    separate temporary directory. In-memory sessions are not rescued.
    """
    session = runtime.current_session
    if session is None or session.save_mode != "auto":
        return
    if session.save_state != "unsaved":
        return
    try:
        directory = Path(tempfile.mkdtemp(prefix="omh-rescue-"))
        destination = directory / f"{session.agent.history.conversation_id}.jsonl"
        await session.export(destination)
    except Exception as error:
        write_diagnostic(
            stderr, f"automatic saving failed and no complete history was rescued: {error}",
        )
        return
    write_diagnostic(stderr, f"automatic saving failed; rescued complete history to {destination}")


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
    runtime: AgentSessionRuntime, tasks: Sequence[PrintTask], stderr: TextIO, *,
    json_mode: bool = False, cancellation: PrintCancellation | None = None,
) -> _RunOutcome:
    """Run tasks serially until completion, reporting the last executed outcome.

    Once a termination signal requested cancellation, no further task is
    admitted or sent; the in-flight task is awaited so the SDK can finish its
    own cooperative cleanup, and a cancellation abort is not reported as a
    model failure.
    """
    outcome = _RunOutcome()
    reported = 0
    for task in tasks:
        if _cancel_requested(cancellation):
            return _RunOutcome()
        await runtime.prompt(task.text, images=list(task.images) or None)
        if _cancel_requested(cancellation):
            return _RunOutcome()
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


def _json_line(value: dict[str, object]) -> str:
    """Render one strict JSON wire record as a single newline-terminated line."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
