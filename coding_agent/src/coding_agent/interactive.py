"""Regular interactive session: scrolling conversation, editor and status.

The screen stays on the primary terminal. A capable terminal pins the status
and editor beneath a scrolling conversation; a plain or dumb terminal prints
the same words without color or cursor addressing. Prompts go through
``AgentSessionRuntime`` and the common host. This module does not start a
second Agent loop, and print execution does not import it.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import select
import signal
import struct
import termios
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from omh.agent import AgentEvent
from omh.agent.execution.events import (
    AgentSettledEvent,
    AgentStartEvent,
    CompactionStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RetryStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnStartEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

from coding_agent.agent_session import AgentSession
from coding_agent.agent_session_runtime import AgentSessionRuntime
from coding_agent.cli import CliArgs
from coding_agent.clipboard import (
    ClipboardError,
    ClipboardSupport,
    ClipboardUnavailable,
    copy_text,
    detect_clipboard,
    read_clipboard_image,
)
from coding_agent.commands import (
    AVAILABLE_COMMANDS,
    COMMANDS_BY_NAME,
    RESERVED_COMMANDS,
    SlashIntent,
    classify,
    command_detail,
    help_lines,
    hotkey_lines,
)
from coding_agent.completion import Completion, CompletionSources, complete
from coding_agent.config import settings_theme
from coding_agent.editor import Editor
from coding_agent.external_editor import editor_command, run_external_editor
from coding_agent.host import CodingAgentHost, SessionSelection
from coding_agent.images import (
    ImageInputError,
    ImageLimits,
    ProcessedImage,
    process_image,
)
from coding_agent.keys import Key, KeyDecoder
from coding_agent.pending import (
    CLIPBOARD_ORIGIN,
    FILE_ORIGIN,
    AttachmentOrigin,
    PendingAttachment,
    PendingAttachments,
    read_pending_image,
)
from coding_agent.resources import ApplicationResources
from coding_agent.terminal import EXIT_OK
from coding_agent.theme import resolve_theme

_EXIT_SIGNALS = {signal.SIGINT: 130, signal.SIGTERM: 143, signal.SIGHUP: 129}
_UNSAVED_GUIDANCE = (
    "In-memory history is retained. New work is paused until the session can be saved."
)
_NO_REQUEST = "No request was sent."
_COLORS = {
    "dark": "\x1b[38;5;117m",
    "light": "\x1b[38;5;25m",
    "plain": "",
}
#: Candidates shown at once; the completion list itself is bounded separately.
_COMPLETION_VISIBLE = 8
_CONTINUATION = "  "


@dataclass(slots=True)
class _Note:
    text: str


@dataclass(slots=True)
class _You:
    text: str
    #: Display labels for attached images: live labels include the name and size,
    #: restored history has only the MIME type.
    image_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class _Assistant:
    text: str = ""
    thinking: str = ""
    error: str | None = None
    live: bool = False


@dataclass(slots=True)
class _Tool:
    tool_id: str
    name: str
    args: dict[str, object]
    state: str
    output: str = ""
    bounded: bool = False


_Item = _Note | _You | _Assistant | _Tool


def run_interactive(
    args: CliArgs, *, host_cls: type[CodingAgentHost], stdin: TextIO, stdout: TextIO, stderr: TextIO,
) -> int:
    """Run one interactive session and return the process exit code."""
    del stderr
    app = _Session(args, host_cls=host_cls, stdin=stdin, stdout=stdout)
    try:
        return asyncio.run(app.run())
    finally:
        app.restore()


class _Session:
    """One terminal conversation bound to a single runtime."""

    def __init__(
        self, args: CliArgs, *, host_cls: type[CodingAgentHost], stdin: TextIO, stdout: TextIO,
    ) -> None:
        self.args = args
        self._host_cls = host_cls
        self._stdin = stdin
        self._stdout = stdout
        self._fd = stdin.fileno()
        self._out = stdout.fileno()
        self.host: CodingAgentHost | None = None
        self.runtime: AgentSessionRuntime | None = None
        self.theme, _notice = resolve_theme(
            requested=args.use_theme, configured=None, environ=os.environ,
        )
        self.exit_code = EXIT_OK
        self._items: list[_Item] = []
        self._tools: dict[str, _Tool] = {}
        self._live: _Assistant | None = None
        self._draft = PendingAttachments()
        self._pending_you: _You | None = None
        self._submitted_entries = 0
        self._clipboard: ClipboardSupport = detect_clipboard()
        self._attachment_task: asyncio.Task[None] | None = None
        self.editor = Editor()
        self._completion: Completion | None = None
        self._phase = "input"
        self._chooser = "prompt"
        self._request_block: str | None = None
        self._thinking_open = False
        self._tools_open = False
        self._busy = False
        self._closing = False
        self._prompt_task: asyncio.Task[None] | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._exit_task: asyncio.Task[None] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._external_task: asyncio.Task[None] | None = None
        self._copy_task: asyncio.Task[None] | None = None
        self._external_running = False
        self._ctrl_c_at = 0.0
        self._termios: list[object] | None = None
        self._flags: int | None = None
        self._previous: tuple[str, ...] = ()
        self._scrollback: tuple[str, ...] = ()
        self._previous_chrome = ""
        self._keys = KeyDecoder()
        self._escape_handle: asyncio.TimerHandle | None = None
        self._read_future: asyncio.Future[bytes] | None = None
        self._done: asyncio.Event | None = None
        self._signals = 0
        self._restored = False

    async def run(self) -> int:
        self._done = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum, code in _EXIT_SIGNALS.items():
            loop.add_signal_handler(signum, self._on_process_signal, code)
        loop.add_signal_handler(signal.SIGWINCH, self._redraw)
        self._enter_terminal()
        self._startup_task = asyncio.create_task(self._startup())
        try:
            await self._startup_task
        except asyncio.CancelledError:
            if not self._closing:
                raise
        self._redraw()
        self._start_reader()
        await self._done.wait()
        await self._cancel_reader()
        if self._external_task is not None:
            try:
                await self._external_task
            except asyncio.CancelledError:
                pass
        if self._exit_task is not None:
            await self._exit_task
        return self.exit_code

    def restore(self) -> None:
        """Return the terminal to the mode captured at entry."""
        if self._restored:
            return
        self._restored = True
        if self._escape_handle is not None:
            self._escape_handle.cancel()
            self._escape_handle = None
        try:
            os.write(self._out, b"\x1b[?2004l")
        except OSError:
            pass
        if self.theme != "plain":
            try:
                os.write(self._out, b"\x1b[0m\x1b[r\x1b[?25h\r\n")
            except OSError:
                pass
        self._restore_cooked()

    # ------------------------------------------------------------------ #
    # Terminal ownership
    # ------------------------------------------------------------------ #

    def _enter_terminal(self) -> None:
        if self._termios is None:
            self._termios = termios.tcgetattr(self._fd)
            self._flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        self._apply_raw_mode()

    def _apply_raw_mode(self) -> None:
        assert self._termios is not None and self._flags is not None
        attrs = list(self._termios)
        attrs[0] = cast(int, attrs[0]) & ~(termios.IXON | termios.IXOFF | termios.ICRNL)
        attrs[3] = cast(int, attrs[3]) & ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
        attrs[6] = list(cast(list[int | bytes], attrs[6]))
        cc = cast(list[int | bytes], attrs[6])
        cc[termios.VMIN] = 1
        cc[termios.VTIME] = 0
        termios.tcsetattr(
            self._fd, termios.TCSADRAIN, cast("list[int | list[bytes | int]]", attrs),
        )
        fcntl.fcntl(self._fd, fcntl.F_SETFL, self._flags | os.O_NONBLOCK)
        try:
            os.write(self._out, b"\x1b[?2004h")
        except OSError:
            pass

    def _restore_cooked(self) -> None:
        if self._flags is not None:
            fcntl.fcntl(self._fd, fcntl.F_SETFL, self._flags)
        if self._termios is not None:
            termios.tcsetattr(
                self._fd, termios.TCSADRAIN, cast("list[int | list[bytes | int]]", self._termios),
            )

    # ------------------------------------------------------------------ #
    # Startup and sessions
    # ------------------------------------------------------------------ #

    async def _startup(self) -> None:
        self._add("interactive")
        startup = Path(self.args.cwd or Path.cwd()).expanduser().resolve()
        try:
            self.host = self._host_cls(
                startup_dir=startup,
                approve=self.args.approve,
                no_approve=self.args.no_approve,
                explicit_skills=tuple(self.args.skills),
                explicit_templates=tuple(self.args.prompt_templates),
                explicit_system_prompt=self.args.system_prompt,
                explicit_system_prompt_file=self.args.system_prompt_file,
                append_system=tuple(self.args.append_system),
                no_skills=self.args.no_skills,
                no_prompt_templates=self.args.no_prompt_templates,
                no_context_files=self.args.no_context_files,
                api_key=self.args.api_key,
                session_dir=self.args.session_dir,
                no_session=self.args.no_session,
            )
        except Exception as error:
            self._prepare_theme(None)
            self._add(f"notice {error}")
            self._add(f"notice {_NO_REQUEST}")
            return
        self._prepare_theme(settings_theme(self.host.settings))
        if self.args.resume:
            self._request_block = "Pass --session <path or id> to reopen one saved session."
            self._add(f"notice {self._request_block}")
            self._add(f"notice {_NO_REQUEST}")
            return
        try:
            await self._apply_selection(self._select(self.args.cwd))
        except Exception as error:
            self._add(f"notice {error}")
            self._add(f"notice {_NO_REQUEST}")

    def _prepare_theme(self, configured: str | None) -> None:
        self.theme, notice = resolve_theme(
            requested=self.args.use_theme, configured=configured, environ=os.environ,
        )
        if notice:
            self._add(f"notice {notice}")

    def _select(self, cwd: str | None) -> SessionSelection:
        assert self.host is not None
        tools = () if self.args.no_tools else self.args.tools
        if self.args.session is not None:
            return self.host.select_open(
                self.args.session, provider=self.args.provider, model=self.args.model,
                thinking=self.args.thinking, tools=tools, cwd=cwd,
            )
        if self.args.continue_session:
            return self.host.select_continue(
                provider=self.args.provider, model=self.args.model,
                thinking=self.args.thinking, tools=tools, cwd=cwd,
            )
        return self.host.select_new(
            provider=self.args.provider, model=self.args.model,
            thinking=self.args.thinking, tools=tools, cwd=cwd,
        )

    async def _apply_selection(self, selection: SessionSelection) -> None:
        assert self.host is not None
        diagnostics = await self.host.readiness(selection)
        for diagnostic in diagnostics:
            self._add(f"notice {diagnostic.message}")
        if not selection.ready:
            missing_cwd = selection.cwd is None and any(
                "directory" in diagnostic.message.lower() for diagnostic in diagnostics
            )
            self._chooser = "cwd" if missing_cwd else "prompt"
            self._request_block = (
                "Saved working directory is unavailable. Enter a replacement directory."
                if missing_cwd else "The session is not ready."
            )
            self._add(f"notice {_NO_REQUEST}")
            return
        blocked = [item for item in diagnostics if item.blocking and item.source != "credentials"]
        credential = next(
            (item for item in diagnostics if item.source == "credentials" and item.blocking), None,
        )
        if blocked:
            self._request_block = blocked[0].message
            self._chooser = "prompt"
            self._add(f"notice {_NO_REQUEST}")
            return
        self._request_block = credential.message if credential is not None else None
        if credential is not None:
            self._add(f"notice {_NO_REQUEST}")
        await self._open_runtime(selection)

    async def _open_runtime(self, selection: SessionSelection) -> None:
        assert self.host is not None
        options = self.host.build_options(selection)
        runtime = AgentSessionRuntime(options)
        self.runtime = runtime
        if selection.session_path is None:
            await runtime.new_session(display_name=self.args.name)
        else:
            await runtime.open_session(selection.session_path)
        runtime.subscribe(self._on_event)
        self._chooser = "prompt"
        self._load_history()
        if selection.session_path is not None and self.args.name is not None:
            await runtime.set_session_name(self.args.name)

    def _current_session(self) -> AgentSession | None:
        return self.runtime.current_session if self.runtime is not None else None

    def _load_history(self) -> None:
        session = self._current_session()
        if session is None:
            return
        calls: dict[str, tuple[str, dict[str, object]]] = {}
        sent: list[str] = []
        for message in session.agent.state.messages:
            if isinstance(message, UserMessage):
                text = _user_text(message)
                labels = _user_image_labels(message)
                if text or labels:
                    self._items.append(_You(text, image_labels=labels))
                if text:
                    sent.append(text)
            elif isinstance(message, AssistantMessage):
                self._items.append(_assistant_from(message))
                for block in message.content:
                    if isinstance(block, ToolCall):
                        calls[block.id] = (block.name, dict(block.arguments))
            elif isinstance(message, ToolResultMessage):
                output = _content_text(message.content)
                name, args = calls.get(message.tool_call_id, (message.tool_name, {}))
                item = _Tool(
                    tool_id=message.tool_call_id, name=name, args=args,
                    state="error" if message.is_error else "ok", output=output,
                    bounded=_text_is_bounded(output, message.details),
                )
                self._tools[item.tool_id] = item
                self._items.append(item)
        self.editor.seed_history(sent)

    # ------------------------------------------------------------------ #
    # Terminal input
    # ------------------------------------------------------------------ #

    def _start_reader(self) -> None:
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.create_task(self._read_loop())

    async def _cancel_reader(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _read_loop(self) -> None:
        while not self._closing:
            data = await self._read_some()
            if self._closing or data is None:
                return
            if data == b"":
                self._request_exit(EXIT_OK)
                return
            self._handle(data)

    async def _read_some(self) -> bytes | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()
        self._read_future = future

        def on_readable() -> None:
            if future.done():
                return
            try:
                chunk = os.read(self._fd, 4096)
            except BlockingIOError:
                return
            except OSError:
                chunk = b""
            if not future.done():
                future.set_result(chunk)

        loop.add_reader(self._fd, on_readable)
        try:
            return await future
        finally:
            loop.remove_reader(self._fd)
            if self._read_future is future:
                self._read_future = None

    def _handle(self, data: bytes) -> None:
        self._cancel_escape_timer()
        for key in self._keys.feed(data):
            self._handle_key(key)
        if self._keys.pending == b"\x1b":
            self._arm_escape_timer()
        self._redraw()

    def _handle_key(self, key: Key) -> None:
        name = key.name
        if name == "escape":
            self._completion = None
            return
        if name == "tab":
            if self._completion is not None:
                self._accept_completion()
            else:
                self._open_completion()
            return
        if name == "enter":
            self._accept_or_submit()
            return
        if name == "newline":
            self.editor.insert("\n")
            self._completion = None
            self._ctrl_c_at = 0.0
            return
        if name == "up":
            if self._completion is not None:
                self._completion.select(-1)
            else:
                self.editor.move_up()
            return
        if name == "down":
            if self._completion is not None:
                self._completion.select(1)
            else:
                self.editor.move_down()
            return
        if name == "left":
            self.editor.move(-1)
            self._completion = None
            return
        if name == "right":
            self.editor.move(1)
            self._completion = None
            return
        if name == "backspace":
            self.editor.backspace()
            self._refresh_completion()
            self._ctrl_c_at = 0.0
            return
        if name == "ctrl_c":
            self._ctrl_c()
            return
        if name == "ctrl_d":
            preparing = self._attachment_task is not None and not self._attachment_task.done()
            if self.editor.empty and len(self._draft) == 0 and not preparing:
                self._request_exit(EXIT_OK)
            elif self.editor.empty:
                self._add(
                    "notice pending attachments remain; /attach remove <n>, /attach clear, "
                    "or submit them before exit"
                )
            return
        if name == "ctrl_o":
            self._tools_open = not self._tools_open
            return
        if name == "ctrl_t":
            self._thinking_open = not self._thinking_open
            return
        if name == "ctrl_g":
            self._open_external_editor()
            return
        if name == "ctrl_x":
            self._start_copy()
            return
        if name == "ctrl_v":
            self._paste_clipboard()
            return
        if key.value:
            self.editor.insert(key.value)
            self._refresh_completion()
            self._ctrl_c_at = 0.0

    def _accept_or_submit(self) -> None:
        completion = self._completion
        if completion is not None:
            self._accept_completion()
            if not completion.submit:
                self._refresh_completion()
                return
        text = self.editor.text
        cursor = self.editor.cursor
        if cursor > 0 and text[cursor - 1] == "\\":
            self.editor.backspace()
            self.editor.insert("\n")
            return
        self._submit()

    def _accept_completion(self) -> None:
        completion = self._completion
        if completion is None:
            return
        self.editor.replace_token(completion.start, completion.end, completion.current.value)
        self._completion = None

    def _arm_escape_timer(self) -> None:
        loop = asyncio.get_running_loop()
        self._escape_handle = loop.call_later(0.05, self._flush_escape)

    def _flush_escape(self) -> None:
        self._escape_handle = None
        key = self._keys.flush_escape()
        if key is None:
            return
        self._handle_key(key)
        self._redraw()

    def _cancel_escape_timer(self) -> None:
        if self._escape_handle is not None:
            self._escape_handle.cancel()
            self._escape_handle = None

    def _ctrl_c(self) -> None:
        now = asyncio.get_running_loop().time()
        if self._ctrl_c_at and now - self._ctrl_c_at <= 0.5:
            self._request_exit(EXIT_OK)
            return
        self.editor.clear()
        self._completion = None
        self._ctrl_c_at = now

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #

    def _open_completion(self) -> None:
        self._completion = complete(self.editor.text, self.editor.cursor, self._completion_sources())

    def _refresh_completion(self) -> None:
        if self._completion is not None:
            self._open_completion()

    def _completion_sources(self) -> CompletionSources:
        session = self._current_session()
        resources = session.resources if session is not None else ApplicationResources()
        cwd = (
            session.cwd if session is not None
            else Path(self.args.cwd or Path.cwd()).expanduser()
        )
        return CompletionSources(
            cwd=cwd,
            commands=AVAILABLE_COMMANDS,
            skills=tuple((skill.name, skill.description) for skill in resources.skills),
            templates=tuple((template.name, template.description) for template in resources.templates),
            reserved=RESERVED_COMMANDS,
            pending_count=len(self._draft),
        )

    # ------------------------------------------------------------------ #
    # Submission and dispatch
    # ------------------------------------------------------------------ #

    def _submit(self) -> None:
        text = self.editor.text
        self._completion = None
        self._ctrl_c_at = 0.0
        if self._chooser == "cwd":
            stripped = text.strip()
            if not stripped:
                return
            self.editor.clear()
            self._choose_cwd(stripped)
            return
        if text.strip():
            intent = classify(text, skills=self._skill_names(), templates=self._template_names())
            if intent.kind == "builtin":
                self.editor.clear()
                self._run_command(intent)
                return
            if intent.kind == "reserved":
                self.editor.clear()
                self._print(f"/{intent.name} is reserved for a later delivery; nothing was sent.")
                return
            if intent.kind == "unknown":
                self._print(f"Unknown command /{intent.name}; sending the text as an ordinary prompt.")
        if self._busy:
            return
        if self._attachment_task is not None and not self._attachment_task.done():
            self._add("notice an attachment is still being prepared; press Enter again")
            return
        if not text.strip():
            self._redraw()
            return
        session = self._current_session()
        if self._request_block is not None or session is None:
            self._keep_refused_submission(
                text, self._request_block or "The session is not ready.",
            )
            return
        if session.save_state == "unsaved":
            self._keep_refused_submission(text, None)
            self._show_unsaved()
            return
        warning = session.unsupported_image_message() if len(self._draft) else None
        if warning is not None:
            self._keep_refused_submission(text, f"image not sent: {warning}")
            return
        self.editor.clear()
        self.editor.remember(text)
        pending = self._draft.pop_all()
        self._pending_you = _You(text, image_labels=tuple(item.label for item in pending))
        self._items.append(self._pending_you)
        self._submitted_entries = len(session.agent.history.entries)
        self._busy = True
        self._set_phase("model")
        self._prompt_task = asyncio.create_task(self._prompt(text, pending))

    def _keep_refused_submission(self, text: str, reason: str | None) -> None:
        """Keep a refused prompt without losing the editor text or a draft image.

        Plain text with no pending image follows the existing display-only
        behavior. With a pending image the text returns to the editor so a
        refused send loses nothing. ``None`` means the caller adds its own notice.
        """
        if len(self._draft) == 0:
            self.editor.clear()
            self.editor.remember(text)
            self._items.append(_You(text))
        if reason is None:
            return
        self._add(f"notice {reason}")
        self._add(f"notice {_NO_REQUEST}")

    def _paste_clipboard(self) -> None:
        """Start reading a clipboard screenshot without submitting anything."""
        if self._attachment_task is not None and not self._attachment_task.done():
            return
        self._ctrl_c_at = 0.0
        self._attachment_task = asyncio.create_task(self._read_clipboard())

    async def _read_clipboard(self) -> None:
        try:
            data = await read_clipboard_image(support=self._clipboard)
        except ClipboardUnavailable as error:
            self._add(f"notice {error}")
            self._redraw()
            return
        except Exception as error:
            self._add(f"notice clipboard image could not be read: {error}")
            self._redraw()
            return
        if data is None:
            self._add("notice clipboard has no image; add one with /attach <image-path> instead")
            self._redraw()
            return
        try:
            processed = await asyncio.to_thread(process_image, data, limits=self._image_limits())
        except ImageInputError as error:
            self._add(f"notice clipboard image: {error}")
            self._redraw()
            return
        except Exception as error:
            self._add(f"notice clipboard image could not be processed: {error}")
            self._redraw()
            return
        self._add_pending(CLIPBOARD_ORIGIN, "clipboard.png", None, processed)

    def _image_limits(self) -> ImageLimits | None:
        return self.runtime.options.image_limits if self.runtime is not None else None

    def _cwd(self) -> Path:
        session = self._current_session()
        if session is not None:
            return session.cwd
        return Path(self.args.cwd or Path.cwd()).expanduser().resolve()

    def _add_pending(
        self, origin: AttachmentOrigin, name: str, source: str | None, image: ProcessedImage,
    ) -> PendingAttachment:
        attachment = self._draft.add(name=name, origin=origin, source=source, image=image)
        self._add(f"attached #{len(self._draft)} {attachment.name} {attachment.status}")
        session = self._current_session()
        warning = session.unsupported_image_message() if session is not None else None
        if warning is not None:
            self._add(f"notice {warning}")
        self._redraw()
        return attachment

    def _attach_command(self, argument: str) -> None:
        """Manage the pending image draft for ``/attach`` with and without a path."""
        if not argument:
            for line in self._draft.describe_lines():
                self._add(line)
            status = self._clipboard
            if status.available:
                self._add(f"clipboard {status.detail}")
            else:
                self._add(f"clipboard unavailable: {status.detail}")
            if len(self._draft):
                self._add("use /attach remove <n> to remove a pending image")
            return
        keyword, _separator, rest = argument.partition(" ")
        if keyword == "remove":
            if rest.strip():
                self._remove_pending(rest.strip())
            else:
                self._add("notice /attach remove needs a pending number")
            return
        if keyword == "clear":
            if rest.strip():
                self._add("notice /attach clear takes no arguments; attachments unchanged")
                return
            if self._attachment_task is not None and not self._attachment_task.done():
                self._attachment_task.cancel()
            self._draft.clear()
            self._add("attachments cleared")
            return
        if self._attachment_task is not None and not self._attachment_task.done():
            self._add("notice an attachment is still being prepared; wait or /attach clear")
            return
        self._attachment_task = asyncio.create_task(self._attach_path(_strip_quotes(argument)))

    def _remove_pending(self, value: str) -> None:
        try:
            index = int(value)
        except ValueError:
            self._add(f"notice /attach remove needs a pending number, not {value!r}")
            return
        removed = self._draft.remove_index(index)
        if removed is None:
            self._add(f"notice no pending attachment #{index}")
            return
        self._add(f"removed #{index} {removed.name}")

    async def _attach_path(self, argument: str) -> None:
        try:
            name, source, image = await asyncio.to_thread(
                read_pending_image, argument, cwd=self._cwd(), limits=self._image_limits(),
            )
        except ImageInputError as error:
            self._add(f"notice {error}")
            self._redraw()
            return
        except Exception as error:
            self._add(f"notice image attachment could not be prepared: {error}")
            self._redraw()
            return
        self._add_pending(FILE_ORIGIN, name, source, image)

    def _skill_names(self) -> tuple[str, ...]:
        session = self._current_session()
        if session is None:
            return ()
        return tuple(skill.name for skill in session.resources.skills)

    def _template_names(self) -> tuple[str, ...]:
        session = self._current_session()
        if session is None:
            return ()
        return tuple(template.name for template in session.resources.templates)

    def _run_command(self, intent: SlashIntent) -> None:
        spec = COMMANDS_BY_NAME.get(intent.name)
        if spec is None:
            return
        argument = intent.argument.strip()
        if spec.max_arguments is not None and len(argument.split()) > spec.max_arguments:
            limit = "no arguments" if spec.max_arguments == 0 else "at most one argument"
            self._print(f"/{spec.name} takes {limit}; nothing was sent.")
            return
        if spec.name == "help":
            if not argument:
                for line in help_lines():
                    self._print(line)
            else:
                name = argument.lstrip("/").split()[0] if argument.split() else ""
                detail = command_detail(name)
                if detail is not None:
                    for line in detail:
                        self._print(line)
                elif name in RESERVED_COMMANDS:
                    self._print(f"/{name} is reserved for a later delivery.")
                else:
                    self._print(f"Unknown command /{name}.")
        elif spec.name == "hotkeys":
            for line in hotkey_lines():
                self._print(line)
        elif spec.name == "copy":
            self._start_copy()
        elif spec.name == "attach":
            self._attach_command(argument)

    def _start_copy(self) -> None:
        if self._copy_task is None or self._copy_task.done():
            self._copy_task = asyncio.create_task(self._copy_answer())

    async def _copy_answer(self) -> None:
        text = _last_assistant_text(self.runtime)
        if not text:
            self._print("No assistant answer to copy yet.")
            self._redraw()
            return
        try:
            await asyncio.to_thread(copy_text, text)
        except ClipboardError as error:
            self._print(str(error))
        except Exception as error:  # the screen must survive any copy failure
            self._print(f"Cannot copy the last assistant answer: {error}")
        else:
            self._print("Copied the last assistant answer to the clipboard.")
        self._redraw()

    def _choose_cwd(self, text: str) -> None:
        candidate = Path(text).expanduser()
        if not candidate.is_dir():
            self._add(f"notice Working directory does not exist: {candidate}")
            return
        self._busy = True
        asyncio.create_task(self._finish_cwd(str(candidate.resolve())))

    async def _finish_cwd(self, cwd: str) -> None:
        try:
            await self._apply_selection(self._select(cwd))
        except Exception as error:
            self._add(f"notice {error}")
            self._add(f"notice {_NO_REQUEST}")
        finally:
            self._busy = False
            self._redraw()

    async def _prompt(self, text: str, pending: tuple[PendingAttachment, ...]) -> None:
        assert self.runtime is not None
        session = self._current_session()
        images = [attachment.content for attachment in pending]
        before = len(session.input_diagnostics) if session is not None else 0
        try:
            await self.runtime.prompt(text, images=images or None)
        except Exception as error:
            started = (
                session is not None
                and len(session.agent.history.entries) > self._submitted_entries
            )
            if not started:
                self._restore_rejected(text, pending)
            self._add(f"notice {error}")
        finally:
            self._pending_you = None
            self._busy = False
            self._remember_save()
            self._show_input_diagnostics(before)
            if not self._closing:
                self._set_phase("input")
            self._redraw()

    def _restore_rejected(self, text: str, pending: tuple[PendingAttachment, ...]) -> None:
        """Return a rejected prompt's text and images to the editable draft."""
        you = self._pending_you
        self._pending_you = None
        if you is not None and you in self._items:
            self._items.remove(you)
        self._draft.restore(pending)
        self.editor.set_text(text + self.editor.text)

    def _show_input_diagnostics(self, before: int) -> None:
        session = self._current_session()
        if session is None:
            return
        for diagnostic in session.input_diagnostics[before:]:
            self._add(f"notice {diagnostic.message}")

    def _remember_save(self) -> None:
        session = self._current_session()
        if session is not None and session.save_state == "unsaved":
            self._show_unsaved()

    def _show_unsaved(self) -> None:
        self._add(f"notice {_UNSAVED_GUIDANCE}")
        session = self._current_session()
        if session is not None and session.save_error is not None:
            self._add(f"notice {session.save_error}")

    # ------------------------------------------------------------------ #
    # External editor
    # ------------------------------------------------------------------ #

    def _open_external_editor(self) -> None:
        if self._external_running:
            return
        self._completion = None
        self._external_running = True
        self._external_task = asyncio.create_task(self._external_editor())

    async def _external_editor(self) -> None:
        command = editor_command(os.environ)
        content = self.editor.text
        await self._cancel_reader()
        self._cancel_escape_timer()
        self._suspend_terminal()
        message: str | None = None
        content_update: str | None = None
        try:
            result = await run_external_editor(
                command, content, stdin=self._fd, stdout=self._out, stderr=self._out,
            )
        except Exception as error:  # keep the editor usable after an unexpected failure
            message = f"External editor failed: {error}"
        else:
            if result.status == "complete":
                content_update = result.content
            else:
                message = result.message
        finally:
            if not self._closing:
                self._apply_raw_mode()
                self._start_reader()
            self._external_running = False
        if content_update is not None:
            self.editor.set_text(content_update)
            self._print("External editor returned; the draft was updated and not submitted.")
        elif message is not None:
            self._print(message)
        self._redraw()

    def _suspend_terminal(self) -> None:
        try:
            os.write(self._out, b"\x1b[?2004l\x1b[0m\x1b[?25h\r\n")
        except OSError:
            pass
        self._restore_cooked()

    # ------------------------------------------------------------------ #
    # Agent events and status
    # ------------------------------------------------------------------ #

    def _on_event(self, event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        if isinstance(event, (AgentStartEvent, TurnStartEvent)):
            self._set_phase("model")
        elif isinstance(event, RetryStartEvent):
            self._phase = "retry"
            self._add(
                f"phase retry {event.scope} {event.attempt}/{event.max_retries} {event.error_message}"
            )
        elif isinstance(event, CompactionStartEvent):
            self._phase = "compact"
            self._add(f"phase compact {event.reason}")
        elif isinstance(event, AgentSettledEvent):
            if event.aborted:
                self._phase = "cancel"
                self._add("phase cancel")
            elif event.error_message:
                self._add(f"model error {event.error_message}")
        elif isinstance(event, MessageStartEvent) and isinstance(event.message, AssistantMessage):
            item = _Assistant(live=True)
            self._items.append(item)
            self._live = item
        elif isinstance(event, (MessageUpdateEvent, MessageEndEvent)) and isinstance(event.message, AssistantMessage):
            self._update_assistant(event.message, final=isinstance(event, MessageEndEvent))
        elif isinstance(event, ToolExecutionStartEvent):
            self._phase = "tool"
            self._upsert_tool(event.tool_call_id, event.tool_name, event.args, "running", "")
            self._add(f"phase tool {event.tool_name}")
        elif isinstance(event, ToolExecutionUpdateEvent):
            self._phase = "tool"
            self._upsert_tool(
                event.tool_call_id, event.tool_name, event.args, "running",
                _content_text(event.partial_result.content),
            )
        elif isinstance(event, ToolExecutionEndEvent):
            self._phase = "tool"
            output = _content_text(event.result.content)
            self._upsert_tool(
                event.tool_call_id, event.tool_name, {},
                "error" if event.is_error else "ok", output,
                bounded=_text_is_bounded(output, event.result.details),
            )
        self._redraw()

    def _update_assistant(self, message: AssistantMessage, *, final: bool) -> None:
        item = self._live if self._live is not None and self._live.live else None
        if item is None:
            item = _Assistant(live=not final)
            self._items.append(item)
            self._live = item
        item.text, item.thinking = _assistant_parts(message)
        item.error = message.error_message
        if final:
            item.live = False
            if message.stop_reason == "aborted":
                self._phase = "cancel"
                self._add("phase cancel")
            elif message.stop_reason == "error" and message.error_message:
                self._add(f"model error {message.error_message}")

    def _upsert_tool(
        self, tool_id: str, name: str, args: dict[str, object], state: str, output: str,
        *, bounded: bool = False,
    ) -> None:
        item = self._tools.get(tool_id)
        if item is None:
            item = _Tool(tool_id=tool_id, name=name, args=dict(args), state=state)
            self._tools[tool_id] = item
            self._items.append(item)
        item.name = name
        if args:
            item.args = dict(args)
        item.state = state
        if output:
            item.output = output
        item.bounded = item.bounded or bounded

    def _set_phase(self, phase: str) -> None:
        if self._phase == phase:
            return
        self._phase = phase
        if phase == "model":
            self._add("phase model")

    def _add(self, text: str) -> None:
        if self._items and isinstance(self._items[-1], _Note) and self._items[-1].text == text:
            return
        self._items.append(_Note(text))

    def _print(self, text: str) -> None:
        """Append one line of command or diagnostic output without deduplication."""
        self._items.append(_Note(text))

    def _request_exit(self, code: int) -> None:
        if self._exit_task is None:
            self._exit_task = asyncio.create_task(self._shutdown(code))

    def _on_process_signal(self, code: int) -> None:
        self._signals += 1
        if self._signals >= 2:
            self.restore()
            os._exit(code)
        self._request_exit(code)

    async def _shutdown(self, code: int) -> None:
        if self._closing:
            return
        self._closing = True
        self.exit_code = code
        try:
            if self._startup_task is not None and not self._startup_task.done():
                self._startup_task.cancel()
                try:
                    await self._startup_task
                except asyncio.CancelledError:
                    pass
            if self._attachment_task is not None and not self._attachment_task.done():
                self._attachment_task.cancel()
                try:
                    await self._attachment_task
                except asyncio.CancelledError:
                    pass
            if self._external_task is not None and not self._external_task.done():
                self._external_task.cancel()
                try:
                    await self._external_task
                except asyncio.CancelledError:
                    pass
            if self._prompt_task is not None and not self._prompt_task.done():
                session = self._current_session()
                if session is not None and session.agent.state.is_busy:
                    self._phase = "cancel"
                    self._add("phase cancel")
                    self._redraw()
                    session.agent.abort()
                await self._prompt_task
            if self.runtime is not None:
                await self.runtime.close()
        except Exception as error:
            self._add(f"notice {error}")
        finally:
            if self._read_future is not None and not self._read_future.done():
                self._read_future.set_result(b"")
            if self._done is not None:
                self._done.set()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _redraw(self) -> None:
        if self._termios is None or self._restored or self._external_running:
            return
        rows, cols = _window_size(self._out)
        lines = _render(self._items, cols, thinking_open=self._thinking_open, tools_open=self._tools_open)
        status = _wrap(self._status(), cols)
        candidates = self._completion_lines(cols)
        entry, cursor_row, cursor_col = self._entry_layout(cols)
        if self.theme == "plain" or rows < 3:
            self._paint_linear(lines, status, candidates, entry)
            return
        self._paint_fancy(lines, status, candidates, entry, cursor_row, cursor_col, rows, cols)

    def _completion_lines(self, cols: int) -> list[str]:
        completion = self._completion
        if completion is None or not completion.items:
            return []
        items = completion.items
        start = max(0, min(completion.index - _COMPLETION_VISIBLE // 2, max(0, len(items) - _COMPLETION_VISIBLE)))
        shown = items[start:start + _COMPLETION_VISIBLE]
        lines = [
            _fit(
                f"{'>' if start + offset == completion.index else ' '} {item.label}"
                f"{'  ' + item.description if item.description else ''}",
                cols,
            )
            for offset, item in enumerate(shown)
        ]
        if len(items) > _COMPLETION_VISIBLE:
            lines.append(_fit(f"  {len(items)} candidates", cols))
        return lines

    def _entry_layout(self, cols: int) -> tuple[list[str], int, int]:
        prefix = "directory> " if self._chooser == "cwd" else "> "
        return _layout_entry(self.editor.text, width=cols, prefix=prefix, cursor=self.editor.cursor)

    def _paint_fancy(
        self, lines: Sequence[str], status: list[str], candidates: list[str], entry: list[str],
        cursor_row: int, cursor_col: int, rows: int, cols: int,
    ) -> None:
        status, candidates, entry, cursor_row = _fit_chrome(status, candidates, entry, cursor_row, rows)
        chrome = len(status) + len(candidates) + len(entry)
        body = max(rows - chrome, 1)
        if len(lines) > body:
            hidden = list(lines[:-body])
            visible = list(lines[-body:])
        else:
            hidden = []
            visible = list(lines)
        self._emit_scrollback(hidden, body)
        parts = ["\x1b[?25l", "\x1b[r"]
        for row in range(1, body + 1):
            text = visible[row - 1] if row - 1 < len(visible) else ""
            parts.append(f"\x1b[{row};1H\x1b[K{self._paint(text)}")
        row = body + 1
        for section in (status, candidates, entry):
            for text in section:
                parts.append(f"\x1b[{row};1H\x1b[K{self._paint(text)}")
                row += 1
        entry_top = body + len(status) + len(candidates) + 1
        column = max(1, min(cursor_col + 1, cols))
        parts.append(f"\x1b[{entry_top + cursor_row};{column}H\x1b[?25h")
        self._write("".join(parts))

    def _emit_scrollback(self, hidden: Sequence[str], body_rows: int) -> None:
        """Write conversation lines that no longer fit so they stay in scrollback."""
        previous = self._scrollback
        index = 0
        limit = min(len(previous), len(hidden))
        while index < limit and previous[index] == hidden[index]:
            index += 1
        fresh = hidden[index:]
        self._scrollback = tuple(hidden)
        if not fresh:
            return
        parts = [f"\x1b[1;{body_rows}r", f"\x1b[{body_rows};1H"]
        for line in fresh:
            parts.append(f"\x1b[K{self._paint(line)}\n")
        parts.append("\x1b[r")
        self._write("".join(parts))

    def _paint_linear(
        self, lines: Sequence[str], status: list[str], candidates: list[str], entry: list[str],
    ) -> None:
        previous = self._previous
        share = 0
        for left, right in zip(previous, lines, strict=False):
            if left != right:
                break
            share += 1
        if share < len(previous):
            share = min(share, len(lines))
        fresh = lines[share:]
        if fresh:
            self._write("\n".join(self._paint(line) for line in fresh) + "\n")
        self._previous = tuple(lines)
        chrome = "\n".join([*status, *candidates, *entry])
        if chrome != self._previous_chrome:
            self._write(self._paint(chrome) + "\n")
            self._previous_chrome = chrome

    def _paint(self, text: str) -> str:
        color = _COLORS[self.theme]
        if not color:
            return text
        return f"{color}{text}\x1b[0m"

    def _status(self) -> str:
        session = self._current_session()
        if session is None:
            identity = save = mode = model = thinking = cwd = path = "none"
        else:
            identity = session.agent.history.conversation_id
            mode = session.save_mode
            save = session.save_state
            selected = session.agent.state.model
            model = f"{selected.provider}/{selected.id}"
            thinking = session.agent.state.thinking_level
            cwd = str(session.cwd)
            path = str(session.path) if session.path is not None else "memory"
        return (
            f"phase {self._phase} | session {identity} | mode {mode} | save {save} | "
            f"model {model} | thinking {thinking} | theme {self.theme} | "
            f"cwd {cwd} | path {path} | pending {len(self._draft)}"
        )

    def _write(self, text: str) -> None:
        view = memoryview(text.encode("utf-8"))
        while view:
            try:
                written = os.write(self._out, view)
            except BlockingIOError:
                _readable, writable, _errors = select.select([], [self._out], [], 0.2)
                if not writable:
                    return
                continue
            except OSError:
                return
            if written <= 0:
                return
            view = view[written:]


def _fit_chrome(
    status: list[str], candidates: list[str], entry: list[str], cursor_row: int, rows: int,
) -> tuple[list[str], list[str], list[str], int]:
    """Trim chrome so the entry keeps a status line and fits above one body row."""
    budget = max(rows - 1, 1)
    status_lines = list(status)
    entry_lines = list(entry)
    # The phase line must stay visible even while a long draft fills the editor.
    status_keep = min(len(status_lines), max(1, budget // 4)) if status_lines else 0
    entry_budget = max(1, budget - status_keep) if entry_lines else 0
    if len(entry_lines) > entry_budget:
        start = max(0, min(cursor_row - entry_budget // 2, len(entry_lines) - entry_budget))
        entry_lines = entry_lines[start:start + entry_budget]
        cursor_row -= start
    remaining = budget - len(entry_lines)
    status_lines = status_lines[:max(0, remaining)]
    remaining -= len(status_lines)
    candidates = list(candidates)[:max(0, remaining)]
    return status_lines, candidates, entry_lines, cursor_row


def _layout_entry(text: str, *, width: int, prefix: str, cursor: int) -> tuple[list[str], int, int]:
    """Wrap the editor buffer and map the cursor to a rendered row and column."""
    lines = text.split("\n")
    cursor_line = 0
    remaining = cursor
    for index, line in enumerate(lines):
        if remaining <= len(line):
            cursor_line, cursor_column = index, remaining
            break
        remaining -= len(line) + 1
    else:
        cursor_line, cursor_column = len(lines) - 1, len(lines[-1])
    rendered: list[str] = []
    cursor_row = cursor_col = 0
    for index, line in enumerate(lines):
        lead = prefix if index == 0 else _CONTINUATION
        chunks = _wrap(lead + line, width)
        start = len(rendered)
        rendered.extend(chunks)
        if index == cursor_line:
            column = _display_width(lead) + _display_width(line[:cursor_column])
            used = 0
            for offset, chunk in enumerate(chunks):
                chunk_width = _display_width(chunk)
                if used + chunk_width >= column or offset == len(chunks) - 1:
                    cursor_row = start + offset
                    cursor_col = max(0, column - used)
                    break
                used += chunk_width
    if not rendered:
        rendered = [""]
    return rendered, cursor_row, cursor_col


def _last_assistant_text(runtime: AgentSessionRuntime | None) -> str:
    session = runtime.current_session if runtime is not None else None
    if session is None:
        return ""
    for message in reversed(session.agent.state.messages):
        if isinstance(message, AssistantMessage):
            text = _assistant_parts(message)[0].strip()
            if text:
                return text
    return ""


def _render(
    items: Sequence[_Item], width: int, *, thinking_open: bool, tools_open: bool,
) -> list[str]:
    lines: list[str] = []
    for item in items:
        if isinstance(item, _Note):
            lines.extend(_wrap(item.text, width))
        elif isinstance(item, _You):
            header = f"you {item.text}"
            lines.extend(_wrap(header.rstrip() if item.image_labels else header, width))
            for label in item.image_labels:
                lines.extend(_wrap(f"image {label}", width))
        elif isinstance(item, _Assistant):
            lines.extend(_render_assistant(item, width, thinking_open=thinking_open))
        else:
            lines.extend(_render_tool(item, width, tools_open=tools_open))
    return lines


def _render_assistant(item: _Assistant, width: int, *, thinking_open: bool) -> list[str]:
    lines = ["assistant"]
    if item.thinking:
        state = "expanded" if thinking_open else "collapsed"
        lines.append(f"thinking {state} ({len(item.thinking)} chars)")
        if thinking_open:
            lines.extend(item.thinking.splitlines() or [""])
    if item.text:
        lines.extend(_markdown_lines(item.text))
    return _flatten(lines, width)


def _flatten(lines: Sequence[str], width: int) -> list[str]:
    rendered: list[str] = []
    for line in lines:
        rendered.extend(_wrap(line, width))
    return rendered


def _markdown_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_code = False
    for raw in text.splitlines() or [""]:
        if raw.startswith("```"):
            if not in_code:
                lines.append(f"code {raw[3:].strip()}".rstrip())
                in_code = True
            else:
                lines.append("code end")
                in_code = False
            continue
        lines.append(f"  {raw}" if in_code else raw)
    return lines


def _render_tool(item: _Tool, width: int, *, tools_open: bool) -> list[str]:
    args = _brief_args(item.name, item.args)
    header = f"tool {item.name} {item.state}"
    if args:
        header = f"{header} {args}"
    lines = [header]
    if item.state == "running":
        snippet = item.output.strip().splitlines()[-1] if item.output.strip() else ""
        if len(snippet) > 48:
            snippet = snippet[:45] + "..."
        lines.append(f"progress {snippet}".rstrip())
    else:
        fold = "expanded" if tools_open else "collapsed"
        bound = " bounded" if item.bounded else ""
        lines.append(f"output {fold} recorded{bound} ({len(item.output)} chars)")
    if tools_open and item.state != "running":
        lines.extend(item.output.splitlines() or [""])
    return _flatten(lines, width)


def _brief_args(name: str, args: dict[str, object]) -> str:
    if not args:
        return ""
    if name == "read":
        return f"path={args.get('path', '')}"
    if name == "write":
        content = args.get("content", "")
        return f"path={args.get('path', '')} content=({len(content) if isinstance(content, str) else 0} chars)"
    if name == "edit":
        edits = args.get("edits")
        count = len(edits) if isinstance(edits, list) else 0
        return f"path={args.get('path', '')} edits={count}"
    if name == "bash":
        command = str(args.get("command", ""))
        if len(command) > 48:
            command = command[:45] + "..."
        return f"command={command}"
    parts: list[str] = []
    for key, value in list(args.items())[:3]:
        text = str(value)
        if len(text) > 24:
            text = text[:21] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _assistant_parts(message: AssistantMessage) -> tuple[str, str]:
    """Return one assistant message's text and thinking content."""
    texts: list[str] = []
    thoughts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            texts.append(block.text)
        elif isinstance(block, ThinkingContent):
            thoughts.append(block.thinking)
    return "".join(texts), "".join(thoughts)


def _assistant_from(message: AssistantMessage) -> _Assistant:
    text, thinking = _assistant_parts(message)
    return _Assistant(text=text, thinking=thinking, error=message.error_message, live=False)


def _user_text(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(block.text for block in message.content if isinstance(block, TextContent))


def _user_image_labels(message: UserMessage) -> tuple[str, ...]:
    """Visible labels for the real images saved in a user message."""
    if not isinstance(message.content, list):
        return ()
    return tuple(block.mime_type for block in message.content if isinstance(block, ImageContent))


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _content_text(content: Sequence[TextContent | ImageContent]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(f"image {block.mime_type}")
    return "\n".join(parts)


def _text_is_bounded(output: str, details: object) -> bool:
    if isinstance(details, dict):
        truncation = details.get("truncation")
        if isinstance(truncation, dict) and truncation.get("truncated") is True:
            return True
    return "Full output:" in output or "more lines in file" in output


def _window_size(fd: int) -> tuple[int, int]:
    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
    except OSError:
        return 24, 80
    rows, cols, _x, _y = struct.unpack("HHHH", packed)
    if rows < 1 or cols < 1:
        return 24, 80
    return int(rows), int(cols)


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _wrap(text: str, width: int) -> list[str]:
    text = _terminal_text(text)
    if width < 1 or not text:
        return [text]
    rendered: list[str] = []
    for source in text.split("\n"):
        current = ""
        used = 0
        for char in source:
            needed = _char_width(char)
            if current and used + needed > width:
                rendered.append(current)
                current = char
                used = needed
            else:
                current += char
                used += needed
        rendered.append(current)
    return rendered


def _fit(text: str, width: int) -> str:
    text = _terminal_text(text)
    if _display_width(text) <= width:
        return text
    kept: list[str] = []
    used = 0
    for char in reversed(text):
        needed = _char_width(char)
        if used + needed > width:
            break
        kept.append(char)
        used += needed
    return "".join(reversed(kept))


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in _terminal_text(text))


def _terminal_text(text: str) -> str:
    """Show untrusted terminal controls as text without changing saved content."""
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in text)
