"""Regular interactive session: scrolling conversation, editor and status.

The screen stays on the primary terminal. A capable terminal pins the status
and editor beneath a scrolling conversation; a plain or dumb terminal prints
the same words without color or cursor addressing. Prompts go through
``AgentSessionRuntime`` and the common host. This module does not start a
second Agent loop, and print execution does not import it.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import os
import select
import shlex
import signal
import struct
import termios
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO, cast

from omh.agent import (
    AgentEvent,
    AgentHistory,
    AgentOptions,
    CustomAgentMessage,
    MessageHistoryEntry,
    PrepareRequestContext,
    validate_history,
)
from omh.agent.conversation.history import history_path
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
from omh.llm import get_supported_thinking_levels
from omh.llm.types import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    ImageContent,
    Model,
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
from coding_agent.config import (
    ConfigError,
    load_settings,
    project_settings_path,
    settings_theme,
    update_settings,
)
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
    HISTORY_ORIGIN,
    AttachmentOrigin,
    PendingAttachment,
    PendingAttachments,
    read_pending_image,
)
from coding_agent.preferences import (
    default_change,
    effect_of,
    parse_tools,
    resolve_model,
)
from coding_agent.resources import ApplicationResources
from coding_agent.session_directory import (
    SESSION_SORTS,
    SessionDirectory,
    SessionInfo,
    SessionSort,
)
from coding_agent.session_manager import ExportFormat, SaveMode
from coding_agent.terminal import EXIT_OK
from coding_agent.theme import resolve_theme
from coding_agent.user_shell import UserShell


@dataclass(frozen=True, slots=True)
class _DerivationRequest:
    entry_id: str | None = None
    cwd: Path | None = None
    save_mode: SaveMode | None = None
    session_dir: str | None = None


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


_Item = _Note | _You | _Assistant | _Tool | UserShell


@dataclass(slots=True)
class _ConversationDraft:
    editor: Editor
    images: PendingAttachments



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
        self._unresolved_selection: SessionSelection | None = None
        self._thinking_open = False
        self._tools_open = False
        self._busy = False
        self._closing = False
        self._prompt_task: asyncio.Task[None] | None = None
        self._shell_task: asyncio.Task[None] | None = None
        self._shell: UserShell | None = None
        self._shell_abort: AbortController | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._exit_task: asyncio.Task[None] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._external_task: asyncio.Task[None] | None = None
        self._copy_task: asyncio.Task[None] | None = None
        self._external_running = False
        self._ctrl_c_at = 0.0
        #: Editor text cleared by the first Ctrl+C, kept only for the exit
        #: confirmation that a second Ctrl+C within 500ms can still undo.
        self._cleared_text: str | None = None
        #: While set, Enter discards the remaining unsubmitted content and any
        #: other key returns to editing; editor-key exits ask before discarding.
        self._exit_confirm = False
        #: Draft attachments captured into queued inputs, paired with the sent
        #: image data so a recall can restore the full original identity.
        self._queued_images: list[tuple[str, PendingAttachment]] = []
        self._drafts: dict[str, _ConversationDraft] = {}
        self._management_task: asyncio.Task[None] | None = None
        self._model_selector: tuple[Model, ...] | None = None
        self._request_selection: str | None = None
        self._fork_selector: tuple[MessageHistoryEntry, ...] | None = None
        self._fork_request: _DerivationRequest | None = None
        self._escape_at = 0.0
        self._selector: tuple[SessionInfo, ...] | None = None
        self._selector_source: Editor | None = None
        self._selector_index = 0
        self._selector_all = False
        self._selector_sort: SessionSort = "mtime"
        self._selector_reverse: bool | None = None
        self._exit_backups: dict[AgentSession, AgentHistory] = {}
        self._exit_discard_unsaved = False
        self._pending_exit_code = EXIT_OK
        #: Whether the run's first user message event is the prompt the editor
        #: already displayed; queued steering and follow-ups display on arrival.
        self._prompt_first_user = False
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
        self._thinking_open = self.host.settings.values.get("hideThinking") is False
        self._tools_open = self.host.settings.values.get("collapseTools") is False
        if self.args.resume:
            self._resume_command("")
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

    async def _apply_selection(
        self, selection: SessionSelection, *, derivation: _DerivationRequest | None = None,
    ) -> None:
        assert self.host is not None
        self._unresolved_selection = selection
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
        await self._open_runtime(selection, derivation=derivation)
        self._prepare_theme(settings_theme(self.host.settings))
        self._thinking_open = self.host.settings.values.get("hideThinking") is False
        self._tools_open = self.host.settings.values.get("collapseTools") is False

    async def _open_runtime(
        self, selection: SessionSelection, *, derivation: _DerivationRequest | None = None,
    ) -> None:
        assert self.host is not None
        options = self.host.build_options(selection, agent_options=AgentOptions(prepare_request=self._capture_request))
        runtime = self.runtime
        if runtime is None:
            runtime = AgentSessionRuntime(options)
            runtime.subscribe(self._on_event)
            self.runtime = runtime
        else:
            runtime.options = options
        old = runtime.current_session
        if derivation is not None:
            if derivation.entry_id is None:
                await runtime.clone_session(cwd=selection.cwd, save_mode=derivation.save_mode,
                                            session_dir=derivation.session_dir)
            else:
                await runtime.fork_session(derivation.entry_id, cwd=selection.cwd, save_mode=derivation.save_mode,
                                           session_dir=derivation.session_dir)
        elif selection.session_path is None:
            await runtime.new_session(display_name=self.args.name if old is None else None)
        else:
            await runtime.open_session(selection.session_path)
        self._chooser = "prompt"
        if old is None:
            self._load_history()
        if old is None and selection.session_path is not None and self.args.name is not None:
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
            if isinstance(message, CustomAgentMessage):
                shell = UserShell.from_message(message)
                if shell is not None:
                    self._items.append(shell)
            elif isinstance(message, UserMessage):
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
        if not self.editor.history:
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
        if key.name != "escape":
            self._escape_at = 0.0
        if self._model_selector is not None:
            self._model_selector_key(key)
            return
        if self._fork_selector is not None:
            self._fork_selector_key(key)
            return
        if self._selector is not None:
            self._selector_key(key)
            return
        if key.name != "ctrl_c":
            # Only consecutive Ctrl+C presses form the exit double-press; any
            # other key starts a fresh pair.
            self._ctrl_c_at = 0.0
        if self._exit_confirm:
            self._answer_exit_confirm(accept=key.name == "enter")
            return
        name = key.name
        if name == "ctrl_l":
            self._open_model_selector()
            return
        if name in {"ctrl_p", "shift_ctrl_p"}:
            self._cycle_model(-1 if name == "shift_ctrl_p" else 1)
            return
        if name == "shift_tab":
            session = self._current_session()
            if session is not None:
                levels = get_supported_thinking_levels(session.agent.state.model)
                if len(levels) > 1:
                    current = session.agent.state.thinking_level
                    self._start_preference("thinking", levels[(levels.index(current) + 1) % len(levels)])
                else:
                    self._add(f"thinking {self._thinking_mode(session.agent.state.model, session.agent.state.thinking_level)}; no other effective level")
            return
        if name == "ctrl_s":
            session = self._current_session()
            if session is not None:
                self._settings_command(f"global thinking {session.agent.state.thinking_level}")
            return
        if name == "escape":
            if self._completion is not None:
                self._completion = None
                return
            self._escape_action()
            return
        if name == "alt_enter":
            self._accept_or_submit(follow_up=True)
            return
        if name == "alt_up":
            self._recall_queued()
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
            self._forget_cleared_line()
            self._completion = None
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
            self._forget_cleared_line()
            self._refresh_completion()
            return
        if name == "ctrl_c":
            self._ctrl_c()
            return
        if name == "ctrl_d":
            if self.editor.empty:
                self._request_exit(EXIT_OK, confirm=True)
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
            self._forget_cleared_line()
            self._refresh_completion()

    def _accept_or_submit(self, *, follow_up: bool = False) -> None:
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
            self._forget_cleared_line()
            return
        self._submit(follow_up=follow_up)

    def _accept_completion(self) -> None:
        completion = self._completion
        if completion is None:
            return
        self.editor.replace_token(completion.start, completion.end, completion.current.value)
        self._forget_cleared_line()
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
            self._ctrl_c_at = 0.0
            self._request_exit(EXIT_OK, confirm=True)
            return
        if self.editor.text:
            # Keep the last cleared line available to the exit decision; only a
            # genuine edit, acceptance or exit resolves it.
            self._cleared_text = self.editor.text
        self.editor.clear()
        self._completion = None
        self._ctrl_c_at = now

    def _forget_cleared_line(self) -> None:
        """Resolve the Ctrl+C-cleared line once real editing replaces it."""
        # A slash command operates on the conversation while its cleared draft
        # stays available for switching, selector cancellation or exit.
        if not self.editor.text.startswith("/"):
            self._cleared_text = None

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
            models=tuple(f"{entry.provider}/{entry.model.id}" for entry in self.host.directory.listings()) if self.host else (),
            thinking_levels=tuple(get_supported_thinking_levels(session.agent.state.model)) if session else (),
        )

    # ------------------------------------------------------------------ #
    # Submission and dispatch
    # ------------------------------------------------------------------ #

    def _submit(self, *, follow_up: bool = False) -> None:
        if self._exit_task is not None or self._closing:
            self._add("notice exit is in progress; no new work was accepted")
            return
        text = self.editor.text
        self._completion = None
        if self._chooser == "cwd":
            stripped = text.strip()
            if not stripped:
                return
            self.editor.clear()
            self._forget_cleared_line()
            self._choose_cwd(stripped)
            return
        if text.startswith("!"):
            self._start_shell(text)
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
            # Only a model run this session started accepts queued input; other
            # busy windows (for example a directory reselection) still ignore Enter.
            if self._prompt_task is not None and not self._prompt_task.done():
                self._queue_input(text, follow_up=follow_up)
            return
        if self._management_task is not None and not self._management_task.done():
            self._add("notice session management is in progress; input stays in the editor")
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
            self._show_unsaved()
            return
        warning = session.unsupported_image_message() if len(self._draft) else None
        if warning is not None:
            self._keep_refused_submission(text, f"image not sent: {warning}")
            return
        self.editor.clear()
        self.editor.remember(text)
        self._forget_cleared_line()
        pending = self._draft.pop_all()
        self._pending_you = _You(text, image_labels=tuple(item.label for item in pending))
        self._items.append(self._pending_you)
        self._submitted_entries = len(session.agent.history.entries)
        self._busy = True
        self._prompt_first_user = True
        self._set_phase("model")
        self._prompt_task = asyncio.create_task(self._prompt(text, pending))

    def _start_shell(self, text: str) -> None:
        if self._management_task is not None and not self._management_task.done():
            self._add("notice session management is in progress; no shell was started")
            return
        if self._shell_task is not None and not self._shell_task.done():
            self._add("notice a user shell is already running; cancel it before starting another")
            return
        session = self._current_session()
        if session is None:
            self._add("notice the session is not ready; no shell was started")
            return
        if session.agent.state.activity_kind == "manual_compaction":
            self._add("notice manual compaction is running; no shell was started")
            return
        try:
            session.ensure_can_accept_work()
        except RuntimeError as error:
            self._add(f"notice {error}")
            if session.save_state == "unsaved":
                self._show_unsaved()
            return
        excluded = text.startswith("!!")
        command = text[2 if excluded else 1:]
        if not command.strip():
            self._add("notice ! or !! needs a command; no shell was started")
            return
        self.editor.clear()
        self.editor.remember(text)
        self._forget_cleared_line()
        shell = UserShell(command, excluded)
        self._shell = shell
        self._shell_abort = AbortController()
        self._items.append(shell)
        self._add("notice model and user shell may operate on the same workspace")
        self._shell_task = asyncio.create_task(self._run_shell(session, shell, self._shell_abort.signal))

    async def _run_shell(self, session: AgentSession, shell: UserShell, signal: AbortSignal) -> None:
        await shell.execute(session.cwd, signal, self._redraw)
        try:
            # This shell was admitted before execution. A later saving failure
            # must not reject its terminal record; Agent owns the safe boundary.
            await session.agent.submit_custom_message(shell.message())
        except Exception as error:
            self._add(f"notice shell record: {error}")
            if session.save_state == "unsaved":
                self._show_unsaved()
        finally:
            self._shell_task = None
            self._add("shell settled")
            self._redraw()

    def _queue_input(self, text: str, *, follow_up: bool) -> None:
        """Accept steering or a follow-up while the model is running.

        The draft images are captured into the queued message, and skills and
        templates expand at acceptance, matching the queued-input contract.
        """
        kind = "follow-up" if follow_up else "steering"
        session = self._current_session()
        if self._attachment_task is not None and not self._attachment_task.done():
            self._add("notice an attachment is still being prepared; press Enter again")
            return
        if not text.strip():
            if len(self._draft):
                self._add(
                    f"notice a queued {kind} needs prompt text; the draft images stay pending"
                )
            return
        if session is None or self._request_block is not None:
            self._add(f"notice {self._request_block or 'The session is not ready.'}")
            self._add(f"notice {_NO_REQUEST}")
            return
        warning = session.unsupported_image_message() if len(self._draft) else None
        if warning is not None:
            self._add(f"notice image not sent: {warning}")
            self._add(f"notice {_NO_REQUEST}")
            return
        pending = self._draft.pop_all()
        images = [attachment.content for attachment in pending]
        try:
            if follow_up:
                session.follow_up(text, images=images or None)
            else:
                session.steer(text, images=images or None)
        except Exception as error:
            self._draft.restore(pending)
            self._add(f"notice {error}")
            self._add("notice the queued input was not accepted")
            if session.save_state == "unsaved":
                self._show_unsaved()
            return
        for attachment in pending:
            self._queued_images.append((attachment.image.data, attachment))
        self.editor.clear()
        self.editor.remember(text)
        self._forget_cleared_line()
        snapshot = session.queued_messages
        # Keep identities only for images still waiting in a queue, so the list
        # cannot grow past what a recall could actually need.
        in_flight = {
            block.data
            for message in (*snapshot.steering, *snapshot.follow_up)
            for block in _message_images(message)
        }
        self._queued_images = [
            (data, attachment)
            for data, attachment in self._queued_images
            if data in in_flight
        ]
        waiting = len(snapshot.follow_up) if follow_up else len(snapshot.steering)
        images_note = f" +{len(pending)} image" if len(pending) == 1 else (
            f" +{len(pending)} images" if pending else ""
        )
        self._add(f"queued {kind} ({waiting} waiting): {text.strip()}{images_note}")

    def _take_queued_image(self, data: str) -> PendingAttachment | None:
        """Take the earliest captured record for one sent image during a recall."""
        for index, (key, item) in enumerate(self._queued_images):
            if key == data:
                return self._queued_images.pop(index)[1]
        return None

    def _recall_queued(self) -> None:
        """Return every queued steering and follow-up to the source editor.

        All steering comes back first, then all follow-ups, merged into the
        editor with blank lines; the current draft and any queued images are
        preserved. The SDK queue is cleared exactly by this explicit recall.
        """
        session = self._current_session()
        if session is None or not self._has_queued_inputs(session):
            self._add("notice no queued inputs to recall")
            return
        snapshot = session.queued_messages
        session.agent.clear_all_queues()
        texts: list[str] = []
        images: list[PendingAttachment] = []
        missing = 0
        for message in (*snapshot.steering, *snapshot.follow_up):
            if not isinstance(message, UserMessage):
                continue
            texts.append(_user_text(message))
            for block in _message_images(message):
                attachment = self._take_queued_image(block.data)
                if attachment is not None:
                    images.append(attachment)
                else:
                    missing += 1
        draft = self.editor.text
        parts = [text for text in texts if text.strip()]
        if draft.strip():
            parts.append(draft)
        self.editor.set_text("\n\n".join(parts))
        self._forget_cleared_line()
        self._draft.restore(tuple(images))
        note = f"recalled {len(texts)} queued input(s) into the editor"
        if images:
            note += f" and {len(images)} image(s) back to the pending draft"
        self._add(note)
        if missing:
            self._add(
                f"notice {missing} queued image(s) could not be restored to the pending draft"
            )

    def _escape_action(self) -> None:
        """Recall queues and cancel the model first, then a still-running shell."""
        if self._management_task is not None and not self._management_task.done():
            if not self._management_task.cancelling():
                self._management_task.cancel()
            else:
                self._add("notice handoff continues; waiting for actual current")
            return
        session = self._current_session()
        if session is not None and self._has_queued_inputs(session):
            self._recall_queued()
        if session is not None and session.agent.state.is_busy:
            self._phase = "cancel"
            self._add("phase cancel")
            session.agent.abort()
        elif self._shell_task is not None and not self._shell_task.done():
            assert self._shell_abort is not None
            self._add("shell cancel requested")
            self._shell_abort.abort()
        elif not self.editor.text:
            now = asyncio.get_running_loop().time()
            if self._escape_at and now - self._escape_at <= 0.5:
                self._escape_at = 0.0
                self._derivation_command("fork", "")
            else:
                self._escape_at = now

    def _unsubmitted_content(self) -> list[str]:
        """Descriptions of unsubmitted content an exit would silently lose."""
        parts: list[str] = []
        if self._cleared_text is not None and self._cleared_text.strip():
            parts.append("cleared editor text")
        if self.editor.text.strip():
            parts.append("editor text")
        if self._attachment_task is not None and not self._attachment_task.done():
            parts.append("an image still being prepared")
        if len(self._draft):
            parts.append(f"{len(self._draft)} pending image(s)")
        session = self._current_session()
        if session is not None:
            snapshot = session.queued_messages
            waiting = len(snapshot.steering) + len(snapshot.follow_up)
            if waiting:
                parts.append(f"{waiting} queued input(s)")
        current_id = session.agent.history.conversation_id if session is not None else None
        for identity, draft in self._drafts.items():
            if (identity != current_id or draft.editor is not self.editor) and (draft.editor.text.strip() or len(draft.images)):
                parts.append(f"draft for {identity}")
        if self.runtime is not None:
            for number, old in enumerate(self.runtime.retained_sessions, 1):
                if self._has_queued_inputs(old):
                    parts.append(f"queued input(s) for retained {number}")
        return parts

    def _has_queued_inputs(self, session: AgentSession) -> bool:
        """Whether the product's queue snapshot still holds steering or follow-up."""
        snapshot = session.queued_messages
        return bool(snapshot.steering or snapshot.follow_up)

    def _begin_exit_confirm(self, parts: Sequence[str]) -> None:
        self._exit_confirm = True
        self._add(
            "notice unsubmitted content remains: " + ", ".join(parts)
            + "; it is not saved to disk"
        )
        self._add("notice Enter discards it and exits; any other key returns to editing")

    def _answer_exit_confirm(self, *, accept: bool) -> None:
        """Resolve the exit confirmation: explicit discard or return to editing."""
        self._exit_confirm = False
        if not accept:
            self._exit_discard_unsaved = False
            if self._cleared_text is not None:
                self.editor.set_text(self._cleared_text)
                self._cleared_text = None
            self._add("notice exit canceled; the unsubmitted content is kept")
            return
        session = self._current_session()
        if session is not None:
            session.agent.clear_all_queues()
        if self._attachment_task is not None and not self._attachment_task.done():
            self._attachment_task.cancel()
        self._draft.clear()
        for draft in self._drafts.values():
            draft.editor.clear()
        self._drafts.clear()
        self._cleared_text = None
        self._add("notice discarding the unsubmitted content; it will not be saved")
        self._request_exit(EXIT_OK)

    def _keep_refused_submission(self, text: str, reason: str | None) -> None:
        """Keep a refused prompt without losing the editor text or a draft image.

        Plain text with no pending image follows the existing display-only
        behavior. With a pending image the text returns to the editor so a
        refused send loses nothing. ``None`` means the caller adds its own notice.
        """
        if len(self._draft) == 0:
            self.editor.clear()
            self.editor.remember(text)
            self._forget_cleared_line()
            self._items.append(_You(text))
        if reason is None:
            return
        self._add(f"notice {reason}")
        self._add(f"notice {_NO_REQUEST}")

    def _draft_edit_allowed(self) -> bool:
        if self._management_task is not None and not self._management_task.done():
            self._add("notice session switch in progress; wait before editing the draft")
            return False
        return True

    def _paste_clipboard(self) -> None:
        """Start reading a clipboard screenshot without submitting anything."""
        if not self._draft_edit_allowed():
            return
        if self._attachment_task is not None and not self._attachment_task.done():
            return
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
        if self._unresolved_selection is not None and self._unresolved_selection.cwd is not None:
            return self._unresolved_selection.cwd
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
        if not self._draft_edit_allowed():
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

    @staticmethod
    def _thinking_mode(model: Model, level: str) -> str:
        return "fixed-on" if model.reasoning and not get_supported_thinking_levels(model) else level

    def _tool_names(self) -> str:
        session = self._current_session()
        return ",".join(tool.name for tool in session.agent.state.tools) if session and session.agent.state.tools else "none"

    def _capture_request(self, request: PrepareRequestContext, signal: AbortSignal | None) -> None:
        del signal
        model = request.model
        thinking = self._thinking_mode(model, request.thinking_level)
        tools = ",".join(tool.name for tool in request.context.tools) or "none"
        self._request_selection = f"{model.provider}/{model.id} thinking {thinking} tools {tools}"
        self._redraw()

    def _show_current_settings(self) -> None:
        session = self._current_session()
        if session is not None:
            state = session.agent.state
            self._add(f"current model {state.model.provider}/{state.model.id} thinking {self._thinking_mode(state.model, state.thinking_level)} tools {self._tool_names()}; next request, defaults unchanged")
        self._add(f"current theme {self.theme} hideThinking {not self._thinking_open} collapseTools {not self._tools_open}")
        self._add("/settings global|project <field> <value> writes only that default; /settings current <field> <value> changes this session")

    def _settings_command(self, argument: str) -> None:
        if self.host is None:
            self._add("notice configuration is unavailable")
            return
        scope, _, rest = (argument or "current").partition(" ")
        if scope not in {"current", "global", "project"}:
            self._add("notice /settings needs current, global or project scope; nothing written")
            return
        field, _, raw = rest.strip().partition(" ")
        if scope == "current":
            if not field:
                self._show_current_settings()
            elif field in {"model", "thinking", "tools", "theme", "hideThinking", "collapseTools"} and raw:
                self._start_preference(field, raw.strip())
            else:
                self._add("notice current fields: model, thinking, tools, theme, hideThinking, collapseTools; other defaults require global/project scope")
            return
        path = self.host.agent_dir / "settings.json" if scope == "global" else project_settings_path(self._cwd())
        if not field:
            snapshot = load_settings(agent_dir=self.host.agent_dir, cwd=self._cwd(), project_trusted=True)
            values = snapshot.global_values if scope == "global" else snapshot.project_values
            self._add(f"{scope} defaults target {path}: {json.dumps(dict(values), ensure_ascii=False)}")
            for item in snapshot.diagnostics:
                self._add(f"notice {item.message}")
            if scope == "project" and not self.host.project_trusted:
                self._add("notice project defaults are not loaded until project trust is granted")
            return
        try:
            if not raw:
                raise ConfigError("settings edit requires a value")
            changes = default_change(self.host.directory, field, raw.strip())
            update_settings(path, changes)
        except (OSError, ValueError, ConfigError) as error:
            self._add(f"notice {scope} write failed at {path}: {error}; no success reported, inspect target before retrying")
            return
        self._add(f"saved {scope} {path}: {field}; {effect_of(field)}")
        if field == "thinking":
            session = self._current_session()
            if session is not None:
                self._add(f"current thinking {self._thinking_mode(session.agent.state.model, session.agent.state.thinking_level)} unchanged")
        if scope == "project" and not self.host.project_trusted:
            self._add("notice project defaults require trust before loading")
        # Refresh default-only state for cycling, future sessions and reload.
        self.host.select_new(cwd=self._cwd())

    def _start_preference(self, field: str, value: str) -> None:
        if self._closing or (self._management_task is not None and not self._management_task.done()):
            self._add("notice session management is in progress; selection unchanged")
            return
        self._management_task = asyncio.create_task(self._change_preference(field, value))

    async def _change_preference(self, field: str, value: str) -> None:
        try:
            if self.host is None:
                raise ConfigError("configuration is unavailable")
            session = self._current_session()
            if field == "theme":
                default_change(self.host.directory, field, value)
                self.theme, _ = resolve_theme(requested=value, configured=None, environ=os.environ)
                self._add(f"current theme {self.theme}; defaults unchanged")
            elif field in {"hideThinking", "collapseTools"}:
                changes = default_change(self.host.directory, field, value)
                if field == "hideThinking":
                    self._thinking_open = not bool(changes[field])
                else:
                    self._tools_open = not bool(changes[field])
                self._add(f"current {field} {value}; defaults unchanged")
            elif field == "model":
                model = resolve_model(self.host.directory, value)
                if session is None:
                    unresolved = self._unresolved_selection
                    selection = (self.host.select_open(unresolved.session_path, model=value, thinking=self.args.thinking, tools=unresolved.tools, cwd=self._cwd())
                                 if unresolved is not None and unresolved.history is not None and unresolved.session_path is not None
                                 else self.host.select_new(model=value, thinking=self.args.thinking, tools=unresolved.tools if unresolved else None, cwd=self._cwd()))
                    await self._apply_selection(selection)
                    session = self._current_session()
                    if session is None:
                        return
                else:
                    session.ensure_can_accept_work()
                    previous = session.agent.state.thinking_level
                    previous_mode = self._thinking_mode(session.agent.state.model, previous)
                    await session.agent.set_model(model)
                    actual = session.agent.state.thinking_level
                    if previous != actual:
                        self._add(f"thinking adjusted from {previous_mode} to {self._thinking_mode(model, actual)}: selected model's effective capabilities")
                readiness = await self.host.readiness(SessionSelection(cwd=session.cwd, model=model, thinking_level=session.agent.state.thinking_level))
                self._request_block = next((item.message for item in readiness if item.blocking), None)
                for item in readiness:
                    self._add(f"notice {item.message}")
                self._add(f"current model {model.provider}/{model.id} thinking {self._thinking_mode(model, session.agent.state.thinking_level)}; next request, defaults unchanged")
            elif session is None:
                raise ConfigError("choose a model first with /model")
            elif field == "thinking":
                session.ensure_can_accept_work()
                levels = get_supported_thinking_levels(session.agent.state.model)
                if value not in levels:
                    raise ConfigError("valid thinking: " + (", ".join(levels) or "fixed-on (no adjustable level)"))
                await session.agent.set_thinking_level(value)
                self._add(f"current thinking {value}; next request, defaults unchanged")
            elif field == "tools":
                await session.set_tools(parse_tools(value))
                self._add(f"current tools {self._tool_names()}; next request, defaults unchanged")
            elif field == "reload":
                assert self.runtime is not None
                state = session.agent.state
                selection = self.host.select_new(cwd=session.cwd, model=f"{state.model.provider}/{state.model.id}", tools=self.runtime.options.tools)
                if not selection.ready:
                    raise ConfigError("; ".join(item.message for item in selection.diagnostics if item.blocking))
                prepared = self.host.build_options(selection)
                options = self.runtime.options
                previous_options = replace(options)
                for name in ("skill_sources", "template_sources", "custom_prompt", "append_system_prompt", "resource_tiers", "load_context_files"):
                    setattr(options, name, getattr(prepared, name))
                try:
                    resources = await session.reload_resources()
                except BaseException:
                    for name in ("skill_sources", "template_sources", "custom_prompt", "append_system_prompt", "resource_tiers", "load_context_files"):
                        setattr(options, name, getattr(previous_options, name))
                    raise
                for diagnostic in resources.diagnostics:
                    self._add(f"notice {diagnostic.message}")
                self._add("resources reloaded; next new prompt; accepted inputs unchanged")
        except Exception as error:
            self._add(f"notice {error}; inspect current selection (/settings current)")
        finally:
            self._remember_save()
            self._redraw()

    def _open_model_selector(self) -> None:
        if self.host is None:
            return
        self._completion = None
        self._model_selector = tuple(entry.model for entry in self.host.directory.listings())
        session = self._current_session()
        selected = session.agent.state.model if session is not None else None
        self._selector_index = next((index for index, model in enumerate(self._model_selector) if selected and (model.provider, model.id) == (selected.provider, selected.id)), 0)

    def _model_selector_key(self, key: Key) -> None:
        assert self._model_selector is not None
        if key.name in {"escape", "ctrl_c"}:
            self._model_selector = None
            self._add("model selection cancelled; draft preserved")
        elif key.name in {"up", "down", "ctrl_p", "shift_ctrl_p"}:
            self._selector_index = (self._selector_index + (-1 if key.name in {"up", "shift_ctrl_p"} else 1)) % len(self._model_selector)
        elif key.name == "enter":
            model = self._model_selector[self._selector_index]
            self._model_selector = None
            self._start_preference("model", f"{model.provider}/{model.id}")

    def _cycle_model(self, delta: int) -> None:
        if self.host is None:
            return
        try:
            configured = self.host.settings.values.get("enabledModels")
            entries = (tuple(resolve_model(self.host.directory, reference) for reference in configured)
                       if isinstance(configured, list) else tuple(entry.model for entry in self.host.directory.listings()))
            if not entries:
                raise ConfigError("model cycle is empty; configure enabledModels via /settings global models")
            session = self._current_session()
            current = session.agent.state.model if session else None
            index = next((i for i, model in enumerate(entries) if current and (model.provider, model.id) == (current.provider, current.id)), -1 if delta > 0 else 0)
            model = entries[(index + delta) % len(entries)]
            self._start_preference("model", f"{model.provider}/{model.id}")
        except (ValueError, ConfigError) as error:
            self._add(f"notice {error}; no provider fallback")

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
        elif spec.name in {"model", "thinking", "tools"}:
            if not argument and spec.name == "model":
                self._open_model_selector()
            elif not argument:
                self._show_current_settings()
            else:
                self._start_preference(spec.name, argument)
        elif spec.name == "settings":
            self._settings_command(argument)
        elif spec.name == "reload":
            if self._management_idle():
                self._start_preference("reload", "")
        elif spec.name == "attach":
            self._attach_command(argument)
        elif spec.name == "session":
            self._session_info()
        elif spec.name == "quit":
            if argument not in {"", "--discard-unsaved"}:
                self._add("notice /quit accepts only --discard-unsaved; nothing was sent")
                return
            self._request_exit(EXIT_OK, confirm=True, discard_unsaved=bool(argument))
        elif spec.name in {"new", "resume"}:
            if spec.name == "resume":
                self._resume_command(argument)
            elif self._management_idle():
                self._busy = True
                self._management_task = asyncio.create_task(self._replace_current(spec.name, argument))
        elif spec.name in {"fork", "clone"}:
            self._derivation_command(spec.name, argument)
        elif spec.name == "name" and not argument:
            session = self._current_session()
            self._add(f"name {session.display_name or '(unnamed)' if session else '(no session)'}")
        elif spec.name in {"name", "save", "export"}:
            if self._management_task is not None and not self._management_task.done():
                self._add("notice session management is already in progress")
                return
            self._management_task = asyncio.create_task(self._save_command(spec.name, argument))

    def _derivation_command(self, command: str, argument: str) -> None:
        if not self._management_idle():
            return
        try:
            tokens = shlex.split(argument)
            entry_id = None
            cwd = None
            mode: SaveMode | None = None
            directory = None
            seen: set[str] = set()
            while tokens:
                token = tokens.pop(0)
                if command == "fork" and not token.startswith("--") and entry_id is None:
                    entry_id = token
                    continue
                if token not in {"--cwd", "--save-mode", "--session-dir"} or token in seen or not tokens:
                    raise ValueError(f"Invalid derivation arguments; see /help {command}")
                seen.add(token)
                value = tokens.pop(0)
                if token == "--cwd":
                    cwd = (self._cwd() / Path(value).expanduser()).resolve()
                elif token == "--session-dir":
                    directory = value
                elif value in {"auto", "memory"}:
                    mode = cast(SaveMode, value)
                else:
                    raise ValueError("--save-mode needs auto or memory")
            if mode == "memory" and directory is not None:
                raise ValueError("Memory mode cannot specify --session-dir")
            request = _DerivationRequest(entry_id=entry_id, cwd=cwd, save_mode=mode, session_dir=directory)
            if command == "fork" and entry_id is None:
                source = self._current_session()
                if source is None:
                    raise ValueError("Create or open a source session first")
                self._fork_selector = tuple(entry for entry in reversed(history_path(source.agent.history))
                                            if isinstance(entry, MessageHistoryEntry) and entry.message.role == "user")
                self._fork_request = request
                self._selector_source = self.editor
                self.editor = Editor()
                self._selector_index = 0
                self._add("Select fork: Up/Down choose a user; Enter copies ancestors and refills input; Escape returns to draft")
                return
            self._busy = True
            self._management_task = asyncio.create_task(self._replace_current(command, "", derivation=request))
        except ValueError as error:
            self._add(f"notice {error}; nothing was sent")

    def _fork_selector_key(self, key: Key) -> None:
        assert self._fork_selector is not None
        if key.name in {"escape", "ctrl_c", "ctrl_d"}:
            self._leave_fork_selector()
            if self._cleared_text is not None:
                self.editor.set_text(self._cleared_text)
                self._cleared_text = None
            self._add("notice fork selector canceled; source draft kept")
        elif key.name in {"up", "down"}:
            if self._fork_selector:
                self._selector_index = (self._selector_index + (1 if key.name == "down" else -1)) % len(self._fork_selector)
        elif key.name == "enter" and self._fork_selector and self._management_idle():
            assert self._fork_request is not None
            request = replace(self._fork_request, entry_id=self._fork_selector[self._selector_index].id)
            self._leave_fork_selector()
            self._busy = True
            self._management_task = asyncio.create_task(self._replace_current("fork", "", derivation=request))

    def _leave_fork_selector(self) -> None:
        assert self._selector_source is not None
        self.editor = self._selector_source
        self._selector_source = None
        self._fork_selector = None
        self._fork_request = None

    def _resume_command(self, argument: str) -> None:
        try:
            tokens = shlex.split(argument)
            reference: str | None = None
            listing = all_projects = False
            reverse_requested = False
            search = ""
            sort: SessionSort = "mtime"
            reverse: bool | None = None
            while tokens:
                token = tokens.pop(0)
                if token == "--list":
                    listing = True
                elif token == "--all-projects":
                    all_projects = True
                elif token == "--reverse":
                    reverse_requested = True
                elif token in {"--search", "--sort"}:
                    if not tokens:
                        raise ValueError(f"{token} needs a value")
                    value = tokens.pop(0)
                    if token == "--search":
                        search = value
                    elif value in SESSION_SORTS:
                        sort = value
                    else:
                        raise ValueError(f"Unknown session sort: {value}")
                elif token.startswith("--") or reference is not None:
                    raise ValueError("Invalid /resume arguments; see /help resume")
                else:
                    reference = token
            reverse = (sort not in {"mtime", "created"}) if reverse_requested else None
            if reference is not None:
                if listing or all_projects or search or sort != "mtime" or reverse is not None:
                    raise ValueError("Use a path/ID alone, or selector/list options")
                if self._management_idle():
                    self._busy = True
                    self._management_task = asyncio.create_task(self._replace_current("resume", shlex.quote(reference)))
                return
            if self.host is None or self.host.session_root is None:
                raise ValueError("Session discovery needs a storage root; --no-session cannot resume")
            directory = SessionDirectory(self.host.session_root)
            entries = directory.list(cwd=None if all_projects else self._cwd(), search=search, sort=sort, reverse=reverse)
            for diagnostic in directory.diagnostics:
                self._print(f"notice {diagnostic}")
            if listing:
                self._print(f"Sessions ({'all projects' if all_projects else 'current project'}, sort {sort}):")
                for item in entries:
                    self._print(f"{item.conversation_id} {item.display_name or '(unnamed)'} {item.cwd} {item.modified_at.isoformat()} {item.path}")
                if not entries:
                    self._print("No saved sessions match")
                return
            if self._management_task is not None and not self._management_task.done():
                self._add("notice session management is in progress; selector was not opened")
                return
            self._selector_source = self.editor
            self.editor = Editor()
            self.editor.set_text(search)
            self._selector_all, self._selector_sort, self._selector_reverse = all_projects, sort, reverse
            self._selector, self._selector_index = entries, 0
            self._add("Select session: type to search; Up/Down choose; Enter opens; Tab toggles current/all projects; Escape returns to draft")
        except ValueError as error:
            self._add(f"notice {error}; nothing was sent")

    def _refresh_selector(self) -> None:
        assert self.host is not None and self.host.session_root is not None
        self._selector = SessionDirectory(self.host.session_root).list(
            cwd=None if self._selector_all else self._cwd(), search=self.editor.text,
            sort=self._selector_sort, reverse=self._selector_reverse,
        )
        self._selector_index = 0

    def _leave_selector(self) -> None:
        assert self._selector_source is not None
        self.editor = self._selector_source
        self._selector_source = None
        self._selector = None

    def _selector_key(self, key: Key) -> None:
        assert self._selector is not None
        if key.name in {"escape", "ctrl_c"}:
            self._leave_selector()
            if self._cleared_text is not None:
                self.editor.set_text(self._cleared_text)
                self._cleared_text = None
            self._add("notice session selector canceled; source draft kept")
        elif key.name in {"up", "down"}:
            if self._selector:
                self._selector_index = (self._selector_index + (1 if key.name == "down" else -1)) % len(self._selector)
        elif key.name == "tab":
            self._selector_all = not self._selector_all
            self._refresh_selector()
        elif key.name == "enter":
            if self._selector and self._management_idle():
                path = self._selector[self._selector_index].path
                self._leave_selector()
                self._busy = True
                self._management_task = asyncio.create_task(self._replace_current("resume", shlex.quote(str(path))))
        elif key.name == "ctrl_d":
            self._leave_selector()
            self._request_exit(EXIT_OK, confirm=True)
        elif key.name == "backspace":
            self.editor.backspace()
            self._refresh_selector()
        elif key.value:
            self.editor.insert(key.value)
            self._refresh_selector()

    def _management_idle(self) -> bool:
        session = self._current_session()
        if (self._busy or self._external_running or (session is not None and session.agent.state.is_busy)
                or (self._shell_task is not None and not self._shell_task.done())
                or (self._management_task is not None and not self._management_task.done())
                or (self._attachment_task is not None and not self._attachment_task.done())):
            self._add("notice model, shell, editor or session preparation is busy; wait or cancel with Escape")
            return False
        return True

    def _session_info(self) -> None:
        session = self._current_session()
        if session is None:
            self._add("notice no current session")
            return
        self._print(f"current {session.agent.history.conversation_id} name {session.display_name or '(unnamed)'}")
        self._print(self._status())
        if session.source is not None:
            source = session.source
            self._print(f"source {source.kind} {source.conversation_id} leaf {source.leaf_id or 'root'} "
                        f"entry {source.entry_id or '(none)'} path {source.path or 'memory'}")
        if session.save_error is not None:
            self._print(f"save error {session.save_error}")
        self._print("usage unknown")
        for diagnostic in session.resources.diagnostics:
            self._print(f"resource {diagnostic}")
        assert self.runtime is not None
        for number, old in enumerate(self.runtime.retained_sessions, 1):
            waiting = old.queued_messages
            self._print(
                f"retained {number} {old.agent.history.conversation_id} name {old.display_name or '(unnamed)'} "
                f"mode {old.save_mode} save {old.save_state} path {old.path or 'memory'} "
                f"queued {len(waiting.steering) + len(waiting.follow_up)}"
            )
            if old.save_error is not None:
                self._print(f"retained {number} save error {old.save_error}")
        self._print("Recovery: /save --retained <n> [path] or /export --retained <n> jsonl <path>")

    async def _save_command(self, command: str, argument: str) -> None:
        session = self._current_session()
        target = "current"
        try:
            if session is None:
                raise ValueError("No current session")
            if command == "name":
                await session.set_name(None if argument == "--clear" else _strip_quotes(argument))
                self._add(f"name {session.display_name or '(unnamed)'}")
                return
            tokens = shlex.split(argument)
            if tokens[:1] == ["--retained"]:
                assert self.runtime is not None
                if len(tokens) < 2 or not tokens[1].isdigit():
                    raise ValueError("--retained needs a session number from /session")
                number = int(tokens[1])
                if number < 1 or number > len(self.runtime.retained_sessions):
                    raise ValueError("Unknown retained session number; see /session")
                session = self.runtime.retained_sessions[number - 1]
                target = f"retained {number}"
                tokens = tokens[2:]
            if command == "save":
                if len(tokens) > 1 or any(token.startswith("--") for token in tokens):
                    raise ValueError("Usage: /save [--retained n] [path]")
                path = await session.save(tokens[0] if tokens else None)
                self._add(f"saved {target} {path}")
            else:
                format: ExportFormat = "jsonl"
                if tokens[:1] in (["jsonl"], ["html"]):
                    format = cast(ExportFormat, tokens.pop(0))
                if len(tokens) != 1 or tokens[0].startswith("--"):
                    raise ValueError("Usage: /export [--retained n] [jsonl | html] <path>")
                exported_history = session.agent.history
                await session.export(tokens[0], format=format)
                destination = Path(tokens[0]).expanduser()
                destination = (session.cwd / destination).resolve()
                if destination == session.path:
                    self._add(f"saved {target} {destination} (bound export writes JSONL)")
                else:
                    self._add(f"exported {format} {target} {destination}; original save state {session.save_state}")
                    if format == "jsonl":
                        self._exit_backups[session] = exported_history
                        self._add(f"complete backup covers {target} history for exit; new work still requires /save")
        except Exception as error:
            self._add(f"notice {command} {target}: {error}")
            self._remember_save()
        finally:
            self._redraw()

    async def _replace_current(
        self, command: str, argument: str, *, derivation: _DerivationRequest | None = None,
    ) -> None:
        old = self._current_session()
        runtime = self.runtime
        previous_options = runtime.options if runtime is not None else None
        previous_block = self._request_block
        source_editor: Editor | None = None
        notices: list[str] = []
        fork_draft: _ConversationDraft | None = None
        try:
            if self.host is None:
                raise ValueError("The host is not ready")
            tools = () if self.args.no_tools else self.args.tools
            if derivation is not None:
                if old is None or runtime is None:
                    raise ValueError("Create or open a source session first")
                history = old.agent.history
                if derivation.entry_id is not None:
                    selected = next((entry for entry in history_path(history) if entry.id == derivation.entry_id), None)
                    if not isinstance(selected, MessageHistoryEntry) or not isinstance(selected.message, UserMessage):
                        raise ValueError("Fork requires a user entry on the current active path")
                    history = replace(history, leaf_id=selected.parent_id)
                    editor = Editor()
                    editor.set_text(_user_text(selected.message))
                    images = PendingAttachments()
                    for number, block in enumerate(_message_images(selected.message), 1):
                        image = process_image(base64.b64decode(block.data, validate=True), limits=self._image_limits())
                        images.add(name=f"history-image-{number}", origin=HISTORY_ORIGIN,
                                   source=f"{history.conversation_id}:{selected.id}", image=image)
                    fork_draft = _ConversationDraft(editor, images)
                settings = validate_history(history)
                selected_model = (f"{settings.provider}/{settings.model_id}" if settings.provider is not None
                                  else f"{old.agent.state.model.provider}/{old.agent.state.model.id}")
                selection = self.host.select_new(
                    cwd=derivation.cwd or old.cwd, model=selected_model,
                    thinking=settings.thinking_level, tools=runtime.options.tools,
                )
            elif command == "new":
                selection = self.host.select_new(
                    cwd=self._cwd(), provider=self.args.provider, model=self.args.model,
                    thinking=self.args.thinking, tools=tools,
                )
            else:
                tokens = shlex.split(argument)
                if len(tokens) != 1:
                    raise ValueError("Usage: /resume <path or unique id>")
                selection = self.host.select_open(
                    tokens[0], cwd=self.args.cwd, provider=self.args.provider,
                    model=self.args.model, thinking=self.args.thinking, tools=tools,
                )
            if not selection.ready:
                raise ValueError("; ".join(item.message for item in selection.diagnostics))
            if old is not None:
                if self._cleared_text is not None:
                    self.editor.set_text("\n\n".join(filter(None, (self.editor.text, self._cleared_text))))
                    self._cleared_text = None
                if self._has_queued_inputs(old):
                    self._recall_queued()
                self._drafts[old.agent.history.conversation_id] = _ConversationDraft(
                    self.editor, self._draft,
                )
                source_editor = self.editor
                self.editor = Editor()
            await self._apply_selection(selection, derivation=derivation)
        except asyncio.CancelledError:
            notices.append("notice session switch wait canceled; checking actual current")
            self._add(notices[-1])
            self._redraw()
            if self.runtime is not None:
                try:
                    await self.runtime.wait_for_switch()
                except Exception as error:
                    notices.append(f"notice handoff: {error}")
        except Exception as error:
            notices.append(f"notice session switch: {error}")
        finally:
            if source_editor is not None:
                if self.editor.text or self._cleared_text:
                    source_editor.set_text("\n\n".join(filter(None, (
                        source_editor.text, self.editor.text, self._cleared_text,
                    ))))
                self.editor = source_editor
                self._cleared_text = None
            current = self._current_session()
            if current is not old and current is not None:
                state = fork_draft or self._drafts.get(current.agent.history.conversation_id)
                if state is not None:
                    self._drafts[current.agent.history.conversation_id] = state
                self.editor = state.editor if state else Editor()
                self._draft = state.images if state else PendingAttachments()
                self._cleared_text = None
                self._queued_images.clear()
                self._items.clear()
                self._tools.clear()
                self._live = self._pending_you = None
                self._shell = None
                self._load_history()
                label = {"new": "new", "resume": "resumed", "clone": "cloned", "fork": "forked"}[command]
                self._add(f"{label} current {current.agent.history.conversation_id}")
                self._remember_save()
            elif runtime is not None and previous_options is not None:
                runtime.options = previous_options
                self._request_block = previous_block
            for notice in notices:
                self._add(notice)
            self._busy = False
            self._phase = "input"
            self._redraw()

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
        self._forget_cleared_line()

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
        if not self._draft_edit_allowed():
            return
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
            self._forget_cleared_line()
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
        elif isinstance(event, MessageStartEvent) and isinstance(event.message, UserMessage):
            if self._prompt_first_user:
                self._prompt_first_user = False
            else:
                message = event.message
                self._items.append(_You(
                    _user_text(message), image_labels=_queued_image_labels(
                        message, self._queued_images,
                    ),
                ))
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

    def _request_exit(
        self, code: int, *, confirm: bool = False, discard_unsaved: bool = False,
    ) -> None:
        """Start the exit, or first ask when unsubmitted content would be lost.

        Editor-key exits ask for an explicit decision when text, images or
        queued inputs remain; process signals and a closed stdin exit directly.
        """
        if discard_unsaved:
            self._exit_discard_unsaved = True
        if code != EXIT_OK:
            self._pending_exit_code = code
        if confirm and self._exit_task is None:
            parts = self._unsubmitted_content()
            if parts:
                self._begin_exit_confirm(parts)
                return
        if self._exit_task is None:
            self._exit_task = asyncio.create_task(self._shutdown(self._pending_exit_code))

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
        complete = False
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
            if self._management_task is not None and not self._management_task.done():
                if not self._management_task.cancelling():
                    self._management_task.cancel()
                try:
                    await self._management_task
                except asyncio.CancelledError:
                    pass
            if self._selector is not None:
                self._leave_selector()
            if self._prompt_task is not None and not self._prompt_task.done():
                session = self._current_session()
                if session is not None and session.agent.state.is_busy:
                    self._phase = "cancel"
                    self._add("phase cancel")
                    self._redraw()
                    session.agent.abort()
            if self._shell_task is not None and not self._shell_task.done():
                assert self._shell_abort is not None
                self._shell_abort.abort()
            if self._prompt_task is not None:
                await self._prompt_task
            if self._shell_task is not None:
                await self._shell_task
            unresolved: list[str] = []
            for label, session in self._sessions_for_exit():
                if session.save_state != "unsaved":
                    continue
                if self._exit_discard_unsaved:
                    self._add(f"notice explicitly abandoning unsaved {label} {session.agent.history.conversation_id}")
                    continue
                if self._exit_backups.get(session) == session.agent.history:
                    self._add(f"notice {label} full history has an independent JSONL backup; original save error remains")
                    continue
                try:
                    await session.save()
                except Exception as error:
                    unresolved.append(label)
                    self._add(f"notice {label} save error: {error}")
            if unresolved:
                self._add("notice exit blocked: unsaved history in " + ", ".join(unresolved))
                self._add("notice use /session, /save [--retained n] [path], /export [--retained n] jsonl <path>, or /quit --discard-unsaved")
                return
            if self.runtime is not None:
                try:
                    await self.runtime.close()
                except Exception as error:
                    # Runtime close still owns writer release after a failed
                    # terminal notification. History decisions happened above.
                    self._add(f"notice close: {error}")
            complete = True
        except Exception as error:
            self._add(f"notice {error}")
        finally:
            if complete:
                if self._read_future is not None and not self._read_future.done():
                    self._read_future.set_result(b"")
                if self._done is not None:
                    self._done.set()
            else:
                self._closing = False
                self._exit_task = None
                self._exit_discard_unsaved = False
                self._phase = "input"
                self._start_reader()
            self._redraw()

    def _sessions_for_exit(self) -> list[tuple[str, AgentSession]]:
        if self.runtime is None:
            return []
        sessions = [(f"retained {number}", session)
                    for number, session in enumerate(self.runtime.retained_sessions, 1)]
        if self.runtime.current_session is not None:
            sessions.insert(0, ("current", self.runtime.current_session))
        return sessions

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
        if self._model_selector is not None:
            start = max(0, self._selector_index - _COMPLETION_VISIBLE // 2)
            lines = ["Select model (Up/Down, Enter; Escape returns)"]
            for number, model in enumerate(self._model_selector[start:start + _COMPLETION_VISIBLE], start):
                levels = ",".join(get_supported_thinking_levels(model)) or "fixed-on"
                lines.append(_fit(f"{'>' if number == self._selector_index else ' '} {model.provider}/{model.id} thinking {levels}", cols))
            return lines
        if self._fork_selector is not None:
            start = max(0, self._selector_index - _COMPLETION_VISIBLE // 2)
            lines = ["Select fork (active user inputs, newest first)"]
            for number, entry in enumerate(self._fork_selector[start:start + _COMPLETION_VISIBLE], start):
                text = _user_text(cast(UserMessage, entry.message)).replace("\n", " ")
                lines.append(_fit(f"{'>' if number == self._selector_index else ' '} {entry.id} {text}", cols))
            if not self._fork_selector:
                lines.append("No active user inputs; Escape returns")
            return lines
        if self._selector is not None:
            index = self._selector_index
            start = max(0, index - _COMPLETION_VISIBLE // 2)
            lines = [f"Select session ({'all projects' if self._selector_all else 'current project'}, sort {self._selector_sort})"]
            for number, item in enumerate(self._selector[start:start + _COMPLETION_VISIBLE], start):
                lines.append(_wrap(f"{'>' if number == index else ' '} {item.display_name or '(unnamed)'} {item.conversation_id} {item.cwd} {item.modified_at.isoformat()}", cols)[0])
            if not self._selector:
                lines.append("No saved sessions match; Escape returns")
            return lines
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
            thinking = self._thinking_mode(selected, session.agent.state.thinking_level)
            cwd = str(session.cwd)
            path = str(session.path) if session.path is not None else "memory"
        # Pending queued inputs stay visible next to the phase, with their text
        # and image counts; empty queues add nothing so the status keeps its
        # established shape.
        queued = self._queued_status(session)
        shell = self._shell.status if self._shell is not None else "idle"
        retained = sum(old.save_state == "unsaved" for old in self.runtime.retained_sessions) if self.runtime is not None else 0
        return (
            f"phase {self._phase} | {queued}"
            f"session {identity} | mode {mode} | save {save} | model {model} | "
            f"thinking {thinking} | theme {self.theme} | "
            + (f"request {self._request_selection} | next {model} thinking {thinking} tools {self._tool_names()} | " if self._busy and self._request_selection else "")
            +             f"cwd {cwd} | path {path} | pending {len(self._draft)} | shell {shell}"
            + (f" | retained unsaved {retained} (/session)" if retained else "")
        )

    def _queued_status(self, session: AgentSession | None) -> str:
        """Preview waiting steering and follow-up inputs for the status line."""
        if session is None or not self._has_queued_inputs(session):
            return ""
        snapshot = session.queued_messages
        segments: list[str] = []
        for label, messages in (("steering", snapshot.steering), ("follow-up", snapshot.follow_up)):
            if not messages:
                continue
            previews: list[str] = []
            for message in messages:
                if not isinstance(message, UserMessage):
                    continue
                text = _user_text(message).strip().replace("\n", " ")
                images = len(_message_images(message))
                if images:
                    text = f"{text} +{images} image" if text else f"+{images} image"
                if text:
                    previews.append(text)
            detail = "; ".join(previews)
            if len(detail) > 60:
                detail = detail[:57] + "..."
            if detail:
                segments.append(f"{label} {len(messages)} ({detail})")
            else:
                segments.append(f"{label} {len(messages)}")
        return "".join(f"{segment} | " for segment in segments)

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
        elif isinstance(item, UserShell):
            prefix = "!!" if item.exclude_context else "!"
            context = "excluded from model context" if item.exclude_context else "included in model context"
            lines.extend(_flatten([
                f"shell {item.status} {prefix}{item.command}", context,
                *item.output.splitlines(),
            ], width))
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


def _message_images(message: object) -> tuple[ImageContent, ...]:
    """The image blocks of one queued message, in content order."""
    if not isinstance(message, UserMessage) or not isinstance(message.content, list):
        return ()
    return tuple(block for block in message.content if isinstance(block, ImageContent))


def _queued_image_labels(
    message: UserMessage, queued_images: list[tuple[str, PendingAttachment]],
) -> tuple[str, ...]:
    """Live labels for images a queued input carries, with the MIME fallback."""
    by_data = {data: attachment for data, attachment in queued_images}
    return tuple(
        by_data[block.data].label if block.data in by_data else block.mime_type
        for block in _message_images(message)
    )


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
