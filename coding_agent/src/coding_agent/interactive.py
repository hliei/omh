"""Regular interactive session: scrolling conversation, editor and status.

The screen stays on the primary terminal. A capable terminal pins the status
and editor beneath a scrolling conversation; a plain or dumb terminal prints
the same words without color or cursor addressing. Prompts go through
``AgentSessionRuntime`` and the common host. This module does not start a
second Agent loop, and print execution does not import it.
"""

from __future__ import annotations

import asyncio
import codecs
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

from coding_agent.agent_session_runtime import AgentSessionRuntime
from coding_agent.cli import CliArgs
from coding_agent.config import settings_theme
from coding_agent.host import CodingAgentHost, SessionSelection
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


@dataclass(slots=True)
class _Note:
    text: str


@dataclass(slots=True)
class _You:
    text: str


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
        self._buffer = ""
        self._cursor = 0
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
        self._ctrl_c_at = 0.0
        self._termios: list[object] | None = None
        self._flags: int | None = None
        self._previous: tuple[str, ...] = ()
        self._scrollback: tuple[str, ...] = ()
        self._previous_status = ""
        self._previous_entry = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = b""
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
        reader = asyncio.create_task(self._read_loop())
        await self._done.wait()
        reader.cancel()
        try:
            await reader
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
        if self.theme != "plain":
            try:
                os.write(self._out, b"\x1b[0m\x1b[r\x1b[?25h\r\n")
            except OSError:
                pass
        if self._flags is not None:
            fcntl.fcntl(self._fd, fcntl.F_SETFL, self._flags)
        if self._termios is not None:
            termios.tcsetattr(
                self._fd, termios.TCSADRAIN, cast("list[int | list[bytes | int]]", self._termios),
            )

    def _enter_terminal(self) -> None:
        self._termios = termios.tcgetattr(self._fd)
        attrs = termios.tcgetattr(self._fd)
        attrs[0] = cast(int, attrs[0]) & ~(termios.IXON | termios.IXOFF)
        attrs[3] = cast(int, attrs[3]) & ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
        cc = cast(list[int | bytes], attrs[6])
        cc[termios.VMIN] = 1
        cc[termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSADRAIN, attrs)
        self._flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, self._flags | os.O_NONBLOCK)

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

    def _load_history(self) -> None:
        session = self.runtime.current_session if self.runtime is not None else None
        if session is None:
            return
        calls: dict[str, tuple[str, dict[str, object]]] = {}
        for message in session.agent.state.messages:
            if isinstance(message, UserMessage):
                text = _user_text(message)
                if text:
                    self._items.append(_You(text))
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
        for key in self._keys(data):
            if key == "\r" or key == "\n":
                self._submit()
            elif key in {"\x7f", "\x08"}:
                self._backspace()
            elif key == "\x03":
                self._ctrl_c()
            elif key == "\x04":
                if self._buffer == "":
                    self._request_exit(EXIT_OK)
            elif key == "\x0f":
                self._tools_open = not self._tools_open
            elif key == "\x14":
                self._thinking_open = not self._thinking_open
            elif key == "left":
                self._cursor = max(0, self._cursor - 1)
            elif key == "right":
                self._cursor = min(len(self._buffer), self._cursor + 1)
            elif key.isprintable():
                self._buffer = self._buffer[:self._cursor] + key + self._buffer[self._cursor:]
                self._cursor += len(key)
                self._ctrl_c_at = 0.0
        self._redraw()

    def _keys(self, data: bytes) -> list[str]:
        blob = self._pending + data
        self._pending = b""
        keys: list[str] = []
        index = 0
        while index < len(blob):
            byte = blob[index]
            if byte == 0x1B:
                if index + 1 >= len(blob):
                    self._pending = blob[index:]
                    break
                if blob[index + 1] != 0x5B:
                    index += 1
                    continue
                final = index + 2
                while final < len(blob) and not 0x40 <= blob[final] <= 0x7E:
                    final += 1
                if final >= len(blob):
                    self._pending = blob[index:]
                    break
                sequence = blob[index:final + 1]
                if sequence == b"\x1b[D":
                    keys.append("left")
                elif sequence == b"\x1b[C":
                    keys.append("right")
                index = final + 1
                continue
            if byte < 0x20 or byte == 0x7F:
                keys.append(chr(byte))
                index += 1
                continue
            character = self._decoder.decode(blob[index:index + 1])
            index += 1
            if character:
                keys.append(character)
        return keys

    def _backspace(self) -> None:
        if self._cursor == 0:
            return
        self._buffer = self._buffer[:self._cursor - 1] + self._buffer[self._cursor:]
        self._cursor -= 1
        self._ctrl_c_at = 0.0

    def _ctrl_c(self) -> None:
        now = asyncio.get_running_loop().time()
        if self._ctrl_c_at and now - self._ctrl_c_at <= 0.5:
            self._request_exit(EXIT_OK)
            return
        self._buffer = ""
        self._cursor = 0
        self._ctrl_c_at = now

    def _submit(self) -> None:
        text = self._buffer
        self._buffer = ""
        self._cursor = 0
        self._ctrl_c_at = 0.0
        if self._busy:
            self._buffer = text
            self._cursor = len(text)
            return
        if not text.strip():
            self._redraw()
            return
        if self._chooser == "cwd":
            self._choose_cwd(text.strip())
            return
        self._items.append(_You(text))
        if self._request_block is not None or self.runtime is None:
            self._add(f"notice {self._request_block or 'The session is not ready.'}")
            self._add(f"notice {_NO_REQUEST}")
            return
        session = self.runtime.current_session
        if session is not None and session.save_state == "unsaved":
            self._show_unsaved()
            return
        self._busy = True
        self._set_phase("model")
        self._prompt_task = asyncio.create_task(self._prompt(text))

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

    async def _prompt(self, text: str) -> None:
        assert self.runtime is not None
        try:
            await self.runtime.prompt(text)
        except Exception as error:
            self._add(f"notice {error}")
        finally:
            self._busy = False
            self._remember_save()
            if not self._closing:
                self._set_phase("input")
            self._redraw()

    def _remember_save(self) -> None:
        session = self.runtime.current_session if self.runtime is not None else None
        if session is not None and session.save_state == "unsaved":
            self._show_unsaved()

    def _show_unsaved(self) -> None:
        self._add(f"notice {_UNSAVED_GUIDANCE}")
        session = self.runtime.current_session if self.runtime is not None else None
        if session is not None and session.save_error is not None:
            self._add(f"notice {session.save_error}")

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
        texts: list[str] = []
        thoughts: list[str] = []
        for block in message.content:
            if isinstance(block, TextContent):
                texts.append(block.text)
            elif isinstance(block, ThinkingContent):
                thoughts.append(block.thinking)
        item.text = "".join(texts)
        item.thinking = "".join(thoughts)
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
            if self._prompt_task is not None and not self._prompt_task.done():
                session = self.runtime.current_session if self.runtime is not None else None
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

    def _redraw(self) -> None:
        if self._termios is None or self._restored:
            return
        rows, cols = _window_size(self._out)
        lines = _render(self._items, cols, thinking_open=self._thinking_open, tools_open=self._tools_open)
        status = _wrap(self._status(), cols)
        entry = _fit(self._entry(), cols)
        if self.theme == "plain" or rows < 3:
            self._paint_linear(lines, status, entry)
            return
        self._paint_fancy(lines, status[: rows - 2], entry, rows, cols)

    def _paint_fancy(
        self, lines: Sequence[str], status: Sequence[str], entry: str, rows: int, cols: int,
    ) -> None:
        chrome = len(status) + 1
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
        for offset, text in enumerate(status):
            parts.append(f"\x1b[{body + 1 + offset};1H\x1b[K{self._paint(text)}")
        parts.append(f"\x1b[{rows};1H\x1b[K{self._paint(entry)}")
        parts.append(f"\x1b[{rows};{self._entry_column(cols)}H\x1b[?25h")
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

    def _paint_linear(self, lines: Sequence[str], status: Sequence[str], entry: str) -> None:
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
        status_text = "\n".join(status)
        if status_text != self._previous_status:
            self._write(self._paint(status_text) + "\n")
            self._previous_status = status_text
        if entry != self._previous_entry:
            self._write(self._paint(entry) + "\n")
            self._previous_entry = entry

    def _paint(self, text: str) -> str:
        color = _COLORS[self.theme]
        if not color:
            return text
        return f"{color}{text}\x1b[0m"

    def _status(self) -> str:
        session = self.runtime.current_session if self.runtime is not None else None
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
            f"cwd {cwd} | path {path}"
        )

    def _entry_column(self, cols: int) -> int:
        prefix = "directory> " if self._chooser == "cwd" else "> "
        raw = prefix + self._buffer
        if _display_width(raw) <= cols:
            column = 1 + _display_width(prefix + self._buffer[:self._cursor])
        else:
            column = _display_width(_fit(raw, cols))
        return max(1, min(column, cols))

    def _entry(self) -> str:
        prefix = "directory> " if self._chooser == "cwd" else "> "
        return prefix + self._buffer

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


def _render(
    items: Sequence[_Item], width: int, *, thinking_open: bool, tools_open: bool,
) -> list[str]:
    lines: list[str] = []
    for item in items:
        if isinstance(item, _Note):
            lines.extend(_wrap(item.text, width))
        elif isinstance(item, _You):
            lines.extend(_wrap(f"you {item.text}", width))
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


def _assistant_from(message: AssistantMessage) -> _Assistant:
    item = _Assistant(error=message.error_message, live=False)
    texts: list[str] = []
    thoughts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            texts.append(block.text)
        elif isinstance(block, ThinkingContent):
            thoughts.append(block.thinking)
    item.text = "".join(texts)
    item.thinking = "".join(thoughts)
    return item


def _user_text(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(block.text for block in message.content if isinstance(block, TextContent))


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
    lines: list[str] = []
    current = ""
    used = 0
    for char in text:
        needed = _char_width(char)
        if current and used + needed > width:
            lines.append(current)
            current = char
            used = needed
        else:
            current += char
            used += needed
    if current or not lines:
        lines.append(current)
    return lines


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
