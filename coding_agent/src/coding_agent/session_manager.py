"""File metadata and persistence for one application conversation."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Literal

from omh.agent import (
    AgentHistory,
    AgentHistoryEntry,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
)

from coding_agent.history import (
    ConversationSource,
    DecodedHistory,
    decode_history,
    encode_entries,
    encode_history,
)
from coding_agent.history_html import encode_history_html
from coding_agent.writer_lock import WriterLock

SaveState = Literal["pending", "saved", "unsaved"]
SaveMode = Literal["auto", "memory"]
ExportFormat = Literal["jsonl", "html"]


def _has_user_activity(entries: tuple[AgentHistoryEntry, ...]) -> bool:
    """Report whether history holds real user activity worth creating a file for.

    A user or assistant message counts, and so does any custom submission (the
    public API the product uses for ``!``/``!!`` shell records) even without an
    assistant reply. Setup-only records and empty conversations stay pending.
    """
    return any(
        isinstance(entry, CustomMessageHistoryEntry)
        or (isinstance(entry, MessageHistoryEntry) and entry.message.role in ("user", "assistant"))
        for entry in entries
    )


def _path(path: str | Path, cwd: Path) -> Path:
    result = Path(path).expanduser()
    return (result if result.is_absolute() else cwd / result).resolve()


class SessionManager:
    """Save supplied SDK snapshots without owning mutable conversation history."""

    def __init__(
        self, *, cwd: Path, path: Path | None = None,
        display_name: str | None = None, source: ConversationSource | None = None,
    ) -> None:
        self.cwd = cwd
        self._path = _path(path, cwd) if path is not None else None
        self._writer = WriterLock(self._path) if self._path is not None else None
        self._closed = False
        self.display_name = display_name
        self.source = source
        self._save_state: SaveState = "pending"
        self._save_error: Exception | None = None
        self._saved_ids: set[str] = set()
        self._file_lock = asyncio.Lock()
        self._needs_separator = False

    @property
    def path(self) -> Path | None:
        return self._path

    @path.setter
    def path(self, value: Path | None) -> None:
        destination = _path(value, self.cwd) if value is not None else None
        if destination == self._path:
            return
        writer = WriterLock(destination) if destination is not None and not self._closed else None
        if self._writer is not None:
            self._writer.close()
        self._path, self._writer = destination, writer
        self._saved_ids.clear()
        if self._save_error is None:
            self._save_state = "pending"

    async def close(self) -> None:
        """Release the writer after its host has finished all history commits."""
        async with self._file_lock:
            self._closed = True
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    @classmethod
    def load(cls, path: Path) -> tuple[SessionManager, DecodedHistory]:
        """Acquire a writer, then read and validate without modifying bytes."""
        return cls._load(path)

    @classmethod
    def _load(
        cls, path: Path, owner: SessionManager | None = None,
    ) -> tuple[SessionManager, DecodedHistory]:
        # Same-path runtime preparation reads under the current writer. The
        # candidate receives ownership only after the old Agent has closed.
        path = _path(path, Path.cwd())
        manager = cls(cwd=path.parent, path=path if owner is None or owner._writer is None else None)
        manager._path = path
        try:
            raw = path.read_bytes()
            decoded = decode_history(raw)
        except BaseException:
            if manager._writer is not None:
                manager._writer.close()
            raise
        manager.cwd = Path(decoded.cwd)
        manager.display_name = decoded.display_name
        manager.source = decoded.source
        manager._save_state = "saved"
        manager._saved_ids = {entry.id for entry in decoded.history.entries}
        manager._needs_separator = bool(raw and not raw.endswith(b"\n"))
        return manager, decoded

    def _take_writer_from(self, owner: SessionManager) -> None:
        assert self.path == owner.path
        if owner._writer is not None:
            assert self._writer is None
            self._writer, owner._writer = owner._writer, None
        owner._closed = True

    async def prepare_append(self) -> None:
        """Separate an unterminated tail after the host has assembled its Agent."""
        async with self._file_lock:
            if self._needs_separator and self.path is not None:
                with self.path.open("ab") as file:
                    file.write(b"\n")
                self._needs_separator = False

    @property
    def save_state(self) -> SaveState:
        return self._save_state

    @property
    def save_mode(self) -> SaveMode:
        """Report ``auto`` for a file-backed session and ``memory`` without a destination."""
        return "memory" if self.path is None else "auto"

    @property
    def save_error(self) -> Exception | None:
        return self._save_error

    async def commit(self, history: AgentHistory, entries: tuple[AgentHistoryEntry, ...]) -> None:
        """Persist newly committed records, using a full snapshot for the first write."""
        if self._closed:
            raise RuntimeError("Session manager is closed")
        if self.path is None:
            return
        async with self._file_lock:
            if self._save_error is not None:
                raise self._save_error
            if self._save_state == "pending" and not _has_user_activity(history.entries):
                return
            try:
                if self._save_state == "pending":
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("x", encoding="utf-8", newline="\n") as file:
                        file.write(encode_history(history, cwd=str(self.cwd), display_name=self.display_name, source=self.source))
                    self._saved_ids = {entry.id for entry in history.entries}
                else:
                    entries = tuple(entry for entry in entries if entry.id not in self._saved_ids)
                    with self.path.open("a", encoding="utf-8", newline="\n") as file:
                        file.write(encode_entries(entries))
                    self._saved_ids.update(entry.id for entry in entries)
            except Exception as error:
                self._save_state, self._save_error = "unsaved", error
                raise
            self._save_state, self._save_error = "saved", None

    async def save(self, history: AgentHistory, path: str | Path | None = None) -> Path:
        """Replace complete history, then bind it; a closed host takes a temporary lock."""
        async with self._file_lock:
            destination = _path(path, self.cwd) if path is not None else self.path
            if destination is None:
                raise ValueError("Saving an in-memory session requires an explicit path")
            writer = None
            try:
                if destination != self.path or self._writer is None:
                    writer = WriterLock(destination)
                text = encode_history(history, cwd=str(self.cwd), display_name=self.display_name, source=self.source)
                _replace_file(destination, text)
            except Exception as error:
                self._save_state, self._save_error = "unsaved", error
                raise
            else:
                if writer is not None and not self._closed:
                    old_writer, self._writer = self._writer, writer
                    writer = None
                    if old_writer is not None:
                        old_writer.close()
                self._path = destination
                self._saved_ids = {entry.id for entry in history.entries}
                self._needs_separator = False
                self._save_state, self._save_error = "saved", None
                return destination
            finally:
                if writer is not None:
                    writer.close()

    async def set_name(self, history: AgentHistory, name: str | None) -> None:
        """Update or clear the display name, saving the complete bound header."""
        self.display_name = name
        if self.path is not None:
            await self.save(history)

    async def export(
        self, history: AgentHistory, path: str | Path | None = None, *, format: ExportFormat = "jsonl",
    ) -> str:
        """Export a full backup; only the bound destination performs repair."""
        if format not in ("jsonl", "html"):
            raise ValueError(f"Unknown export format: {format}")
        destination = _path(path, self.cwd) if path is not None else None
        if destination is not None and destination == self.path:
            await self.save(history)
            # A bound history destination always stays reopenable JSONL.
            format = "jsonl"
        text = (encode_history(history, cwd=str(self.cwd), display_name=self.display_name, source=self.source)
                if format == "jsonl" else
                encode_history_html(history, cwd=str(self.cwd), display_name=self.display_name))
        if destination is not None and destination != self.path:
            writer = WriterLock(destination)
            try:
                _replace_file(destination, text)
            finally:
                writer.close()
        return text


def _replace_file(destination: Path, text: str) -> None:
    """Finish a same-directory temporary snapshot before replacing old bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    temporary = Path(name)
    os.close(descriptor)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            file.write(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
