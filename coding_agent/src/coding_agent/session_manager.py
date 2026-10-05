"""File metadata and persistence for one application conversation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from omh.agent import (
    AgentHistory,
    AgentHistoryEntry,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
)

from coding_agent.history import (
    DecodedHistory,
    decode_history,
    encode_entries,
    encode_history,
)

SaveState = Literal["pending", "saved", "unsaved"]
SaveMode = Literal["auto", "memory"]


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
        display_name: str | None = None,
    ) -> None:
        self.cwd = cwd
        self.path = path
        self.display_name = display_name
        self._save_state: SaveState = "pending"
        self._save_error: Exception | None = None
        self._saved_ids: set[str] = set()
        self._file_lock = asyncio.Lock()
        self._needs_separator = False

    @classmethod
    def load(cls, path: Path) -> tuple[SessionManager, DecodedHistory]:
        """Read and validate a saved file without modifying its bytes."""
        raw = path.read_bytes()
        decoded = decode_history(raw)
        manager = cls(cwd=Path(decoded.cwd), path=path, display_name=decoded.display_name)
        manager._save_state = "saved"
        manager._saved_ids = {entry.id for entry in decoded.history.entries}
        manager._needs_separator = bool(raw and not raw.endswith(b"\n"))
        return manager, decoded

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
                        file.write(encode_history(history, cwd=str(self.cwd), display_name=self.display_name))
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
        """Explicitly write the complete history, even before a first prompt."""
        async with self._file_lock:
            destination = _path(path, self.cwd) if path is not None else self.path
            if destination is None:
                raise ValueError("Saving an in-memory session requires an explicit path")
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("w", encoding="utf-8", newline="\n") as file:
                    file.write(encode_history(history, cwd=str(self.cwd), display_name=self.display_name))
            except Exception as error:
                self._save_state, self._save_error = "unsaved", error
                raise
            self.path = destination
            self._saved_ids = {entry.id for entry in history.entries}
            self._save_state, self._save_error = "saved", None
            return destination

    async def export(self, history: AgentHistory, path: str | Path | None = None) -> str:
        """Return a full JSONL snapshot; an optional separate file receives it."""
        if path is not None and _path(path, self.cwd) == self.path:
            await self.save(history)
        text = encode_history(history, cwd=str(self.cwd), display_name=self.display_name)
        if path is not None and _path(path, self.cwd) != self.path:
            destination = _path(path, self.cwd)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8", newline="\n")
        return text
