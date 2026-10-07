"""Editor text state, editor history and pasted text normalization.

The editor is plain data: one string, a cursor offset and a bounded list of
previously submitted entries. It receives already decoded keys and never reads
or writes a terminal, so interactive mode owns rendering while this module owns
the editing rules. Editor history belongs to the running or reopened session;
it is never written to a cross-session input file.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Submitted entries the editor keeps for Up/Down navigation, newest first.
HISTORY_LIMIT = 100


class Editor:
    """A multiline text buffer with cursor movement and session history."""

    def __init__(self, *, history_limit: int = HISTORY_LIMIT) -> None:
        self._text = ""
        self._cursor = 0
        self._history: list[str] = []
        self._history_limit = max(history_limit, 1)
        self._history_index = -1
        self._draft: tuple[str, int] | None = None

    # ------------------------------------------------------------------ #
    # Observable state
    # ------------------------------------------------------------------ #

    @property
    def text(self) -> str:
        return self._text

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def empty(self) -> bool:
        return self._text == ""

    @property
    def history(self) -> tuple[str, ...]:
        """Submitted entries, newest first."""
        return tuple(self._history)

    @property
    def browsing_history(self) -> bool:
        """Whether Up/Down is currently showing an entry instead of the draft."""
        return self._history_index != -1

    @property
    def line(self) -> int:
        """Zero-based logical line of the cursor."""
        return self._text.count("\n", 0, self._cursor)

    @property
    def line_count(self) -> int:
        return self._text.count("\n") + 1

    @property
    def column(self) -> int:
        """Cursor offset within its logical line."""
        return self._cursor - self._line_start()

    # ------------------------------------------------------------------ #
    # Editing
    # ------------------------------------------------------------------ #

    def set_text(self, text: str, cursor: int | None = None) -> None:
        """Replace the whole buffer and leave history browsing."""
        self._history_index = -1
        self._draft = None
        self._text = text
        self._cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))

    def clear(self) -> None:
        self.set_text("")

    def insert(self, text: str) -> None:
        """Insert literal text at the cursor, newlines included."""
        if not text:
            return
        self._history_index = -1
        self._draft = None
        self._text = self._text[: self._cursor] + text + self._text[self._cursor :]
        self._cursor += len(text)

    def backspace(self) -> None:
        if self._cursor == 0:
            return
        self._history_index = -1
        self._draft = None
        self._text = self._text[: self._cursor - 1] + self._text[self._cursor :]
        self._cursor -= 1

    def delete(self) -> None:
        if self._cursor >= len(self._text):
            return
        self._history_index = -1
        self._draft = None
        self._text = self._text[: self._cursor] + self._text[self._cursor + 1 :]

    def replace_token(self, start: int, end: int, value: str) -> None:
        """Replace one completion range and place the cursor after the value."""
        start = max(0, min(start, len(self._text)))
        end = max(start, min(end, len(self._text)))
        self._text = self._text[:start] + value + self._text[end:]
        self._cursor = start + len(value)
        self._history_index = -1
        self._draft = None

    # ------------------------------------------------------------------ #
    # Cursor movement
    # ------------------------------------------------------------------ #

    def move(self, delta: int) -> None:
        self._cursor = max(0, min(len(self._text), self._cursor + delta))

    def move_home(self) -> None:
        self._cursor = self._line_start()

    def move_end(self) -> None:
        self._cursor = self._line_end()

    def move_up(self) -> None:
        """Move between lines, then browse history at the top boundary."""
        if self.line == 0:
            if self._text == "" or self.browsing_history or self.column == 0:
                self._history_previous()
            else:
                self.move_home()
            return
        self._move_line(-1)

    def move_down(self) -> None:
        """Move between lines, then browse history at the bottom boundary."""
        if self.browsing_history and self.line == self.line_count - 1:
            self._history_next()
        elif self.line == self.line_count - 1:
            self.move_end()
        else:
            self._move_line(1)

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    def remember(self, text: str) -> None:
        """Record one submitted entry, newest first, without consecutive repeats."""
        trimmed = text.strip()
        if not trimmed:
            return
        if self._history and self._history[0] == trimmed:
            return
        self._history.insert(0, trimmed)
        if len(self._history) > self._history_limit:
            del self._history[self._history_limit :]
        self._history_index = -1
        self._draft = None

    def seed_history(self, texts: Iterable[str]) -> None:
        """Fill history from a reopened conversation's user texts, oldest first."""
        for text in texts:
            self.remember(text)
        self._history_index = -1
        self._draft = None

    def leave_history(self) -> None:
        """Stop browsing and restore the unsubmitted draft, if one was saved."""
        if self._history_index == -1:
            return
        self._history_index = -1
        draft = self._draft
        self._draft = None
        if draft is not None:
            self._text, self._cursor = draft

    def _history_previous(self) -> None:
        if not self._history:
            return
        index = self._history_index + 1
        if index >= len(self._history):
            return
        if self._history_index == -1:
            self._draft = (self._text, self._cursor)
        self._history_index = index
        self._set_browse_text(self._history[index])

    def _history_next(self) -> None:
        if self._history_index == -1:
            return
        index = self._history_index - 1
        self._history_index = index
        if index == -1:
            draft = self._draft
            self._draft = None
            if draft is not None:
                self._text, self._cursor = draft
            else:
                self._text = ""
                self._cursor = 0
            return
        self._set_browse_text(self._history[index])

    def _set_browse_text(self, text: str) -> None:
        self._text = text
        self._cursor = len(text)

    def _move_line(self, direction: int) -> None:
        lines = self._text.split("\n")
        target = self.line + direction
        if target < 0 or target >= len(lines):
            return
        column = min(self.column, len(lines[target]))
        self._cursor = sum(len(line) + 1 for line in lines[:target]) + column

    def _line_start(self) -> int:
        return self._text.rfind("\n", 0, self._cursor) + 1

    def _line_end(self) -> int:
        end = self._text.find("\n", self._cursor)
        return len(self._text) if end == -1 else end


def normalize_paste(text: str) -> str:
    """Return pasted text with terminal line endings as plain newlines."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


__all__ = ["HISTORY_LIMIT", "Editor", "normalize_paste"]
