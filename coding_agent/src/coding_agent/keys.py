"""Terminal key decoding for the interactive editor.

The decoder turns the raw bytes read from a terminal into named keys and
literal text. It owns the byte-level state a key sequence needs: an incomplete
escape sequence, a bracketed paste that spans reads, and an incremental UTF-8
decoder. Interactive mode owns every action a decoded key performs.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass

from coding_agent.editor import normalize_paste

_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"
_CONTROL_KEYS = {
    0x0D: "enter",
    0x0A: "newline",
    0x09: "tab",
    0x7F: "backspace",
    0x08: "backspace",
    0x03: "ctrl_c",
    0x04: "ctrl_d",
    0x07: "ctrl_g",
    0x0F: "ctrl_o",
    0x14: "ctrl_t",
    0x18: "ctrl_x",
}
_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


@dataclass(frozen=True, slots=True)
class Key:
    """One decoded terminal key: a named action or literal inserted text."""

    name: str | None = None
    value: str = ""


class KeyDecoder:
    """Incremental decoder for one terminal's raw input bytes."""

    def __init__(self) -> None:
        self._text = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = b""
        self._paste = b""
        self._in_paste = False

    @property
    def pending(self) -> bytes:
        """Bytes held back because they may still start a longer sequence."""
        return self._pending

    def flush_escape(self) -> Key | None:
        """Resolve a lone pending Escape once no further byte followed it."""
        if self._pending != b"\x1b":
            return None
        self._pending = b""
        return Key(name="escape")

    def feed(self, data: bytes) -> list[Key]:
        blob = self._pending + data
        self._pending = b""
        keys: list[Key] = []
        index = 0
        while index < len(blob):
            if self._in_paste:
                self._paste += blob[index:]
                end = self._paste.find(_PASTE_END)
                if end == -1:
                    break
                keys.append(Key(value=normalize_paste(self._paste[:end].decode("utf-8", "replace"))))
                blob = self._paste[end + len(_PASTE_END):]
                self._in_paste = False
                self._paste = b""
                index = 0
                continue
            byte = blob[index]
            if byte == 0x1B:
                if blob.startswith(_PASTE_START, index):
                    self._in_paste = True
                    index += len(_PASTE_START)
                    continue
                matched = _escape_sequence(blob, index)
                if matched is None:
                    self._pending = blob[index:]
                    break
                name, consumed = matched
                if name is not None:
                    keys.append(Key(name=name))
                index += consumed
                continue
            if byte < 0x20 or byte == 0x7F:
                name = _CONTROL_KEYS.get(byte)
                if name is not None:
                    keys.append(Key(name=name))
                index += 1
                continue
            character = self._text.decode(blob[index:index + 1])
            index += 1
            if character:
                keys.append(Key(value=character))
        return keys


def _escape_sequence(blob: bytes, index: int) -> tuple[str | None, int] | None:
    """Decode one escape sequence, or ``None`` while it may be incomplete."""
    if index + 1 >= len(blob):
        return None
    second = blob[index + 1]
    if second != 0x5B:
        # A bare Escape or an unbound modifier sequence; a later delivery owns
        # Alt+Enter (follow-up), so ESC CR is not treated as a newline here.
        return (None, 2)
    final = index + 2
    while final < len(blob) and not 0x40 <= blob[final] <= 0x7E:
        final += 1
    if final >= len(blob):
        return None
    consumed = final + 1 - index
    body = blob[index + 2:final].decode("ascii", "replace")
    final_byte = chr(blob[final])
    if not body and final_byte in _ARROWS:
        return (_ARROWS[final_byte], consumed)
    if final_byte == "Z":
        return (None, consumed)
    parts = body.split(";")
    if final_byte in {"u", "~"}:
        if parts[0] == "13" and "2" in parts[1:]:
            return ("newline", consumed)
        if final_byte == "~" and len(parts) >= 3 and parts[2] == "13" and "2" in parts[1:]:
            return ("newline", consumed)
    return (None, consumed)


__all__ = ["Key", "KeyDecoder"]
