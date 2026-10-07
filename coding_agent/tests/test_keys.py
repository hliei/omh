"""Terminal key decoding: newline keys, arrows, escape and bracketed paste."""

from __future__ import annotations

import pytest

from coding_agent.keys import Key, KeyDecoder


def names(decoder: KeyDecoder, data: bytes) -> list[str | None]:
    return [key.name for key in decoder.feed(data)]


def values(decoder: KeyDecoder, data: bytes) -> list[str]:
    return [key.value for key in decoder.feed(data) if key.value]


def test_enter_and_ctrl_j_are_different_keys() -> None:
    decoder = KeyDecoder()
    assert names(decoder, b"\r") == ["enter"]
    assert names(decoder, b"\x0a") == ["newline"]


def test_shift_enter_encodings_insert_a_newline() -> None:
    assert names(KeyDecoder(), b"\x1b[13;2u") == ["newline"]
    assert names(KeyDecoder(), b"\x1b[13;2~") == ["newline"]
    assert names(KeyDecoder(), b"\x1b[27;2;13~") == ["newline"]


def test_alt_enter_and_alt_up_encodings_are_bound() -> None:
    # Alt+Enter queues a follow-up; Alt+Up recalls queued inputs.
    assert names(KeyDecoder(), b"\x1b\r") == ["alt_enter"]
    assert names(KeyDecoder(), b"\x1b\n") == ["alt_enter"]
    assert names(KeyDecoder(), b"\x1b[13;3u") == ["alt_enter"]
    assert names(KeyDecoder(), b"\x1b[1;3A") == ["alt_up"]


def test_other_modifier_keys_are_left_unbound() -> None:
    assert names(KeyDecoder(), b"\x1b[13;2A") == []
    assert names(KeyDecoder(), b"\x1b[1;3B") == []
    assert names(KeyDecoder(), b"\x1b[13;5u") == []
    assert KeyDecoder().flush_escape() is None


def test_arrows_and_control_keys() -> None:
    decoder = KeyDecoder()
    assert names(decoder, b"\x1b[A\x1b[B\x1b[C\x1b[D") == ["up", "down", "right", "left"]
    assert names(decoder, b"\x07\x18\t\x7f") == ["ctrl_g", "ctrl_x", "tab", "backspace"]
    assert names(decoder, b"\x16") == ["ctrl_v"]


def test_text_and_multibyte_characters_are_decoded() -> None:
    decoder = KeyDecoder()
    assert values(decoder, "中文".encode()) == ["中", "文"]
    assert values(decoder, b"plain") == list("plain")


def test_split_escape_sequence_waits_for_the_rest() -> None:
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b[") == []
    assert decoder.pending == b"\x1b["
    assert names(decoder, b"A") == ["up"]
    assert decoder.pending == b""


def test_lone_escape_resolves_through_flush() -> None:
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.pending == b"\x1b"
    assert decoder.flush_escape() == Key(name="escape")
    assert decoder.flush_escape() is None


def test_bracketed_paste_keeps_newlines_and_never_submits() -> None:
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b[200~first\n") == []
    keys = decoder.feed("中文\nlast\x1b[201~".encode())
    assert keys == [Key(value="first\n中文\nlast")]
    assert all(key.name != "enter" for key in keys)


@pytest.mark.parametrize("split", range(1, 6))
def test_split_paste_end_preserves_text_and_following_keys(split: int) -> None:
    decoder = KeyDecoder()
    marker = b"\x1b[201~"
    assert decoder.feed("\x1b[200~中文\nlast".encode() + marker[:split]) == []
    assert decoder.feed(marker[split:] + b"\r\x03") == [
        Key(value="中文\nlast"), Key(name="enter"), Key(name="ctrl_c"),
    ]


def test_paste_end_can_arrive_one_byte_at_a_time() -> None:
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b[200~text\x1b[20") == []
    assert decoder.feed(b"x") == []
    for byte in b"\x1b[201":
        assert decoder.feed(bytes([byte])) == []
    assert decoder.feed(b"~") == [Key(value="text\x1b[20x")]
