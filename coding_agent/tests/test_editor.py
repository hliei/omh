"""Editor text state, history browsing and pasted-text normalization."""

from __future__ import annotations

from coding_agent.editor import HISTORY_LIMIT, Editor, normalize_paste


def test_newline_editing_keeps_every_character() -> None:
    editor = Editor()
    editor.insert("first")
    editor.insert("\n")
    editor.insert("中文")
    assert editor.text == "first\n中文"
    assert editor.line == 1
    assert editor.column == 2
    editor.move_up()
    assert editor.line == 0
    assert editor.column == 2
    editor.insert("!")
    assert editor.text == "fi!rst\n中文"


def test_cursor_movement_and_deletion() -> None:
    editor = Editor()
    editor.set_text("abc\ndef", cursor=4)
    editor.insert("X")
    assert editor.text == "abc\nXdef"
    editor.backspace()
    assert editor.text == "abc\ndef"
    editor.move(1)
    editor.delete()
    assert editor.text == "abc\ndf"


def test_history_is_bounded_and_skips_consecutive_repeats() -> None:
    editor = Editor(history_limit=3)
    for text in ("one", "two", "two", "three", "four"):
        editor.remember(text)
    assert editor.history == ("four", "three", "two")
    assert HISTORY_LIMIT == 100


def test_up_browses_history_and_down_restores_the_draft() -> None:
    editor = Editor()
    editor.remember("sent one")
    editor.remember("sent two")
    editor.insert("unfinished draft")
    editor.move_up()  # single line, cursor not at column 0 -> line start
    assert editor.text == "unfinished draft"
    assert editor.cursor == 0
    editor.move_up()
    assert editor.text == "sent two"
    assert editor.browsing_history
    editor.move_up()
    assert editor.text == "sent one"
    editor.move_down()
    assert editor.text == "sent two"
    editor.move_down()
    assert editor.text == "unfinished draft"
    assert not editor.browsing_history


def test_up_uses_history_immediately_when_the_editor_is_empty() -> None:
    editor = Editor()
    editor.remember("earlier")
    editor.move_up()
    assert editor.text == "earlier"
    editor.move_down()
    assert editor.text == ""


def test_multiline_up_and_down_move_between_lines() -> None:
    editor = Editor()
    editor.set_text("one\ntwo", cursor=1)
    editor.move_down()
    assert editor.line == 1
    assert editor.column == 1
    editor.move_up()
    assert editor.line == 0
    assert editor.column == 1


def test_seed_history_uses_reopened_session_order() -> None:
    editor = Editor()
    editor.seed_history(["oldest", "middle", "newest"])
    assert editor.history == ("newest", "middle", "oldest")
    editor.move_up()
    assert editor.text == "newest"


def test_leave_history_restores_the_draft_only_while_browsing() -> None:
    editor = Editor()
    editor.remember("committed")
    editor.insert("draft")
    editor.move_up()
    assert editor.text == "draft"
    editor.move_up()
    assert editor.text == "committed"
    editor.leave_history()
    assert editor.text == "draft"
    editor.leave_history()
    assert editor.text == "draft"


def test_paste_normalizes_terminal_line_endings() -> None:
    assert normalize_paste("a\r\nb\rc") == "a\nb\nc"
    assert normalize_paste("plain") == "plain"
