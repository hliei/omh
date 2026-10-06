"""Slash-command classification, help metadata and hotkey listing."""

from __future__ import annotations

from pathlib import Path

from coding_agent.commands import (
    AVAILABLE_COMMANDS,
    COMMANDS_BY_NAME,
    RESERVED_COMMANDS,
    classify,
    command_detail,
    help_lines,
    hotkey_lines,
)
from coding_agent.completion import CompletionSources


def test_classify_distinguishes_every_dispatch_kind() -> None:
    assert classify("/help").kind == "builtin"
    assert classify("/help extra words").argument == "extra words"
    assert classify("/new").kind == "reserved"
    assert classify("/skill:tdd do it").kind == "resource"
    assert classify("/notes topic", templates=["notes"]).kind == "resource"
    assert classify("/mystery").kind == "unknown"
    assert classify("ordinary text").kind == "plain"
    assert classify("!shell").kind == "plain"


def test_reserved_names_win_over_same_name_templates() -> None:
    intent = classify("/new hello", templates=["new"])
    assert intent.kind == "reserved"


def test_help_lists_only_runnable_commands() -> None:
    text = "\n".join(help_lines())
    assert "/help" in text
    assert "/hotkeys" in text
    assert "/copy" in text
    for spec in AVAILABLE_COMMANDS:
        assert spec.usage in text
    for reserved in ("/new", "/model", "/login", "/quit", "/resume"):
        assert reserved not in text


def test_command_detail_only_resolves_runnable_commands() -> None:
    detail = command_detail("/help")
    assert detail is not None
    assert "Usage: /help [command]" in detail[1]
    assert command_detail("new") is None
    assert command_detail("mystery") is None


def test_hotkeys_describe_the_delivered_defaults() -> None:
    text = "\n".join(hotkey_lines())
    assert "Shift+Enter, Ctrl+J" in text
    assert "Ctrl+G" in text
    assert "Ctrl+X, /copy" in text
    assert "Ctrl+L" not in text
    assert "Shift+Tab" not in text


def test_help_argument_completion_uses_the_runnable_surface() -> None:
    spec = COMMANDS_BY_NAME["help"]
    sources = CompletionSources(cwd=Path("."), commands=AVAILABLE_COMMANDS)
    assert [item.value for item in spec.argument_items("he", sources)] == ["help"]
    assert spec.argument_items("model", sources) == ()
    assert "help" in RESERVED_COMMANDS
