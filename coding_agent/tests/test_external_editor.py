"""External editor command resolution and draft round-tripping."""

from __future__ import annotations

from pathlib import Path

from coding_agent.external_editor import (
    DEFAULT_EDITOR,
    editor_command,
    parse_editor_command,
    run_external_editor,
)


def write_editor(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-editor"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return script


def test_editor_command_prefers_visual_then_editor_then_default() -> None:
    assert editor_command({"VISUAL": "vim", "EDITOR": "nano"}) == "vim"
    assert editor_command({"EDITOR": "nano"}) == "nano"
    assert editor_command({}) == DEFAULT_EDITOR
    assert editor_command({"VISUAL": "   ", "EDITOR": "  "}) == DEFAULT_EDITOR


def test_editor_command_splits_arguments() -> None:
    assert parse_editor_command("code --wait") == ["code", "--wait"]
    assert parse_editor_command("'unterminated") == []


async def test_success_refills_the_draft_without_a_trailing_newline(tmp_path: Path) -> None:
    script = write_editor(tmp_path, "printf 'edited draft\\n' > \"$1\"")
    result = await run_external_editor(str(script), "original draft")
    assert result.status == "complete"
    assert result.content == "edited draft"


async def test_success_strips_a_byte_order_mark(tmp_path: Path) -> None:
    script = write_editor(tmp_path, "printf '\\357\\273\\277edited' > \"$1\"")
    result = await run_external_editor(str(script), "original")
    assert result.status == "complete"
    assert result.content == "edited"


async def test_missing_editor_reports_a_start_diagnostic() -> None:
    result = await run_external_editor("/definitely/missing/editor", "draft")
    assert result.status == "failed"
    assert "command not found" in result.message


async def test_nonzero_exit_keeps_the_draft() -> None:
    result = await run_external_editor("sh -c 'exit 3'", "draft")
    assert result.status == "failed"
    assert "status 3" in result.message
    assert result.content == ""
