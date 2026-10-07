"""External editor command resolution and draft round-tripping."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

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


async def test_cancelled_editor_is_stopped_before_returning(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    script = tmp_path / "editor.py"
    script.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    task = asyncio.create_task(run_external_editor(command, "draft"))
    pid: int | None = None
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
        pid = int(ready.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
