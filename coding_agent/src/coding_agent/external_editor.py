"""Edit the current interactive draft in the user's external editor.

Ctrl+G writes the draft to a temporary Markdown file, hands the terminal to
``$VISUAL``, ``$EDITOR`` or ``nano``, and reads the file back. A successful exit
only refills the editor; the caller never submits. Start, exit and read errors
are returned as a diagnostic so the terminal and editor stay usable.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Editor used when neither VISUAL nor EDITOR is set.
DEFAULT_EDITOR = "nano"


@dataclass(frozen=True, slots=True)
class ExternalEditorResult:
    """The outcome of one external-editor run."""

    status: str
    content: str = ""
    message: str = ""


def editor_command(environ: Mapping[str, str] | None = None) -> str:
    """Return VISUAL, then EDITOR, then the product default."""
    environment = os.environ if environ is None else environ
    for name in ("VISUAL", "EDITOR"):
        value = environment.get(name)
        if value and value.strip():
            return value.strip()
    return DEFAULT_EDITOR


def parse_editor_command(command: str) -> list[str]:
    """Split the configured command into an executable and its arguments."""
    try:
        return shlex.split(command)
    except ValueError:
        return []


async def run_external_editor(
    command: str, content: str, *, stdin: Any = None, stdout: Any = None, stderr: Any = None,
) -> ExternalEditorResult:
    """Run ``command`` on ``content`` and return the edited draft or a diagnostic."""
    parts = parse_editor_command(command)
    if not parts:
        return ExternalEditorResult("failed", message=f"External editor command {command!r} is empty")
    with tempfile.TemporaryDirectory(prefix="omh-editor-") as directory:
        path = Path(directory) / "prompt.md"
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as error:
            return ExternalEditorResult("failed", message=f"Cannot write the external editor draft: {error}")
        try:
            process = await asyncio.create_subprocess_exec(
                *parts, str(path), stdin=stdin, stdout=stdout, stderr=stderr,
            )
        except FileNotFoundError:
            return ExternalEditorResult(
                "failed", message=f"Cannot start external editor {parts[0]!r}: command not found",
            )
        except OSError as error:
            return ExternalEditorResult(
                "failed", message=f"Cannot start external editor {parts[0]!r}: {error}",
            )
        status = await process.wait()
        if status != 0:
            return ExternalEditorResult(
                "failed",
                message=f"External editor {parts[0]!r} exited with status {status}; the draft is unchanged",
            )
        try:
            updated = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            return ExternalEditorResult("failed", message=f"Cannot read the external editor result: {error}")
    if updated.endswith("\n"):
        updated = updated[:-1]
    return ExternalEditorResult("complete", content=updated)


__all__ = ["DEFAULT_EDITOR", "ExternalEditorResult", "editor_command", "parse_editor_command", "run_external_editor"]
