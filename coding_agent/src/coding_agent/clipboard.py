"""Clipboard backend detection for ``/copy`` and Ctrl+X.

The product copies the last assistant answer through a desktop clipboard
command that already exists on the machine. It never installs one: when no
backend is present the caller reports the diagnostic and keeps the screen
usable, and the actual terminal or desktop behaviour is verified separately.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping


class ClipboardError(RuntimeError):
    """No usable clipboard backend, or the chosen backend failed."""


def clipboard_command(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> list[str] | None:
    """Return the first available platform clipboard command, or ``None``."""
    selected = sys.platform if platform is None else platform
    environment = os.environ if environ is None else environ
    finder = shutil.which if which is None else which
    if selected == "darwin":
        candidates: list[list[str]] = [["pbcopy"]]
    elif selected.startswith("win"):
        candidates = [["clip"]]
    else:
        candidates = []
        if environment.get("TERMUX_VERSION"):
            candidates.append(["termux-clipboard-set"])
        if environment.get("WAYLAND_DISPLAY"):
            candidates.append(["wl-copy"])
        if environment.get("DISPLAY"):
            candidates.append(["xclip", "-selection", "clipboard"])
            candidates.append(["xsel", "--clipboard", "--input"])
    for candidate in candidates:
        if finder(candidate[0]):
            return candidate
    return None


def copy_text(
    text: str, *, platform: str | None = None, environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> None:
    """Copy ``text`` to the desktop clipboard or raise :class:`ClipboardError`."""
    command = clipboard_command(platform=platform, environ=environ, which=which)
    if command is None:
        raise ClipboardError(
            "No clipboard backend is available; install pbcopy, wl-copy, xclip or xsel "
            "to enable /copy. omh does not install one automatically."
        )
    run = subprocess.run if runner is None else runner
    try:
        result = run(
            command, input=text, text=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ClipboardError(f"Cannot run clipboard backend {command[0]}: {error}") from error
    if result.returncode != 0:
        raise ClipboardError(
            f"Clipboard backend {command[0]} failed with status {result.returncode}"
        )


__all__ = ["ClipboardError", "clipboard_command", "copy_text"]
