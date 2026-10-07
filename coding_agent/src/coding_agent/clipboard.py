"""Desktop clipboard image backends for pasted screenshots.

The product reads an image from the desktop clipboard without installing system
tools. Detection is explicit per platform: macOS uses the system ``osascript``
against the AppKit clipboard, Linux prefers ``wl-paste`` from ``wl-clipboard``
on Wayland and ``xclip`` on X11. A missing backend never breaks the screen: the
caller explains the dependency and keeps explicit file input available. The
runner boundary is injectable so a controlled fixture can prove acceptance and
fallback without a real desktop clipboard.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: Default seconds allowed for one clipboard command before it is stopped.
CLIPBOARD_TIMEOUT_SECONDS = 5.0

#: Image types accepted from a clipboard, most preferred first.
_MIME_PREFERENCE = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp")

#: Runner contract: command arguments in, ``(returncode, stdout, stderr)`` out.
Runner = Callable[[Sequence[str]], Awaitable[tuple[int, bytes, bytes]]]

_MACOS_SCRIPT = """\
set png to the clipboard as \u00abclass PNGf\u00bb
set target to open for access POSIX file "{output}" with write permission
set eof target to 0
write png to target
close access target
"""


class ClipboardUnavailable(Exception):
    """No clipboard image backend is available; explicit file input still works."""


@dataclass(frozen=True, slots=True)
class ClipboardBackend:
    """One detected platform backend and the dependency it comes from."""

    name: str
    dependency: str
    kind: Literal["macos", "wayland", "x11"]


@dataclass(frozen=True, slots=True)
class ClipboardSupport:
    """The detection result for the current platform and environment."""

    platform: str
    backend: ClipboardBackend | None
    detail: str
    #: A second backend tried only when the Wayland backend itself fails.
    fallback: ClipboardBackend | None = None

    @property
    def available(self) -> bool:
        return self.backend is not None


class _BackendFailed(Exception):
    """A backend command failed, which may allow the X11 fallback to run."""


_MACOS = ClipboardBackend("macOS AppKit (osascript)", "osascript", "macos")
_WAYLAND = ClipboardBackend("wl-paste (wl-clipboard)", "wl-clipboard", "wayland")
_X11 = ClipboardBackend("xclip", "xclip", "x11")


def detect_clipboard(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> ClipboardSupport:
    """Choose the clipboard image backend for this platform without installing anything.

    A backend is available only when its executable is on ``PATH`` and the
    desktop session variable it needs is present. A headless environment can
    never claim clipboard support from a TTY alone.
    """
    selected_platform = platform if platform is not None else sys.platform
    variables = os.environ if environ is None else environ
    find = shutil.which if which is None else which

    if selected_platform == "darwin":
        if find("osascript"):
            return ClipboardSupport(
                selected_platform, _MACOS, "osascript reads a macOS clipboard image",
            )
        return ClipboardSupport(
            selected_platform, None,
            "no clipboard image backend: the macOS osascript command was not found",
        )
    if selected_platform.startswith("linux"):
        wayland = bool(variables.get("WAYLAND_DISPLAY"))
        display = bool(variables.get("DISPLAY"))
        wl_paste = find("wl-paste")
        xclip = find("xclip")
        if wayland and wl_paste:
            fallback = _X11 if (display and xclip) else None
            detail = "wl-paste (wl-clipboard) reads a Wayland clipboard image"
            if fallback is not None:
                detail += "; xclip is the fallback"
            return ClipboardSupport(selected_platform, _WAYLAND, detail, fallback)
        if display and xclip:
            return ClipboardSupport(
                selected_platform, _X11, "xclip reads an X11 clipboard image",
            )
        missing = [name for name, found in (("wl-clipboard (wl-paste)", wl_paste), ("xclip", xclip)) if not found]
        if wayland or display:
            detail = f"no clipboard image backend: install {' or '.join(missing)}"
        else:
            detail = "no clipboard image backend: no Wayland or X11 display and no wl-clipboard or xclip"
        return ClipboardSupport(selected_platform, None, detail)
    return ClipboardSupport(
        selected_platform, None, f"clipboard image paste is not supported on {selected_platform}",
    )


async def read_clipboard_image(
    *, support: ClipboardSupport | None = None, runner: Runner | None = None,
    environ: Mapping[str, str] | None = None, which: Callable[[str], str | None] | None = None,
) -> bytes | None:
    """Return the raw clipboard image bytes, or ``None`` when the clipboard has none.

    Raises :class:`ClipboardUnavailable` when no backend exists so the caller can
    explain the dependency and the file input path instead of failing the screen.
    """
    resolved = support if support is not None else detect_clipboard(environ=environ, which=which)
    backend = resolved.backend
    if backend is None:
        raise ClipboardUnavailable(
            f"{resolved.detail}. Add an image file with /attach <image-path> instead."
        )
    run = runner if runner is not None else _default_runner
    if backend.kind == "macos":
        return await _read_macos(run)
    if backend.kind == "wayland":
        try:
            return await _read_wayland(run)
        except _BackendFailed:
            if resolved.fallback is None:
                return None
            return await _read_x11(run)
    return await _read_x11(run)


async def _read_macos(run: Runner) -> bytes | None:
    directory = tempfile.mkdtemp(prefix="omh-clipboard-")
    try:
        target = os.path.join(directory, "clipboard.png")
        code, _stdout, _stderr = await run(("osascript", "-e", _MACOS_SCRIPT.format(output=target)))
        if code != 0:
            return None
        try:
            data = Path(target).read_bytes()
        except OSError:
            return None
        return data or None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


async def _read_wayland(run: Runner) -> bytes | None:
    code, stdout, _stderr = await run(("wl-paste", "--list-types"))
    if code != 0:
        raise _BackendFailed("wl-paste --list-types failed")
    mime_type = _preferred_mime(stdout.decode("utf-8", "replace").split())
    if mime_type is None:
        return None
    code, stdout, _stderr = await run(("wl-paste", "--no-newline", "--type", mime_type))
    if code != 0:
        raise _BackendFailed("wl-paste image read failed")
    return stdout or None


async def _read_x11(run: Runner) -> bytes | None:
    for mime_type in _MIME_PREFERENCE:
        code, stdout, _stderr = await run(
            ("xclip", "-selection", "clipboard", "-t", mime_type, "-o"),
        )
        if code == 0 and stdout:
            return stdout
    return None


def _preferred_mime(available: Sequence[str]) -> str | None:
    present = set(available)
    return next((mime_type for mime_type in _MIME_PREFERENCE if mime_type in present), None)


async def _default_runner(command: Sequence[str]) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=CLIPBOARD_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, b"", b"clipboard backend timed out"
    except asyncio.CancelledError:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    return int(process.returncode or 0), stdout, stderr
