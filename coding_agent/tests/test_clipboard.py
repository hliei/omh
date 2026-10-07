"""Clipboard image backend detection and controlled read/fallback behavior.

The tests use an injected ``which`` and runner: no real desktop clipboard is
touched. A successful real desktop paste is proved separately by the manual
two-platform acceptance, not here.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from support import png_bytes

from coding_agent.clipboard import (
    ClipboardUnavailable,
    detect_clipboard,
    read_clipboard_image,
)


class FakeRunner:
    """Controlled subprocess boundary returning queued results and recording commands."""

    def __init__(self, results: list[tuple[int, bytes, bytes]]) -> None:
        self.results = list(results)
        self.commands: list[tuple[str, ...]] = []
        self.on_call = None

    async def __call__(self, command: Sequence[str]) -> tuple[int, bytes, bytes]:
        self.commands.append(tuple(command))
        if self.on_call is not None:
            self.on_call(tuple(command))
        return self.results.pop(0)


def which_from(mapping: dict[str, str]) -> Callable[[str], str | None]:
    return lambda name: mapping.get(name)


def test_macos_backend_is_detected_only_with_osascript() -> None:
    detected = detect_clipboard(platform="darwin", which=which_from({"osascript": "osascript"}))
    assert detected.available
    assert detected.backend is not None and detected.backend.kind == "macos"
    assert "osascript" in detected.detail

    missing = detect_clipboard(platform="darwin", which=which_from({}))
    assert not missing.available
    assert "osascript" in missing.detail


def test_linux_prefers_wayland_then_x11() -> None:
    wayland = detect_clipboard(
        platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"},
        which=which_from({"wl-paste": "wl-paste", "xclip": "xclip"}),
    )
    assert wayland.available and wayland.backend is not None and wayland.backend.kind == "wayland"
    assert "wl-clipboard" in wayland.detail
    assert wayland.fallback is not None and wayland.fallback.kind == "x11"

    x11 = detect_clipboard(
        platform="linux", environ={"DISPLAY": ":0"},
        which=which_from({"xclip": "xclip"}),
    )
    assert x11.available and x11.backend is not None and x11.backend.kind == "x11"
    assert "xclip" in x11.detail


def test_linux_headless_or_missing_dependencies_never_claim_support() -> None:
    headless = detect_clipboard(platform="linux", environ={}, which=which_from({"wl-paste": "x"}))
    assert not headless.available
    assert "Wayland or X11 display" in headless.detail

    missing = detect_clipboard(
        platform="linux", environ={"DISPLAY": ":0"}, which=which_from({}),
    )
    assert not missing.available
    assert "xclip" in missing.detail


def test_other_platforms_report_no_clipboard_paste() -> None:
    detected = detect_clipboard(platform="win32", which=which_from({}))
    assert not detected.available
    assert "not supported" in detected.detail


async def test_unavailable_backend_names_the_file_fallback() -> None:
    support = detect_clipboard(platform="linux", environ={}, which=which_from({}))

    with pytest.raises(ClipboardUnavailable) as failure:
        await read_clipboard_image(support=support)

    message = str(failure.value)
    assert "no clipboard image backend" in message
    assert "/attach <image-path>" in message


async def test_macos_reads_the_png_the_backend_writes(tmp_path: Path) -> None:
    expected = png_bytes()
    support = detect_clipboard(platform="darwin", which=which_from({"osascript": "osascript"}))
    runner = FakeRunner([(0, b"", b"")])

    def write_target(command: tuple[str, ...]) -> None:
        script = command[2]
        target = script.split('POSIX file "', 1)[1].split('"', 1)[0]
        Path(target).write_bytes(expected)

    runner.on_call = write_target
    assert await read_clipboard_image(support=support, runner=runner) == expected
    assert runner.commands[0][0] == "osascript"


async def test_wayland_lists_types_then_reads_the_preferred_image() -> None:
    expected = png_bytes()
    support = detect_clipboard(
        platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0"}, which=which_from({"wl-paste": "x"}),
    )
    runner = FakeRunner([
        (0, b"text/plain\nimage/png\n", b""),
        (0, expected, b""),
    ])

    assert await read_clipboard_image(support=support, runner=runner) == expected
    assert runner.commands[1] == ("wl-paste", "--no-newline", "--type", "image/png")


async def test_wayland_backend_failure_falls_back_to_x11() -> None:
    expected = png_bytes()
    support = detect_clipboard(
        platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"},
        which=which_from({"wl-paste": "x", "xclip": "x"}),
    )
    runner = FakeRunner([(1, b"", b"compositor unavailable"), (0, expected, b"")])

    assert await read_clipboard_image(support=support, runner=runner) == expected
    assert runner.commands[0] == ("wl-paste", "--list-types")
    assert runner.commands[1][0] == "xclip"


async def test_wayland_empty_result_never_reads_the_stale_x11_clipboard() -> None:
    support = detect_clipboard(
        platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"},
        which=which_from({"wl-paste": "x", "xclip": "x"}),
    )
    runner = FakeRunner([(0, b"text/plain\n", b"")])

    assert await read_clipboard_image(support=support, runner=runner) is None
    assert len(runner.commands) == 1


async def test_wayland_without_an_image_returns_none_without_reading() -> None:
    support = detect_clipboard(
        platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0"}, which=which_from({"wl-paste": "x"}),
    )
    runner = FakeRunner([(0, b"text/plain\n", b"")])

    assert await read_clipboard_image(support=support, runner=runner) is None
    assert len(runner.commands) == 1


async def test_x11_tries_supported_types_and_skips_failure() -> None:
    expected = png_bytes()
    support = detect_clipboard(platform="linux", environ={"DISPLAY": ":0"}, which=which_from({"xclip": "x"}))
    runner = FakeRunner([(1, b"", b"no target"), (0, expected, b"")])

    assert await read_clipboard_image(support=support, runner=runner) == expected
    assert runner.commands[0][-3:] == ("-t", "image/png", "-o")
    assert runner.commands[1][-3:] == ("-t", "image/jpeg", "-o")


async def test_backend_failure_or_empty_output_returns_none() -> None:
    support = detect_clipboard(platform="linux", environ={"DISPLAY": ":0"}, which=which_from({"xclip": "x"}))
    runner = FakeRunner([(1, b"", b"")] * 5)
    assert await read_clipboard_image(support=support, runner=runner) is None


async def test_default_runner_stops_a_hanging_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from coding_agent import clipboard

    monkeypatch.setattr(clipboard, "CLIPBOARD_TIMEOUT_SECONDS", 0.2)
    code, _stdout, stderr = await clipboard._default_runner(
        (sys.executable, "-c", "import time; time.sleep(30)"),
    )
    assert code == 124
    assert b"timed out" in stderr


async def test_cancelled_image_read_stops_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = tmp_path / "ready"
    backend = tmp_path / "wl-paste"
    backend.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, time\n"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    backend.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    support = detect_clipboard(platform="linux", environ={"WAYLAND_DISPLAY": "test"})
    task = asyncio.create_task(read_clipboard_image(support=support))
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
