"""Clipboard backend detection and the missing-backend diagnostic."""

from __future__ import annotations

import subprocess

import pytest

from coding_agent.clipboard import ClipboardError, clipboard_command, copy_text


def test_darwin_prefers_pbcopy() -> None:
    found = clipboard_command(platform="darwin", environ={}, which=lambda name: f"/usr/bin/{name}")
    assert found == ["pbcopy"]


def test_linux_uses_the_display_environment() -> None:
    assert clipboard_command(
        platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0"},
        which=lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None,
    ) == ["wl-copy"]
    assert clipboard_command(
        platform="linux", environ={"DISPLAY": ":0"},
        which=lambda name: "/usr/bin/xsel" if name == "xsel" else None,
    ) == ["xsel", "--clipboard", "--input"]
    assert clipboard_command(platform="linux", environ={}, which=lambda name: None) is None


def test_copy_text_reports_the_missing_backend() -> None:
    with pytest.raises(ClipboardError) as error:
        copy_text("answer", platform="linux", environ={}, which=lambda name: None)
    message = str(error.value)
    assert "No clipboard backend" in message
    assert "does not install" in message


def test_missing_backend_names_the_platform_command() -> None:
    with pytest.raises(ClipboardError) as error:
        copy_text("answer", platform="darwin", environ={}, which=lambda name: None)
    assert "install pbcopy" in str(error.value)
    with pytest.raises(ClipboardError) as error:
        copy_text("answer", platform="linux", environ={"DISPLAY": ":0"}, which=lambda name: None)
    assert "xclip" in str(error.value)


def test_copy_text_pipes_the_answer_to_the_backend() -> None:
    calls: list[tuple[list[str], str]] = []

    def runner(command, *, input, **kwargs):
        calls.append((command, input))
        return subprocess.CompletedProcess(command, 0)

    copy_text("the answer", platform="darwin", environ={}, which=lambda name: "/usr/bin/pbcopy", runner=runner)
    assert calls == [(["pbcopy"], "the answer")]


def test_failing_backend_is_a_diagnostic() -> None:
    def runner(command, *, input, **kwargs):
        return subprocess.CompletedProcess(command, 1)

    with pytest.raises(ClipboardError) as error:
        copy_text("answer", platform="darwin", environ={}, which=lambda name: "/usr/bin/pbcopy", runner=runner)
    assert "failed with status 1" in str(error.value)
