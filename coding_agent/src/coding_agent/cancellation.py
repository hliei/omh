"""Cooperative print cancellation and complete-history rescue.

The first SIGINT/SIGTERM/SIGHUP of a print run stops admitting and sending the
remaining tasks and cooperatively aborts the current Agent activity through its
public ``abort`` boundary. The run still awaits Agent, tool, retry and summary
cleanup, the existing history save and the ordinary close path, then exits with
130/143/129. A second termination signal forces the process to stop and says on
stderr that saving may be incomplete.

Cancelling a prompt waiter is not a way to stop the work it owns, so this module
never cancels the waiter, never waits for idle/close from inside an awaited
callback handled here, and sets no automatic hard timeout. The host keeps
waiting for the cleanup it started and never rolls back completed side effects.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable
from typing import TextIO

from coding_agent.terminal import write_diagnostic

#: The three termination signals the print command handles cooperatively.
_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def signal_exit_code(signal_number: int) -> int:
    """Return the cooperative first-signal exit code (128 + signal number)."""
    return 128 + int(signal_number)


class PrintCancellation:
    """Own first/second termination-signal behavior for one print run."""

    def __init__(self, stderr: TextIO) -> None:
        self._stderr = stderr
        self._signal_number: int | None = None
        self._installed: list[int] = []

    @property
    def requested(self) -> bool:
        return self._signal_number is not None

    @property
    def exit_code(self) -> int:
        assert self._signal_number is not None
        return signal_exit_code(self._signal_number)

    def install(self, loop: asyncio.AbstractEventLoop, abort: Callable[[], None]) -> None:
        """Install the three handlers when the loop and thread allow it.

        Unsupported platforms or a non-main thread keep the previous process
        behavior rather than failing the print run.
        """
        def handler(signal_number: int) -> None:
            self._handle(signal_number, abort)

        for signal_number in _SIGNALS:
            try:
                loop.add_signal_handler(signal_number, handler, signal_number)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            self._installed.append(signal_number)

    def uninstall(self, loop: asyncio.AbstractEventLoop) -> None:
        for signal_number in self._installed:
            try:
                loop.remove_signal_handler(signal_number)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
        self._installed.clear()

    def _handle(self, signal_number: int, abort: Callable[[], None]) -> None:
        if self._signal_number is None:
            self._signal_number = signal_number
            write_diagnostic(
                self._stderr,
                f"received {_signal_name(signal_number)}; stopping remaining tasks "
                "and waiting for cleanup and saving",
            )
            try:
                abort()
            except Exception:
                # A failed abort must not break signal delivery; close still
                # awaits the Agent-owned cleanup it started.
                pass
            return
        write_diagnostic(
            self._stderr,
            f"received a second {_signal_name(signal_number)}; saving may be incomplete",
        )
        _flush(self._stderr)
        os._exit(signal_exit_code(signal_number))


def _signal_name(signal_number: int) -> str:
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return str(signal_number)


def _flush(stream: TextIO) -> None:
    try:
        stream.flush()
    except Exception:
        pass
