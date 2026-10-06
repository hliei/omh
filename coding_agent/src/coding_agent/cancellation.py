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
from types import FrameType
from typing import TextIO

from coding_agent.terminal import plain_text, write_diagnostic

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
        self._previous: dict[int, signal._HANDLER] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def requested(self) -> bool:
        return self._signal_number is not None

    @property
    def exit_code(self) -> int:
        assert self._signal_number is not None
        return signal_exit_code(self._signal_number)

    def install(
        self, abort: Callable[[], None], *, loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Handle signals even while stdout backpressure blocks the event loop."""
        self._loop = loop

        def handler(signal_number: int, frame: FrameType | None) -> None:
            self._handle(signal_number, abort)

        for signal_number in _SIGNALS:
            previous = signal.getsignal(signal_number)
            try:
                signal.signal(signal_number, handler)
            except (OSError, ValueError):
                continue
            self._previous[signal_number] = previous

    def uninstall(self) -> None:
        for signal_number, previous in self._previous.items():
            signal.signal(signal_number, previous)
        self._previous.clear()
        self._loop = None

    def _handle(self, signal_number: int, abort: Callable[[], None]) -> None:
        if self._signal_number is None:
            self._signal_number = signal_number
            try:
                try:
                    if self._loop is not None:
                        # Wake a sleeping selector so syscall retry cannot delay
                        # the cleanup scheduled by the public abort callback.
                        self._loop.call_soon_threadsafe(lambda: None)
                    abort()
                except Exception:
                    # Close still awaits the Agent-owned cleanup it started.
                    pass
            finally:
                _signal_diagnostic(
                    self._stderr,
                    f"received {_signal_name(signal_number)}; stopping remaining tasks "
                    "and waiting for cleanup and saving",
                )
            return
        try:
            _signal_diagnostic(
                self._stderr,
                f"received a second {_signal_name(signal_number)}; saving may be incomplete",
            )
        finally:
            os._exit(signal_exit_code(signal_number))


def _signal_name(signal_number: int) -> str:
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return str(signal_number)


def _signal_diagnostic(stream: TextIO, message: str) -> None:
    """Use a best-effort unbuffered write; stderr cannot hold signal exit hostage."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        try:
            write_diagnostic(stream, message)
        except Exception:
            pass
        return
    try:
        blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)
        try:
            os.write(fd, f"omh: {plain_text(message)}\n".encode("utf-8"))
        finally:
            os.set_blocking(fd, blocking)
    except Exception:
        pass


class PrintPreparationCancelled(BaseException):
    """A termination signal interrupted print before the runtime owned work."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
