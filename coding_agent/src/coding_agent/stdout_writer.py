"""Serial stdout records with defined backpressure and write-failure outcomes.

Print text and print JSON share one writer so ordering, temporary-error retry
and permanent-error behavior cannot diverge. When the stream exposes a file
descriptor, each record is one ordered ``os.write`` loop instead of an
unbounded user-space queue behind a slow consumer; ``write`` returns only after
the operating system accepted every byte. ``ENOBUFS``, ``EAGAIN`` and
``EWOULDBLOCK`` retry the same pending bytes after the fixed delay, so no
second writer is started and no data is dropped or duplicated. Any other write
error is permanent: the writer records the failure and raises
:class:`StdoutWriteError`, which tells print to exit 1 without waiting for
cooperative cleanup, complete saving or rescue.
"""

from __future__ import annotations

import errno
import io
import os
import time
from typing import TextIO

#: ``os.write`` errors that mean the consumer is temporarily not accepting data.
TEMPORARY_WRITE_ERRNOS = frozenset({errno.ENOBUFS, errno.EAGAIN, errno.EWOULDBLOCK})
#: Fixed delay before retrying a temporarily rejected write.
WRITE_RETRY_DELAY_SECONDS = 0.01


class StdoutWriteError(Exception):
    """A permanent stdout write failure that must end print with exit 1."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"stdout write failed: {cause}")
        self.cause = cause


def _file_descriptor(stream: TextIO) -> int | None:
    """Return the stream's open descriptor, or ``None`` for a memory stream."""
    try:
        return stream.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return None


class StdoutWriter:
    """Write text records serially to one stdout stream.

    ``write`` returns only after the record reached the operating system, so a
    slow consumer blocks the producer instead of growing an unbounded queue.
    ``failed`` stays true after a permanent error; callers use it to skip the
    cooperative cleanup that a broken stdout must not wait for.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._fd = _file_descriptor(stream)
        self._encoding = getattr(stream, "encoding", None) or "utf-8"
        self._errors = getattr(stream, "errors", None) or "strict"
        self.failed = False

    def write(self, text: str) -> None:
        """Write one text record, blocking until the OS accepted every byte."""
        if not text:
            return
        fd = self._fd
        if fd is None:
            self._write_stream(text)
        else:
            self._write_bytes(fd, text.encode(self._encoding, self._errors))

    def flush(self) -> None:
        """Wait for any buffered stream state after the final record."""
        while True:
            try:
                self._stream.flush()
                return
            except OSError as error:
                self._retry_or_fail(error)
                continue

    def _write_stream(self, text: str) -> None:
        remaining = text
        while remaining:
            try:
                written = self._stream.write(remaining)
            except OSError as error:
                self._retry_or_fail(error)
                continue
            if written is None:
                written = len(remaining)
            if written <= 0:
                self._wait()
                continue
            remaining = remaining[written:]

    def _write_bytes(self, fd: int, data: bytes) -> None:
        pending = memoryview(data)
        while pending:
            try:
                written = os.write(fd, pending)
            except OSError as error:
                self._retry_or_fail(error)
                continue
            if written <= 0:
                self._wait()
                continue
            pending = pending[written:]

    def _retry_or_fail(self, error: OSError) -> None:
        """Retry a temporarily rejected write in place, or fail permanently.

        Returns after sleeping for a temporary error so the caller retries the
        same pending bytes; otherwise records the permanent failure and raises.
        """
        if error.errno in TEMPORARY_WRITE_ERRNOS:
            self._wait()
            return
        self.failed = True
        raise StdoutWriteError(error) from error

    def _wait(self) -> None:
        time.sleep(WRITE_RETRY_DELAY_SECONDS)
