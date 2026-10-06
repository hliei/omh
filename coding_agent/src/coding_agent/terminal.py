"""Process exit codes and single-line terminal diagnostics.

The installed command and print execution share one definition of the process
outcome constants and one control-sanitized diagnostic writer, so no mode grows
its own convention. Every diagnostic is one ``omh: ...`` line: control
characters are shown as escapes and a newline never splits a terminal field.
"""

from __future__ import annotations

import os
from typing import TextIO

#: A command completed normally.
EXIT_OK = 0
#: Execution failed after a request started, or the final assistant ended in error.
EXIT_FAILURE = 1
#: Invalid command line, input or configuration rejected before a request.
EXIT_USAGE = 2


def plain_text(value: str) -> str:
    """Keep one terminal field on one line, with controls shown as escapes."""
    return "".join(
        char if char.isprintable() and char != "\\" else ascii(char)[1:-1]
        for char in value
    )


def write_diagnostic(stream: TextIO, error: Exception | str) -> None:
    """Write one diagnostic line to a diagnostic stream."""
    stream.write(f"omh: {plain_text(str(error))}\n")


def write_best_effort_diagnostic(stream: TextIO, error: Exception | str) -> None:
    """Report an immediate exit without blocking or reentering buffered stderr."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        try:
            write_diagnostic(stream, error)
        except Exception:
            pass
        return
    try:
        blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)
        try:
            os.write(fd, f"omh: {plain_text(str(error))}\n".encode("utf-8"))
        finally:
            os.set_blocking(fd, blocking)
    except Exception:
        pass
