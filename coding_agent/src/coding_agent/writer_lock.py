"""Stable per-path advisory locks for the supported macOS/Linux hosts."""

from __future__ import annotations

import fcntl
import os
from hashlib import sha256
from pathlib import Path


class SessionWriterError(RuntimeError):
    """Another application writer owns the requested session path."""


class WriterLock:
    def __init__(self, path: Path) -> None:
        # Use one stable location across environments and leave its inodes in
        # place: unlinking a lock file would let later writers lock another inode.
        directory = Path("/tmp") / f"omh-coding-agent-writers-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        name = sha256(os.fsencode(path)).hexdigest()
        self._file = open(directory / name, "a+b")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._file.close()
            raise SessionWriterError(f"Session already has a writer: {path}") from error
        except BaseException:
            self._file.close()
            raise

    def close(self) -> None:
        self._file.close()
