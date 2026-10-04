"""Per-event-loop canonical-path coordination for write and edit adapters."""

import asyncio
import os
import sys
import unicodedata
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from omh.agent.tools.file_io import run_file_io
from omh.llm.types import AbortSignal


@dataclass(slots=True)
class _FileLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass(slots=True)
class _MutationQueues:
    registration: asyncio.Lock = field(default_factory=asyncio.Lock)
    files: dict[Path, _FileLock] = field(default_factory=dict)


_queues: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _MutationQueues] = weakref.WeakKeyDictionary()


def _canonical_key(path: Path) -> Path:
    resolved = path.resolve()
    if sys.platform == "darwin":
        existing = resolved
        while not existing.exists():
            existing = existing.parent
        # Darwin's _PC_CASE_SENSITIVE is 11 (sys/unistd.h); Python's named
        # pathconf table omits it. Query the volume, not the platform default.
        if os.pathconf(existing, 11) == 0:
            return Path(unicodedata.normalize("NFD", str(resolved).casefold()))
    return resolved


async def with_file_mutation_queue[T](
    path: Path, operation: Callable[[], Awaitable[T]], signal: AbortSignal | None
) -> T:
    """Hold one file's coordination until the adapter's operation settles.

    Non-strict resolution also resolves existing symlink parents for new files.
    Factories share this registry regardless of their cwd or tool instance.
    """
    if signal is not None:
        signal.throw_if_aborted()
    queues = _queues.setdefault(asyncio.get_running_loop(), _MutationQueues())
    # Slow canonicalization must not let a later submission overtake this one.
    async with queues.registration:
        key = await run_file_io(lambda: _canonical_key(path))
        state = queues.files.setdefault(key, _FileLock())
        state.users += 1
    try:
        async with state.lock:
            if signal is not None:
                signal.throw_if_aborted()
            return await operation()
    finally:
        state.users -= 1
        if state.users == 0:
            del queues.files[key]
