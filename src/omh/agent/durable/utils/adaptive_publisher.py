from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptivePublisherOptions[TValue, TUpdate]:
    snapshot: Callable[[], TValue]
    update: Callable[[TValue | None, TValue], TUpdate | None]
    measure: Callable[[TUpdate], int]
    publish: Callable[[TUpdate], None]
    on_error: Callable[[BaseException], None]
    min_interval_ms: float = 100.0
    target_bytes_per_second: float = 100 * 1024


class AdaptivePublisher[TValue, TUpdate]:
    """Publish the latest state without queueing intermediate mutations.

    The first dirty state after idle is immediate. Each publication then buys a
    delay proportional to its encoded size, with a minimum interval that also
    bounds event count. A single trailing timer guarantees eventual publication.
    """

    def __init__(self, options: AdaptivePublisherOptions[TValue, TUpdate]) -> None:
        self._options = options
        self._loop = asyncio.get_running_loop()
        self._published: TValue | None = None
        self._dirty = False
        self._next_emit_at = 0.0
        self._timer: asyncio.TimerHandle | None = None
        self._disposed = False

    def mark_dirty(self) -> None:
        if self._disposed:
            return
        self._dirty = True
        wait = self._next_emit_at - self._now()
        if wait <= 0:
            self.flush()
            return
        self._arm_timer(wait)

    def flush(self, force: bool = False) -> None:
        if self._disposed or not self._dirty:
            return
        now = self._now()
        if not force and now < self._next_emit_at:
            self._arm_timer(self._next_emit_at - now)
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        current = self._options.snapshot()
        update = self._options.update(self._published, current)
        if update is None:
            self._published = current
            self._dirty = False
            return
        encoded_bytes = self._options.measure(update)
        self._published = current
        self._dirty = False
        self._next_emit_at = now + max(
            self._options.min_interval_ms,
            (encoded_bytes * 1000) / self._options.target_bytes_per_second,
        )
        # Commit before delivery. A consumer may apply the update and then throw
        # or reenter the producer; retaining the old baseline would duplicate
        # that delta.
        self._options.publish(update)

    def dispose(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._disposed = True

    def _now(self) -> float:
        return self._loop.time() * 1000

    def _arm_timer(self, wait_ms: float) -> None:
        if self._timer is not None:
            return

        def on_timer() -> None:
            self._timer = None
            try:
                self.flush()
            except BaseException as error:  # noqa: BLE001 - forwarded to the owner
                self._options.on_error(error)

        self._timer = self._loop.call_later(max(wait_ms, 0) / 1000, on_timer)

