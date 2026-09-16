from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from omh.llm.types import AbortError, AbortSignal


def operation_signal(signal: AbortSignal | None) -> AbortSignal:
    return signal if signal is not None else AbortSignal()


def abort_reason(signal: AbortSignal) -> BaseException:
    if signal.reason is not None:
        return signal.reason
    return AbortError()


async def race_with_abort_signal[T](operation: Awaitable[T], signal: AbortSignal) -> T:
    task = asyncio.ensure_future(operation)
    if signal.aborted:
        task.cancel()
        try:
            await task
        except (Exception, asyncio.CancelledError):
            pass
        raise abort_reason(signal)

    aborted = asyncio.get_running_loop().create_future()

    def on_abort() -> None:
        if not aborted.done():
            aborted.set_result(None)

    signal.add_callback(on_abort)
    if signal.aborted and not aborted.done():
        aborted.set_result(None)
    try:
        done, _pending = await asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except (Exception, asyncio.CancelledError):
            pass
        raise abort_reason(signal)
    finally:
        if not aborted.done():
            aborted.cancel()
