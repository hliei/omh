"""Awaitable adaptation shared by the Agent and loop."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from omh.llm.types import AbortSignal

_T = TypeVar("_T")


async def maybe_await(value: _T | Awaitable[_T]) -> _T:
    """Await ``value`` when it is awaitable; otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


async def call_with_signal(call: Callable[[], _T | Awaitable[_T]], signal: AbortSignal | None) -> _T:
    """Cancel pending preparation; retain completed values for caller cleanup.

    Cancellation can race completion, or a callback can return a resource from
    its cancellation handler. Keep that value so its owner can finish cleanup;
    the caller checks the signal before admitting the next operation.
    """
    if signal is not None:
        signal.throw_if_aborted()
    value = call()
    if signal is None:
        return await maybe_await(value)
    if not inspect.isawaitable(value):
        return value

    task: asyncio.Future[_T] = asyncio.ensure_future(value)
    aborted = asyncio.get_running_loop().create_future()

    def on_abort() -> None:
        if not aborted.done():
            aborted.set_result(None)

    signal.add_callback(on_abort)
    try:
        await asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if task.cancelled():
            signal.throw_if_aborted()
        return task.result()
    finally:
        signal.remove_callback(on_abort)
        aborted.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def gather_settled(awaitables: Iterable[Awaitable[_T]]) -> list[_T]:
    """Keep ownership of every child even when one raises during notification."""
    tasks = [asyncio.ensure_future(value) for value in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
