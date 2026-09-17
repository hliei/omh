from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast


class _Cancellation:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.reason: BaseException | None = None

    def cancel(self, reason: BaseException) -> None:
        if self.event.is_set():
            return
        self.reason = reason
        self.event.set()


@dataclass(frozen=True, slots=True)
class Context:
    """Explicit process-local context for one harness call.

    Cancellation belongs to one invocation and is never durable operation state.
    """

    _cancellations: tuple[_Cancellation, ...] = ()

    def raise_if_cancelled(self) -> None:
        for cancellation in self._cancellations:
            if cancellation.event.is_set():
                reason = cancellation.reason
                if reason is None:
                    raise RuntimeError("Context cancellation is missing its reason")
                raise reason


@dataclass(frozen=True, slots=True)
class CancelScope:
    context: Context
    _cancellation: _Cancellation

    def cancel(self, reason: BaseException | None = None) -> None:
        self._cancellation.cancel(
            reason if reason is not None else asyncio.CancelledError("Context cancelled")
        )


def with_cancel(context: Context) -> CancelScope:
    cancellation = _Cancellation()
    return CancelScope(
        context=Context((*context._cancellations, cancellation)),
        _cancellation=cancellation,
    )


async def await_with_context[T](awaitable: Awaitable[T], context: Context) -> T:
    return await _race_with_context(awaitable, context, cancel_target=False)


async def cancel_on_context[T](awaitable: Awaitable[T], context: Context) -> T:
    return await _race_with_context(awaitable, context, cancel_target=True)


async def _race_with_context[T](
    awaitable: Awaitable[T], context: Context, *, cancel_target: bool
) -> T:
    context.raise_if_cancelled()
    if not context._cancellations:
        return await awaitable

    target = asyncio.ensure_future(
        awaitable if cancel_target else asyncio.shield(awaitable)
    )
    cancellation_waiters = [
        asyncio.create_task(cancellation.event.wait())
        for cancellation in context._cancellations
    ]
    waiters = {
        cast(asyncio.Future[object], target),
        *(cast(asyncio.Future[object], waiter) for waiter in cancellation_waiters),
    }
    try:
        done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if target in done:
            return target.result()
        if cancel_target:
            _cancel_and_observe(target)
        context.raise_if_cancelled()
        raise RuntimeError("Context cancellation completed without a reason")
    except asyncio.CancelledError:
        if cancel_target:
            _cancel_and_observe(target)
        raise
    finally:
        for waiter in cancellation_waiters:
            waiter.cancel()
        if not cancel_target and not target.done():
            target.cancel()
        await asyncio.gather(*cancellation_waiters, return_exceptions=True)


def _cancel_and_observe[T](task: asyncio.Future[T]) -> None:
    if task.done():
        return
    task.cancel()
    task.add_done_callback(_observe_completion)


def _observe_completion[T](task: asyncio.Future[T]) -> None:
    if not task.cancelled():
        task.exception()


BACKGROUND_CONTEXT = Context()
