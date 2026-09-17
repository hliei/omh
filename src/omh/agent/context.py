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
    context.raise_if_cancelled()
    if not context._cancellations:
        return await awaitable

    observation = asyncio.ensure_future(asyncio.shield(awaitable))
    cancellation_waiters = [
        asyncio.create_task(cancellation.event.wait())
        for cancellation in context._cancellations
    ]
    try:
        waiters = {
            cast(asyncio.Future[object], observation),
            *(cast(asyncio.Future[object], waiter) for waiter in cancellation_waiters),
        }
        done, _ = await asyncio.wait(
            waiters,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if observation in done:
            return observation.result()
        context.raise_if_cancelled()
        raise RuntimeError("Context cancellation completed without a reason")
    finally:
        for waiter in cancellation_waiters:
            waiter.cancel()
        if not observation.done():
            observation.cancel()
        await asyncio.gather(*cancellation_waiters, return_exceptions=True)


async def cancel_on_context[T](awaitable: Awaitable[T], context: Context) -> T:
    context.raise_if_cancelled()
    if not context._cancellations:
        return await awaitable

    operation = asyncio.ensure_future(awaitable)
    cancellation_waiters = [
        asyncio.create_task(cancellation.event.wait())
        for cancellation in context._cancellations
    ]
    waiters = {
        cast(asyncio.Future[object], operation),
        *(cast(asyncio.Future[object], waiter) for waiter in cancellation_waiters),
    }
    try:
        done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if operation in done:
            return operation.result()
        operation.cancel()
        operation.add_done_callback(_observe_completion)
        context.raise_if_cancelled()
        raise RuntimeError("Context cancellation completed without a reason")
    except asyncio.CancelledError:
        operation.cancel()
        operation.add_done_callback(_observe_completion)
        raise
    finally:
        for waiter in cancellation_waiters:
            waiter.cancel()
        await asyncio.gather(*cancellation_waiters, return_exceptions=True)


def _observe_completion[T](task: asyncio.Future[T]) -> None:
    if not task.cancelled():
        task.exception()


BACKGROUND_CONTEXT = Context()
