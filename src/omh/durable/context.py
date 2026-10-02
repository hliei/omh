from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True, slots=True)
class ContextKey[T]:
    name: str


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
    _values: tuple[tuple[ContextKey[object], object], ...] = ()

    def value[T](self, key: ContextKey[T]) -> T | None:
        for stored_key, stored_value in reversed(self._values):
            if stored_key is key:
                return cast(T, stored_value)
        return None

    def raise_if_cancelled(self) -> None:
        for cancellation in self._cancellations:
            if cancellation.event.is_set():
                reason = cancellation.reason
                if reason is None:
                    raise RuntimeError("Context cancellation is missing its reason")
                raise reason

    @property
    def cancelled(self) -> bool:
        """Whether this context was cancelled, without consuming the reason."""
        return any(cancellation.event.is_set() for cancellation in self._cancellations)


@dataclass(frozen=True, slots=True)
class CancelScope:
    context: Context
    _cancellation: _Cancellation

    def cancel(self, reason: BaseException | None = None) -> None:
        self._cancellation.cancel(
            reason
            if reason is not None
            else asyncio.CancelledError("Context cancelled")
        )


def with_cancel(context: Context) -> CancelScope:
    cancellation = _Cancellation()
    return CancelScope(
        context=Context(
            _cancellations=(*context._cancellations, cancellation),
            _values=context._values,
        ),
        _cancellation=cancellation,
    )


def without_cancel(context: Context) -> Context:
    return Context(_values=context._values)


async def await_with_context[T](awaitable: Awaitable[T], context: Context) -> T:
    return await _race_with_context(awaitable, context, cancel_target=False)


async def cancel_on_context[T](awaitable: Awaitable[T], context: Context) -> T:
    return await _race_with_context(awaitable, context, cancel_target=True)


async def wait_for_cancellation(context: Context) -> None:
    """Wait until this context is cancelled; never resolves without a cancellation."""
    if not context._cancellations:
        await asyncio.Future()
        return
    waiters = [
        asyncio.create_task(cancellation.event.wait())
        for cancellation in context._cancellations
    ]
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


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


def create_context_key[T](name: str) -> ContextKey[T]:
    return ContextKey(name)


def with_context_value[T](key: ContextKey[T], value: T, context: Context) -> Context:
    return Context(
        _cancellations=context._cancellations,
        _values=(*context._values, (cast(ContextKey[object], key), value)),
    )
