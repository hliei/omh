from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from omh.llm.types import AssistantMessage, AssistantMessageEvent, DoneEvent, ErrorEvent


class EventStream[T, R]:
    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        self._queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._done = False
        self._result: asyncio.Future[R] | None = None
        self._is_complete = is_complete
        self._extract_result = extract_result

    def _future(self) -> asyncio.Future[R]:
        if self._result is None:
            self._result = asyncio.get_running_loop().create_future()
        return self._result

    def push(self, event: T) -> None:
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            future = self._future()
            if not future.done():
                future.set_result(self._extract_result(event))
        self._queue.put_nowait(event)

    def end(self, result: R | None = None) -> None:
        self._done = True
        future = self._future()
        if result is not None and not future.done():
            future.set_result(result)
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    def result(self) -> asyncio.Future[R]:
        return self._future()


def _extract_result(event: AssistantMessageEvent) -> AssistantMessage:
    if isinstance(event, DoneEvent):
        return event.message
    if isinstance(event, ErrorEvent):
        return event.error
    raise RuntimeError(f"Unexpected event type for final result: {event.type}")


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(
            lambda event: event.type in {"done", "error"},
            _extract_result,
        )


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    return AssistantMessageEventStream()
