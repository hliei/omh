from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable

from omh.llm.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ErrorEvent,
    Model,
    empty_usage,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream


def create_setup_error_message(model: Model, error: object) -> AssistantMessage:
    message = str(error)
    return AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=empty_usage(),
        stop_reason="error",
        timestamp=_now_ms(),
        error_message=message,
    )


def _now_ms() -> int:
    import time

    return time.time_ns() // 1_000_000


async def _forward_stream(
    target: AssistantMessageEventStream,
    source: AsyncIterable[AssistantMessageEvent],
) -> None:
    async for event in source:
        target.push(event)
    result = await source.result() if hasattr(source, "result") else None
    target.end(result)


def lazy_stream(
    model: Model,
    setup: Callable[[], Awaitable[AsyncIterable[AssistantMessageEvent]]],
) -> AssistantMessageEventStream:
    outer = AssistantMessageEventStream()

    async def run() -> None:
        try:
            inner = await setup()
            await _forward_stream(outer, inner)
        except Exception as error:
            message = create_setup_error_message(model, error)
            outer.push(ErrorEvent(reason="error", error=message))
            outer.end(message)

    asyncio.get_running_loop().create_task(run())
    return outer
