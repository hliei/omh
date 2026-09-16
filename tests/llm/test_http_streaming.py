from __future__ import annotations

import asyncio

import httpx
import pytest

from omh.llm.models import create_models
from omh.llm.providers.deepseek import deepseek_provider
from omh.llm.types import (
    AbortController,
    Context,
    SimpleStreamOptions,
    TextDeltaEvent,
    UserMessage,
)


class GatedBody(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        self.waiting.set()
        await self.release.wait()
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

    async def aclose(self) -> None:
        self.closed.set()


def configure_http(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    models = create_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None
    return models, model


async def test_default_http_delivers_delta_before_response_finishes(monkeypatch):
    body = GatedBody()
    models, model = configure_http(monkeypatch, lambda request: httpx.Response(200, stream=body))
    stream = models.stream_simple(model, Context(messages=[UserMessage(content="hello", timestamp=1)]), SimpleStreamOptions(api_key="test"))

    async def first_delta():
        async for event in stream:
            if isinstance(event, TextDeltaEvent):
                return event.delta

    try:
        assert await asyncio.wait_for(first_delta(), 1) == "hello"
        assert not body.release.is_set()
    finally:
        body.release.set()
        result = await asyncio.wait_for(stream.result(), 1)
    assert result.stop_reason == "stop"
    assert body.closed.is_set()


@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_abort_interrupts_inflight_http(monkeypatch, phase):
    body = GatedBody()
    requested = asyncio.Event()
    allow_headers = asyncio.Event()

    async def handler(request):
        requested.set()
        if phase == "headers":
            await allow_headers.wait()
        return httpx.Response(200, stream=body)

    models, model = configure_http(monkeypatch, handler)
    controller = AbortController()
    stream = models.stream_simple(model, Context(messages=[]), SimpleStreamOptions(api_key="test", signal=controller.signal))
    await asyncio.wait_for(requested.wait() if phase == "headers" else body.waiting.wait(), 1)
    controller.abort()
    try:
        result = await asyncio.wait_for(asyncio.shield(stream.result()), 1)
        assert result.stop_reason == "aborted"
        if phase == "body":
            assert result.content[0].text == "hello"
            assert body.closed.is_set()
    finally:
        allow_headers.set()
        body.release.set()
        await asyncio.wait_for(stream.result(), 1)
