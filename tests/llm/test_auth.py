from __future__ import annotations

import asyncio

import pytest

from omh.llm.auth.credential_store import InMemoryCredentialStore
from omh.llm.auth.types import ApiKeyCredential, AuthOperationOptions
from omh.llm.models import create_models
from omh.llm.providers.deepseek import deepseek_provider
from omh.llm.types import AbortController, AbortError, Context, StreamOptions

from .http_samples import RecordingFetch, sse_response


@pytest.mark.asyncio
async def test_concurrent_credential_modifications_observe_previous_write() -> None:
    store = InMemoryCredentialStore()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    observed: list[str | None] = []

    async def first(current: ApiKeyCredential | None) -> ApiKeyCredential:
        first_started.set()
        await release_first.wait()
        return ApiKeyCredential(key="first")

    async def second(current: ApiKeyCredential | None) -> ApiKeyCredential:
        observed.append(current.key if current else None)
        return ApiKeyCredential(key="second")

    first_task = asyncio.create_task(store.modify("deepseek", first))
    await first_started.wait()
    second_task = asyncio.create_task(store.modify("deepseek", second))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert observed == ["first"]
    current = await store.read("deepseek")
    assert current is not None and current.key == "second"


@pytest.mark.asyncio
async def test_aborted_queued_delete_preserves_running_modification() -> None:
    store = InMemoryCredentialStore()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def first(current: ApiKeyCredential | None) -> ApiKeyCredential:
        first_started.set()
        await release_first.wait()
        return ApiKeyCredential(key="retained")

    first_task = asyncio.create_task(store.modify("deepseek", first))
    await first_started.wait()
    controller = AbortController()
    deletion = asyncio.create_task(store.delete("deepseek", AuthOperationOptions(signal=controller.signal)))
    await asyncio.sleep(0)
    controller.abort()
    with pytest.raises(AbortError):
        await deletion
    release_first.set()
    await first_task
    current = await store.read("deepseek")
    assert current is not None and current.key == "retained"


@pytest.mark.asyncio
async def test_reused_stream_options_resolve_current_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    models = create_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None
    fetch = RecordingFetch(sse_response({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    options = StreamOptions(fetch=fetch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first-key")
    first = await models.complete(model, Context(messages=[]), options)
    assert first.stop_reason == "stop"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "second-key")
    second = await models.complete(model, Context(messages=[]), options)
    assert second.stop_reason == "stop"
    assert fetch.requests[-1].headers["authorization"] == "Bearer second-key"
    assert options.api_key is None
    assert options.headers is None
    assert options.env is None
