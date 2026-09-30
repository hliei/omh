"""Provider-boundary tests for normalized transcripts, projection, and options."""

from __future__ import annotations

from typing import Any

import pytest

from omh.llm.auth.helpers import env_api_key_auth
from omh.llm.auth.types import ProviderAuth
from omh.llm.models import CreateProviderOptions, create_models, create_provider
from omh.llm.providers.deepseek import deepseek_provider
from omh.llm.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    Model,
    ModelCost,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    SystemMessage,
    TextContent,
    ThinkingBudgets,
    Tool,
    ToolReference,
    TranscriptContext,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

from .http_samples import RecordingFetch, sse_response

CALCULATOR = Tool(
    name="math_operation",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
)


def _deepseek() -> tuple[Any, Model]:
    models = create_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None
    return models, model


def _final_message() -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="deepseek",
        model="deepseek-flash",
        usage=empty_usage(),
        stop_reason="stop",
        timestamp=1,
        content=[TextContent(text="ok")],
    )


def _completed_stream() -> AssistantMessageEventStream:
    stream = create_assistant_message_event_stream()
    stream.push(StartEvent(partial=_final_message()))
    stream.push(DoneEvent(reason="stop", message=_final_message()))
    return stream


class _RecordingApi:
    def __init__(self) -> None:
        self.contexts: list[TranscriptContext] = []

    def _record(self, context: TranscriptContext) -> AssistantMessageEventStream:
        self.contexts.append(context)
        return _completed_stream()

    def stream(self, model: Model, context: TranscriptContext, options: object = None) -> AssistantMessageEventStream:
        return self._record(context)

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: object = None
    ) -> AssistantMessageEventStream:
        return self._record(context)


def _recording_provider() -> tuple[Any, _RecordingApi]:
    api = _RecordingApi()
    provider = create_provider(
        CreateProviderOptions(
            id="recording",
            auth=ProviderAuth(api_key=env_api_key_auth("Recording API key", ("RECORDING_API_KEY",))),
            models=[
                Model(
                    id="recording-model",
                    name="Recording",
                    api="openai-completions",
                    provider="recording",
                    base_url="https://example.invalid",
                    reasoning=False,
                    input=("text",),
                    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                    context_window=1000,
                    max_tokens=100,
                )
            ],
            api=api,
        )
    )
    models = create_models()
    models.set_provider(provider)
    return models, api


@pytest.mark.asyncio
async def test_models_normalizes_legacy_context_before_provider() -> None:
    models, api = _recording_provider()
    model = models.get_model("recording", "recording-model")
    assert model is not None

    await models.stream_simple(
        model,
        Context(messages=[UserMessage(content="hi", timestamp=1)], system_prompt="Be terse", tools=[CALCULATOR]),
        SimpleStreamOptions(api_key="test"),
    ).result()

    assert len(api.contexts) == 1
    messages = api.contexts[0].messages
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Be terse"
    assert messages[0].tools_added == [CALCULATOR]
    assert messages[1].role == "user"


@pytest.mark.asyncio
async def test_deepseek_projects_legacy_prompt_and_tools_into_request() -> None:
    models, model = _deepseek()
    fetch = RecordingFetch(sse_response({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}))

    await models.stream_simple(
        model,
        Context(
            messages=[UserMessage(content="Hello", timestamp=1)],
            system_prompt="Be terse",
            tools=[CALCULATOR],
        ),
        SimpleStreamOptions(api_key="test-key", fetch=fetch),
    ).result()

    body = fetch.body
    assert body["messages"][0] == {"role": "system", "content": "Be terse"}
    assert body["tools"][0]["function"]["name"] == "math_operation"


@pytest.mark.asyncio
async def test_deepseek_folds_mid_conversation_system_messages() -> None:
    models, model = _deepseek()
    fetch = RecordingFetch(sse_response({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}))

    context = Context(
        messages=[
            SystemMessage(
                content="Base",
                timestamp=0,
                sections={"style": "concise", "tone": "warm"},
                tools_added=[Tool(name="run", description="v1", parameters={"type": "object"})],
            ),
            UserMessage(content="Hello", timestamp=1),
            SystemMessage(
                content="Added later",
                timestamp=2,
                sections={"style": None},
                tools_removed=[ToolReference(name="run")],
                tools_added=[Tool(name="run", description="v2", parameters={"type": "object"})],
            ),
        ]
    )

    await models.stream_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch)).result()

    body = fetch.body
    system_messages = [message for message in body["messages"] if message["role"] == "system"]
    assert len(system_messages) == 1
    assert "Base" in system_messages[0]["content"]
    assert "Added later" in system_messages[0]["content"]
    assert "warm" in system_messages[0]["content"]
    assert body["tools"][0]["function"]["description"] == "v2"


@pytest.mark.asyncio
async def test_on_response_and_provider_stream_event_are_invoked() -> None:
    models, model = _deepseek()
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"delta": {"content": "one"}}]},
            {"choices": [{"delta": {"content": " two"}, "finish_reason": "stop"}]},
        )
    )
    responses: list[ProviderResponse] = []
    chunks: list[object] = []

    await models.stream_simple(
        model,
        Context(messages=[UserMessage(content="Hello", timestamp=1)]),
        SimpleStreamOptions(
            api_key="test-key",
            fetch=fetch,
            on_response=lambda response, request_model: responses.append(response),
            on_provider_stream_event=lambda chunk, request_model: chunks.append(chunk),
        ),
    ).result()

    assert len(responses) == 1
    assert responses[0].status == 200
    assert len(chunks) == 2
    assert isinstance(chunks[0], dict)
    assert "choices" in chunks[0]


@pytest.mark.asyncio
async def test_session_and_thinking_options_survive_auth_resolution() -> None:
    models, model = _deepseek()
    captured: list[SimpleStreamOptions] = []

    def _capture(
        request_model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        assert options is not None
        captured.append(options)
        return _completed_stream()

    provider = models.get_provider("deepseek")
    assert provider is not None
    provider.stream_simple = _capture  # type: ignore[method-assign]

    options = SimpleStreamOptions(
        api_key="test-key",
        session_id="session-1",
        thinking_budgets=ThinkingBudgets(low=32),
        transport="sse",
        max_retry_delay_ms=500,
    )
    await models.stream_simple(model, Context(messages=[]), options).result()

    assert captured[0].session_id == "session-1"
    assert captured[0].thinking_budgets == ThinkingBudgets(low=32)
    assert captured[0].transport == "sse"
    assert captured[0].max_retry_delay_ms == 500
    assert captured[0].api_key == "test-key"


def _mid_convo_models() -> tuple[Any, Model]:
    from omh.llm.api.openai_completions import openai_completions_api
    from omh.llm.types import OpenAICompletionsCompat

    provider = create_provider(
        CreateProviderOptions(
            id="midconvo",
            auth=ProviderAuth(api_key=env_api_key_auth("Midconvo API key", ("MIDCONVO_API_KEY",))),
            models=[
                Model(
                    id="midconvo-model",
                    name="Midconvo",
                    api="openai-completions",
                    provider="midconvo",
                    base_url="https://example.invalid",
                    reasoning=False,
                    input=("text",),
                    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                    context_window=1000,
                    max_tokens=100,
                    compat=OpenAICompletionsCompat(supports_mid_convo_system_messages=True),
                )
            ],
            api=openai_completions_api(),
        )
    )
    models = create_models()
    models.set_provider(provider)
    model = models.get_model("midconvo", "midconvo-model")
    assert model is not None
    return models, model


@pytest.mark.asyncio
async def test_mid_conversation_system_messages_are_retained_when_supported() -> None:
    models, model = _mid_convo_models()
    fetch = RecordingFetch(sse_response({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}))

    context = Context(
        messages=[
            SystemMessage(content="Base", timestamp=0, sections={"style": "concise"}),
            UserMessage(content="Hello", timestamp=1),
            SystemMessage(content="Added later", timestamp=2, sections={"style": None}),
        ]
    )

    await models.stream_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch)).result()

    system_messages = [message for message in fetch.body["messages"] if message["role"] == "system"]
    assert len(system_messages) == 2
    assert system_messages[0]["content"] == "Base\n\nconcise"
    assert system_messages[1]["content"] == 'Added later\n\nRemoved system prompt section "style".'
    assert [message["role"] for message in fetch.body["messages"]] == ["system", "user", "system"]


@pytest.mark.asyncio
async def test_on_response_fires_for_http_error_responses() -> None:
    from .http_samples import json_error_response

    models, model = _deepseek()
    fetch = RecordingFetch(json_error_response(401, {"error": "unauthorized"}))
    responses: list[ProviderResponse] = []

    result = await models.stream_simple(
        model,
        Context(messages=[]),
        SimpleStreamOptions(
            api_key="test-key",
            fetch=fetch,
            on_response=lambda response, request_model: responses.append(response),
        ),
    ).result()

    assert result.stop_reason == "error"
    assert len(responses) == 1
    assert responses[0].status == 401
