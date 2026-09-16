from __future__ import annotations

from typing import Any

import pytest

from omh.llm.models import CreateModelsOptions, create_models
from omh.llm.providers.deepseek import deepseek_provider
from omh.llm.types import Context, SimpleStreamOptions, Tool, UserMessage

from .http_samples import RecordingFetch, json_error_response, sse_response

CALCULATOR = Tool(
    name="math_operation",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
        },
        "required": ["a", "b", "operation"],
    },
)


def _models():
    models = create_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None
    return models, model


async def _collect(stream: Any) -> tuple[list[Any], Any]:
    events = [event async for event in stream]
    return events, await stream.result()


def test_deepseek_catalog_is_configurable() -> None:
    models, model = _models()
    assert models.get_provider("deepseek") is not None
    assert model.id == "deepseek-flash"
    assert model.api == "openai-completions"
    assert model.provider == "deepseek"
    assert model.base_url == "https://api.deepseek.com"


@pytest.mark.asyncio
async def test_text_stream_and_final_result_from_http_sample() -> None:
    models, model = _models()
    fetch = RecordingFetch(
        sse_response(
            {
                "id": "chatcmpl-text",
                "model": "deepseek-flash",
                "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            },
            {
                "id": "chatcmpl-text",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "prompt_cache_hit_tokens": 4,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
        )
    )
    context = Context(messages=[UserMessage(content="Say hello", timestamp=1)])
    events, result = await _collect(
        models.stream_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch))
    )

    types = [event.type for event in events]
    assert types == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
    assert result.stop_reason == "stop"
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hello world"
    assert result.response_id == "chatcmpl-text"
    assert result.usage.input == 8
    assert result.usage.output == 3
    assert result.usage.cache_read == 4
    assert result.usage.reasoning == 1
    assert result.usage.total_tokens == 15
    assert fetch.body["model"] == "deepseek-flash"
    assert fetch.body["stream"] is True
    assert fetch.body["max_tokens"] == model.max_tokens
    assert "max_completion_tokens" not in fetch.body
    assert fetch.requests[0].url == "https://api.deepseek.com/chat/completions"
    assert fetch.requests[0].headers["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_thinking_stream_uses_deepseek_reasoning_payload() -> None:
    models, model = _models()
    fetch = RecordingFetch(
        sse_response(
            {
                "choices": [{"index": 0, "delta": {"reasoning_content": "plan"}}],
            },
            {
                "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}],
            },
        )
    )
    context = Context(messages=[UserMessage(content="Think then answer", timestamp=1)])
    events, result = await _collect(
        models.stream_simple(
            model,
            context,
            SimpleStreamOptions(api_key="test-key", fetch=fetch, reasoning="high"),
        )
    )

    types = [event.type for event in events]
    assert types == [
        "start",
        "thinking_start",
        "thinking_delta",
        "text_start",
        "text_delta",
        "thinking_end",
        "text_end",
        "done",
    ]
    assert result.content[0].type == "thinking"
    assert result.content[0].thinking == "plan"
    assert result.content[1].type == "text"
    assert result.content[1].text == "done"
    assert fetch.body["thinking"] == {"type": "enabled"}
    assert fetch.body["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_tool_call_stream_and_stop_reason() -> None:
    models, model = _models()
    fetch = RecordingFetch(
        sse_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "math_operation", "arguments": ""},
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"a": 15, "b": 27, "operation": "add"}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )
    )
    context = Context(
        messages=[UserMessage(content="Add 15 and 27", timestamp=1)],
        tools=[CALCULATOR],
    )
    events, result = await _collect(
        models.stream_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch))
    )

    types = [event.type for event in events]
    assert types[0] == "start"
    assert "toolcall_start" in types
    assert "toolcall_delta" in types
    assert types[-2] == "toolcall_end"
    assert types[-1] == "done"
    assert result.stop_reason == "toolUse"
    tool_call = result.content[0]
    assert tool_call.type == "toolCall"
    assert tool_call.id == "call_1"
    assert tool_call.name == "math_operation"
    assert tool_call.arguments == {"a": 15, "b": 27, "operation": "add"}
    assert fetch.body["tools"][0]["function"]["name"] == "math_operation"


@pytest.mark.asyncio
async def test_complete_returns_final_assistant_message() -> None:
    models, model = _models()
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
        )
    )
    result = await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch),
    )
    assert result.role == "assistant"
    assert result.stop_reason == "stop"
    assert result.content[0].text == "ok"


@pytest.mark.asyncio
async def test_replayed_assistant_includes_empty_reasoning_content() -> None:
    models, model = _models()
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"index": 0, "delta": {"content": "next"}, "finish_reason": "stop"}]},
        )
    )
    from omh.llm.types import AssistantMessage, TextContent, empty_usage

    context = Context(
        messages=[
            UserMessage(content="first", timestamp=1),
            AssistantMessage(
                content=[TextContent(text="hello")],
                api="openai-completions",
                provider="deepseek",
                model="deepseek-flash",
                usage=empty_usage(),
                stop_reason="stop",
                timestamp=2,
            ),
            UserMessage(content="second", timestamp=3),
        ]
    )
    await models.complete_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch))
    assistant = fetch.body["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "hello"
    assert assistant["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_image_content_is_sent_as_data_url() -> None:
    models, model = _models()
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"index": 0, "delta": {"content": "see"}, "finish_reason": "stop"}]},
        )
    )
    from omh.llm.types import ImageContent, TextContent

    context = Context(
        messages=[
            UserMessage(
                content=[
                    TextContent(text="look"),
                    ImageContent(data="QQ==", mime_type="image/png"),
                ],
                timestamp=1,
            )
        ]
    )
    await models.complete_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch))
    user = fetch.body["messages"][0]
    assert user["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QQ=="},
    }


@pytest.mark.asyncio
async def test_missing_auth_becomes_stream_error() -> None:
    class EmptyAuthContext:
        async def env(self, name: str) -> str | None:
            return None

        async def file_exists(self, path: str) -> bool:
            return False

    models = create_models(CreateModelsOptions(auth_context=EmptyAuthContext()))
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None
    events, result = await _collect(
        models.stream_simple(
            model,
            Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        )
    )
    assert events[0].type == "error"
    assert result.stop_reason == "error"
    assert result.error_message is not None


@pytest.mark.asyncio
async def test_env_api_key_is_used_when_request_omits_api_key() -> None:
    class EnvAuthContext:
        async def env(self, name: str) -> str | None:
            return "from-env" if name == "DEEPSEEK_API_KEY" else None

        async def file_exists(self, path: str) -> bool:
            return False

    models = create_models(CreateModelsOptions(auth_context=EnvAuthContext()))
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
        )
    )
    await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(fetch=fetch),
    )
    assert fetch.requests[0].headers["authorization"] == "Bearer from-env"


@pytest.mark.asyncio
async def test_http_error_body_is_surfaced() -> None:
    models, model = _models()
    fetch = RecordingFetch(json_error_response(401, {"error": {"message": "bad key"}}))
    result = await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch),
    )
    assert result.stop_reason == "error"
    assert "401" in (result.error_message or "")
    assert "bad key" in (result.error_message or "")
