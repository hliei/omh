"""Offline behavior tests for the OpenCode Go GLM/Kimi Completions models.

The tests drive the public ``Models`` registry, the built-in
``opencode_go_provider`` catalog, the injectable HTTP transport and the public
``Agent``. They never send a real request and do not touch private provider
seams.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from omh.llm.models import create_models, get_supported_thinking_levels
from omh.llm.providers.opencode_go import opencode_go_provider
from omh.llm.types import (
    AbortController,
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    OpenAICompletionsOptions,
    SimpleStreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    empty_usage,
)

from .http_samples import (
    RecordingFetch,
    SequencedFetch,
    json_error_response,
    sse_response,
)

GO_BASE_URL = "https://opencode.ai/zen/go/v1"

#: The four GLM/Kimi models this ticket adds to the Go route.
GLM_KIMI_MODEL_IDS = ("glm-5.3", "glm-5.3-flash", "kimi-k3", "kimi-k2.7-code")


def _models() -> Any:
    models = create_models()
    models.set_provider(opencode_go_provider())
    return models


def _model(models: Any, model_id: str) -> Model:
    model = models.get_model("opencode-go", model_id)
    assert model is not None
    return model


async def _collect(stream: Any) -> tuple[list[Any], Any]:
    events = [event async for event in stream]
    return events, await stream.result()


def _tool_call_response(tool_call_id: str, name: str, arguments: dict[str, object]) -> Any:
    return sse_response(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
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
                        "tool_calls": [{"index": 0, "function": {"arguments": json.dumps(arguments)}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )


def test_go_catalog_registers_the_four_glm_kimi_models_with_their_own_capabilities() -> None:
    models = _models()

    glm = _model(models, "glm-5.3")
    glm_flash = _model(models, "glm-5.3-flash")
    kimi_k3 = _model(models, "kimi-k3")
    kimi_code = _model(models, "kimi-k2.7-code")

    for model in (glm, glm_flash, kimi_k3, kimi_code):
        assert model.provider == "opencode-go"
        assert model.api == "openai-completions"
        assert model.base_url == GO_BASE_URL
        assert model.reasoning is True

    assert glm.input == ("text",)
    assert glm_flash.input == ("text", "image")
    assert kimi_k3.input == ("text", "image")
    assert kimi_code.input == ("text", "image")

    assert get_supported_thinking_levels(glm) == ["low", "high", "max"]
    assert get_supported_thinking_levels(glm_flash) == ["low", "high", "max"]
    assert get_supported_thinking_levels(kimi_k3) == ["max"]
    # Fixed-on with no adjustable tier: an empty set, not a fabricated ``off``.
    assert get_supported_thinking_levels(kimi_code) == []

    assert (glm.context_window, glm.max_tokens) == (1_000_000, 131_072)
    assert (glm_flash.context_window, glm_flash.max_tokens) == (1_000_000, 131_072)
    assert (kimi_k3.context_window, kimi_k3.max_tokens) == (1_048_576, 131_072)
    assert (kimi_code.context_window, kimi_code.max_tokens) == (262_144, 262_144)

    assert (glm.cost.input, glm.cost.output) == (1.40, 4.40)
    assert (glm_flash.cost.input, glm_flash.cost.output) == (0.15, 0.50)
    assert (kimi_k3.cost.input, kimi_k3.cost.output) == (3.00, 15.00)
    assert (kimi_code.cost.input, kimi_code.cost.output) == (0.95, 4.00)


@pytest.mark.parametrize(
    ("model_id", "reasoning", "expected_effort"),
    [
        ("glm-5.3", "low", "low"),
        ("glm-5.3", "high", "high"),
        ("glm-5.3", "max", "max"),
        ("glm-5.3", "medium", "high"),
        ("glm-5.3-flash", "low", "low"),
        ("glm-5.3-flash", "high", "high"),
        ("glm-5.3-flash", "max", "max"),
        ("kimi-k3", "max", "max"),
        ("kimi-k3", "high", "max"),
    ],
)
async def test_adjustable_go_models_send_only_their_effective_effort(
    model_id: str, reasoning: str, expected_effort: str,
) -> None:
    models = _models()
    model = _model(models, model_id)
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch, reasoning=reasoning),  # type: ignore[arg-type]
    )

    body = fetch.body
    assert body["model"] == model_id
    assert body["max_tokens"] == model.max_tokens
    assert "max_completion_tokens" not in body
    assert "store" not in body
    assert "thinking" not in body
    assert body["reasoning_effort"] == expected_effort
    assert fetch.requests[0].headers["user-agent"].startswith("omh (")


@pytest.mark.parametrize(
    ("model_id", "requested"),
    [
        ("glm-5.3", "medium"),
        ("glm-5.3", "minimal"),
        ("glm-5.3", "xhigh"),
        ("glm-5.3-flash", "medium"),
        ("kimi-k3", "low"),
        ("kimi-k3", "high"),
    ],
)
async def test_unsupported_go_effort_is_omitted_not_fabricated(model_id: str, requested: str) -> None:
    models = _models()
    model = _model(models, model_id)
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        OpenAICompletionsOptions(api_key="test-key", fetch=fetch, reasoning_effort=requested),  # type: ignore[arg-type]
    )

    assert "reasoning_effort" not in fetch.body
    assert "thinking" not in fetch.body


@pytest.mark.parametrize("model_id", ["glm-5.3", "glm-5.3-flash", "kimi-k3", "kimi-k2.7-code"])
async def test_go_glm_kimi_never_send_the_deepseek_thinking_object(model_id: str) -> None:
    models = _models()
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete(
        _model(models, model_id),
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        OpenAICompletionsOptions(api_key="test-key", fetch=fetch, reasoning_effort="max"),
    )

    assert "thinking" not in fetch.body


async def test_kimi_k2_7_code_is_fixed_on_without_any_effort_field() -> None:
    models = _models()
    model = _model(models, "kimi-k2.7-code")
    context = Context(messages=[UserMessage(content="Hi", timestamp=1)])

    simple_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        model, context, SimpleStreamOptions(api_key="test-key", fetch=simple_fetch, reasoning="max")
    )
    assert "reasoning_effort" not in simple_fetch.body
    assert "thinking" not in simple_fetch.body

    full_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete(
        model, context, OpenAICompletionsOptions(api_key="test-key", fetch=full_fetch, reasoning_effort="max")
    )
    assert "reasoning_effort" not in full_fetch.body
    assert "thinking" not in full_fetch.body


@pytest.mark.parametrize(
    ("model_id", "expects_image"),
    [
        ("glm-5.3", False),
        ("glm-5.3-flash", True),
        ("kimi-k3", True),
        ("kimi-k2.7-code", True),
    ],
)
async def test_go_glm_kimi_project_the_declared_input_modality(model_id: str, expects_image: bool) -> None:
    models = _models()
    context = Context(
        messages=[
            UserMessage(
                content=[TextContent(text="look"), ImageContent(data="QQ==", mime_type="image/png")],
                timestamp=1,
            )
        ]
    )
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "seen"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(_model(models, model_id), context, SimpleStreamOptions(api_key="test-key", fetch=fetch))

    content = fetch.body["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    if expects_image:
        assert image_parts == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}}
        ]
    else:
        assert image_parts == []
        assert any("does not support images" in part["text"] for part in content)

    # The text-only GLM model still routes to the same Go service; it never
    # silently switches provider or model.
    assert fetch.body["model"] == model_id


@pytest.mark.parametrize(
    ("model_id", "expects_image"),
    [
        ("glm-5.3", False),
        ("glm-5.3-flash", True),
        ("kimi-k3", True),
        ("kimi-k2.7-code", True),
    ],
)
async def test_go_tool_result_images_follow_the_declared_modality(model_id: str, expects_image: bool) -> None:
    models = _models()
    context = Context(
        messages=[
            UserMessage(content="Read the image", timestamp=1),
            AssistantMessage(
                api="openai-completions",
                provider="opencode-go",
                model=model_id,
                usage=empty_usage(),
                stop_reason="toolUse",
                timestamp=2,
                content=[ToolCall(id="call_1", name="read", arguments={"path": "shot.png"})],
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[ImageContent(data="QQ==", mime_type="image/png")],
                timestamp=3,
            ),
        ]
    )
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "seen"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(_model(models, model_id), context, SimpleStreamOptions(api_key="test-key", fetch=fetch))

    messages = fetch.body["messages"]
    image_parts = [
        part
        for message in messages
        if message["role"] == "user" and isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    tool = next(message for message in messages if message["role"] == "tool")
    if expects_image:
        assert image_parts == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}}
        ]
        assert tool["content"] == "(see attached image)"
    else:
        assert image_parts == []
        assert "does not support images" in tool["content"]


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_go_reasoning_stream_is_normalized_for_every_glm_kimi_model(model_id: str) -> None:
    models = _models()
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"index": 0, "delta": {"reasoning": "plan"}}]},
            {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]},
        )
    )

    events, result = await _collect(
        models.stream_simple(
            _model(models, model_id),
            Context(messages=[UserMessage(content="Think then answer", timestamp=1)]),
            SimpleStreamOptions(api_key="test-key", fetch=fetch),
        )
    )

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "text_start",
        "text_delta",
        "thinking_end",
        "text_end",
        "done",
    ]
    thinking = result.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.thinking == "plan"
    assert thinking.thinking_signature == "reasoning_content"


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_go_tool_continuation_replays_reasoning_and_tool_results(model_id: str) -> None:
    models = _models()
    context = Context(
        messages=[
            UserMessage(content="Read the note", timestamp=1),
            AssistantMessage(
                api="openai-completions",
                provider="opencode-go",
                model=model_id,
                usage=empty_usage(),
                stop_reason="toolUse",
                timestamp=2,
                content=[
                    ThinkingContent(thinking="plan", thinking_signature="reasoning_content"),
                    ToolCall(id="call_1", name="read", arguments={"path": "note.txt"}),
                ],
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[TextContent(text="file body")],
                timestamp=3,
            ),
        ]
    )
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(_model(models, model_id), context, SimpleStreamOptions(api_key="test-key", fetch=fetch))

    messages = fetch.body["messages"]
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "plan"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    tool = next(message for message in messages if message["role"] == "tool")
    assert tool["tool_call_id"] == "call_1"
    assert tool["content"] == "file body"


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_agent_drives_the_four_tools_for_every_glm_kimi_model(tmp_path: Path, model_id: str) -> None:
    models = _models()
    model = _model(models, model_id)
    # The adjustable models select an explicit tier; the fixed Kimi model keeps
    # no adjustable selection and must still omit the effort field.
    thinking_level = None if model_id == "kimi-k2.7-code" else "high"
    expected_effort = {
        "glm-5.3": "high",
        "glm-5.3-flash": "high",
        "kimi-k3": "max",
        "kimi-k2.7-code": None,
    }[model_id]
    fetch = SequencedFetch(
        _tool_call_response("w1", "write", {"path": "note.txt", "content": "alpha\n"}),
        _tool_call_response("r1", "read", {"path": "note.txt"}),
        _tool_call_response("e1", "edit", {"path": "note.txt", "edits": [{"oldText": "alpha", "newText": "beta"}]}),
        _tool_call_response("b1", "bash", {"command": "cat note.txt"}),
        sse_response({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}),
    )

    def stream_fn(request_model: Model, context: Any, options: SimpleStreamOptions | None) -> Any:
        assert options is not None
        return models.stream_simple(
            request_model, Context(messages=list(context.messages)), replace(options, fetch=fetch)
        )

    agent = Agent(
        AgentOptions(
            initial_state=AgentInitialState(
                model=model,
                tools=[
                    create_write_tool(tmp_path),
                    create_read_tool(tmp_path),
                    create_edit_tool(tmp_path),
                    create_bash_tool(tmp_path),
                ],
                system_prompt="Base",
                thinking_level=thinking_level,  # type: ignore[arg-type]
            ),
            stream_fn=stream_fn,
            session_id="conversation-tools",
            api_key="test-key",
        )
    )

    await agent.prompt("Write, read, edit, then cat the note")

    assert len(fetch.requests) == 5
    assert [request.headers["x-opencode-session"] for request in fetch.requests] == [
        "conversation-tools"
    ] * 5
    results = [message for message in agent.state.messages if message.role == "toolResult"]
    assert [message.tool_name for message in results] == ["write", "read", "edit", "bash"]
    assert [message.is_error for message in results] == [False, False, False, False]
    assert results[1].content[0].text == "alpha\n"
    assert "beta" in results[3].content[0].text

    final = agent.state.messages[-1]
    assert final.role == "assistant"
    assert final.content[0].text == "done"

    # The Agent's effective thinking selection reaches every request on this
    # route without inventing an unsupported field.
    for request in fetch.requests:
        assert request.json_body["model"] == model_id
        assert "thinking" not in request.json_body
        if expected_effort is None:
            assert "reasoning_effort" not in request.json_body
        else:
            assert request.json_body["reasoning_effort"] == expected_effort


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_go_glm_kimi_report_reasoning_usage_and_cost(model_id: str) -> None:
    models = _models()
    model = _model(models, model_id)
    fetch = RecordingFetch(
        sse_response(
            {
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 10},
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            }
        )
    )

    result = await models.complete_simple(
        model, Context(messages=[UserMessage(content="Hi", timestamp=1)]), SimpleStreamOptions(api_key="test-key", fetch=fetch)
    )

    assert result.usage.input == 90
    assert result.usage.output == 20
    assert result.usage.cache_read == 10
    assert result.usage.reasoning == 8
    assert result.usage.reasoning <= result.usage.output
    assert result.usage.total_tokens == 120
    assert result.usage.cost.output == pytest.approx(model.cost.output / 1_000_000 * 20)
    assert result.usage.reported is True


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_go_glm_kimi_http_error_stays_a_provider_error(model_id: str) -> None:
    models = _models()
    fetch = RecordingFetch(json_error_response(401, {"error": {"message": "bad key"}}))

    result = await models.complete_simple(
        _model(models, model_id),
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="nope", fetch=fetch),
    )

    assert result.stop_reason == "error"
    assert "401" in (result.error_message or "")
    assert result.usage.reported is False


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_go_glm_kimi_cancel_marks_the_stream_aborted(model_id: str) -> None:
    models = _models()
    controller = AbortController()
    response = sse_response(
        {"choices": [{"index": 0, "delta": {"content": "never"}, "finish_reason": "stop"}]}
    )

    async def aborting_fetch(request: Any) -> Any:
        controller.abort()
        return response

    events, result = await _collect(
        models.stream_simple(
            _model(models, model_id),
            Context(messages=[UserMessage(content="Hi", timestamp=1)]),
            SimpleStreamOptions(api_key="test-key", fetch=aborting_fetch, signal=controller.signal),
        )
    )

    assert events[0].type == "start"
    assert events[-1].type == "error"
    assert result.stop_reason == "aborted"
    assert result.usage.reported is False


@pytest.mark.parametrize("model_id", GLM_KIMI_MODEL_IDS)
async def test_go_glm_kimi_explicit_output_cap_is_sent_as_max_tokens(model_id: str) -> None:
    models = _models()
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(
        _model(models, model_id),
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch, max_tokens=4096),
    )

    assert fetch.body["max_tokens"] == 4096
