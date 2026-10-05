"""Offline behavior tests for the OpenCode Go Chat Completions route.

The tests drive the public ``Models`` registry, the built-in
``opencode_go_provider`` catalog, and the injectable HTTP transport. They never
send a real request and do not touch private provider seams.
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
    CompactionHistoryEntry,
    CompactionSettings,
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
    Usage,
    UsageCost,
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


def test_opencode_go_registers_the_two_deepseek_completions_models() -> None:
    models = _models()
    provider = models.get_provider("opencode-go")
    assert provider is not None
    assert provider.base_url == GO_BASE_URL

    flash = _model(models, "deepseek-v4.1-flash")
    pro = _model(models, "deepseek-v4-pro")

    for model in (flash, pro):
        assert model.provider == "opencode-go"
        assert model.api == "openai-completions"
        assert model.base_url == GO_BASE_URL
        assert model.reasoning is True

    assert flash.input == ("text", "image")
    assert pro.input == ("text",)
    assert get_supported_thinking_levels(flash) == ["low", "high", "max"]
    assert get_supported_thinking_levels(pro) == ["high", "max"]


@pytest.mark.asyncio
async def test_request_uses_go_route_max_tokens_and_own_user_agent() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)], system_prompt="Be terse"),
        SimpleStreamOptions(api_key="test-key", fetch=fetch, reasoning="high"),
    )

    request = fetch.requests[0]
    assert request.url == f"{GO_BASE_URL}/chat/completions"
    assert request.headers["user-agent"].startswith("omh (")
    assert request.json_body["model"] == "deepseek-v4.1-flash"
    assert request.json_body["stream"] is True
    assert request.json_body["max_tokens"] == model.max_tokens
    assert "max_completion_tokens" not in request.json_body
    assert "store" not in request.json_body
    assert request.json_body["messages"][0] == {"role": "system", "content": "Be terse"}
    assert request.json_body["reasoning_effort"] == "high"
    assert "thinking" not in request.json_body


@pytest.mark.asyncio
async def test_pro_sends_only_its_supported_thinking_levels() -> None:
    models = _models()
    pro = _model(models, "deepseek-v4-pro")
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(
        pro,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch, reasoning="max"),
    )
    assert fetch.requests[0].json_body["reasoning_effort"] == "max"

    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        pro,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch, reasoning="low"),
    )
    body = fetch.requests[0].json_body
    assert body["reasoning_effort"] == "high"
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_session_id_becomes_the_per_conversation_header_on_every_path() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    context = Context(messages=[UserMessage(content="Hi", timestamp=1)])

    simple_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        model, context, SimpleStreamOptions(api_key="test-key", fetch=simple_fetch, session_id="conversation-1")
    )
    assert simple_fetch.requests[0].headers["x-opencode-session"] == "conversation-1"

    full_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete(
        model, context, SimpleStreamOptions(api_key="test-key", fetch=full_fetch, session_id="conversation-1")
    )
    assert full_fetch.requests[0].headers["x-opencode-session"] == "conversation-1"


@pytest.mark.asyncio
async def test_explicit_session_header_override_is_respected() -> None:
    models = _models()
    base = _model(models, "deepseek-v4.1-flash")
    context = Context(messages=[UserMessage(content="Hi", timestamp=1)])

    model_header_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        replace(base, headers={"x-opencode-session": "from-model"}),
        context,
        SimpleStreamOptions(api_key="test-key", fetch=model_header_fetch, session_id="conversation-1"),
    )
    assert model_header_fetch.requests[0].headers["x-opencode-session"] == "from-model"

    options_header_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        base,
        context,
        SimpleStreamOptions(
            api_key="test-key",
            fetch=options_header_fetch,
            session_id="conversation-1",
            headers={"x-opencode-session": "explicit"},
        ),
    )
    assert options_header_fetch.requests[0].headers["x-opencode-session"] == "explicit"


@pytest.mark.asyncio
async def test_session_header_is_absent_without_a_session_id() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        model, Context(messages=[UserMessage(content="Hi", timestamp=1)]), SimpleStreamOptions(api_key="test-key", fetch=fetch)
    )
    assert "x-opencode-session" not in fetch.requests[0].headers


@pytest.mark.asyncio
async def test_reasoning_field_is_normalized_to_reasoning_content() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    fetch = RecordingFetch(
        sse_response(
            {"choices": [{"index": 0, "delta": {"reasoning": "plan"}}]},
            {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]},
        )
    )

    events, result = await _collect(
        models.stream_simple(
            model,
            Context(messages=[UserMessage(content="Think then answer", timestamp=1)]),
            SimpleStreamOptions(api_key="test-key", fetch=fetch, reasoning="high"),
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
    assert fetch.body["reasoning_effort"] == "high"
    assert "thinking" not in fetch.body


@pytest.mark.asyncio
async def test_tool_continuation_replays_reasoning_content_and_tool_results() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    context = Context(
        messages=[
            UserMessage(content="Read the note", timestamp=1),
            AssistantMessage(
                api="openai-completions",
                provider="opencode-go",
                model="deepseek-v4.1-flash",
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

    await models.complete_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch))

    messages = fetch.body["messages"]
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "plan"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "read"
    tool = next(message for message in messages if message["role"] == "tool")
    assert tool["tool_call_id"] == "call_1"
    assert tool["content"] == "file body"


@pytest.mark.asyncio
async def test_legacy_reasoning_signature_is_replayed_as_reasoning_content() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    context = Context(
        messages=[
            UserMessage(content="Continue", timestamp=1),
            AssistantMessage(
                api="openai-completions",
                provider="opencode-go",
                model="deepseek-v4.1-flash",
                usage=empty_usage(),
                stop_reason="toolUse",
                timestamp=2,
                content=[
                    ThinkingContent(thinking="earlier plan", thinking_signature="reasoning"),
                    ToolCall(id="call_9", name="read", arguments={"path": "note.txt"}),
                ],
            ),
            ToolResultMessage(
                tool_call_id="call_9",
                tool_name="read",
                content=[TextContent(text="body")],
                timestamp=3,
            ),
        ]
    )
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(model, context, SimpleStreamOptions(api_key="test-key", fetch=fetch))

    assistant = next(message for message in fetch.body["messages"] if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "earlier plan"
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_flash_projects_image_content_and_pro_keeps_text_capability() -> None:
    models = _models()
    context = Context(
        messages=[
            UserMessage(
                content=[TextContent(text="look"), ImageContent(data="QQ==", mime_type="image/png")],
                timestamp=1,
            )
        ]
    )

    flash_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "seen"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        _model(models, "deepseek-v4.1-flash"), context, SimpleStreamOptions(api_key="test-key", fetch=flash_fetch)
    )
    flash_content = flash_fetch.body["messages"][0]["content"]
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}} in flash_content

    pro_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "read"}, "finish_reason": "stop"}]})
    )
    await models.complete_simple(
        _model(models, "deepseek-v4-pro"), context, SimpleStreamOptions(api_key="test-key", fetch=pro_fetch)
    )
    pro_content = pro_fetch.body["messages"][0]["content"]
    assert all(part["type"] != "image_url" for part in pro_content)
    assert any("does not support images" in part["text"] for part in pro_content)


@pytest.mark.asyncio
async def test_usage_reports_reasoning_as_an_output_subset() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
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
    assert result.usage.cost.output == pytest.approx(1.20 / 1_000_000 * 20)
    assert result.usage.cost.total > 0


@pytest.mark.asyncio
async def test_http_error_stays_a_provider_error() -> None:
    models = _models()
    model = _model(models, "deepseek-v4-pro")
    fetch = RecordingFetch(json_error_response(401, {"error": {"message": "bad key"}}))

    result = await models.complete_simple(
        model, Context(messages=[UserMessage(content="Hi", timestamp=1)]), SimpleStreamOptions(api_key="nope", fetch=fetch)
    )

    assert result.stop_reason == "error"
    assert "401" in (result.error_message or "")
    assert "bad key" in (result.error_message or "")


@pytest.mark.asyncio
async def test_cancel_marks_the_go_stream_aborted() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    controller = AbortController()
    response = sse_response(
        {"choices": [{"index": 0, "delta": {"content": "never"}, "finish_reason": "stop"}]}
    )

    async def aborting_fetch(request: Any) -> Any:
        controller.abort()
        return response

    events, result = await _collect(
        models.stream_simple(
            model,
            Context(messages=[UserMessage(content="Hi", timestamp=1)]),
            SimpleStreamOptions(api_key="test-key", fetch=aborting_fetch, signal=controller.signal),
        )
    )

    assert events[0].type == "start"
    assert events[-1].type == "error"
    assert result.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_explicit_output_cap_is_sent_as_max_tokens() -> None:
    models = _models()
    model = _model(models, "deepseek-v4-pro")
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=1)]),
        SimpleStreamOptions(api_key="test-key", fetch=fetch, max_tokens=4096),
    )

    assert fetch.body["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_unsupported_explicit_effort_is_omitted_not_fabricated() -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    context = Context(messages=[UserMessage(content="Hi", timestamp=1)])

    unsupported_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete(
        model,
        context,
        OpenAICompletionsOptions(api_key="test-key", fetch=unsupported_fetch, reasoning_effort="medium"),
    )
    assert "reasoning_effort" not in unsupported_fetch.body

    supported_fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    await models.complete(
        model,
        context,
        OpenAICompletionsOptions(api_key="test-key", fetch=supported_fetch, reasoning_effort="max"),
    )
    assert supported_fetch.body["reasoning_effort"] == "max"


@pytest.mark.asyncio
async def test_opencode_api_key_env_is_the_auth_entry() -> None:
    class EnvAuthContext:
        async def env(self, name: str) -> str | None:
            return "from-env" if name == "OPENCODE_API_KEY" else None

        async def file_exists(self, path: str) -> bool:
            return False

    from omh.llm.models import CreateModelsOptions

    models = create_models(CreateModelsOptions(auth_context=EnvAuthContext()))
    models.set_provider(opencode_go_provider())
    model = _model(models, "deepseek-v4-pro")
    fetch = RecordingFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )

    await models.complete_simple(model, Context(messages=[UserMessage(content="Hi", timestamp=1)]), SimpleStreamOptions(fetch=fetch))

    assert fetch.requests[0].headers["authorization"] == "Bearer from-env"


@pytest.mark.asyncio
async def test_agent_tool_roundtrip_replays_go_reasoning_and_tool_results(tmp_path: Path) -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    (tmp_path / "note.txt").write_text("hello from disk\n", encoding="utf-8")

    tool_call_response = sse_response(
        {"choices": [{"index": 0, "delta": {"reasoning": "checking the note"}}]},
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
                                "function": {"name": "read", "arguments": ""},
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
                                "function": {"arguments": '{"path": "note.txt"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    final_response = sse_response(
        {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
    )
    fetch = SequencedFetch(tool_call_response, final_response)

    def stream_fn(model: Model, context: Any, options: SimpleStreamOptions | None) -> Any:
        assert options is not None
        return models.stream_simple(model, Context(messages=list(context.messages)), replace(options, fetch=fetch))

    agent = Agent(
        AgentOptions(
            initial_state=AgentInitialState(model=model, tools=[create_read_tool(tmp_path)], system_prompt="Base"),
            stream_fn=stream_fn,
            session_id="conversation-42",
            api_key="test-key",
        )
    )

    await agent.prompt("Read the note")

    assert [request.headers["x-opencode-session"] for request in fetch.requests] == [
        "conversation-42",
        "conversation-42",
    ]
    results = [message for message in agent.state.messages if message.role == "toolResult"]
    assert [message.content[0].text for message in results] == ["hello from disk\n"]

    continuation = fetch.bodies[1]["messages"]
    assistant = next(message for message in continuation if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "checking the note"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    tool = next(message for message in continuation if message["role"] == "tool")
    assert tool["tool_call_id"] == "call_1"
    assert "hello from disk" in tool["content"]

    final = agent.state.messages[-1]
    assert final.role == "assistant"
    assert final.content[0].text == "done"


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
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": json.dumps(arguments)}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_agent_drives_write_read_edit_bash_flow_through_go(tmp_path: Path) -> None:
    models = _models()
    model = _model(models, "deepseek-v4.1-flash")
    fetch = SequencedFetch(
        _tool_call_response("w1", "write", {"path": "note.txt", "content": "alpha\n"}),
        _tool_call_response("r1", "read", {"path": "note.txt"}),
        _tool_call_response("e1", "edit", {"path": "note.txt", "edits": [{"oldText": "alpha", "newText": "beta"}]}),
        _tool_call_response("b1", "bash", {"command": "cat note.txt"}),
        sse_response({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}),
    )

    def stream_fn(request_model: Model, context: Any, options: SimpleStreamOptions | None) -> Any:
        assert options is not None
        return models.stream_simple(request_model, Context(messages=list(context.messages)), replace(options, fetch=fetch))

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
            ),
            stream_fn=stream_fn,
            session_id="conversation-tools",
            api_key="test-key",
        )
    )

    await agent.prompt("Write, read, edit, then cat the note")

    assert len(fetch.requests) == 5
    results = [message for message in agent.state.messages if message.role == "toolResult"]
    assert [message.tool_name for message in results] == ["write", "read", "edit", "bash"]
    assert [message.is_error for message in results] == [False, False, False, False]
    assert results[1].content[0].text == "alpha\n"
    assert "beta" in results[3].content[0].text

    continuation = fetch.bodies[4]["messages"]
    assert [message["tool_call_id"] for message in continuation if message["role"] == "tool"] == [
        "w1",
        "r1",
        "e1",
        "b1",
    ]

    final = agent.state.messages[-1]
    assert final.role == "assistant"
    assert final.content[0].text == "done"


@pytest.mark.asyncio
async def test_compaction_summary_request_carries_the_go_session_header() -> None:
    models = _models()
    model = replace(_model(models, "deepseek-v4.1-flash"), context_window=1000)
    usage = Usage(input=900, output=0, cache_read=0, cache_write=0, total_tokens=900, cost=UsageCost())
    fetch = SequencedFetch(
        sse_response({"choices": [{"index": 0, "delta": {"content": "summary"}, "finish_reason": "stop"}]}),
        sse_response({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}),
    )

    def stream_fn(request_model: Model, context: Any, options: SimpleStreamOptions | None) -> Any:
        assert options is not None
        return models.stream_simple(request_model, Context(messages=list(context.messages)), replace(options, fetch=fetch))

    agent = Agent(
        AgentOptions(
            initial_state=AgentInitialState(
                model=model,
                messages=[
                    UserMessage(content="old question", timestamp=1),
                    replace(
                        AssistantMessage(
                            api=model.api,
                            provider=model.provider,
                            model=model.id,
                            usage=usage,
                            stop_reason="stop",
                            timestamp=2,
                            content=[TextContent(text="old answer")],
                        )
                    ),
                    UserMessage(content="tail", timestamp=3),
                ],
            ),
            stream_fn=stream_fn,
            session_id="conversation-summary",
            api_key="test-key",
            compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
        )
    )

    await agent.prompt("new question")

    assert len(fetch.requests) == 2
    assert [request.headers["x-opencode-session"] for request in fetch.requests] == [
        "conversation-summary",
        "conversation-summary",
    ]
    assert any(isinstance(entry, CompactionHistoryEntry) for entry in agent.history.entries)
