"""Model-integration tests for the traditional Agent: custom messages, transcripts,
tool-declaration replay, default StreamFn, dynamic credentials, and request options."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentMessage,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    clear_default_stream_fn,
    set_default_stream_fn,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
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
    TranscriptContext,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)
from omh.llm.utils.transcript import get_current_tools

NOW = 1_700_000_000_000


def make_model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="test",
        base_url="https://example.invalid",
        reasoning=False,
        input=("text",),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def final_message(text: str = "ok") -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason="stop",
        timestamp=NOW,
        content=[TextContent(text=text)],
    )


@dataclass
class CapturedRequest:
    model: Model
    context: TranscriptContext
    options: SimpleStreamOptions | None


class ScriptedStreamFn:
    def __init__(self, result_text: str = "ok") -> None:
        self.requests: list[CapturedRequest] = []
        self.calls = 0
        self.result_text = result_text

    def __call__(
        self, model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        self.calls += 1
        self.requests.append(CapturedRequest(model=model, context=context, options=options))
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=final_message("")))
        stream.push(DoneEvent(reason="stop", message=final_message(self.result_text)))
        return stream


@pytest.fixture(autouse=True)
def _clear_default_stream_fn() -> None:
    clear_default_stream_fn()
    yield
    clear_default_stream_fn()


@dataclass
class NoticeMessage:
    text: str
    role: str = "notice"
    timestamp: int = NOW


# ---------------------------------------------------------------------------
# A08: custom messages and conversion order
# ---------------------------------------------------------------------------


async def test_custom_messages_stay_in_history_and_default_conversion_filters_them() -> None:
    stream = ScriptedStreamFn()
    agent = Agent(
        AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model(), system_prompt="Base"))
    )

    await agent.prompt([NoticeMessage(text="app state"), UserMessage(content="hello", timestamp=NOW)])

    roles = [message.role for message in agent.state.messages]
    assert "notice" in roles
    request_roles = [message.role for message in stream.requests[0].context.messages]
    assert "notice" not in request_roles
    assert request_roles == ["system", "user"]


async def test_transform_then_convert_then_normalize_order_and_original_history_preserved() -> None:
    stream = ScriptedStreamFn()
    order: list[str] = []

    async def transform_context(
        messages: list[AgentMessage], signal: AbortSignal | None
    ) -> list[AgentMessage]:
        order.append("transform")
        assert any(isinstance(message, NoticeMessage) for message in messages)
        return [message for message in messages if not isinstance(message, NoticeMessage)]

    async def convert_to_llm(messages: list[AgentMessage]) -> list[object]:
        order.append("convert")
        return [message for message in messages if getattr(message, "role", None) != "notice"]  # type: ignore[misc]

    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            transform_context=transform_context,
            convert_to_llm=convert_to_llm,  # type: ignore[arg-type]
            initial_state=AgentInitialState(model=make_model(), system_prompt="Base"),
        )
    )

    await agent.prompt([NoticeMessage(text="app state"), UserMessage(content="hello", timestamp=NOW)])

    assert order == ["transform", "convert"]
    original_roles = [message.role for message in agent.state.messages]
    assert "notice" in original_roles
    request_roles = [message.role for message in stream.requests[0].context.messages]
    assert request_roles == ["system", "user"]


# ---------------------------------------------------------------------------
# A09: system transcript replay reaches a custom StreamFn
# ---------------------------------------------------------------------------


async def test_stream_fn_receives_full_system_transcript_without_early_collapse() -> None:
    stream = ScriptedStreamFn()
    tool_v2 = AgentTool(
        name="run",
        description="v2",
        parameters={"type": "object"},
        label="Run",
        execute=lambda *args: AgentToolResult(content=[TextContent(text="done")], details={}),
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                messages=[
                    SystemMessage(
                        content="Base",
                        timestamp=0,
                        sections={"style": "concise", "tone": "warm"},
                        tools_added=[Tool(name="run", description="v1", parameters={"type": "object"})],
                    ),
                    UserMessage(content="hello", timestamp=NOW),
                    SystemMessage(
                        content="",
                        timestamp=1,
                        sections={"style": None},
                        tools_added=[Tool(name="run", description="v2", parameters={"type": "object"})],
                    ),
                ],
                tools=[tool_v2],
            ),
        )
    )

    await agent.prompt("go")

    request_messages = stream.requests[0].context.messages
    system_messages = [message for message in request_messages if isinstance(message, SystemMessage)]
    assert len(system_messages) == 2
    assert agent.state.system_prompt == "Base\n\nwarm"
    assert [tool.description for tool in get_current_tools(request_messages)] == ["v2"]


async def test_executable_tool_delta_is_announced_and_resolves_to_executable_set() -> None:
    stream = ScriptedStreamFn()
    tool_v1 = Tool(name="run", description="v1", parameters={"type": "object"})
    tool_v2 = Tool(name="run", description="v2", parameters={"type": "object"})
    executable = AgentTool(
        name="run",
        description="v2",
        parameters={"type": "object"},
        label="Run",
        execute=lambda *args: AgentToolResult(content=[TextContent(text="done")], details={}),
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                messages=[
                    SystemMessage(content="Base", timestamp=0, tools_added=[tool_v1]),
                    UserMessage(content="seed", timestamp=NOW),
                ],
                tools=[executable],
            ),
        )
    )

    await agent.prompt("go")

    assert agent.state.system_prompt == "Base"
    request_messages = stream.requests[0].context.messages
    assert get_current_tools(request_messages) == [tool_v2]


async def test_tool_declaration_change_emits_message_lifecycle_events() -> None:
    stream = ScriptedStreamFn()
    executable = AgentTool(
        name="run",
        description="v2",
        parameters={"type": "object"},
        label="Run",
        execute=lambda *args: AgentToolResult(content=[TextContent(text="done")], details={}),
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                messages=[
                    SystemMessage(
                        content="Base",
                        timestamp=0,
                        tools_added=[Tool(name="run", description="v1", parameters={"type": "object"})],
                    ),
                    UserMessage(content="seed", timestamp=NOW),
                ],
                tools=[executable],
            ),
        )
    )
    events: list[object] = []
    agent.subscribe(lambda event, signal: events.append(event))

    await agent.prompt("go")

    starts = [event for event in events if isinstance(event, MessageStartEvent)]
    assert any(
        isinstance(event.message, SystemMessage) and event.message.content == "" for event in starts
    )
    ends = [event for event in events if isinstance(event, MessageEndEvent)]
    assert len(ends) >= len(starts)


# ---------------------------------------------------------------------------
# M01: default StreamFn, dynamic credentials, and request options
# ---------------------------------------------------------------------------


def test_unconfigured_default_stream_fn_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="No default StreamFn configured"):
        Agent(AgentOptions(initial_state=AgentInitialState(model=make_model())))


async def test_explicit_stream_fn_overrides_installed_default() -> None:
    default_stream = ScriptedStreamFn("default")
    explicit_stream = ScriptedStreamFn("explicit")
    set_default_stream_fn(default_stream)

    agent = Agent(AgentOptions(stream_fn=explicit_stream, initial_state=AgentInitialState(model=make_model())))
    await agent.prompt("hello")

    assert explicit_stream.calls == 1
    assert default_stream.calls == 0


async def test_installed_default_stream_fn_is_used_and_can_be_cleared() -> None:
    default_stream = ScriptedStreamFn("default")
    set_default_stream_fn(default_stream)

    agent = Agent(AgentOptions(initial_state=AgentInitialState(model=make_model())))
    await agent.prompt("hello")
    assert default_stream.calls == 1

    clear_default_stream_fn()
    with pytest.raises(RuntimeError):
        Agent(AgentOptions(initial_state=AgentInitialState(model=make_model())))


async def test_dynamic_api_key_is_resolved_each_request_with_static_fallback() -> None:
    stream = ScriptedStreamFn()
    keys = iter(["key-1", "key-2", None])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            api_key="static",
            get_api_key=lambda provider: next(keys),
            initial_state=AgentInitialState(model=make_model()),
        )
    )

    await agent.prompt("one")
    await agent.prompt("two")
    await agent.prompt("three")

    assert [request.options.api_key for request in stream.requests] == ["key-1", "key-2", "static"]  # type: ignore[union-attr]


async def test_model_and_request_options_are_forwarded_to_stream_fn() -> None:
    stream = ScriptedStreamFn()
    payloads: list[dict[str, object]] = []
    responses: list[ProviderResponse] = []
    provider_events: list[object] = []
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            api_key="static",
            session_id="session-1",
            thinking_budgets=ThinkingBudgets(low=16, high=64),
            transport="sse",
            max_retry_delay_ms=250,
            on_payload=lambda payload, model: payloads.append(payload) or payload,
            on_response=lambda response, model: responses.append(response),
            on_provider_stream_event=lambda event, model: provider_events.append(event),
            initial_state=AgentInitialState(model=make_model(), thinking_level="high"),
        )
    )

    await agent.prompt("hello")

    options = stream.requests[0].options
    assert options is not None
    assert options.session_id == "session-1"
    assert options.thinking_budgets == ThinkingBudgets(low=16, high=64)
    assert options.transport == "sse"
    assert options.max_retry_delay_ms == 250
    assert options.reasoning == "high"
    assert options.api_key == "static"
    assert options.on_payload is not None
    assert options.on_response is not None
    assert options.on_provider_stream_event is not None
    assert getattr(agent.state.messages[-1], "thinking_level") == "high"


async def test_pending_system_message_tool_fields_track_the_delta() -> None:
    stream = ScriptedStreamFn()
    executable = AgentTool(
        name="run",
        description="v2",
        parameters={"type": "object"},
        label="Run",
        execute=lambda *args: AgentToolResult(content=[TextContent(text="done")], details={}),
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                messages=[SystemMessage(content="Base", timestamp=0)],
                tools=[executable],
            ),
        )
    )

    await agent.prompt(
        [SystemMessage(content="Intent", timestamp=1, tools_added=[Tool(name="other", description="x", parameters={})]), UserMessage(content="go", timestamp=NOW)]
    )

    request_messages = stream.requests[0].context.messages
    pending = [message for message in request_messages if isinstance(message, SystemMessage) and message.content == "Intent"]
    assert len(pending) == 1
    assert "Intent" in agent.state.system_prompt
    assert [tool.name for tool in get_current_tools(request_messages)] == ["run"]
