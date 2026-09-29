"""Public behavior tests for the traditional Agent tool roundtrip."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from omh.agent import (
    Agent,
    AgentEndEvent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    to_tool_declaration,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

NOW = 1_700_000_000_000


def make_model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="test",
        base_url="https://example.invalid",
        reasoning=False,
        input=("text", "image"),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def text_message(text: str, stop_reason: StopReason = "stop") -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason=stop_reason,
        timestamp=NOW,
        content=[TextContent(text=text)],
    )


def tool_call_message(
    name: str,
    arguments: dict[str, object],
    call_id: str = "c1",
    stop_reason: StopReason = "toolUse",
) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason=stop_reason,
        timestamp=NOW,
        content=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


@dataclass
class CapturedRequest:
    model: Model
    context: TranscriptContext
    options: SimpleStreamOptions | None


class ScriptedStreamFn:
    """Controlled StreamFn that records requests and replays one message per call."""

    def __init__(self, turns: list[Callable[[], AssistantMessage]]) -> None:
        self.turns = turns
        self.requests: list[CapturedRequest] = []
        self.calls = 0

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        self.calls += 1
        self.requests.append(CapturedRequest(model=model, context=context, options=options))
        final = self.turns[min(self.calls - 1, len(self.turns) - 1)]()
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        if final.stop_reason in {"error", "aborted"}:
            stream.push(ErrorEvent(reason=final.stop_reason, error=final))  # type: ignore[arg-type]
        else:
            stream.push(DoneEvent(reason=final.stop_reason, message=final))  # type: ignore[arg-type]
        return stream


@dataclass
class RecordingTool:
    """Controlled executable tool recording real call arguments and side effects."""

    name: str = "echo"
    parameters: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
    )
    result: AgentToolResult = field(
        default_factory=lambda: AgentToolResult(content=[TextContent(text="ok")], details={})
    )
    error: Exception | None = None
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del signal, on_update
        self.calls.append((tool_call_id, args))
        if self.error is not None:
            raise self.error
        return self.result

    def agent_tool(self, **overrides: object) -> AgentTool:
        values: dict[str, object] = {
            "name": self.name,
            "description": "controlled tool",
            "parameters": self.parameters,
            "label": self.name.title(),
            "execute": self.execute,
        }
        values.update(overrides)
        return AgentTool(**values)  # type: ignore[arg-type]


def collect_events(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    return events


def event_types(events: list[AgentEvent]) -> list[type[AgentEvent]]:
    return [type(event) for event in events]


# ---------------------------------------------------------------------------
# Complete roundtrip
# ---------------------------------------------------------------------------


async def test_tool_call_roundtrip_executes_and_continues_model() -> None:
    tool = RecordingTool(result=AgentToolResult(content=[TextContent(text="42")], details={"answer": 42}))
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"value": "7"}),
            lambda: text_message("The value was 7"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(system_prompt="sys", model=make_model(), tools=[tool.agent_tool()]),
        )
    )
    events = collect_events(agent)

    await agent.prompt("go")

    assert stream.calls == 2
    assert tool.calls == [("c1", {"value": "7"})]

    first_system = stream.requests[0].context.messages[0]
    assert isinstance(first_system, SystemMessage)
    assert [declaration.name for declaration in (first_system.tools_added or [])] == ["echo"]
    assert first_system.tools_added is not None
    assert first_system.tools_added[0].description == "controlled tool"

    second_request_messages = stream.requests[1].context.messages
    assert isinstance(second_request_messages[-1], ToolResultMessage)
    assert second_request_messages[-1].tool_call_id == "c1"
    assert second_request_messages[-1].content == [TextContent(text="42")]
    assert second_request_messages[-1].is_error is False

    assert [message.role for message in agent.state.messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.content == [TextContent(text="The value was 7")]

    assert event_types(events) == [
        AgentStartEvent,
        TurnStartEvent,
        MessageStartEvent,
        MessageEndEvent,
        MessageStartEvent,
        MessageEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionEndEvent,
        MessageStartEvent,
        MessageEndEvent,
        TurnEndEvent,
        TurnStartEvent,
        MessageStartEvent,
        MessageEndEvent,
        TurnEndEvent,
        AgentEndEvent,
    ]
    start_event = next(event for event in events if isinstance(event, ToolExecutionStartEvent))
    assert start_event.args == {"value": "7"}
    end_event = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
    assert end_event.result.content == [TextContent(text="42")]
    assert end_event.is_error is False
    turn_end = next(event for event in events if isinstance(event, TurnEndEvent))
    assert [message.tool_call_id for message in turn_end.tool_results] == ["c1"]


def test_tool_declaration_strips_execution_and_display_fields() -> None:
    tool = RecordingTool()
    agent_tool = tool.agent_tool(label="Fancy Label")

    declaration = to_tool_declaration(agent_tool)

    assert isinstance(declaration, Tool)
    assert declaration.name == "echo"
    assert declaration.description == "controlled tool"
    assert declaration.parameters == tool.parameters
    assert not hasattr(declaration, "label")
    assert not hasattr(declaration, "execute")
    assert not hasattr(declaration, "prepare_arguments")

    tool.parameters["properties"] = {"changed": {"type": "string"}}
    assert declaration.parameters != tool.parameters


# ---------------------------------------------------------------------------
# Error tool results that never execute
# ---------------------------------------------------------------------------


async def test_unknown_tool_becomes_error_result_without_execution() -> None:
    tool = RecordingTool()
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("missing", {"value": "x"}),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == []
    assert stream.calls == 2
    result = stream.requests[1].context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert "not found" in result.content[0].text
    assert isinstance(agent.state.messages[-1], AssistantMessage)
    assert agent.state.messages[-1].content == [TextContent(text="recovered")]


async def test_invalid_arguments_become_error_result_without_execution() -> None:
    tool = RecordingTool()
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"unexpected": True}),
            lambda: text_message("retried"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == []
    result = stream.requests[1].context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert "Validation failed" in result.content[0].text


async def test_prepare_arguments_runs_before_validation_and_keeps_raw_call() -> None:
    tool = RecordingTool()
    prepared_calls: list[dict[str, object]] = []

    def prepare(arguments: dict[str, object]) -> dict[str, object]:
        prepared_calls.append(arguments)
        return {"value": str(arguments.get("raw", ""))}

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"raw": "payload"}),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[tool.agent_tool(prepare_arguments=prepare)],
            ),
        )
    )
    events = collect_events(agent)

    await agent.prompt("go")

    assert prepared_calls == [{"raw": "payload"}]
    assert tool.calls == [("c1", {"value": "payload"})]
    start_event = next(event for event in events if isinstance(event, ToolExecutionStartEvent))
    assert start_event.args == {"raw": "payload"}
    assistant = agent.state.messages[2]
    assert isinstance(assistant, AssistantMessage)
    raw_call = assistant.content[0]
    assert isinstance(raw_call, ToolCall)
    assert raw_call.arguments == {"raw": "payload"}


async def test_prepare_arguments_exception_becomes_error_result() -> None:
    tool = RecordingTool()

    def prepare(arguments: dict[str, object]) -> dict[str, object]:
        del arguments
        raise RuntimeError("preprocess exploded")

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"value": "x"}),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[tool.agent_tool(prepare_arguments=prepare)],
            ),
        )
    )

    await agent.prompt("go")

    assert tool.calls == []
    result = stream.requests[1].context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert result.content[0].text == "preprocess exploded"


async def test_tool_execution_exception_becomes_error_result() -> None:
    tool = RecordingTool(error=RuntimeError("tool broke"))
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"value": "x"}),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == [("c1", {"value": "x"})]
    result = stream.requests[1].context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert result.content[0].text == "tool broke"
    assert isinstance(agent.state.messages[-1], AssistantMessage)


# ---------------------------------------------------------------------------
# Truncated responses
# ---------------------------------------------------------------------------


async def test_truncated_tool_call_is_not_executed_and_model_can_reissue() -> None:
    tool = RecordingTool()
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"value": "complete"}, stop_reason="length"),
            lambda: tool_call_message("echo", {"value": "complete"}, call_id="c2"),
            lambda: text_message("finished"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert stream.calls == 3
    assert tool.calls == [("c2", {"value": "complete"})]
    truncation_result = stream.requests[1].context.messages[-1]
    assert isinstance(truncation_result, ToolResultMessage)
    assert truncation_result.is_error is True
    assert "output token limit" in truncation_result.content[0].text
    assert isinstance(agent.state.messages[-1], AssistantMessage)
    assert agent.state.messages[-1].content == [TextContent(text="finished")]


# ---------------------------------------------------------------------------
# Schema conversion, optional nulls, nested and composition constraints
# ---------------------------------------------------------------------------


async def test_argument_coercion_optional_null_and_nested_values() -> None:
    tool = RecordingTool(
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
                "label": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
                "optional": {"type": "string"},
            },
            "required": ["count", "flag", "label", "nested"],
        }
    )
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message(
                "echo",
                {
                    "count": "3",
                    "flag": "true",
                    "label": 5,
                    "nested": {"value": "2.5"},
                    "optional": None,
                },
            ),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == [
        (
            "c1",
            {"count": 3, "flag": True, "label": "5", "nested": {"value": 2.5}},
        )
    ]


async def test_integral_float_satisfies_integer_schema() -> None:
    tool = RecordingTool(
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
    )
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"count": 3.0}),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == [("c1", {"count": 3.0})]


async def test_argument_coercion_does_not_rewrite_raw_tool_call_history() -> None:
    tool = RecordingTool(
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
    )
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"value": "7"}),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == [("c1", {"value": 7})]
    assistant = agent.state.messages[2]
    assert isinstance(assistant, AssistantMessage)
    raw_call = assistant.content[0]
    assert isinstance(raw_call, ToolCall)
    assert raw_call.arguments == {"value": "7"}


async def test_combination_constraints_are_enforced() -> None:
    tool = RecordingTool(
        parameters={
            "type": "object",
            "properties": {
                "mode": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "choice": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["mode", "choice"],
        }
    )
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"mode": [], "choice": "ok"}),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == []
    result = stream.requests[1].context.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error is True
    assert "anyOf" in result.content[0].text


# ---------------------------------------------------------------------------
# Tool declaration boundaries
# ---------------------------------------------------------------------------


async def test_tools_assigned_after_construction_are_declared_to_model() -> None:
    tool = RecordingTool()
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    agent.state.tools = [tool.agent_tool()]

    await agent.prompt("go")

    system = stream.requests[0].context.messages[0]
    assert isinstance(system, SystemMessage)
    assert [declaration.name for declaration in (system.tools_added or [])] == ["echo"]


async def test_removed_tools_are_declared_to_model() -> None:
    tool = RecordingTool()
    stream = ScriptedStreamFn(
        [
            lambda: text_message("first"),
            lambda: text_message("second"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("one")
    agent.state.tools = []
    await agent.prompt("two")

    second_request = stream.requests[1].context.messages
    removals = [
        message
        for message in second_request
        if isinstance(message, SystemMessage) and message.tools_removed
    ]
    assert len(removals) == 1
    assert removals[0].tools_removed is not None
    assert [removed.name for removed in removals[0].tools_removed] == ["echo"]
