"""Public behavior tests for request preparation and turn-continuation control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentLoopTurnUpdate,
    AgentOptions,
    AgentRequestUpdate,
    AgentSettledEvent,
    AgentTool,
    AgentToolResult,
    AgentTurnContext,
    PrepareRequestContext,
)
from omh.agent.context import AgentContext
from omh.agent.loop import AgentLoopConfig, run_agent_loop
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
    ToolCall,
    TranscriptContext,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

NOW = 1_700_000_000_000


def make_model(model_id: str = "test-model") -> Model:
    return Model(
        id=model_id,
        name=f"Test Model {model_id}",
        api="openai-completions",
        provider="test",
        base_url="https://example.invalid",
        reasoning=True,
        input=("text",),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def text_message(
    text: str, stop_reason: StopReason = "stop", error_message: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason=stop_reason,
        timestamp=NOW,
        content=[TextContent(text=text)],
        error_message=error_message,
    )


def tool_call_message(name: str, arguments: dict[str, object], call_id: str = "c1") -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason="toolUse",
        timestamp=NOW,
        content=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


@dataclass(slots=True)
class CapturedRequest:
    model: Model
    context: TranscriptContext
    options: SimpleStreamOptions | None


class ScriptedStreamFn:
    """Controlled StreamFn that records each request and replays one message per call."""

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


def echo_tool(name: str = "echo", result_text: str = "ok") -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: object,
    ) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text=result_text)], details={})

    return AgentTool(
        name=name,
        description="controlled echo",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        label=name.title(),
        execute=execute,
    )


def recorder(log: list[tuple[str, str]]) -> Callable[[AgentEvent, AbortSignal], None]:
    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        log.append(("event", type(event).__name__))

    return listener


def event_names(events: list[AgentEvent]) -> list[type[AgentEvent]]:
    return [type(event) for event in events]


# ---------------------------------------------------------------------------
# A06: prepare_request
# ---------------------------------------------------------------------------


async def test_prepare_request_runs_before_first_request_after_pending_messages() -> None:
    log: list[tuple[str, str]] = []
    seen: list[list[str]] = []
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(system_prompt="base", model=make_model()),
        )
    )
    agent.subscribe(recorder(log))

    def prepare_request(request: PrepareRequestContext, signal: AbortSignal | None) -> None:
        del signal
        log.append(("hook", "prepare_request"))
        seen.append([message.role for message in request.context.messages])

    agent.prepare_request = prepare_request

    await agent.prompt("hello")

    hook_index = log.index(("hook", "prepare_request"))
    assert log[:hook_index] == [
        ("event", "AgentStartEvent"),
        ("event", "TurnStartEvent"),
        ("event", "MessageStartEvent"),
        ("event", "HistoryCommitEvent"),
        ("event", "MessageEndEvent"),
    ]
    assert seen == [["system", "user"]]
    assert stream.calls == 1


async def test_prepare_request_update_applies_to_this_and_later_requests() -> None:
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message("echo", {"value": "1"}),
            lambda: text_message("done"),
        ]
    )
    calls = 0

    def prepare_request(
        request: PrepareRequestContext, signal: AbortSignal | None
    ) -> AgentRequestUpdate | None:
        del signal
        nonlocal calls
        calls += 1
        if calls > 1:
            return None
        replacement = AgentContext(
            messages=[SystemMessage(content="prepared context", timestamp=1), *request.context.messages],
            tools=list(request.context.tools),
        )
        return AgentRequestUpdate(context=replacement, model=make_model("next-model"), thinking_level="high")

    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            prepare_request=prepare_request,
            initial_state=AgentInitialState(system_prompt="base", model=make_model(), tools=[echo_tool()]),
        )
    )

    await agent.prompt("hello")

    assert calls == 2
    assert [request.model.id for request in stream.requests] == ["next-model", "next-model"]
    assert [request.options.reasoning for request in stream.requests if request.options] == ["high", "high"]
    requested = agent.state.messages[-1]
    assert isinstance(requested, AssistantMessage)
    assert requested.thinking_level == "high"

    # The first replacement context persists into the later request.
    second_prompt = stream.requests[1].context.messages[0]
    assert isinstance(second_prompt, SystemMessage)
    assert second_prompt.content == "prepared context"


async def test_prepare_request_thinking_level_off_clears_reasoning() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            prepare_request=lambda request, signal: AgentRequestUpdate(thinking_level="off"),
            initial_state=AgentInitialState(model=make_model(), thinking_level="high"),
        )
    )

    await agent.prompt("hello")

    assert stream.requests[0].options is not None
    assert stream.requests[0].options.reasoning is None
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.thinking_level == "off"


# ---------------------------------------------------------------------------
# A06/A07: prepare_next_turn and continuation scheduling
# ---------------------------------------------------------------------------


async def test_prepare_next_turn_only_runs_when_continuing() -> None:
    no_tools = ScriptedStreamFn([lambda: text_message("ok")])
    single_agent = Agent(
        AgentOptions(stream_fn=no_tools, initial_state=AgentInitialState(model=make_model()))
    )
    single_calls = 0

    def single_prepare(context: AgentTurnContext) -> None:
        del context
        nonlocal single_calls
        single_calls += 1

    single_agent.prepare_next_turn = single_prepare
    await single_agent.prompt("hello")
    assert single_calls == 0

    tool_stream = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    tool_agent = Agent(
        AgentOptions(
            stream_fn=tool_stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    tool_calls = 0

    def tool_prepare(context: AgentTurnContext) -> None:
        del context
        nonlocal tool_calls
        tool_calls += 1

    tool_agent.prepare_next_turn = tool_prepare
    await tool_agent.prompt("hello")
    assert tool_calls == 1


async def test_prepare_next_turn_runs_before_next_turn_start() -> None:
    log: list[tuple[str, str]] = []
    stream = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    agent.subscribe(recorder(log))

    def prepare_next_turn(context: AgentTurnContext) -> None:
        del context
        log.append(("hook", "prepare_next_turn"))

    agent.prepare_next_turn = prepare_next_turn
    await agent.prompt("hello")

    first_turn_end = log.index(("event", "TurnEndEvent"))
    hook_index = log.index(("hook", "prepare_next_turn"))
    second_turn_start = log.index(("event", "TurnStartEvent"), hook_index)
    assert first_turn_end < hook_index < second_turn_start


async def test_prepare_next_turn_with_context_takes_priority_and_receives_signal() -> None:
    stream = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    calls: list[str] = []
    captured_signals: list[AbortSignal | None] = []

    def prepare_with_signal(signal: AbortSignal | None) -> None:
        calls.append("signal")
        captured_signals.append(signal)

    def prepare_with_context(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        calls.append("context")
        captured_signals.append(signal)
        assert isinstance(context.message, AssistantMessage)

    agent.prepare_next_turn = prepare_with_signal
    agent.prepare_next_turn_with_context = prepare_with_context
    await agent.prompt("hello")

    assert calls == ["context"]
    assert len(captured_signals) == 1
    assert captured_signals[0] is not None


async def test_agent_prepare_next_turn_signal_only_receives_active_run_signal() -> None:
    stream = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    captured: list[tuple[AbortSignal | None, AbortSignal | None]] = []

    def prepare_with_signal(signal: AbortSignal | None) -> None:
        captured.append((signal, agent.signal))

    agent.prepare_next_turn = prepare_with_signal
    await agent.prompt("hello")

    assert len(captured) == 1
    assert captured[0][0] is captured[0][1]


async def test_prepare_next_turn_messages_get_lifecycle_events_and_tool_declarations() -> None:
    log: list[tuple[str, str]] = []
    stream = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    agent.subscribe(recorder(log))

    def prepare_with_context(context: AgentTurnContext, signal: AbortSignal | None) -> AgentLoopTurnUpdate:
        del signal
        log.append(("hook", "prepare_next_turn"))
        replacement = AgentContext(
            messages=list(context.context.messages),
            tools=[*context.context.tools, echo_tool("late")],
        )
        return AgentLoopTurnUpdate(
            context=replacement,
            messages=[UserMessage(content="prepared", timestamp=NOW + 5)],
        )

    agent.prepare_next_turn_with_context = prepare_with_context
    await agent.prompt("hello")

    prepared_start = log.index(("event", "MessageStartEvent"), log.index(("hook", "prepare_next_turn")))
    # A prepared message uses the ordinary lifecycle before the assistant response.
    assert log[prepared_start + 1] == ("event", "HistoryCommitEvent")
    assert log[prepared_start + 2] == ("event", "MessageEndEvent")

    second_turn_start = log.index(("event", "TurnStartEvent"), log.index(("hook", "prepare_next_turn")))
    assert log[second_turn_start + 1] == ("event", "MessageStartEvent")
    assert log[second_turn_start + 2] == ("event", "HistoryCommitEvent")
    assert log[second_turn_start + 3] == ("event", "MessageEndEvent")

    second_request = stream.requests[1].context.messages
    prepared = [message for message in second_request if isinstance(message, UserMessage)]
    assert prepared[-1].content == "prepared"
    system_messages = [message for message in second_request if isinstance(message, SystemMessage)]
    added = [declaration.name for message in system_messages for declaration in (message.tools_added or [])]
    assert "late" in added
    assert "echo" in added


# ---------------------------------------------------------------------------
# A07/A06: finish_turn
# ---------------------------------------------------------------------------


async def test_finish_turn_observes_completed_turn_before_turn_end() -> None:
    log: list[tuple[str, str]] = []
    stream = ScriptedStreamFn([lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    agent.subscribe(recorder(log))
    observed: list[AgentTurnContext] = []
    snapshots: list[list[str]] = []

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del signal
        log.append(("hook", "finish_turn"))
        observed.append(context)
        snapshots.append([message.role for message in context.context.messages])

    agent.finish_turn = finish_turn
    await agent.prompt("hello")

    first = observed[0]
    assert isinstance(first.message, AssistantMessage)
    assert first.message.stop_reason == "toolUse"
    assert [result.tool_call_id for result in first.tool_results] == ["c1"]
    # At hook time the completed turn's assistant message and tool result are present.
    assert snapshots[0][-2:] == ["assistant", "toolResult"]
    assert any(result is message for result in first.tool_results for message in first.context.messages)
    first_turn_end = log.index(("event", "TurnEndEvent"))
    finish_index = log.index(("hook", "finish_turn"))
    assert finish_index < first_turn_end
    assert len(observed) == 2


async def test_finish_turn_end_stops_before_natural_tool_continuation() -> None:
    stream = ScriptedStreamFn([lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )
    decisions: list[str] = []

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str:
        del context, signal
        decisions.append("end")
        return "end"

    agent.finish_turn = finish_turn
    await agent.prompt("hello")

    assert stream.calls == 1
    assert decisions == ["end"]
    assert [message.role for message in agent.state.messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
    ]


async def test_finish_turn_continue_forces_one_request_without_tools() -> None:
    stream = ScriptedStreamFn([lambda: text_message("first"), lambda: text_message("second")])
    agent = Agent(
        AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model()))
    )
    decisions: list[str | None] = []

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del signal
        decision = "continue" if len(decisions) == 0 else None
        decisions.append(decision)
        return decision

    agent.finish_turn = finish_turn
    await agent.prompt("hello")

    assert stream.calls == 2
    assert decisions == ["continue", None]
    assert [message.role for message in agent.state.messages] == ["user", "assistant", "assistant"]


async def test_finish_turn_continue_is_satisfied_by_natural_tool_continuation() -> None:
    stream = ScriptedStreamFn([lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[echo_tool()]),
        )
    )

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del signal
        return "continue" if context.message.stop_reason == "toolUse" else None

    agent.finish_turn = finish_turn
    await agent.prompt("hello")

    # The tool result already schedules the next request; ``continue`` adds none.
    assert stream.calls == 2


async def test_finish_turn_continue_runs_prepare_next_turn_for_the_extra_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("first"), lambda: text_message("second")])
    agent = Agent(
        AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model()))
    )
    prepare_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del signal
        return "continue" if context.message.content == [TextContent(text="first")] else None

    def prepare(context: AgentTurnContext) -> None:
        del context
        nonlocal prepare_calls
        prepare_calls += 1

    agent.finish_turn = finish_turn
    agent.prepare_next_turn = prepare
    await agent.prompt("hello")

    assert stream.calls == 2
    assert prepare_calls == 1


async def test_finish_turn_continue_is_ignored_on_error_and_aborted() -> None:
    for stop_reason in ("error", "aborted"):
        def failing() -> AssistantMessage:
            return text_message("failed", stop_reason, error_message="failed")  # type: ignore[arg-type]

        stream = ScriptedStreamFn([failing])
        agent = Agent(
            AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model()))
        )
        decisions: list[str] = []

        def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str:
            del context, signal
            decisions.append("continue")
            return "continue"

        agent.finish_turn = finish_turn
        await agent.prompt("hello")

        assert stream.calls == 1
        assert decisions == ["continue"]
        assert agent.state.error_message == "failed"
        final = agent.state.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert final.stop_reason == stop_reason


# ---------------------------------------------------------------------------
# Hook failure boundaries
# ---------------------------------------------------------------------------


async def test_agent_prepare_request_failure_uses_agent_error_lifecycle() -> None:
    stream = ScriptedStreamFn([lambda: text_message("never")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model()),
        )
    )

    def prepare_request(request: PrepareRequestContext, signal: AbortSignal | None) -> None:
        del request, signal
        raise RuntimeError("prepare failed")

    agent.prepare_request = prepare_request
    await agent.prompt("hello")

    assert stream.calls == 0
    assert agent.state.error_message == "prepare failed"
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "error"
    assert agent.state.is_streaming is False


async def test_agent_finish_turn_failure_uses_agent_error_lifecycle() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(
        AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model()))
    )
    events: list[type[AgentEvent]] = []
    agent.subscribe(lambda event, signal: events.append(type(event)))

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del context, signal
        raise RuntimeError("finish failed")

    agent.finish_turn = finish_turn
    await agent.prompt("hello")

    assert stream.calls == 1
    assert agent.state.error_message == "finish failed"
    assert events[-1] is AgentSettledEvent


async def test_direct_loop_hook_failure_propagates_to_caller() -> None:
    stream = ScriptedStreamFn([lambda: text_message("never")])

    def prepare_request(request: PrepareRequestContext, signal: AbortSignal | None) -> None:
        del request, signal
        raise RuntimeError("prepare failed")

    context = AgentContext(messages=[UserMessage(content="hello", timestamp=NOW)], tools=[])
    config = AgentLoopConfig(model=make_model(), prepare_request=prepare_request)

    with pytest.raises(RuntimeError, match="prepare failed"):
        await run_agent_loop(
            [UserMessage(content="hello", timestamp=NOW)],
            context,
            config,
            lambda event: None,
            None,
            stream,
        )


# ---------------------------------------------------------------------------
# Hooks on the continuation entry
# ---------------------------------------------------------------------------


async def test_hooks_run_through_continue_entry() -> None:
    stream = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[echo_tool()],
                messages=[UserMessage(content="seeded", timestamp=NOW)],
            ),
        )
    )
    prepare_roles: list[list[str]] = []
    prepared_turns: list[AgentTurnContext] = []
    finished_turns: list[AgentTurnContext] = []

    def prepare_request(request: PrepareRequestContext, signal: AbortSignal | None) -> None:
        del signal
        prepare_roles.append([message.role for message in request.context.messages])

    def prepare_next_turn(context: AgentTurnContext) -> None:
        prepared_turns.append(context)

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del signal
        finished_turns.append(context)

    agent.prepare_request = prepare_request
    agent.prepare_next_turn = prepare_next_turn
    agent.finish_turn = finish_turn

    await agent.continue_()

    assert stream.calls == 2
    assert prepare_roles[0] == ["system", "user"]
    assert len(prepared_turns) == 1
    assert len(finished_turns) == 2
    # Continuation returns only the messages this run added, not the seeded history.
    assert [message.role for message in finished_turns[-1].new_messages] == [
        "assistant",
        "toolResult",
        "assistant",
    ]
