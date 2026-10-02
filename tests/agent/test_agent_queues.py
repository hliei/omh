"""Public behavior tests for traditional Agent steering and follow-up queues."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

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
    AgentTurnContext,
    CustomAgentMessage,
    ToolExecutionEndEvent,
    ToolExecutionMode,
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
    TextContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
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
        input=("text",),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def text_message(text: str, stop_reason: StopReason = "stop", error_message: str | None = None) -> AssistantMessage:
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


def tool_call_message(
    calls: list[tuple[str, str, dict[str, object]]], stop_reason: StopReason = "toolUse"
) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason=stop_reason,
        timestamp=NOW,
        content=[ToolCall(id=call_id, name=name, arguments=arguments) for call_id, name, arguments in calls],
    )


def user_message(text: str, timestamp: int = NOW) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


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
class ControlledTool:
    """Tool with an explicit start/release gate and recorded side effects."""

    name: str
    parameters: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
    )
    execution_mode: ToolExecutionMode | None = None
    gated: bool = True
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del signal, on_update
        self.calls.append((tool_call_id, args))
        self.started.set()
        if self.gated:
            await self.release.wait()
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    def agent_tool(self) -> AgentTool:
        return AgentTool(
            name=self.name,
            description="controlled tool",
            parameters=self.parameters,
            label=self.name.title(),
            execute=self.execute,
            execution_mode=self.execution_mode,
        )


def collect_events(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    return events


def request_users(request: CapturedRequest) -> list[UserMessage]:
    return [message for message in request.context.messages if isinstance(message, UserMessage)]


def user_text(message: UserMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))


def last_user(request: CapturedRequest) -> UserMessage:
    return request_users(request)[-1]


def history_users(agent: Agent) -> list[UserMessage]:
    return [message for message in agent.state.messages if isinstance(message, UserMessage)]


def event_count(events: list[AgentEvent], kind: type[AgentEvent]) -> int:
    return sum(1 for event in events if isinstance(event, kind))


def make_agent(
    stream: ScriptedStreamFn,
    *,
    tools: list[AgentTool] | None = None,
    system_prompt: str | None = None,
    messages: list[object] | None = None,
) -> Agent:
    return Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=tools,
                system_prompt=system_prompt,
                messages=messages,  # type: ignore[arg-type]
            ),
        )
    )


# ---------------------------------------------------------------------------
# A04: queue interface, modes, preview, clearing
# ---------------------------------------------------------------------------


def test_queue_interface_defaults_and_preview_priority() -> None:
    agent = make_agent(ScriptedStreamFn([lambda: text_message("ok")]))
    steer_one = user_message("s1", 1)
    steer_two = user_message("s2", 2)
    follow_up = user_message("f1", 3)

    assert agent.has_queued_messages() is False
    assert agent.peek_queued_messages() == []
    assert agent.steering_mode == "one-at-a-time"
    assert agent.follow_up_mode == "one-at-a-time"

    agent.steer(steer_one)
    agent.steer(steer_two)
    agent.follow_up(follow_up)

    assert agent.has_queued_messages() is True
    # Steering takes priority and one-at-a-time previews only the oldest message.
    assert agent.peek_queued_messages() == [steer_one]

    agent.steering_mode = "all"
    assert agent.peek_queued_messages() == [steer_one, steer_two]

    agent.follow_up_mode = "all"
    assert agent.peek_queued_messages()[0] == steer_one


def test_peek_does_not_consume_and_returns_isolated_messages() -> None:
    agent = make_agent(ScriptedStreamFn([lambda: text_message("ok")]))
    steer = user_message("s1", 1)
    agent.steer(steer)

    first = agent.peek_queued_messages()
    second = agent.peek_queued_messages()
    assert first[0] == steer
    assert second[0] == steer
    assert first[0] is not steer
    assert second[0] is not first[0]
    assert agent.has_queued_messages() is True


def test_clear_operations_are_independent() -> None:
    agent = make_agent(ScriptedStreamFn([lambda: text_message("ok")]))
    agent.steer(user_message("s1", 1))
    agent.follow_up(user_message("f1", 2))

    agent.clear_steering_queue()
    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [user_message("f1", 2)]

    agent.steer(user_message("s2", 3))
    agent.clear_all_queues()
    assert agent.has_queued_messages() is False
    assert agent.peek_queued_messages() == []


# ---------------------------------------------------------------------------
# A04/A05: initial poll, turn-boundary consumption, and tool batches
# ---------------------------------------------------------------------------


async def test_initial_steering_poll_injects_before_first_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream, system_prompt="base")
    steer = user_message("steer", 2)
    agent.steer(steer)

    await agent.prompt("hello")

    assert stream.calls == 1
    request = stream.requests[0]
    assert [message.role for message in request.context.messages] == ["system", "user", "user"]
    assert [user_text(message) for message in request_users(request)] == ["hello", "steer"]
    assert request_users(request)[1] == steer
    assert [message.role for message in agent.state.messages] == ["system", "user", "user", "assistant"]


async def test_one_at_a_time_consumes_one_steering_message_per_boundary() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    first = user_message("first", 1)
    second = user_message("second", 2)
    agent.steer(first)
    agent.steer(second)

    await agent.prompt("hello")

    assert stream.calls == 2
    assert request_users(stream.requests[0])[1] == first
    assert second not in request_users(stream.requests[0])
    assert last_user(stream.requests[1]) == second


async def test_all_mode_drains_every_queued_steering_message_at_once() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    agent.steering_mode = "all"
    agent.steer(user_message("first", 1))
    agent.steer(user_message("second", 2))

    await agent.prompt("hello")

    assert stream.calls == 1
    assert [user_text(message) for message in request_users(stream.requests[0])] == ["hello", "first", "second"]


async def test_steering_during_tool_batch_does_not_skip_remaining_tools() -> None:
    first = ControlledTool(name="first")
    second = ControlledTool(name="second")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = make_agent(stream, tools=[first.agent_tool(), second.agent_tool()])
    events = collect_events(agent)
    steer = user_message("steer", 5)
    enqueued = False

    def enqueue_on_tool_end(event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        nonlocal enqueued
        if isinstance(event, ToolExecutionEndEvent) and not enqueued:
            enqueued = True
            agent.steer(steer)

    agent.subscribe(enqueue_on_tool_end)

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(asyncio.gather(first.started.wait(), second.started.wait()), timeout=1)
    first.release.set()
    second.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert first.calls == [("c1", {"value": "a"})]
    assert second.calls == [("c2", {"value": "b"})]
    assert [event.tool_name for event in events if isinstance(event, ToolExecutionEndEvent)] == [
        "first",
        "second",
    ]
    assert stream.calls == 2
    # The tool results stay in source order; the steer arrives only in the next request.
    assert [message.tool_call_id for message in agent.state.messages if isinstance(message, ToolResultMessage)] == [
        "c1",
        "c2",
    ]
    assert steer not in request_users(stream.requests[0])
    assert last_user(stream.requests[1]) == steer


async def test_consumed_queue_order_matches_preview_and_history() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream, system_prompt="base")
    first = user_message("first", 1)
    second = user_message("second", 2)
    agent.steer(first)
    agent.steer(second)

    assert agent.peek_queued_messages() == [first]

    await agent.prompt("hello")

    assert stream.calls == 2
    users = history_users(agent)
    assert user_text(users[0]) == "hello"
    assert users[1] == first
    assert users[2] == second
    assert request_users(stream.requests[0])[1] == first
    assert last_user(stream.requests[1]) == second


# ---------------------------------------------------------------------------
# A05: steering during turn preparation
# ---------------------------------------------------------------------------


async def test_steer_queued_during_turn_preparation_is_picked_up_when_none_was_taken() -> None:
    stream = ScriptedStreamFn(
        [lambda: tool_call_message([("c1", "echo", {"value": "a"})]), lambda: text_message("done")]
    )
    agent = make_agent(
        stream,
        tools=[
            ControlledTool(name="echo", gated=False).agent_tool(),
        ],
    )
    steer = user_message("prepared-steer", 5)
    prepare_calls = 0

    async def prepare_next_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del context, signal
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            # Preparation can be long-running; steering queued while it runs is
            # picked up by the follow-up poll before the next request.
            await asyncio.sleep(0)
            agent.steer(steer)

    agent.prepare_next_turn_with_context = prepare_next_turn

    await agent.prompt("go")

    assert stream.calls == 2
    assert steer not in request_users(stream.requests[0])
    assert last_user(stream.requests[1]) == steer


async def test_steer_queued_during_turn_preparation_is_not_double_drained() -> None:
    stream = ScriptedStreamFn(
        [lambda: tool_call_message([("c1", "echo", {"value": "a"})]), lambda: text_message("done")]
    )
    agent = make_agent(
        stream,
        tools=[
            ControlledTool(name="echo", gated=False).agent_tool(),
        ],
    )
    first = user_message("first", 5)
    second = user_message("second", 6)
    prepare_calls = 0

    def enqueue_on_tool_end(event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        if isinstance(event, ToolExecutionEndEvent):
            agent.steer(first)

    def prepare_next_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del context, signal
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            agent.steer(second)

    agent.subscribe(enqueue_on_tool_end)
    agent.prepare_next_turn_with_context = prepare_next_turn

    await agent.prompt("go")

    # One-at-a-time already took ``first`` for this turn, so preparation must not
    # drain ``second`` into the same request.
    assert stream.calls == 3
    assert user_text(last_user(stream.requests[1])) == "first"
    assert user_text(last_user(stream.requests[2])) == "second"


# ---------------------------------------------------------------------------
# A05/A07: follow-up consumption and continuation scheduling
# ---------------------------------------------------------------------------


async def test_follow_up_waits_for_natural_tool_continuation() -> None:
    tool = ControlledTool(name="echo", gated=False)
    stream = ScriptedStreamFn(
        [lambda: tool_call_message([("c1", "echo", {"value": "a"})]), lambda: text_message("done")]
    )
    agent = make_agent(stream, tools=[tool.agent_tool()])
    events = collect_events(agent)
    follow_up = user_message("later", 5)

    def enqueue_on_tool_end(event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        if isinstance(event, ToolExecutionEndEvent):
            agent.follow_up(follow_up)

    agent.subscribe(enqueue_on_tool_end)

    await agent.prompt("go")

    assert stream.calls == 3
    assert follow_up not in request_users(stream.requests[1])
    assert last_user(stream.requests[2]) == follow_up
    # Follow-up continues the same run, so the lifecycle stays a single cycle.
    assert event_count(events, AgentStartEvent) == 1
    assert event_count(events, AgentEndEvent) == 1


async def test_follow_up_consumed_only_when_run_would_stop() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    events = collect_events(agent)
    follow_up = user_message("later", 5)
    finish_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del context, signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            agent.follow_up(follow_up)

    agent.finish_turn = finish_turn

    await agent.prompt("hello")

    assert stream.calls == 2
    assert last_user(stream.requests[1]) == follow_up
    assert event_count(events, AgentStartEvent) == 1
    assert event_count(events, AgentEndEvent) == 1


async def test_all_mode_drains_every_queued_follow_up_at_stop() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    agent.follow_up_mode = "all"
    first = user_message("first", 1)
    second = user_message("second", 2)
    finish_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del context, signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            agent.follow_up(first)
            agent.follow_up(second)

    agent.finish_turn = finish_turn

    await agent.prompt("hello")

    assert stream.calls == 2
    assert [user_text(message) for message in request_users(stream.requests[1])] == [
        "hello",
        "first",
        "second",
    ]


async def test_steering_is_consumed_before_follow_up() -> None:
    tool = ControlledTool(name="echo", gated=False)
    stream = ScriptedStreamFn(
        [lambda: tool_call_message([("c1", "echo", {"value": "a"})]), lambda: text_message("done")]
    )
    agent = make_agent(stream, tools=[tool.agent_tool()])
    steer = user_message("steer", 5)
    follow_up = user_message("later", 6)

    def enqueue_on_tool_end(event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        if isinstance(event, ToolExecutionEndEvent):
            agent.steer(steer)
            agent.follow_up(follow_up)

    agent.subscribe(enqueue_on_tool_end)

    await agent.prompt("go")

    assert stream.calls == 3
    assert last_user(stream.requests[1]) == steer
    assert last_user(stream.requests[2]) == follow_up


async def test_finish_turn_end_leaves_queues_untouched() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str:
        del context, signal
        agent.steer(steer)
        agent.follow_up(follow_up)
        return "end"

    agent.finish_turn = finish_turn

    await agent.prompt("hello")

    assert stream.calls == 1
    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [steer]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


async def test_finish_turn_continue_satisfied_by_steering_adds_no_extra_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    steer = user_message("steer", 1)
    finish_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del context, signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            agent.steer(steer)
            return "continue"
        return None

    agent.finish_turn = finish_turn

    await agent.prompt("hello")

    assert stream.calls == 2
    assert last_user(stream.requests[1]) == steer


async def test_finish_turn_continue_satisfied_by_follow_up_adds_no_extra_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    follow_up = user_message("later", 1)
    finish_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del context, signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            agent.follow_up(follow_up)
            return "continue"
        return None

    agent.finish_turn = finish_turn

    await agent.prompt("hello")

    assert stream.calls == 2
    assert last_user(stream.requests[1]) == follow_up


# ---------------------------------------------------------------------------
# A03: continue_ with queues at an assistant tail
# ---------------------------------------------------------------------------


async def test_continue_assistant_tail_prefers_steering_and_skips_initial_poll() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream, messages=[user_message("seed", 0), text_message("done")])
    first = user_message("first", 1)
    second = user_message("second", 2)
    agent.steer(first)
    agent.steer(second)

    await agent.continue_()

    # ``first`` becomes the new prompt; the skipped initial poll must not also
    # consume ``second`` into the same request.
    assert stream.calls == 2
    assert stream.requests[0].context.messages[-1] == first
    assert second not in request_users(stream.requests[0])
    assert last_user(stream.requests[1]) == second


async def test_continue_assistant_tail_uses_follow_up_when_no_steering() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream, messages=[user_message("seed", 0), text_message("done")])
    follow_up = user_message("later", 1)
    agent.follow_up(follow_up)

    await agent.continue_()

    assert stream.calls == 1
    assert stream.requests[0].context.messages[-1] == follow_up


async def test_continue_assistant_tail_prefers_steering_over_follow_up() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream, messages=[user_message("seed", 0), text_message("done")])
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)
    agent.steer(steer)
    agent.follow_up(follow_up)

    await agent.continue_()

    assert stream.requests[0].context.messages[-1] == steer
    assert follow_up not in request_users(stream.requests[0])


async def test_continue_assistant_tail_rejects_when_no_queued_input() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream, messages=[user_message("seed", 0), text_message("done")])

    with pytest.raises(ValueError, match="assistant"):
        await agent.continue_()

    assert stream.calls == 0


# ---------------------------------------------------------------------------
# A15: failure and abort retain unconsumed queues; reset clears them
# ---------------------------------------------------------------------------


async def test_error_run_retains_unconsumed_queues() -> None:
    holder: dict[str, Agent] = {}
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)

    def stream_fn(
        model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        del model, context, options
        holder["agent"].steer(steer)
        holder["agent"].follow_up(follow_up)
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        stream.push(
            ErrorEvent(reason="error", error=text_message("failed", "error", error_message="provider failed"))  # type: ignore[arg-type]
        )
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    holder["agent"] = agent

    await agent.prompt("hello")

    assert agent.state.error_message == "provider failed"
    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [steer]


async def test_abort_run_retains_unconsumed_queues() -> None:
    entered = asyncio.Event()
    holder: dict[str, Agent] = {}
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)

    def stream_fn(
        model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        del model, context
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        agent = holder["agent"]
        agent.steer(steer)
        agent.follow_up(follow_up)
        aborted = text_message("", "aborted", error_message="aborted by caller")
        assert options is not None
        assert options.signal is not None

        def on_abort() -> None:
            stream.push(ErrorEvent(reason="aborted", error=aborted))  # type: ignore[arg-type]

        options.signal.add_callback(on_abort)
        entered.set()
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    holder["agent"] = agent

    run = asyncio.create_task(agent.prompt("hello"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    agent.abort()
    await asyncio.wait_for(run, timeout=1)

    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"
    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [steer]


async def test_new_instance_has_separate_queues_and_the_host_system_baseline() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    old = make_agent(stream, system_prompt="Stay")
    old.steer(user_message("steer", 1))
    old.follow_up(user_message("later", 2))
    agent = make_agent(stream, system_prompt="Stay")
    assert agent.has_queued_messages() is False
    assert old.has_queued_messages() is True
    assert agent.state.system_prompt == "Stay"


# ---------------------------------------------------------------------------
# A04/A08: queued custom messages keep the application-ownership contract
# ---------------------------------------------------------------------------


async def test_queued_custom_message_stays_in_history_and_respects_conversion() -> None:
    notice = CustomAgentMessage(custom_type="notice", content="heads up", timestamp=7)

    default_stream = ScriptedStreamFn([lambda: text_message("ok")])
    default_agent = make_agent(default_stream)
    default_agent.steer(notice)
    await default_agent.prompt("hello")

    # The default conversion includes custom content as a user message.
    assert [message.role for message in default_stream.requests[0].context.messages] == ["user", "user"]
    assert notice not in default_stream.requests[0].context.messages
    assert notice in default_agent.state.messages

    mapped_stream = ScriptedStreamFn([lambda: text_message("ok")])
    mapped_agent = Agent(
        AgentOptions(
            stream_fn=mapped_stream,
            initial_state=AgentInitialState(model=make_model()),
            convert_to_llm=lambda messages: [
                user_message(message.content, 7) if isinstance(message, CustomAgentMessage) else message  # type: ignore[arg-type]
                for message in messages
            ],
        )
    )
    mapped_agent.steer(notice)
    await mapped_agent.prompt("hello")

    request = mapped_stream.requests[0]
    assert "notice" not in [message.role for message in request.context.messages]
    assert [user_text(message) for message in request_users(request)] == ["hello", "heads up"]
