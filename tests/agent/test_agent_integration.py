"""Integrated acceptance tests that combine the traditional Agent capabilities.

The per-ticket suites prove each capability in isolation. These tests exercise
the cross-feature scenarios the specification accepts as a whole: custom
messages, system sections, tool add/remove/redefine, dynamic model/thinking,
out-of-order tool completion, steering and follow-up, turn decisions,
observation cancellation, explicit abort, and the standalone loop entries.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from omh.agent import (
    Agent,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentInitialState,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentOptions,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    AgentTurnContext,
    PrepareRequestContext,
    ToolExecutionEndEvent,
    agent_loop,
    run_agent_loop,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    SystemMessage,
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


def make_model(model_id: str = "test-model", reasoning: bool = False) -> Model:
    return Model(
        id=model_id,
        name=f"Test Model {model_id}",
        api="openai-completions",
        provider="test",
        base_url="https://example.invalid",
        reasoning=reasoning,
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


@dataclass(slots=True)
class NoticeMessage:
    """Application-owned message that is not part of the LLM message union."""

    text: str
    role: str = "notice"


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


@dataclass
class ControlledTool:
    """Tool with an explicit start/release gate and a recorded completion order."""

    name: str
    description: str = "controlled tool"
    completions: list[str] = field(default_factory=list)
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    gated: bool = True

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
        self.completions.append(self.name)
        return AgentToolResult(content=[TextContent(text=f"{self.name}:{args['value']}")], details={})

    def agent_tool(self, description: str | None = None) -> AgentTool:
        return AgentTool(
            name=self.name,
            description=description or self.description,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            label=self.name.title(),
            execute=self.execute,
        )


def collect_events(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    return events


async def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not met before the timeout")
        await asyncio.sleep(0)


def request_users(request: CapturedRequest) -> list[UserMessage]:
    return [message for message in request.context.messages if isinstance(message, UserMessage)]


def user_text(message: UserMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))


def declared_tools(context: TranscriptContext) -> dict[str, str]:
    tools: dict[str, str] = {}
    for message in context.messages:
        if not isinstance(message, SystemMessage):
            continue
        for removed in message.tools_removed or []:
            tools.pop(removed.name, None)
        for added in message.tools_added or []:
            tools[added.name] = added.description
    return tools


def merged_sections(context: TranscriptContext) -> dict[str, str | None]:
    sections: dict[str, str | None] = {}
    for message in context.messages:
        if isinstance(message, SystemMessage):
            sections.update(message.sections or {})
    return sections


def tool_results(agent: Agent) -> list[ToolResultMessage]:
    return [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]


def end_event_ids(events: list[AgentEvent]) -> list[str]:
    return [event.tool_call_id for event in events if isinstance(event, ToolExecutionEndEvent)]


# ---------------------------------------------------------------------------
# Spec acceptance: one combined Agent run
# ---------------------------------------------------------------------------


async def test_combined_agent_run_tracks_requests_history_and_tools() -> None:
    agent_ref: list[Agent] = []
    slow = ControlledTool("slow")
    fast = ControlledTool("fast")
    gamma = ControlledTool("gamma", gated=False)
    completions: list[str] = []
    slow.completions = completions
    fast.completions = completions
    gamma.completions = completions

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "slow", {"value": "a"}), ("c2", "fast", {"value": "b"})]),
            lambda: text_message("after tools"),
            lambda: text_message("after follow"),
            lambda: text_message("extra"),
        ]
    )

    steer = user_message("steer", 10)
    follow_up = user_message("later", 11)
    finish_calls = 0
    prepare_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 2:
            agent_ref[0].follow_up(follow_up)
            return "continue"
        return None

    def prepare_request(request: PrepareRequestContext, signal: AbortSignal | None) -> AgentRequestUpdate | None:
        del request, signal
        if stream.calls == 0:
            return AgentRequestUpdate(model=make_model("dyn-model", reasoning=True), thinking_level="high")
        return None

    def prepare_next_turn(context: AgentTurnContext, signal: AbortSignal | None) -> AgentLoopTurnUpdate | None:
        del signal
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls > 1:
            return None
        replacement = AgentContext(
            messages=list(context.context.messages),
            tools=[slow.agent_tool("slow redefined"), gamma.agent_tool()],
        )
        return AgentLoopTurnUpdate(
            context=replacement,
            messages=[SystemMessage(content="", timestamp=20, sections={"style": None, "tone": "warm"})],
        )

    def transform_context(messages: list[object], signal: AbortSignal | None) -> list[object]:
        del signal
        return [message for message in messages if not (isinstance(message, NoticeMessage) and message.text == "drop")]

    def convert_to_llm(messages: list[object]) -> list[Message]:
        return [
            user_message(f"notice:{message.text}", 5) if isinstance(message, NoticeMessage) else message  # type: ignore[arg-type]
            for message in messages
        ]

    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            transform_context=transform_context,  # type: ignore[arg-type]
            convert_to_llm=convert_to_llm,  # type: ignore[arg-type]
            prepare_request=prepare_request,
            prepare_next_turn_with_context=prepare_next_turn,
            finish_turn=finish_turn,
            initial_state=AgentInitialState(
                system_prompt="base prompt",
                model=make_model(),
                tools=[slow.agent_tool(), fast.agent_tool()],
            ),
        )
    )
    agent_ref.append(agent)
    events = collect_events(agent)

    prompts: list[object] = [
        SystemMessage(content="", timestamp=1, sections={"style": "brief"}),
        NoticeMessage("keep"),
        NoticeMessage("drop"),
        user_message("work", 2),
    ]

    run = asyncio.create_task(agent.prompt(prompts))  # type: ignore[arg-type]
    await asyncio.wait_for(slow.started.wait(), timeout=1)
    await asyncio.wait_for(fast.started.wait(), timeout=1)
    agent.steer(steer)
    fast.release.set()
    await wait_until(lambda: "c2" in end_event_ids(events))
    slow.release.set()
    await asyncio.wait_for(run, timeout=1)

    # Each request carries the transcript the loop said it would.
    assert stream.calls == 3
    assert [request.model.id for request in stream.requests] == ["dyn-model"] * 3
    assert [request.options.reasoning for request in stream.requests if request.options] == ["high"] * 3

    assert declared_tools(stream.requests[0].context) == {
        "slow": "controlled tool",
        "fast": "controlled tool",
    }
    assert merged_sections(stream.requests[0].context) == {"style": "brief"}

    # The second request announces the added, removed, and redefined tools.
    assert declared_tools(stream.requests[1].context) == {"slow": "slow redefined", "gamma": "controlled tool"}
    second_system = [message for message in stream.requests[1].context.messages if isinstance(message, SystemMessage)]
    removed = {removed.name for message in second_system for removed in (message.tools_removed or [])}
    assert removed == {"slow", "fast"}
    assert merged_sections(stream.requests[1].context) == {"style": None, "tone": "warm"}
    assert [user_text(message) for message in request_users(stream.requests[1])] == [
        "notice:keep",
        "work",
        "steer",
    ]

    assert declared_tools(stream.requests[2].context) == {"slow": "slow redefined", "gamma": "controlled tool"}
    assert [user_text(message) for message in request_users(stream.requests[2])] == [
        "notice:keep",
        "work",
        "steer",
        "later",
    ]

    # Out-of-order completion is visible on events; final tool messages keep source order.
    assert [event.tool_call_id for event in events if isinstance(event, ToolExecutionEndEvent)] == ["c2", "c1"]
    assert [message.tool_call_id for message in tool_results(agent)] == ["c1", "c2"]
    assert completions == ["fast", "slow"]
    assert gamma.calls == []

    # The custom message conversion is a request projection only; application history keeps both notices.
    assert any(isinstance(message, NoticeMessage) and message.text == "keep" for message in agent.state.messages)
    assert any(isinstance(message, NoticeMessage) and message.text == "drop" for message in agent.state.messages)
    assert not any(isinstance(message, NoticeMessage) for message in stream.requests[0].context.messages)
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.content == [TextContent(text="after follow")]
    assert final.thinking_level == "high"


# ---------------------------------------------------------------------------
# Spec acceptance: finish_turn decisions with natural continuation and queues
# ---------------------------------------------------------------------------


async def test_finish_turn_decisions_do_not_duplicate_requests_or_consume_queues_twice() -> None:
    echo = ControlledTool("echo", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("after steer"),
            lambda: text_message("after follow"),
        ]
    )
    agent_ref: list[Agent] = []
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)
    finish_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str | None:
        del context, signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            agent_ref[0].steer(steer)
            agent_ref[0].follow_up(follow_up)
            return "continue"
        return None

    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            finish_turn=finish_turn,
            initial_state=AgentInitialState(model=make_model(), tools=[echo.agent_tool()]),
        )
    )
    agent_ref.append(agent)

    await asyncio.wait_for(agent.prompt("go"), timeout=1)

    # finish_turn "continue" plus the natural tool continuation plus both queues
    # produce exactly one request per new input and no extra scheduling request.
    assert stream.calls == 3
    assert finish_calls == 3
    assert [[user_text(message) for message in request_users(request)] for request in stream.requests] == [
        ["go"],
        ["go", "steer"],
        ["go", "steer", "later"],
    ]
    assert [message.tool_call_id for message in tool_results(agent)] == ["c1"]
    assert echo.calls == [("c1", {"value": "x"})]
    assert agent.has_queued_messages() is False


async def test_finish_turn_end_with_natural_continuation_stops_and_keeps_queues() -> None:
    echo = ControlledTool("echo", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("never requested"),
        ]
    )
    agent_ref: list[Agent] = []
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> str:
        del context, signal
        agent_ref[0].steer(steer)
        agent_ref[0].follow_up(follow_up)
        return "end"

    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            finish_turn=finish_turn,
            initial_state=AgentInitialState(model=make_model(), tools=[echo.agent_tool()]),
        )
    )
    agent_ref.append(agent)

    await asyncio.wait_for(agent.prompt("go"), timeout=1)

    assert stream.calls == 1
    assert [message.tool_call_id for message in tool_results(agent)] == ["c1"]
    # "end" stops before any queue poll, so both messages stay queued.
    assert agent.peek_queued_messages() == [steer]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [follow_up]


# ---------------------------------------------------------------------------
# Spec acceptance: observation cancellation and explicit abort in one run
# ---------------------------------------------------------------------------


async def test_cancelled_waiter_keeps_observer_and_abort_settles_with_remaining_queue() -> None:
    first = ControlledTool("first")
    second = ControlledTool("second", gated=False)
    side_effects: list[str] = []
    first.completions = side_effects
    second.completions = side_effects

    def stream_fn(
        model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        del model, context
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        if options is not None and options.signal is not None and options.signal.aborted:
            stream.push(
                ErrorEvent(  # type: ignore[arg-type]
                    reason="aborted",
                    error=text_message("", "aborted", error_message="aborted by caller"),
                )
            )
        else:
            stream.push(
                DoneEvent(  # type: ignore[arg-type]
                    reason="toolUse",
                    message=tool_call_message(
                        [("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]
                    ),
                )
            )
        return stream

    agent_ref: list[Agent] = []
    steer = user_message("steer", 1)
    follow_up = user_message("later", 2)
    finish_calls = 0

    def finish_turn(context: AgentTurnContext, signal: AbortSignal | None) -> None:
        del context, signal
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            agent_ref[0].steer(steer)
            agent_ref[0].follow_up(follow_up)

    agent = Agent(
        AgentOptions(
            stream_fn=stream_fn,
            finish_turn=finish_turn,
            tool_execution="sequential",
            initial_state=AgentInitialState(model=make_model(), tools=[first.agent_tool(), second.agent_tool()]),
        )
    )
    agent_ref.append(agent)

    terminal_settled = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        del signal
        if isinstance(event, AgentEndEvent):
            await asyncio.sleep(0)
            terminal_settled.set()

    agent.subscribe(listener)

    waiter_a = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(first.started.wait(), timeout=1)
    waiter_b = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)

    waiter_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_a
    # Cancelling the first waiter does not cancel the run or the other observer.
    assert waiter_b.done() is False
    assert agent.state.is_streaming is True

    agent.abort()
    first.release.set()
    await asyncio.wait_for(waiter_b, timeout=1)
    await asyncio.wait_for(terminal_settled.wait(), timeout=1)

    # The running tool kept its side effect; the next serial call was skipped
    # once the abort signal was observed.
    assert first.calls == [("c1", {"value": "a"})]
    assert second.calls == []
    assert side_effects == ["first"]
    assert agent.state.pending_tool_calls == frozenset()
    assert agent.state.is_streaming is False
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"
    # Abort does not drain the queues; the follow-up queued during the run remains.
    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [follow_up]


# ---------------------------------------------------------------------------
# Spec acceptance: standalone loop combined capabilities and cancellation
# ---------------------------------------------------------------------------


def _combined_loop_config(pending_steers: list[object]) -> AgentLoopConfig:
    def get_steering() -> list[object]:
        return [pending_steers.pop(0)] if pending_steers else []

    def convert_to_llm(messages: list[object]) -> list[object]:
        return [
            user_message(f"notice:{message.text}", 4) if isinstance(message, NoticeMessage) else message
            for message in messages
        ]

    return AgentLoopConfig(
        model=make_model(),
        get_steering_messages=get_steering,  # type: ignore[arg-type]
        convert_to_llm=convert_to_llm,  # type: ignore[arg-type]
    )


def _request_user_texts(request: CapturedRequest) -> list[str]:
    return [user_text(message) for message in request_users(request)]


async def test_standalone_loop_combined_capabilities_match_across_entries() -> None:
    direct_tool = ControlledTool("echo", gated=False)
    direct_stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    direct_events: list[AgentEvent] = []
    direct_messages = list(
        await run_agent_loop(
            [user_message("go", 1), NoticeMessage("note")],  # type: ignore[list-item]
            AgentContext(messages=[], tools=[direct_tool.agent_tool()]),
            _combined_loop_config([user_message("steer", 5)]),
            direct_events.append,
            None,
            direct_stream,
        )
    )

    tool = ControlledTool("echo", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    loop_stream = agent_loop(
        [user_message("go", 1), NoticeMessage("note")],  # type: ignore[list-item]
        AgentContext(messages=[], tools=[tool.agent_tool()]),
        _combined_loop_config([user_message("steer", 5)]),
        None,
        stream,
    )
    streamed_events = [event async for event in loop_stream]
    streamed_messages = await loop_stream.result()

    # Both entries make the same two requests, project the custom message the same
    # way, and inject the steering message through the initial poll.
    assert direct_stream.calls == stream.calls == 2
    assert [_request_user_texts(request) for request in direct_stream.requests] == [
        ["go", "notice:note", "steer"],
        ["go", "notice:note", "steer"],
    ]
    assert [_request_user_texts(request) for request in stream.requests] == [
        ["go", "notice:note", "steer"],
        ["go", "notice:note", "steer"],
    ]
    assert [type(event) for event in streamed_events] == [type(event) for event in direct_events]
    assert [getattr(message, "role", None) for message in streamed_messages] == [
        getattr(message, "role", None) for message in direct_messages
    ]
    assert [message.tool_call_id for message in tool_results_from(streamed_messages)] == ["c1"]
    assert direct_tool.calls == [("c1", {"value": "x"})]
    assert tool.calls == [("c1", {"value": "x"})]


def tool_results_from(messages: list[object]) -> list[ToolResultMessage]:
    return [message for message in messages if isinstance(message, ToolResultMessage)]


async def test_standalone_loop_direct_cancel_interrupts_combined_run_without_hang() -> None:
    tool = ControlledTool("slow")
    stream = ScriptedStreamFn([lambda: tool_call_message([("c1", "slow", {"value": "a"})])])
    context = AgentContext(messages=[user_message("go")], tools=[tool.agent_tool()])
    config = AgentLoopConfig(model=make_model())
    events: list[AgentEvent] = []

    task = asyncio.create_task(
        run_agent_loop(list(context.messages), context, config, events.append, None, stream)
    )
    await asyncio.wait_for(tool.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # A cancelled direct run does not fabricate a normal terminal event sequence.
    assert not any(isinstance(event, AgentEndEvent) for event in events)
    assert tool.calls == [("c1", {"value": "a"})]


async def test_standalone_loop_reader_cancel_does_not_stop_combined_producer() -> None:
    tool = ControlledTool("echo", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    loop_stream = agent_loop(
        [user_message("go", 1)],
        AgentContext(messages=[], tools=[tool.agent_tool()]),
        AgentLoopConfig(model=make_model()),
        None,
        stream,
    )

    iterator = loop_stream.__aiter__()
    await anext(iterator)
    await iterator.aclose()

    # Stopping the read does not cancel the producer or break its final result.
    messages = await asyncio.wait_for(loop_stream.result(), timeout=1)
    assert [getattr(message, "role", None) for message in messages][-1] == "assistant"
    assert stream.calls == 2
    assert tool.calls == [("c1", {"value": "x"})]


async def test_standalone_loop_producer_failure_converges_without_hang() -> None:
    def boom_stream_fn(
        model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        del model, context, options
        raise RuntimeError("producer boom")

    loop_stream = agent_loop(
        [user_message("go", 1)],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        None,
        boom_stream_fn,
    )

    with pytest.raises(RuntimeError, match="producer boom"):
        async for _ in loop_stream:
            pass
    with pytest.raises(RuntimeError, match="producer boom"):
        await asyncio.wait_for(loop_stream.result(), timeout=1)
