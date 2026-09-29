"""Public behavior tests for traditional Agent tool batches, hooks, and progress."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from omh.agent import (
    AfterToolCallContext,
    AfterToolCallResult,
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
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
    Usage,
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
    calls: list[tuple[str, str, dict[str, object]]],
    stop_reason: StopReason = "toolUse",
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
    result: AgentToolResult = field(
        default_factory=lambda: AgentToolResult(content=[TextContent(text="ok")], details={})
    )
    execution_mode: ToolExecutionMode | None = None
    error: Exception | None = None
    gated: bool = True
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    update_sink: AgentToolUpdateCallback | None = None
    seen_signal: AbortSignal | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        self.calls.append((tool_call_id, args))
        self.seen_signal = signal
        self.update_sink = on_update
        self.started.set()
        if self.gated:
            await self.release.wait()
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
            "execution_mode": self.execution_mode,
        }
        values.update(overrides)
        return AgentTool(**values)  # type: ignore[arg-type]


def collect_events(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    return events


def tool_end_names(events: list[AgentEvent]) -> list[str]:
    return [event.tool_name for event in events if isinstance(event, ToolExecutionEndEvent)]


def result_messages(agent: Agent) -> list[ToolResultMessage]:
    return [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]


# ---------------------------------------------------------------------------
# A11: batch scheduling and ordering
# ---------------------------------------------------------------------------


async def test_parallel_batch_completes_out_of_order_with_source_order_results() -> None:
    slow = ControlledTool(name="slow")
    fast = ControlledTool(name="fast")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "slow", {"value": "a"}), ("c2", "fast", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[slow.agent_tool(), fast.agent_tool()]),
        )
    )
    events = collect_events(agent)
    fast_ended = asyncio.Event()

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, ToolExecutionEndEvent) and event.tool_name == "fast":
            fast_ended.set()

    agent.subscribe(observe)

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(asyncio.gather(slow.started.wait(), fast.started.wait()), timeout=1)

    fast.release.set()
    await asyncio.wait_for(fast_ended.wait(), timeout=1)
    slow.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert slow.calls == [("c1", {"value": "a"})]
    assert fast.calls == [("c2", {"value": "b"})]
    assert tool_end_names(events) == ["fast", "slow"]
    assert [message.tool_call_id for message in result_messages(agent)] == ["c1", "c2"]
    assert stream.calls == 2


async def test_tool_sequential_override_makes_whole_batch_serial() -> None:
    first = ControlledTool(name="first", execution_mode="sequential")
    second = ControlledTool(name="second")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[first.agent_tool(), second.agent_tool()],
            ),
        )
    )

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(first.started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert second.started.is_set() is False

    first.release.set()
    await asyncio.wait_for(second.started.wait(), timeout=1)
    second.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert first.calls == [("c1", {"value": "a"})]
    assert second.calls == [("c2", {"value": "b"})]
    assert [message.tool_call_id for message in result_messages(agent)] == ["c1", "c2"]


async def test_global_sequential_runs_batch_serially() -> None:
    first = ControlledTool(name="first")
    second = ControlledTool(name="second")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            tool_execution="sequential",
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[first.agent_tool(), second.agent_tool()],
            ),
        )
    )

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(first.started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert second.started.is_set() is False

    first.release.set()
    await asyncio.wait_for(second.started.wait(), timeout=1)
    second.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert [message.tool_call_id for message in result_messages(agent)] == ["c1", "c2"]


async def test_parallel_preflight_finishes_in_source_order_before_any_execution() -> None:
    first = ControlledTool(name="first")
    second = ControlledTool(name="second")
    preflight_order: list[str] = []

    def before(context: BeforeToolCallContext, signal: AbortSignal | None) -> BeforeToolCallResult:
        preflight_order.append(context.tool_call.name)
        assert first.started.is_set() is False
        assert second.started.is_set() is False
        return BeforeToolCallResult()

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            before_tool_call=before,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[first.agent_tool(), second.agent_tool()],
            ),
        )
    )

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(asyncio.gather(first.started.wait(), second.started.wait()), timeout=1)
    first.release.set()
    second.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert preflight_order == ["first", "second"]


async def test_unknown_tool_in_parallel_batch_produces_error_without_execution() -> None:
    known = ControlledTool(name="known", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "missing", {"value": "a"}), ("c2", "known", {"value": "b"})]),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[known.agent_tool()]),
        )
    )
    events = collect_events(agent)

    await agent.prompt("go")

    assert known.calls == [("c2", {"value": "b"})]
    messages = result_messages(agent)
    assert [message.tool_call_id for message in messages] == ["c1", "c2"]
    assert messages[0].is_error is True
    assert "not found" in messages[0].content[0].text
    assert messages[1].is_error is False
    assert tool_end_names(events) == ["missing", "known"]


async def test_sequential_immediate_failure_produces_events_and_continues() -> None:
    known = ControlledTool(name="known", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "missing", {"value": "a"}), ("c2", "known", {"value": "b"})]),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            tool_execution="sequential",
            initial_state=AgentInitialState(model=make_model(), tools=[known.agent_tool()]),
        )
    )
    events = collect_events(agent)

    await agent.prompt("go")

    assert known.calls == [("c2", {"value": "b"})]
    messages = result_messages(agent)
    assert [message.tool_call_id for message in messages] == ["c1", "c2"]
    assert messages[0].is_error is True
    assert [event.tool_name for event in events if isinstance(event, ToolExecutionStartEvent)] == [
        "missing",
        "known",
    ]
    assert tool_end_names(events) == ["missing", "known"]


# ---------------------------------------------------------------------------
# A10 / A12: before and after tool hooks
# ---------------------------------------------------------------------------


async def test_before_tool_call_blocks_with_reason_and_skips_execution() -> None:
    tool = ControlledTool(name="echo")
    seen: list[dict[str, object]] = []

    def before(context: BeforeToolCallContext, signal: AbortSignal | None) -> BeforeToolCallResult:
        seen.append(context.args)
        assert isinstance(context.assistant_message, AssistantMessage)
        return BeforeToolCallResult(block=True, reason="blocked by policy")

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            before_tool_call=before,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == []
    assert seen == [{"value": "x"}]
    result = result_messages(agent)[0]
    assert result.is_error is True
    assert result.content[0].text == "blocked by policy"
    assert isinstance(agent.state.messages[-1], AssistantMessage)


async def test_before_tool_call_block_can_request_terminate() -> None:
    tool = ControlledTool(name="echo")

    def before(context: BeforeToolCallContext, signal: AbortSignal | None) -> BeforeToolCallResult:
        return BeforeToolCallResult(block=True, terminate=True)

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("should not run"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            before_tool_call=before,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert tool.calls == []
    assert stream.calls == 1
    assert result_messages(agent)[0].is_error is True


async def test_after_tool_call_replaces_fields_and_keeps_omitted() -> None:
    original_usage = Usage(
        input=1,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=2,
        cost=empty_usage().cost,
    )
    tool = ControlledTool(
        name="echo",
        gated=False,
        result=AgentToolResult(
            content=[TextContent(text="original")],
            details={"original": True},
            usage=original_usage,
        ),
    )
    captured: list[AfterToolCallContext] = []

    def after(context: AfterToolCallContext, signal: AbortSignal | None) -> AfterToolCallResult:
        captured.append(context)
        return AfterToolCallResult(
            content=[TextContent(text="replaced")],
            is_error=True,
            terminate=True,
        )

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            after_tool_call=after,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    # terminate replaces the flag; the batch stops before another model request.
    await agent.prompt("go")

    assert captured[0].result.content == [TextContent(text="original")]
    assert captured[0].is_error is False
    assert captured[0].args == {"value": "x"}
    assert stream.calls == 1
    result = result_messages(agent)[0]
    assert result.content == [TextContent(text="replaced")]
    assert result.is_error is True
    # Untouched fields keep the executed result, without a deep merge.
    assert result.details == {"original": True}
    assert result.usage == original_usage


async def test_after_tool_call_exception_becomes_error_result() -> None:
    tool = ControlledTool(name="echo", gated=False)

    def after(context: AfterToolCallContext, signal: AbortSignal | None) -> AfterToolCallResult:
        raise RuntimeError("after hook exploded")

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("recovered"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            after_tool_call=after,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    await agent.prompt("go")

    result = result_messages(agent)[0]
    assert result.is_error is True
    assert result.content[0].text == "after hook exploded"


# ---------------------------------------------------------------------------
# A12: progress
# ---------------------------------------------------------------------------


async def test_progress_updates_settle_before_end_and_late_updates_are_ignored() -> None:
    tool = ControlledTool(name="echo", gated=False)

    async def execute_with_progress(
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        on_update(AgentToolResult(content=[TextContent(text="half")], details={"step": 1}))
        on_update(AgentToolResult(content=[TextContent(text="done")], details={"step": 2}))
        tool.calls.append((tool_call_id, args))
        tool.update_sink = on_update
        return tool.result

    tool.execute = execute_with_progress  # type: ignore[method-assign]

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )
    events = collect_events(agent)

    await agent.prompt("go")

    updates = [event for event in events if isinstance(event, ToolExecutionUpdateEvent)]
    ends = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
    assert [event.partial_result.details for event in updates] == [{"step": 1}, {"step": 2}]
    assert events.index(updates[-1]) < events.index(ends[0])

    assert tool.update_sink is not None
    tool.update_sink(AgentToolResult(content=[TextContent(text="late")], details={"step": 3}))
    late_updates = [event for event in events if isinstance(event, ToolExecutionUpdateEvent)]
    assert len(late_updates) == 2


async def test_progress_is_delivered_while_tool_is_still_running() -> None:
    tool = ControlledTool(name="echo")
    update_sent = asyncio.Event()
    update_received = asyncio.Event()

    async def execute_with_progress(
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        on_update(AgentToolResult(content=[TextContent(text="working")], details={"step": 1}))
        update_sent.set()
        await tool.release.wait()
        tool.calls.append((tool_call_id, args))
        return tool.result

    tool.execute = execute_with_progress  # type: ignore[method-assign]

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, ToolExecutionUpdateEvent):
            update_received.set()

    agent.subscribe(observe)

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(update_sent.wait(), timeout=1)
    # The update must be observable before the tool is released.
    await asyncio.wait_for(update_received.wait(), timeout=1)
    assert tool.release.is_set() is False

    tool.release.set()
    await asyncio.wait_for(run, timeout=1)
    assert tool.calls == [("c1", {"value": "x"})]


# ---------------------------------------------------------------------------
# A14: pending tool state
# ---------------------------------------------------------------------------


async def test_pending_tool_calls_tracks_tool_event_order() -> None:
    tool = ControlledTool(name="echo")
    observed: list[frozenset[str]] = []
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, (ToolExecutionStartEvent, ToolExecutionEndEvent)):
            observed.append(agent.state.pending_tool_calls)

    agent.subscribe(observe)

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(tool.started.wait(), timeout=1)
    assert agent.state.pending_tool_calls == frozenset({"c1"})

    tool.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert observed[0] == frozenset({"c1"})
    assert observed[1] == frozenset()
    assert agent.state.pending_tool_calls == frozenset()


# ---------------------------------------------------------------------------
# A13: terminate aggregation
# ---------------------------------------------------------------------------


async def test_all_terminate_stops_subsequent_model_request() -> None:
    terminating = ControlledTool(
        name="terminating",
        gated=False,
        result=AgentToolResult(content=[TextContent(text="stop")], details={}, terminate=True),
    )
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "terminating", {"value": "x"})]),
            lambda: text_message("should not run"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[terminating.agent_tool()]),
        )
    )

    await agent.prompt("go")

    assert stream.calls == 1
    assert result_messages(agent)[0].tool_call_id == "c1"
    assert isinstance(agent.state.messages[-1], ToolResultMessage)


async def test_partial_terminate_continues_model_request() -> None:
    terminating = ControlledTool(
        name="terminating",
        gated=False,
        result=AgentToolResult(content=[TextContent(text="stop")], details={}, terminate=True),
    )
    plain = ControlledTool(name="plain", gated=False)
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message(
                [("c1", "terminating", {"value": "a"}), ("c2", "plain", {"value": "b"})]
            ),
            lambda: text_message("continued"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[terminating.agent_tool(), plain.agent_tool()],
            ),
        )
    )

    await agent.prompt("go")

    assert stream.calls == 2
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.content == [TextContent(text="continued")]


# ---------------------------------------------------------------------------
# A15: abort and observer cancellation
# ---------------------------------------------------------------------------


async def test_abort_during_preflight_marks_call_and_skips_remaining() -> None:
    first = ControlledTool(name="first")
    second = ControlledTool(name="second")
    agent_ref: list[Agent] = []

    def before(context: BeforeToolCallContext, signal: AbortSignal | None) -> BeforeToolCallResult:
        if context.tool_call.name == "first":
            agent_ref[0].abort()
        return BeforeToolCallResult()

    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            before_tool_call=before,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[first.agent_tool(), second.agent_tool()],
            ),
        )
    )
    agent_ref.append(agent)
    events = collect_events(agent)

    await asyncio.wait_for(agent.prompt("go"), timeout=1)

    assert first.calls == []
    assert second.calls == []
    assert [event.tool_name for event in events if isinstance(event, ToolExecutionStartEvent)] == ["first"]
    results = result_messages(agent)
    assert [message.tool_call_id for message in results] == ["c1"]
    assert results[0].is_error is True
    assert results[0].content[0].text == "Operation aborted"


async def test_abort_during_sequential_stops_remaining_calls() -> None:
    first = ControlledTool(name="first")
    second = ControlledTool(name="second")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            tool_execution="sequential",
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[first.agent_tool(), second.agent_tool()],
            ),
        )
    )

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(first.started.wait(), timeout=1)
    agent.abort()
    first.release.set()
    await asyncio.wait_for(run, timeout=1)

    assert first.calls == [("c1", {"value": "a"})]
    assert second.calls == []
    assert [message.tool_call_id for message in result_messages(agent)] == ["c1"]


async def test_abort_during_parallel_tool_execution_still_finalizes_running_tools() -> None:
    first = ControlledTool(name="first")
    second = ControlledTool(name="second")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "first", {"value": "a"}), ("c2", "second", {"value": "b"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                model=make_model(),
                tools=[first.agent_tool(), second.agent_tool()],
            ),
        )
    )

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(asyncio.gather(first.started.wait(), second.started.wait()), timeout=1)

    agent.abort()
    first.release.set()
    second.release.set()
    await asyncio.wait_for(run, timeout=1)

    # Cancellation is cooperative: tools already running are not preempted.
    assert first.calls == [("c1", {"value": "a"})]
    assert second.calls == [("c2", {"value": "b"})]
    assert [message.tool_call_id for message in result_messages(agent)] == ["c1", "c2"]
    assert agent.state.pending_tool_calls == frozenset()
    assert agent.state.is_streaming is False


async def test_cancelling_idle_waiter_does_not_stop_running_tool() -> None:
    tool = ControlledTool(name="echo")
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "echo", {"value": "x"})]),
            lambda: text_message("done"),
        ]
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
        )
    )

    run = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(tool.started.wait(), timeout=1)

    waiter = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    waiter.cancel()

    tool.release.set()
    await asyncio.wait_for(run, timeout=1)
    assert tool.calls == [("c1", {"value": "x"})]
    assert isinstance(agent.state.messages[-1], AssistantMessage)
