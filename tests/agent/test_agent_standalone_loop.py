"""Public behavior tests for the standalone agent-loop entries.

The standalone loop is a second public boundary beside the Agent: it runs a
model conversation without a Session or Agent and owns its producer or caller
task according to the entry used.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from omh.agent import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    agent_loop,
    agent_loop_continue,
    clear_default_stream_fn,
    run_agent_loop,
    run_agent_loop_continue,
    set_default_stream_fn,
)
from omh.llm.types import (
    AbortController,
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


def user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=NOW)


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


class BlockingStreamFn:
    """StreamFn whose request stays open until the test pushes events into it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stream: AssistantMessageEventStream | None = None

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        del model, context, options
        stream = create_assistant_message_event_stream()
        self.stream = stream
        self.started.set()
        return stream


class CooperativeSignalStreamFn:
    """StreamFn that completes with an aborted message when its signal fires."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        del model, context
        stream = create_assistant_message_event_stream()
        self.started.set()
        signal = options.signal if options is not None else None

        def on_abort() -> None:
            stream.push(
                ErrorEvent(
                    reason="aborted",
                    error=text_message("", stop_reason="aborted", error_message="aborted"),
                )
            )

        if signal is not None:
            signal.add_callback(on_abort)
        return stream


def echo_tool(executed: list[dict[str, object]]) -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: Callable[[AgentToolResult], None],
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        executed.append(args)
        return AgentToolResult(content=[TextContent(text=f"echo:{args['value']}")], details={})

    return AgentTool(
        name="echo",
        description="Echo a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        label="Echo",
        execute=execute,
    )


def event_types(events: list[AgentEvent]) -> list[str]:
    return [type(event).__name__ for event in events]


def assert_no_terminal_end(events: list[AgentEvent]) -> None:
    assert not any(isinstance(event, AgentEndEvent) for event in events)


# ---------------------------------------------------------------------------
# L01: direct prompt/continue return values and context ownership
# ---------------------------------------------------------------------------


async def test_direct_prompt_returns_accepted_messages_and_copies_inputs() -> None:
    stream = ScriptedStreamFn([lambda: text_message("hi back")])
    prompts = [user_message("hello")]
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=make_model())

    result = await run_agent_loop(prompts, context, config, lambda event: None, None, stream)

    assert [message.role for message in result] == ["user", "assistant"]
    assert result[0] is prompts[0]
    assert result[-1].content == [TextContent(text="hi back")]
    # The prompt list and the passed context are not mutated by a new prompt.
    assert prompts == [result[0]]
    assert context.messages == []


async def test_direct_prompt_includes_needed_tool_declaration() -> None:
    executed: list[dict[str, object]] = []
    stream = ScriptedStreamFn([lambda: text_message("done")])
    prompts = [user_message("hello")]
    context = AgentContext(messages=[], tools=[echo_tool(executed)])
    config = AgentLoopConfig(model=make_model())

    result = await run_agent_loop(prompts, context, config, lambda event: None, None, stream)

    assert [message.role for message in result] == ["system", "user", "assistant"]
    declaration = result[0]
    assert isinstance(declaration, SystemMessage)
    assert declaration.tools_added is not None
    assert declaration.tools_added[0].name == "echo"


async def test_direct_continue_appends_to_context_and_returns_only_new_messages() -> None:
    stream = ScriptedStreamFn([lambda: text_message("continued")])
    context = AgentContext(messages=[user_message("hello")], tools=[])
    config = AgentLoopConfig(model=make_model())

    result = await run_agent_loop_continue(context, config, lambda event: None, None, stream)

    assert [message.role for message in result] == ["assistant"]
    assert result[-1].content == [TextContent(text="continued")]
    # Continuation appends the new messages to the caller's list in place.
    assert [message.role for message in context.messages] == ["user", "assistant"]
    assert context.messages[-1] is result[-1]


async def test_direct_continue_rejects_empty_and_assistant_tail() -> None:
    config = AgentLoopConfig(model=make_model())
    stream = ScriptedStreamFn([lambda: text_message("never")])

    with pytest.raises(ValueError, match="Cannot continue: no messages"):
        await run_agent_loop_continue(AgentContext(messages=[], tools=[]), config, lambda event: None, None, stream)

    with pytest.raises(ValueError, match="Cannot continue from message role: assistant"):
        await run_agent_loop_continue(
            AgentContext(messages=[text_message("done")], tools=[]),
            config,
            lambda event: None,
            None,
            stream,
        )
    assert stream.calls == 0


# ---------------------------------------------------------------------------
# L02: stream entries mirror the direct entries
# ---------------------------------------------------------------------------


async def test_stream_prompt_matches_direct_prompt() -> None:
    direct_stream = ScriptedStreamFn([lambda: text_message("hi back")])
    direct_events: list[AgentEvent] = []
    direct_result = await run_agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        direct_events.append,
        None,
        direct_stream,
    )

    stream_stream = ScriptedStreamFn([lambda: text_message("hi back")])
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        None,
        stream_stream,
    )
    streamed_events = [event async for event in loop_stream]
    streamed_result = await loop_stream.result()

    assert event_types(streamed_events) == event_types(direct_events)
    assert [message.role for message in streamed_result] == [message.role for message in direct_result]
    assert streamed_result[-1].content == direct_result[-1].content
    assert stream_stream.calls == direct_stream.calls == 1


async def test_stream_continue_matches_direct_continue() -> None:
    direct_stream = ScriptedStreamFn([lambda: text_message("continued")])
    direct_events: list[AgentEvent] = []
    direct_context = AgentContext(messages=[user_message("hello")], tools=[])
    direct_result = await run_agent_loop_continue(
        direct_context,
        AgentLoopConfig(model=make_model()),
        direct_events.append,
        None,
        direct_stream,
    )

    stream_stream = ScriptedStreamFn([lambda: text_message("continued")])
    stream_context = AgentContext(messages=[user_message("hello")], tools=[])
    loop_stream = agent_loop_continue(
        stream_context,
        AgentLoopConfig(model=make_model()),
        None,
        stream_stream,
    )
    streamed_events = [event async for event in loop_stream]
    streamed_result = await loop_stream.result()

    assert event_types(streamed_events) == event_types(direct_events)
    assert [message.role for message in streamed_result] == [message.role for message in direct_result]
    assert [message.role for message in stream_context.messages] == ["user", "assistant"]


def test_stream_continue_rejects_before_starting_a_producer() -> None:
    config = AgentLoopConfig(model=make_model())

    with pytest.raises(ValueError, match="Cannot continue: no messages"):
        agent_loop_continue(AgentContext(messages=[], tools=[]), config)

    with pytest.raises(ValueError, match="Cannot continue from message role: assistant"):
        agent_loop_continue(AgentContext(messages=[text_message("done")], tools=[]), config)


async def test_direct_entry_keeps_local_request_overrides_for_later_requests() -> None:
    """The standalone loop keeps its local replacement value across a run."""
    stream = ScriptedStreamFn([lambda: text_message("first"), lambda: text_message("second")])
    turns = 0

    def finish_turn(turn: object, signal: AbortSignal | None) -> str | None:
        del turn, signal
        nonlocal turns
        turns += 1
        return "continue" if turns == 1 else None

    def prepare_request(request: object, signal: AbortSignal | None) -> AgentRequestUpdate | None:
        del request, signal
        if stream.calls == 0:
            return AgentRequestUpdate(model=make_model("override-model"))
        return None

    result = await run_agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model(), prepare_request=prepare_request, finish_turn=finish_turn),
        lambda event: None,
        None,
        stream,
    )

    assert [message.role for message in result] == ["user", "assistant", "assistant"]
    assert [request.model.id for request in stream.requests] == ["override-model", "override-model"]


async def test_stream_producer_failure_reaches_reader_and_result_waiter() -> None:
    def prepare_request(request: object, signal: AbortSignal | None) -> None:
        del request, signal
        raise RuntimeError("prepare failed")

    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model(), prepare_request=prepare_request),
        None,
        ScriptedStreamFn([lambda: text_message("never")]),
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        async for _ in loop_stream:
            pass

    with pytest.raises(RuntimeError, match="prepare failed"):
        await asyncio.wait_for(loop_stream.result(), 1)


class _ProducerBoom(BaseException):
    """A non-Exception failure that must still converge instead of hanging."""


async def test_stream_producer_base_exception_does_not_hang_result() -> None:
    def boom_stream_fn(
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        del model, context, options
        raise _ProducerBoom("boom")

    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        None,
        boom_stream_fn,
    )

    with pytest.raises(_ProducerBoom, match="boom"):
        await asyncio.wait_for(loop_stream.result(), 1)


# ---------------------------------------------------------------------------
# L03: cancellation and producer ownership
# ---------------------------------------------------------------------------


async def test_stopping_stream_read_does_not_stop_producer() -> None:
    stream_fn = ScriptedStreamFn([lambda: text_message("done")])
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        None,
        stream_fn,
    )

    reader = loop_stream.__aiter__()
    first = await asyncio.wait_for(reader.__anext__(), 1)
    assert type(first).__name__ == "AgentStartEvent"
    await reader.aclose()

    result = await asyncio.wait_for(loop_stream.result(), 1)
    assert [message.role for message in result] == ["user", "assistant"]
    assert stream_fn.calls == 1
    assert loop_stream.task is not None and not loop_stream.task.cancelled()


async def test_cancelling_a_reader_does_not_stop_producer() -> None:
    stream_fn = BlockingStreamFn()
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        None,
        stream_fn,
    )
    await asyncio.wait_for(stream_fn.started.wait(), 1)

    async def drain() -> list[AgentEvent]:
        return [event async for event in loop_stream]

    reader_task = asyncio.ensure_future(drain())
    await asyncio.sleep(0)
    reader_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader_task

    assert not loop_stream.task.cancelled()  # type: ignore[union-attr]
    assert stream_fn.stream is not None
    stream_fn.stream.push(StartEvent(partial=text_message("")))
    stream_fn.stream.push(DoneEvent(reason="stop", message=text_message("done")))

    result = await asyncio.wait_for(loop_stream.result(), 1)
    assert [message.role for message in result] == ["user", "assistant"]


async def test_cancelling_one_result_waiter_keeps_the_producer_and_other_waiters() -> None:
    stream_fn = BlockingStreamFn()
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        None,
        stream_fn,
    )
    await asyncio.wait_for(stream_fn.started.wait(), 1)

    first_waiter = asyncio.ensure_future(loop_stream.result())
    second_waiter = asyncio.ensure_future(loop_stream.result())
    await asyncio.sleep(0)
    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    assert not loop_stream.task.cancelled()  # type: ignore[union-attr]
    assert stream_fn.stream is not None
    stream_fn.stream.push(StartEvent(partial=text_message("")))
    stream_fn.stream.push(DoneEvent(reason="stop", message=text_message("done")))

    result = await asyncio.wait_for(second_waiter, 1)
    assert [message.role for message in result] == ["user", "assistant"]
    assert not loop_stream.task.cancelled()  # type: ignore[union-attr]


async def test_cancelling_direct_loop_task_interrupts_without_terminal_events() -> None:
    stream_fn = BlockingStreamFn()
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(
        run_agent_loop(
            [user_message("hello")],
            AgentContext(messages=[], tools=[]),
            AgentLoopConfig(model=make_model()),
            events.append,
            None,
            stream_fn,
        )
    )
    await asyncio.wait_for(stream_fn.started.wait(), 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert event_types(events) == [
        "AgentStartEvent",
        "TurnStartEvent",
        "MessageStartEvent",
        "MessageEndEvent",
    ]
    assert_no_terminal_end(events)


async def test_explicit_signal_lets_direct_entry_finish_cooperatively() -> None:
    controller = AbortController()
    stream_fn = CooperativeSignalStreamFn()
    task = asyncio.ensure_future(
        run_agent_loop(
            [user_message("hello")],
            AgentContext(messages=[], tools=[]),
            AgentLoopConfig(model=make_model()),
            lambda event: None,
            controller.signal,
            stream_fn,
        )
    )
    await asyncio.wait_for(stream_fn.started.wait(), 1)

    controller.abort()
    result = await asyncio.wait_for(task, 1)

    assert isinstance(result[-1], AssistantMessage)
    assert result[-1].stop_reason == "aborted"


async def test_explicit_signal_lets_stream_producer_finish_cooperatively() -> None:
    controller = AbortController()
    stream_fn = CooperativeSignalStreamFn()
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model()),
        controller.signal,
        stream_fn,
    )
    await asyncio.wait_for(stream_fn.started.wait(), 1)

    controller.abort()
    result = await asyncio.wait_for(loop_stream.result(), 1)

    assert isinstance(result[-1], AssistantMessage)
    assert result[-1].stop_reason == "aborted"
    assert loop_stream.task is not None and not loop_stream.task.cancelled()


# ---------------------------------------------------------------------------
# Integration: tools, hooks, and queues through the standalone entries
# ---------------------------------------------------------------------------


async def test_stream_entry_executes_tools_and_runs_turn_hooks() -> None:
    executed: list[dict[str, object]] = []
    seen_turns: list[object] = []

    def finish_turn(turn: object, signal: AbortSignal | None) -> None:
        del signal
        seen_turns.append(turn)

    stream_fn = ScriptedStreamFn(
        [lambda: tool_call_message("echo", {"value": "1"}), lambda: text_message("done")]
    )
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[echo_tool(executed)]),
        AgentLoopConfig(model=make_model(), finish_turn=finish_turn),
        None,
        stream_fn,
    )

    await asyncio.wait_for(loop_stream.result(), 1)

    assert executed == [{"value": "1"}]
    assert stream_fn.calls == 2
    assert len(seen_turns) == 2


async def test_stream_entry_polls_steering_messages() -> None:
    queued = [user_message("steer me")]

    def get_steering_messages() -> list[object]:
        return [queued.pop(0)] if queued else []

    stream_fn = ScriptedStreamFn([lambda: text_message("done")])
    loop_stream = agent_loop(
        [user_message("hello")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=make_model(), get_steering_messages=get_steering_messages),
        None,
        stream_fn,
    )

    result = await asyncio.wait_for(loop_stream.result(), 1)

    assert [message.role for message in result] == ["user", "user", "assistant"]
    assert result[1].content == "steer me"


# ---------------------------------------------------------------------------
# Default StreamFn fallback for the standalone entries
# ---------------------------------------------------------------------------


async def test_direct_entry_uses_installed_default_stream_fn() -> None:
    stream = ScriptedStreamFn([lambda: text_message("default")])
    set_default_stream_fn(stream)
    try:
        result = await run_agent_loop(
            [user_message("hello")],
            AgentContext(messages=[], tools=[]),
            AgentLoopConfig(model=make_model()),
            lambda event: None,
        )
    finally:
        clear_default_stream_fn()

    assert result[-1].content == [TextContent(text="default")]


async def test_direct_entry_fails_clearly_without_a_stream_fn() -> None:
    clear_default_stream_fn()
    with pytest.raises(RuntimeError, match="No default StreamFn"):
        await run_agent_loop(
            [user_message("hello")],
            AgentContext(messages=[], tools=[]),
            AgentLoopConfig(model=make_model()),
            lambda event: None,
        )
