from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping

from omh.agent import (
    BACKGROUND_CONTEXT,
    AfterToolHook,
    AfterToolResult,
    AgentHarness,
    AgentHarnessOptions,
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentHarnessToolUpdateOptions,
    AgentToolResult,
    BeforeDriveHook,
    BeforeRequestHook,
    BeforeRunEndHook,
    BeforeRunEndResult,
    BeforeRunHook,
    BeforeRunResult,
    BeforeToolHook,
    BeforeToolResult,
    ConfigUpdateEvent,
    Context,
    EntryAddedEvent,
    HandlerErrorEvent,
    HookOptions,
    MemorySessionRepo,
    MessageEndEvent,
    MessageStartEvent,
    ModelIdentity,
    PromptRequest,
    RetryEndEvent,
    RetryPolicy,
    RetryScheduledEvent,
    RetryStartEvent,
    RunEndEvent,
    RunningToolSnapshot,
    RunStartEvent,
    SessionCreateOptions,
    ToolEndEvent,
    ToolStartEvent,
    ToolUpdateEvent,
    TransformContextHook,
    TransformContextResult,
    UsageEvent,
    with_telemetry_context,
)
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    Model,
    StartEvent,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
    UserMessage,
)
from omh.llm import Context as LlmContext

NOW = 1_700_000_000_000
MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="test",
    base_url="https://example.invalid",
    reasoning=False,
    input=("text",),
    cost=UsageCost(),
    context_window=8_192,
    max_tokens=1_024,
)
USAGE = Usage(
    input=4,
    output=2,
    cache_read=0,
    cache_write=0,
    total_tokens=6,
    cost=UsageCost(),
)


class Models:
    def get_model(self, provider: str, model_id: str) -> Model | None:
        return MODEL if (provider, model_id) == (MODEL.provider, MODEL.id) else None


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status = "ok"

    def set_attributes(self, attributes: Mapping[str, object]) -> None:
        self.attributes.update(attributes)

    def set_status(self, status: str) -> None:
        self.status = status


class RecordingTelemetryContext:
    def __init__(self) -> None:
        self.started: list[tuple[str, dict[str, object], RecordingSpan]] = []

    async def start_span(self, name, attributes, callback):
        span = RecordingSpan()
        self.started.append((name, dict(attributes), span))
        result = callback(span, self)
        return await result if inspect.isawaitable(result) else result


def _stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


class RetryingModels(Models):
    def __init__(self) -> None:
        self.attempts = 0

    def stream_simple(self, model: Model, context: object, options: object):
        del model, context, options
        self.attempts += 1
        if self.attempts == 1:
            return _stream(
                AssistantMessage(
                    api=MODEL.api,
                    provider=MODEL.provider,
                    model=MODEL.id,
                    usage=USAGE,
                    stop_reason="error",
                    error_message="rate limit",
                    timestamp=NOW,
                    content=[],
                )
            )
        return _stream(
            AssistantMessage(
                api=MODEL.api,
                provider=MODEL.provider,
                model=MODEL.id,
                usage=USAGE,
                stop_reason="stop",
                timestamp=NOW + 1,
                content=[TextContent(text="done")],
            )
        )


class ToolCallingModels(Models):
    def __init__(self) -> None:
        self.requests = 0

    def stream_simple(self, model: Model, context: object, options: object):
        del model, context, options
        self.requests += 1
        if self.requests == 1:
            return _stream(
                AssistantMessage(
                    api=MODEL.api,
                    provider=MODEL.provider,
                    model=MODEL.id,
                    usage=USAGE,
                    stop_reason="toolUse",
                    timestamp=NOW,
                    content=[
                        ToolCall(
                            id="call-1",
                            name="add",
                            arguments={"left": 2, "right": 3},
                        )
                    ],
                )
            )
        return _stream(
            AssistantMessage(
                api=MODEL.api,
                provider=MODEL.provider,
                model=MODEL.id,
                usage=USAGE,
                stop_reason="stop",
                timestamp=NOW + 1,
                content=[TextContent(text="done")],
            )
        )


class CapturingModels(Models):
    def __init__(self) -> None:
        self.contexts: list[LlmContext] = []

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del model, options
        self.contexts.append(context)
        return _stream(
            AssistantMessage(
                api=MODEL.api,
                provider=MODEL.provider,
                model=MODEL.id,
                usage=USAGE,
                stop_reason="stop",
                timestamp=NOW,
                content=[TextContent(text="done")],
            )
        )


async def test_config_listener_error_does_not_roll_back_and_preserves_context() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    harness = created.harness
    lane = await harness.lane("main", BACKGROUND_CONTEXT)
    config_events: list[tuple[ConfigUpdateEvent, Context]] = []
    handler_errors: list[tuple[HandlerErrorEvent, Context]] = []

    async def fail_listener(event: ConfigUpdateEvent, context: Context) -> None:
        config_events.append((event, context))
        raise RuntimeError("display failed")

    async def capture_error(event: HandlerErrorEvent, context: Context) -> None:
        handler_errors.append((event, context))

    harness.events.on("config_update", fail_listener)
    harness.events.on("handler_error", capture_error)
    call_context = Context()

    await lane.set_model(
        ModelIdentity(provider="missing", model_id="replacement"), call_context
    )

    assert await lane.get_model(BACKGROUND_CONTEXT) is None
    assert config_events == [
        (
            ConfigUpdateEvent(
                lane="main",
                property="model",
                previous=ModelIdentity(provider="test", model_id="test-model"),
                value=ModelIdentity(provider="missing", model_id="replacement"),
            ),
            call_context,
        )
    ]
    assert handler_errors[0][0].kind == "event"
    assert handler_errors[0][0].event == "config_update"
    assert handler_errors[0][0].error == "display failed"
    assert handler_errors[0][0].lane == "main"
    assert handler_errors[0][1] is call_context

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_lane_watch_buffers_events_after_its_coherent_snapshot() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    watch = await lane.watch(BACKGROUND_CONTEXT)
    replacement = ModelIdentity(provider="missing", model_id="replacement")
    await lane.set_model(replacement, BACKGROUND_CONTEXT)
    buffered: list[ConfigUpdateEvent] = []
    delivered = asyncio.Event()

    async def capture(event: ConfigUpdateEvent, _context: Context) -> None:
        buffered.append(event)
        delivered.set()

    watch.start(capture)
    await delivered.wait()

    assert watch.snapshot.configuration.model == ModelIdentity(
        provider="test", model_id="test-model"
    )
    assert buffered == [
        ConfigUpdateEvent(
            lane="main",
            property="model",
            previous=ModelIdentity(provider="test", model_id="test-model"),
            value=replacement,
        )
    ]
    assert (
        await watch.resnapshot(BACKGROUND_CONTEXT)
    ).configuration.model == replacement

    delivered.clear()
    await created.harness.set_tools((), BACKGROUND_CONTEXT)
    await delivered.wait()
    assert buffered[-1] == ConfigUpdateEvent(property="tools")

    watch.unsubscribe()
    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_accept_emits_committed_prompt_lifecycle_in_order() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    observed: list[tuple[object, Context]] = []

    async def capture(event: object, context: Context) -> None:
        observed.append((event, context))

    for event_type in ("run_start", "message_start", "message_end", "entry_added"):
        created.harness.events.on(event_type, capture)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    call_context = Context()

    admission = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run-1"), call_context
    )

    assert admission.ok
    assert [event.type for event, _ in observed] == [
        "run_start",
        "message_start",
        "message_end",
        "entry_added",
    ]
    assert isinstance(observed[0][0], RunStartEvent)
    assert observed[0][0].run_id == "run-1"
    assert isinstance(observed[1][0], MessageStartEvent)
    assert isinstance(observed[2][0], MessageEndEvent)
    assert isinstance(observed[3][0], EntryAddedEvent)
    assert observed[2][0].entry_id == observed[3][0].entry.id
    assert all(context is call_context for _, context in observed)

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_hook_order_and_retry_boundary_preserve_context() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RetryingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
        ),
        BACKGROUND_CONTEXT,
    )
    calls: list[tuple[str, int | None, Context]] = []

    async def before_drive(event: object, context: Context) -> None:
        assert isinstance(event, BeforeDriveHook)
        calls.append(("drive", None, context))

    async def first_request(event: object, context: Context) -> None:
        assert isinstance(event, BeforeRequestHook)
        calls.append(("first", event.attempt, context))

    async def second_request(event: object, context: Context) -> None:
        assert isinstance(event, BeforeRequestHook)
        calls.append(("second", event.attempt, context))

    created.harness.hooks.on("before_drive", before_drive)
    created.harness.hooks.on("before_request", first_request)
    created.harness.hooks.on("before_request", second_request)
    retry_events: list[object] = []

    async def capture_retry(event: object, _context: Context) -> None:
        retry_events.append(event)

    for event_type in ("retry_scheduled", "retry_start", "retry_end"):
        created.harness.events.on(event_type, capture_retry)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    call_context = Context()

    result = await lane.prompt("hello", call_context)

    assert result.ok
    hook_context = calls[0][2]
    assert calls == [
        ("drive", None, hook_context),
        ("first", 1, hook_context),
        ("second", 1, hook_context),
        ("first", 2, hook_context),
        ("second", 2, hook_context),
    ]
    assert [event.type for event in retry_events] == [
        "retry_scheduled",
        "retry_start",
        "retry_end",
    ]
    assert isinstance(retry_events[0], RetryScheduledEvent)
    assert retry_events[0].attempt == 2
    assert isinstance(retry_events[1], RetryStartEvent)
    assert retry_events[1].attempt == 2
    assert isinstance(retry_events[2], RetryEndEvent)
    assert retry_events[2].success

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_before_run_and_transform_context_hooks_chain_in_order() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = CapturingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    hook_order: list[str] = []
    first = UserMessage(content="first injection", timestamp=NOW + 1)
    second = UserMessage(content="second injection", timestamp=NOW + 2)

    async def before_one(event: object, _context: Context) -> BeforeRunResult:
        assert isinstance(event, BeforeRunHook)
        hook_order.append("before-one")
        assert len(event.prompt) == 1
        return BeforeRunResult(messages=(first,))

    async def before_two(event: object, _context: Context) -> BeforeRunResult:
        assert isinstance(event, BeforeRunHook)
        hook_order.append("before-two")
        assert event.prompt[-1] == first
        return BeforeRunResult(messages=(second,))

    async def transform_one(event: object, _context: Context) -> TransformContextResult:
        assert isinstance(event, TransformContextHook)
        hook_order.append("transform-one")
        assert event.messages[-2:] == (first, second)
        return TransformContextResult(
            messages=event.messages[-2:], system_prompt="policy"
        )

    async def transform_two(event: object, _context: Context) -> TransformContextResult:
        assert isinstance(event, TransformContextHook)
        hook_order.append("transform-two")
        assert event.system_prompt == "policy"
        return TransformContextResult(messages=event.messages[-1:])

    created.harness.hooks.on("before_run", before_one)
    created.harness.hooks.on("before_run", before_two)
    created.harness.hooks.on("transform_context", transform_one)
    created.harness.hooks.on("transform_context", transform_two)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("original", BACKGROUND_CONTEXT)

    assert result.ok
    assert hook_order == [
        "before-one",
        "before-two",
        "transform-one",
        "transform-two",
    ]
    assert len(models.contexts) == 1
    assert models.contexts[0].messages == [second]
    assert models.contexts[0].system_prompt == "policy"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_before_run_end_follow_up_reenters_the_existing_run() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = CapturingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    calls = 0

    async def continue_once(
        event: object, _context: Context
    ) -> BeforeRunEndResult | None:
        nonlocal calls
        assert isinstance(event, BeforeRunEndHook)
        calls += 1
        if calls == 1:
            return BeforeRunEndResult(follow_up="one more turn")
        return None

    created.harness.hooks.on("before_run_end", continue_once)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("original", BACKGROUND_CONTEXT)

    assert result.ok
    assert calls == 2
    assert len(models.contexts) == 2
    follow_up = models.contexts[1].messages[-1]
    assert isinstance(follow_up, UserMessage)
    assert follow_up.content == "one more turn"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_tool_hooks_progress_usage_and_terminal_events_are_observable() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    hook_order: list[str] = []
    executed_args: list[dict[str, object]] = []

    async def execute(
        _tool_call_id: str,
        arguments: dict[str, object],
        on_update: AgentHarnessToolUpdateCallback,
        _tool_context: object,
        _invocation: AgentHarnessToolInvocation,
        _context: Context,
    ) -> AgentToolResult:
        executed_args.append(arguments)
        on_update(AgentToolResult(content=[TextContent(text="working")]))
        return AgentToolResult(content=[TextContent(text="raw")])

    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=ToolCallingModels(),
            model=MODEL,
            tools=(
                AgentHarnessTool(
                    name="add",
                    description="Add numbers",
                    parameters={"type": "object"},
                    execute=execute,
                ),
            ),
        ),
        BACKGROUND_CONTEXT,
    )

    async def before_one(event: object, _context: Context) -> BeforeToolResult:
        assert isinstance(event, BeforeToolHook)
        hook_order.append("before-one")
        assert event.args == {"left": 2, "right": 3}
        return BeforeToolResult(args={"left": 2, "right": 4})

    async def before_two(event: object, _context: Context) -> BeforeToolResult:
        assert isinstance(event, BeforeToolHook)
        hook_order.append("before-two")
        assert event.args == {"left": 2, "right": 4}
        return BeforeToolResult(args={"left": 2, "right": 5})

    async def after_one(event: object, _context: Context) -> AfterToolResult:
        assert isinstance(event, AfterToolHook)
        hook_order.append("after-one")
        assert event.content == [TextContent(text="raw")]
        return AfterToolResult(content=[TextContent(text="hook-one")])

    async def after_two(event: object, _context: Context) -> AfterToolResult:
        assert isinstance(event, AfterToolHook)
        hook_order.append("after-two")
        assert event.content == [TextContent(text="hook-one")]
        return AfterToolResult(content=[TextContent(text="hook-two")])

    created.harness.hooks.on("before_tool", before_one, HookOptions(id="before-one"))
    created.harness.hooks.on("before_tool", before_two)
    created.harness.hooks.on("after_tool", after_one)
    created.harness.hooks.on("after_tool", after_two)
    observed: list[object] = []

    async def capture(event: object, _context: Context) -> None:
        observed.append(event)

    for event_type in (
        "tool_start",
        "tool_update",
        "tool_end",
        "entry_added",
        "usage",
        "run_end",
    ):
        created.harness.events.on(event_type, capture)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    telemetry = RecordingTelemetryContext()
    invocation_context = with_telemetry_context(telemetry, BACKGROUND_CONTEXT)

    result = await lane.prompt("calculate", invocation_context)

    assert result.ok
    assert executed_args == [{"left": 2, "right": 5}]
    assert hook_order == ["before-one", "before-two", "after-one", "after-two"]
    tool_events = [
        event
        for event in observed
        if isinstance(event, ToolStartEvent | ToolUpdateEvent | ToolEndEvent)
    ]
    assert [event.type for event in tool_events] == [
        "tool_start",
        "tool_update",
        "tool_end",
    ]
    assert isinstance(tool_events[1], ToolUpdateEvent)
    assert tool_events[1].partial_result.content == [TextContent(text="working")]
    assert isinstance(tool_events[2], ToolEndEvent)
    assert tool_events[2].result.content == [TextContent(text="hook-two")]
    tool_entry = next(
        event
        for event in observed
        if isinstance(event, EntryAddedEvent)
        and event.entry.type == "message"
        and event.entry.message.role == "toolResult"
    )
    assert observed.index(tool_events[2]) < observed.index(tool_entry)
    usage_events = [event for event in observed if isinstance(event, UsageEvent)]
    assert len(usage_events) == 2
    assert usage_events[-1].totals.total_tokens == 12
    terminal = next(event for event in observed if isinstance(event, RunEndEvent))
    assert terminal.status == "completed"
    assert [item[0] for item in telemetry.started] == [
        "pi.harness.hook",
        "pi.harness.hook",
        "pi.harness.hook",
        "pi.harness.hook",
    ]
    assert telemetry.started[0][1]["pi.hook.registration_id"] == "before-one"
    assert [item[2].attributes["pi.hook.outcome"] for item in telemetry.started] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_lane_snapshot_reconstructs_a_running_tool_checkpoint() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    checkpoint_requested = asyncio.Event()
    release = asyncio.Event()
    checkpoint = AgentToolResult(content=[TextContent(text="halfway")])

    async def execute(
        _tool_call_id: str,
        _arguments: dict[str, object],
        on_update: AgentHarnessToolUpdateCallback,
        _tool_context: object,
        _invocation: AgentHarnessToolInvocation,
        _context: Context,
    ) -> AgentToolResult:
        on_update(checkpoint, AgentHarnessToolUpdateOptions(checkpoint=True))
        checkpoint_requested.set()
        await release.wait()
        return AgentToolResult(content=[TextContent(text="complete")])

    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=ToolCallingModels(),
            model=MODEL,
            tools=(
                AgentHarnessTool(
                    name="add",
                    description="Add numbers",
                    parameters={"type": "object"},
                    execute=execute,
                ),
            ),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    prompt = asyncio.create_task(lane.prompt("calculate", BACKGROUND_CONTEXT))
    await checkpoint_requested.wait()

    running_tool: RunningToolSnapshot | None = None
    for _ in range(20):
        watch = await lane.watch(BACKGROUND_CONTEXT)
        operation = watch.snapshot.operation
        watch.unsubscribe()
        if operation is not None and operation.running_tools:
            tool = operation.running_tools[0]
            if isinstance(tool, RunningToolSnapshot) and tool.result is not None:
                running_tool = tool
                break
        await asyncio.sleep(0)

    assert running_tool == RunningToolSnapshot(
        tool_call_id="call-1",
        tool_name="add",
        args={"left": 2, "right": 3},
        result=checkpoint,
    )

    release.set()
    assert (await prompt).ok
    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
