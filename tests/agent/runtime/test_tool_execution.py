from __future__ import annotations

import asyncio

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    TOOL_MEMO_UNSET,
    AgentHarness,
    AgentHarnessOptions,
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentToolResult,
    BranchScan,
    Context,
    DriveOptions,
    MemorySessionRepo,
    PromptRequest,
    Session,
    SessionCreateOptions,
    SessionMutator,
    ToolReplayPolicy,
    Write,
)
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    Model,
    StartEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
)
from omh.llm import (
    Context as LlmContext,
)
from omh.session_backends.sqlite import SqliteSessionRepo

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


def _stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


class ToolCallingModels:
    def __init__(self) -> None:
        self.contexts: list[LlmContext] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return MODEL if (provider, model_id) == (MODEL.provider, MODEL.id) else None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is MODEL
        self.contexts.append(context)
        if len(self.contexts) == 1:
            return _stream(
                AssistantMessage(
                    api=MODEL.api,
                    provider=MODEL.provider,
                    model=MODEL.id,
                    usage=USAGE,
                    stop_reason="toolUse",
                    timestamp=NOW + 1,
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
                timestamp=NOW + 2,
                content=[TextContent(text="The answer is 5")],
            )
        )


class InvalidToolArgumentsModels(ToolCallingModels):
    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        if self.contexts:
            return super().stream_simple(model, context, options)
        del options
        assert model is MODEL
        self.contexts.append(context)
        return _stream(
            AssistantMessage(
                api=MODEL.api,
                provider=MODEL.provider,
                model=MODEL.id,
                usage=USAGE,
                stop_reason="toolUse",
                timestamp=NOW + 1,
                content=[
                    ToolCall(
                        id="call-1",
                        name="add",
                        arguments={"left": 2},
                    )
                ],
            )
        )


class FinishingModels:
    def __init__(self) -> None:
        self.contexts: list[LlmContext] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return MODEL if (provider, model_id) == (MODEL.provider, MODEL.id) else None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is MODEL
        self.contexts.append(context)
        return _stream(
            AssistantMessage(
                api=MODEL.api,
                provider=MODEL.provider,
                model=MODEL.id,
                usage=USAGE,
                stop_reason="stop",
                timestamp=NOW + 2,
                content=[TextContent(text="finished")],
            )
        )


class PauseAfterToolStagingSession:
    def __init__(self, session: Session) -> None:
        self._session = session
        self.metadata = session.metadata
        self.id_generator = session.id_generator
        self.staged = asyncio.Event()

    def __getattr__(self, name: str) -> object:
        return getattr(self._session, name)

    async def mutate(self, mutation, context):
        owner = self

        class Mutator:
            def __init__(self, inner: SessionMutator) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

            async def commit(self, writes: list[Write], commit_context: Context):
                result = await self._inner.commit(writes, commit_context)
                if any(
                    write.kind == "value"
                    and write.op == "set"
                    and write.namespace == "pi.pending.entry"
                    for write in writes
                ):
                    owner.staged.set()
                    await asyncio.Future[None]()
                return result

        async def wrapped(mutator: SessionMutator, mutation_context: Context):
            return await mutation(Mutator(mutator), mutation_context)

        return await self._session.mutate(wrapped, context)


async def test_prompt_executes_active_tool_then_continues_model() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = ToolCallingModels()
    calls: list[tuple[str, dict[str, object], str, str, str]] = []

    async def execute_add(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        calls.append(
            (
                tool_call_id,
                params,
                invocation.invocation_id,
                invocation.operation_id,
                invocation.turn_id,
            )
        )
        assert context is BACKGROUND_CONTEXT
        return AgentToolResult(content=[TextContent(text="5")], details={"sum": 5})

    tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        execute=execute_add,
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            tools=(tool,),
            active_tool_names=("add",),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("what is 2 + 3?", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.status == "completed"
    assert await lane.get_active_tools(BACKGROUND_CONTEXT) == ("add",)
    assert await created.harness.get_tools(BACKGROUND_CONTEXT) == (tool,)
    assert len(calls) == 1
    assert calls[0][0:2] == ("call-1", {"left": 2, "right": 3})
    assert calls[0][2] != "call-1"
    assert calls[0][3] == result.value.operation_id
    assert models.contexts[0].tools is not None
    assert [item.name for item in models.contexts[0].tools] == ["add"]
    assert any(
        isinstance(message, ToolResultMessage)
        and message.tool_call_id == "call-1"
        and message.content == [TextContent(text="5")]
        for message in models.contexts[1].messages
    )
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [entry.message.role for entry in history if entry.type == "message"] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_invalid_tool_arguments_become_error_result_without_execution() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = InvalidToolArgumentsModels()
    calls = 0

    async def execute_add(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, invocation, context
        nonlocal calls
        calls += 1
        return AgentToolResult(content=[TextContent(text="unreachable")])

    tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        execute=execute_add,
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            tools=(tool,),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("what is 2 + 3?", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.status == "completed"
    assert calls == 0
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    tool_result = next(
        entry.message
        for entry in history
        if entry.type == "message" and isinstance(entry.message, ToolResultMessage)
    )
    assert tool_result.is_error is True
    assert tool_result.details is None
    assert "missing required property 'right'" in tool_result.content[0].text

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_terminating_tool_result_is_durable_and_skips_another_model_request(
    tmp_path,
) -> None:
    repo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = ToolCallingModels()

    async def execute_add(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, invocation, context
        return AgentToolResult(
            content=[TextContent(text="5")],
            terminate=True,
        )

    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            tools=(
                AgentHarnessTool(
                    name="add",
                    description="Add two integers",
                    parameters={"type": "object"},
                    execute=execute_add,
                ),
            ),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("what is 2 + 3?", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.status == "completed"
    assert len(models.contexts) == 1
    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path, now=lambda: NOW + 1)
    reopened_session = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=FinishingModels(),
            model=MODEL,
            tools=(),
        ),
        BACKGROUND_CONTEXT,
    )
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    tool_entry = next(
        entry
        for entry in history
        if entry.type == "message" and isinstance(entry.message, ToolResultMessage)
    )
    assert tool_entry.terminate is True

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_reopen_replays_only_safe_tool_with_stable_invocation_and_memo(
    tmp_path,
) -> None:
    repo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    first_identity: tuple[str, str, str] | None = None

    async def interrupt_after_memo(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, context
        nonlocal first_identity
        first_identity = (
            invocation.invocation_id,
            invocation.operation_id,
            invocation.turn_id,
        )
        await invocation.set_memo("request", {"id": "durable-request"})
        await invocation.set_memo("nullable", None)
        started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    first_tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={"type": "object"},
        execute=interrupt_after_memo,
        replay="safe",
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=ToolCallingModels(),
            model=MODEL,
            tools=(first_tool,),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="what is 2 + 3?", operation_id="run"),
        BACKGROUND_CONTEXT,
    )
    assert admitted.ok is True
    observing = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await started.wait()
    observing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observing
    assert not cancelled.is_set()
    await created.harness.close(BACKGROUND_CONTEXT)
    assert cancelled.is_set()
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path, now=lambda: NOW + 1)
    reopened_session = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    replayed: list[
        tuple[str, str, str, dict[str, object], object, object]
    ] = []

    async def finish_replay(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, context
        replayed.append(
            (
                invocation.invocation_id,
                invocation.operation_id,
                invocation.turn_id,
                params,
                await invocation.get_memo("request"),
                await invocation.get_memo("nullable"),
            )
        )
        assert await invocation.get_memo("missing") is TOOL_MEMO_UNSET
        return AgentToolResult(content=[TextContent(text="5")])

    replay_tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={"type": "object"},
        execute=finish_replay,
        replay="safe",
    )
    models = FinishingModels()
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=models,
            model=MODEL,
            tools=(replay_tool,),
        ),
        BACKGROUND_CONTEXT,
    )

    assert [(item.lane, item.operation_id) for item in reopened.open] == [
        ("main", "run")
    ]
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)
    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok is True
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.status == "completed"
    assert first_identity is not None
    assert replayed == [
        (
            *first_identity,
            {"left": 2, "right": 3},
            {"id": "durable-request"},
            None,
        )
    ]
    assert len(models.contexts) == 1

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


@pytest.mark.parametrize(
    ("stored_replay", "current_replay"),
    [("never", "safe"), ("safe", "never")],
)
async def test_reopen_interrupts_tool_unless_both_replay_declarations_are_safe(
    tmp_path,
    stored_replay: ToolReplayPolicy,
    current_replay: ToolReplayPolicy,
) -> None:
    repo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    started = asyncio.Event()

    async def interrupted_tool(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, invocation, context
        started.set()
        await asyncio.Future[None]()
        raise AssertionError("unreachable")

    first_tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={"type": "object"},
        execute=interrupted_tool,
        replay=stored_replay,
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=ToolCallingModels(),
            model=MODEL,
            tools=(first_tool,),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="what is 2 + 3?", operation_id="run"),
        BACKGROUND_CONTEXT,
    )
    assert admitted.ok is True
    observing = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await started.wait()
    await created.harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(asyncio.CancelledError):
        await observing
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path, now=lambda: NOW + 1)
    reopened_session = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    replay_calls = 0

    async def must_not_replay(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, invocation, context
        nonlocal replay_calls
        replay_calls += 1
        return AgentToolResult(content=[TextContent(text="unexpected")])

    current_tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={"type": "object"},
        execute=must_not_replay,
        replay=current_replay,
    )
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=FinishingModels(),
            model=MODEL,
            tools=(current_tool,),
        ),
        BACKGROUND_CONTEXT,
    )
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok is True
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.status == "completed"
    assert replay_calls == 0
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    tool_result = next(
        entry.message
        for entry in history
        if entry.type == "message" and isinstance(entry.message, ToolResultMessage)
    )
    assert tool_result.is_error is True
    assert tool_result.details is None
    assert "external outcome is unknown" in tool_result.content[0].text

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_reopen_materializes_staged_tool_result_without_rerunning_tool(
    tmp_path,
) -> None:
    repo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    stored_session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    session = PauseAfterToolStagingSession(stored_session)
    executions = 0

    async def execute_once(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, invocation, context
        nonlocal executions
        executions += 1
        return AgentToolResult(content=[TextContent(text="5")])

    tool = AgentHarnessTool(
        name="add",
        description="Add two integers",
        parameters={"type": "object"},
        execute=execute_once,
        replay="safe",
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,  # type: ignore[arg-type]
            models=ToolCallingModels(),
            model=MODEL,
            tools=(tool,),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="what is 2 + 3?", operation_id="run"),
        BACKGROUND_CONTEXT,
    )
    assert admitted.ok is True
    observing = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await session.staged.wait()
    assert executions == 1
    await created.harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(asyncio.CancelledError):
        await observing
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path, now=lambda: NOW + 1)
    reopened_session = await reopened_repo.open(
        stored_session.metadata, BACKGROUND_CONTEXT
    )
    replay_calls = 0

    async def must_not_execute(
        tool_call_id: str,
        params: dict[str, object],
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, params, invocation, context
        nonlocal replay_calls
        replay_calls += 1
        return AgentToolResult(content=[TextContent(text="unexpected")])

    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=FinishingModels(),
            model=MODEL,
            tools=(
                AgentHarnessTool(
                    name="add",
                    description="Add two integers",
                    parameters={"type": "object"},
                    execute=must_not_execute,
                    replay="safe",
                ),
            ),
        ),
        BACKGROUND_CONTEXT,
    )
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok is True
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.status == "completed"
    assert replay_calls == 0
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [entry.message.role for entry in history if entry.type == "message"] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)
