"""Public behavior tests for live configuration and per-request refresh.

These cover ARD-05: model/thinking/tool configuration changing during a run,
per-request ``prepare_request`` refresh, expected base system sections, and the
capacity preconditions for starting an activity. All tests use offline
``StreamFn``s and controlled tools.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentLoopTurnUpdate,
    AgentOptions,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    AgentTurnContext,
    HistoryCommitEvent,
    ModelChangeEvent,
    PrepareRequestContext,
    ThinkingLevelChangeEvent,
)
from omh.agent.conversation.history import (
    ModelChangeHistoryEntry,
    ThinkingLevelChangeHistoryEntry,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
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


def make_model(model_id: str = "test-model", reasoning: bool = True) -> Model:
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


def make_capacity_less_model() -> Model:
    return Model(
        id="no-capacity",
        name="No Capacity",
        api="openai-completions",
        provider="test",
        base_url="",
        reasoning=True,
        input=("text",),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=0,
        max_tokens=0,
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
    def __init__(self, turns: list[Callable[[], AssistantMessage]]) -> None:
        self.turns = turns
        self.requests: list[CapturedRequest] = []
        self.calls = 0

    def __call__(
        self, model: Model, context: TranscriptContext, options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        self.calls += 1
        self.requests.append(CapturedRequest(model=model, context=context, options=options))
        final = self.turns[min(self.calls - 1, len(self.turns) - 1)]()
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        stream.push(DoneEvent(reason=final.stop_reason, message=final))  # type: ignore[arg-type]
        return stream


def merged_sections(request: CapturedRequest) -> dict[str, str | None]:
    sections: dict[str, str | None] = {}
    for message in request.context.messages:
        if isinstance(message, SystemMessage):
            sections.update(message.sections or {})
    return sections


def declared_tools(request: CapturedRequest) -> dict[str, str]:
    tools: dict[str, str] = {}
    for message in request.context.messages:
        if not isinstance(message, SystemMessage):
            continue
        for removed in message.tools_removed or []:
            tools.pop(removed.name, None)
        for added in message.tools_added or []:
            tools[added.name] = added.description
    return tools


def simple_tool(name: str, execute) -> AgentTool:  # type: ignore[no-untyped-def]
    return AgentTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        label=name.title(),
        execute=execute,
    )


# ---------------------------------------------------------------------------
# Capacity preconditions
# ---------------------------------------------------------------------------


async def test_unconfigured_agent_is_inspectable_but_prompt_requires_capacity() -> None:
    stream = ScriptedStreamFn([lambda: text_message("never")])
    agent = Agent(AgentOptions(stream_fn=stream))

    assert agent.state.model.id == "unknown"
    assert agent.state.model.context_window == 0

    with pytest.raises(ValueError, match="context_window"):
        await agent.prompt("go")
    assert stream.calls == 0


async def test_continue_requires_a_capacity_bearing_model() -> None:
    stream = ScriptedStreamFn([lambda: text_message("never")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(messages=[UserMessage(content="seeded", timestamp=NOW)]),
    ))

    with pytest.raises(ValueError, match="max_tokens"):
        await agent.continue_()
    assert stream.calls == 0


async def test_host_custom_model_executes_without_a_registered_catalog() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model("custom-host-model")),
    ))

    await agent.prompt("go")

    assert stream.calls == 1
    assert stream.requests[0].model.id == "custom-host-model"


async def test_prepare_request_model_without_capacity_fails_the_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("never")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        prepare_request=lambda request, signal: AgentRequestUpdate(model=make_capacity_less_model()),
        initial_state=AgentInitialState(model=make_model()),
    ))

    await agent.prompt("go")

    assert stream.calls == 0
    assert agent.state.error_message is not None
    assert "context_window" in agent.state.error_message


# ---------------------------------------------------------------------------
# Live model and thinking selection
# ---------------------------------------------------------------------------


async def test_set_model_inside_tool_applies_to_next_request_and_records_change() -> None:
    agent_ref: list[Agent] = []

    async def switch(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        await agent_ref[0].set_model(make_model("second-model"))
        return AgentToolResult(content=[TextContent(text="switched")], details={})

    stream = ScriptedStreamFn([lambda: tool_call_message("switch", {"value": "x"}), lambda: text_message("done")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[simple_tool("switch", switch)]),
    ))
    agent_ref.append(agent)
    events: list[object] = []
    agent.subscribe(lambda event, signal: events.append(event))

    await agent.prompt("go")

    assert [request.model.id for request in stream.requests] == ["test-model", "second-model"]
    assert agent.state.model.id == "second-model"
    changes = [entry for entry in agent.history.entries if isinstance(entry, ModelChangeHistoryEntry)]
    assert [(entry.provider, entry.model_id) for entry in changes] == [("test", "test-model"), ("test", "second-model")]
    assert any(isinstance(event, ModelChangeEvent) for event in events)
    # The record precedes the configuration notification.
    model_event_index = next(index for index, event in enumerate(events) if isinstance(event, ModelChangeEvent))
    commit = events[model_event_index - 1]
    assert isinstance(commit, HistoryCommitEvent)
    assert isinstance(commit.entries[-1], ModelChangeHistoryEntry)


async def test_repeated_identical_model_selection_adds_no_record() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
    ))

    await agent.set_model(make_model())
    await agent.set_model(make_model("different-provider") if False else make_model())

    changes = [entry for entry in agent.history.entries if isinstance(entry, ModelChangeHistoryEntry)]
    assert len(changes) == 1


async def test_set_thinking_level_records_change_and_is_idempotent() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
    ))
    events: list[object] = []
    agent.subscribe(lambda event, signal: events.append(event))

    await agent.set_thinking_level("high")
    await agent.set_thinking_level("high")

    assert agent.state.thinking_level == "high"
    changes = [entry for entry in agent.history.entries if isinstance(entry, ThinkingLevelChangeHistoryEntry)]
    assert [entry.thinking_level for entry in changes] == ["off", "high"]
    assert sum(isinstance(event, ThinkingLevelChangeEvent) for event in events) == 1


async def test_set_thinking_level_clamps_to_a_non_reasoning_model() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model(reasoning=False)),
    ))

    await agent.set_thinking_level("high")

    assert agent.state.thinking_level == "off"
    changes = [entry for entry in agent.history.entries if isinstance(entry, ThinkingLevelChangeHistoryEntry)]
    assert [entry.thinking_level for entry in changes] == ["off"]


async def test_prepare_request_reads_live_selection_after_an_async_hook() -> None:
    agent_ref: list[Agent] = []

    async def prepare(request: PrepareRequestContext, signal: AbortSignal | None) -> None:
        del request, signal
        await agent_ref[0].set_model(make_model("switched-model"))
        return None

    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare,
        initial_state=AgentInitialState(model=make_model(), thinking_level="high"),
    ))
    agent_ref.append(agent)

    await agent.prompt("go")

    assert stream.requests[0].model.id == "switched-model"
    assert stream.requests[0].options is not None
    assert stream.requests[0].options.reasoning == "high"


# ---------------------------------------------------------------------------
# Live tools
# ---------------------------------------------------------------------------


async def test_set_tools_inside_a_tool_changes_the_next_request_declaration() -> None:
    agent_ref: list[Agent] = []

    async def swap(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        await agent_ref[0].set_tools([simple_tool("beta", args_echo)])
        return AgentToolResult(content=[TextContent(text="swapped")], details={})

    async def args_echo(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="beta")], details={})

    stream = ScriptedStreamFn([lambda: tool_call_message("swap", {"value": "x"}), lambda: text_message("done")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[simple_tool("swap", swap), simple_tool("alpha", args_echo)]),
    ))
    agent_ref.append(agent)

    await agent.prompt("go")

    assert declared_tools(stream.requests[0]) == {"swap": "swap tool", "alpha": "alpha tool"}
    assert declared_tools(stream.requests[1]) == {"beta": "beta tool"}
    # The declaration delta is committed as ordinary history before the request.
    delta_messages = [
        message for message in stream.requests[1].context.messages
        if isinstance(message, SystemMessage) and (message.tools_removed or message.tools_added)
    ]
    assert delta_messages
    removed = {removed.name for message in delta_messages for removed in (message.tools_removed or [])}
    assert removed == {"swap", "alpha"}


async def test_set_tools_while_idle_is_notified_before_the_next_prompt() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))

    async def echo(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    await agent.set_tools([simple_tool("late", echo)])
    await agent.prompt("go")

    assert declared_tools(stream.requests[0]) == {"late": "late tool"}


# ---------------------------------------------------------------------------
# Base system sections
# ---------------------------------------------------------------------------


async def test_set_system_sections_syncs_replace_and_delete_at_next_prompt() -> None:
    stream = ScriptedStreamFn([lambda: text_message("one"), lambda: text_message("two")])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(system_prompt="base", model=make_model()),
    ))

    await agent.set_system_sections({"style": "brief", "tone": "warm"})
    await agent.prompt("first")
    assert merged_sections(stream.requests[0]) == {"style": "brief", "tone": "warm"}

    await agent.set_system_sections({"style": "terse"})
    await agent.prompt("second")
    assert merged_sections(stream.requests[1]) == {"style": "terse", "tone": None}
    assert agent.state.system_sections == {"style": "terse"}


async def test_set_system_sections_during_a_prompt_applies_at_the_next_new_prompt() -> None:
    agent_ref: list[Agent] = []

    async def update(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        await agent_ref[0].set_system_sections({"style": "late"})
        return AgentToolResult(content=[TextContent(text="updated")], details={})

    stream = ScriptedStreamFn([lambda: tool_call_message("update", {"value": "x"}), lambda: text_message("done"), lambda: text_message("next")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(system_prompt="base", model=make_model(), tools=[simple_tool("update", update)]),
    ))
    agent_ref.append(agent)

    await agent.prompt("first")
    # The running prompt keeps its snapshot: no section change reaches the mid-run request.
    assert "style" not in merged_sections(stream.requests[1])
    # The next new prompt synchronizes the expected set.
    await agent.prompt("second")
    assert merged_sections(stream.requests[2]) == {"style": "late"}


async def test_bare_system_content_keeps_appending_alongside_sections() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(system_prompt="base", model=make_model()),
    ))
    await agent.set_system_sections({"style": "brief"})

    await agent.prompt([SystemMessage(content="extra", timestamp=1), UserMessage(content="go", timestamp=2)])

    assert agent.state.system_prompt == "base\n\nextra\n\nbrief"
    assert merged_sections(stream.requests[0]) == {"style": "brief"}


async def test_restored_agent_keeps_historical_base_sections() -> None:
    stream = ScriptedStreamFn([lambda: text_message("one")])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(system_prompt="base", model=make_model()),
    ))
    await agent.set_system_sections({"style": "brief"})
    await agent.prompt("first")

    restored_stream = ScriptedStreamFn([lambda: text_message("restored")])
    restored = Agent.from_history(agent.history, AgentOptions(
        stream_fn=restored_stream, initial_state=AgentInitialState(model=make_model()),
    ))

    assert restored.state.system_sections == {"style": "brief"}
    await restored.prompt("next")
    # The next prompt does not delete sections that history already carries.
    assert merged_sections(restored_stream.requests[0]) == {"style": "brief"}


# ---------------------------------------------------------------------------
# Next-turn migration and finish boundaries
# ---------------------------------------------------------------------------


async def test_prepare_next_turn_request_overrides_are_rejected() -> None:
    stream = ScriptedStreamFn([lambda: tool_call_message("echo", {"value": "x"}), lambda: text_message("never")])

    async def echo(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    def prepare_next_turn(context: AgentTurnContext, signal: AbortSignal | None) -> AgentLoopTurnUpdate:
        del context, signal
        return AgentLoopTurnUpdate(model=make_model("other"))

    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_next_turn_with_context=prepare_next_turn,
        initial_state=AgentInitialState(model=make_model(), tools=[simple_tool("echo", echo)]),
    ))

    await agent.prompt("go")

    assert stream.calls == 1
    assert agent.state.error_message is not None
    assert "prepare_request" in agent.state.error_message
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "error"


async def test_tool_batch_terminate_still_consumes_queued_steering() -> None:
    agent_ref: list[Agent] = []

    async def stop(tool_call_id: str, args: dict[str, object], signal: AbortSignal | None, on_update: object) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        agent_ref[0].steer(UserMessage(content="steer-after-terminate", timestamp=NOW + 1))
        return AgentToolResult(content=[TextContent(text="stop")], details={}, terminate=True)

    stream = ScriptedStreamFn([lambda: tool_call_message("stop", {"value": "x"}), lambda: text_message("after")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[simple_tool("stop", stop)]),
    ))
    agent_ref.append(agent)

    await agent.prompt("go")

    # terminate suppresses the natural tool continuation but does not cancel the activity or its queues.
    assert stream.calls == 2
    second_users = [
        message for message in stream.requests[1].context.messages if isinstance(message, UserMessage)
    ]
    assert second_users[-1].content == "steer-after-terminate"
