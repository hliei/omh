"""Conversation history and data ownership through the public Agent boundary."""

from datetime import UTC
from uuid import UUID

import pytest

from omh.agent import (
    AfterToolCallResult,
    Agent,
    AgentContext,
    AgentEvent,
    AgentInitialState,
    AgentLoopConfig,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CustomAgentMessage,
    run_agent_loop,
)
from omh.llm.types import (
    AbortSignal,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tests.agent.test_agent_conversation import (
    RecordingStreamFn,
    make_model,
    text_message,
)


def test_new_conversation_has_stable_identity_and_linked_initial_records() -> None:
    seed = UserMessage(content="seed", timestamp=123)
    agent = Agent(AgentOptions(
        stream_fn=RecordingStreamFn(),
        initial_state=AgentInitialState(model=make_model(), thinking_level="medium", messages=[seed]),
    ))

    history = agent.history
    assert UUID(history.conversation_id).version == 7
    assert history.created_at.tzinfo == UTC
    assert [entry.type for entry in history.entries] == [
        "model_change", "thinking_level_change", "message",
    ]
    assert history.entries[0].provider == "test"
    assert history.entries[0].model_id == "test-model"
    assert history.entries[1].thinking_level == "medium"
    assert history.entries[2].message == seed
    assert history.entries[2].message is not seed
    assert history.entries[0].parent_id is None
    assert [entry.parent_id for entry in history.entries[1:]] == [
        history.entries[0].id, history.entries[1].id,
    ]
    assert len({entry.id for entry in history.entries}) == 3
    assert all(entry.timestamp.tzinfo == UTC for entry in history.entries)
    assert history.leaf_id == history.entries[-1].id
    assert agent.history == history

    explicit = Agent(AgentOptions(stream_fn=RecordingStreamFn(), conversation_id="host-conversation"))
    assert explicit.history.conversation_id == "host-conversation"
    assert explicit.history.conversation_id != history.conversation_id


async def test_final_messages_commit_before_message_end_with_current_context() -> None:
    stream = RecordingStreamFn()
    stream.set_script("update")

    def response():
        message = text_message("done")
        if stream.calls == 1:
            message.content = [ToolCall(id="call", name="echo", arguments={})]
            message.stop_reason = "toolUse"
        return message

    stream.set_result(response)

    def execute(call_id, args, signal, on_update):
        on_update(AgentToolResult(content=[TextContent(text="progress")], details={"step": 1}))
        return AgentToolResult(content=[TextContent(text="result")], details={"answer": [42]})

    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[AgentTool(
            name="echo", description="Echo", parameters={"type": "object"}, label="Echo", execute=execute,
        )]),
    ))
    initial = agent.history
    events: list[AgentEvent] = []
    committed = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if event.type == "history_commit":
            committed.extend(event.entries)
            assert event.conversation_id == initial.conversation_id
            assert event.leaf_id == agent.history.leaf_id
            assert event.entries[-1] == agent.history.entries[-1]
        if event.type == "message_end":
            assert events[-2].type == "history_commit"
            assert agent.history.entries[-1].message == event.message
            assert agent.state.messages[-1] == event.message

    agent.subscribe(listener)
    await agent.prompt("go")

    assert tuple(committed) == agent.history.entries[len(initial.entries):]
    assert [entry.message.role for entry in committed] == ["user", "assistant", "toolResult", "assistant"]
    assert agent.history.leaf_id != initial.leaf_id
    assert not any(
        entry.message.content == [TextContent(text="progress")]
        for entry in committed
    )
    assert agent.history.entries[:len(initial.entries)] == initial.entries


async def test_inputs_queues_readers_and_each_listener_have_isolated_messages() -> None:
    seed = ToolResultMessage(
        tool_call_id="seed", tool_name="echo", timestamp=1,
        content=[TextContent(text="seed")], details={"nested": ["original"]},
    )
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model(), messages=[seed]),
    ))
    seed.details["nested"].append("caller mutation")
    seed.content[0].text = "changed"

    queued = UserMessage(content=[TextContent(text="queued")], timestamp=2)
    agent.steer(queued)
    queued.content[0].text = "caller mutation"
    peek = agent.peek_queued_messages()[0]
    peek.content[0].text = "peek mutation"

    observed = []

    def mutate(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "message_end" and event.message.role == "user":
            event.message.content[0].text = "listener mutation"
        if event.type == "history_commit":
            event.entries[-1].message.content[0].text = "history listener mutation"

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "message_end" and event.message.role == "user":
            observed.append(event.message.content[0].text)

    agent.subscribe(mutate)
    agent.subscribe(observe)
    prompt = UserMessage(content=[TextContent(text="prompt")], timestamp=3)
    await agent.prompt(prompt)
    prompt.content[0].text = "changed later"

    snapshot = agent.state.messages
    snapshot[0].details["nested"].append("reader mutation")
    agent.history.entries[-1].message.content[0].text = "reader mutation"
    with pytest.raises(AttributeError):
        agent.state.messages = []
    with pytest.raises(AttributeError):
        snapshot.append(prompt)
    for name in ("model", "thinking_level", "tools", "is_streaming", "streaming_message", "error_message"):
        with pytest.raises(AttributeError):
            setattr(agent.state, name, None)
    assert not hasattr(agent, "reset")

    await agent.prompt("again")
    request = stream.requests[-1].context.messages
    assert request[0].content == [TextContent(text="seed")]
    assert request[0].details == {"nested": ["original"]}
    assert request[1].content == [TextContent(text="prompt")]
    assert request[2].content == [TextContent(text="queued")]
    assert observed == ["prompt", "queued", "again"]


async def test_custom_content_enters_request_without_display_metadata() -> None:
    stream = RecordingStreamFn()
    custom = CustomAgentMessage(
        custom_type="notice", content=[TextContent(text="context")], display=False,
        details={"private": ["display data"]}, timestamp=123456,
    )
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    await agent.prompt(custom)

    request = stream.requests[0].context.messages
    assert request == [UserMessage(content=[TextContent(text="context")], timestamp=123456)]
    entry = next(entry for entry in agent.history.entries if entry.type == "custom_message")
    assert entry.custom_type == "notice"
    assert entry.content == [TextContent(text="context")]
    assert entry.details == {"private": ["display data"]}
    assert entry.display is False
    assert int(entry.timestamp.timestamp() * 1000) == 123456
    assert agent.state.messages[0] == custom


@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf"), {1: "non-string key"}, (1, 2)])
async def test_invalid_history_payload_is_rejected_before_input_or_queue_commit(bad) -> None:
    message = ToolResultMessage(
        tool_call_id="call", tool_name="echo", timestamp=1,
        content=[TextContent(text="result")], details={"nested": bad},
    )
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    initial = agent.history
    with pytest.raises(ValueError, match="JSON"):
        await agent.prompt(message)
    with pytest.raises(ValueError, match="JSON"):
        agent.steer(message)
    with pytest.raises(ValueError, match="JSON"):
        agent.follow_up(message)
    with pytest.raises(ValueError, match="JSON"):
        Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(messages=[message])))
    assert agent.history == initial
    assert not agent.has_queued_messages()
    assert stream.calls == 0


async def test_request_hooks_and_provider_cannot_change_later_requests_or_history() -> None:
    stream = RecordingStreamFn()
    seed = UserMessage(content=[TextContent(text="original")], timestamp=1)
    held = []

    def prepare(request, signal):
        request.context.messages[0].content[0].text = "prepare mutation"
        held.append(request.context)

    def transform(messages, signal):
        messages[0].content[0].text = "projection"
        return messages

    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare, transform_context=transform,
        initial_state=AgentInitialState(model=make_model(), messages=[seed]),
    ))
    await agent.prompt("first")
    stream.requests[0].context.messages[1].content[0].text = "provider mutation"
    held[0].messages[1].content[0].text = "retained hook mutation"
    await agent.prompt("second")

    assert stream.requests[-1].context.messages[1].content == [TextContent(text="first")]
    assert agent.state.messages[0] == seed
    assert agent.history.entries[2].message == seed


@pytest.mark.parametrize("source", ["tool", "after_hook"])
async def test_invalid_tool_details_become_saved_errors_without_undoing_effect(tmp_path, source) -> None:
    effect = tmp_path / "effect.txt"
    stream = RecordingStreamFn()

    def response():
        message = text_message("done")
        if stream.calls == 1:
            message.content = [ToolCall(id="call", name="write", arguments={})]
            message.stop_reason = "toolUse"
        return message

    def execute(call_id, args, signal, on_update):
        effect.write_text("written")
        return AgentToolResult(
            content=[TextContent(text="written")], details=object() if source == "tool" else {},
        )

    stream.set_result(response)
    agent = Agent(AgentOptions(
        stream_fn=stream,
        after_tool_call=(lambda context, signal: AfterToolCallResult(details=object())) if source == "after_hook" else None,
        initial_state=AgentInitialState(model=make_model(), tools=[AgentTool(
            name="write", description="Write", parameters={"type": "object"}, label="Write", execute=execute,
        )]),
    ))
    await agent.prompt("write")

    results = [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "JSON" in results[0].content[0].text
    assert results[0].details == {}
    assert effect.read_text() == "written"
    assert stream.calls == 2
    assert next(entry.message for entry in agent.history.entries if entry.type == "message" and entry.message.role == "toolResult") == results[0]


async def test_model_turns_keep_nested_message_views_isolated() -> None:
    stream = RecordingStreamFn()
    returned = []
    seen = []

    def response():
        message = text_message("done")
        if stream.calls == 1:
            message.content = [ToolCall(id="call", name="echo", arguments={"nested": {"value": "raw"}})]
            message.stop_reason = "toolUse"
        returned.append(message)
        return message

    def before(context, signal):
        context.tool_call.arguments["nested"]["value"] = "before mutation"
        context.assistant_message.content[0].arguments["nested"]["value"] = "before mutation"
        context.context.messages[1].content[0].text = "before mutation"

    def after(context, signal):
        context.result.details["nested"].append("after mutation")
        context.context.messages[1].content[0].text = "after mutation"

    def finish(context, signal):
        context.context.messages[1].content[0].text = "finish mutation"
        if context.tool_results:
            context.tool_results[0].details["nested"].append("finish mutation")

    def execute(call_id, args, signal, on_update):
        seen.append(args)
        # A provider may retain its result and the request it received.
        returned[0].content[0].arguments["nested"]["value"] = "provider result mutation"
        stream.requests[0].context.messages[1].content[0].text = "provider request mutation"
        return AgentToolResult(content=[TextContent(text="result")], details={"nested": ["original"]})

    stream.set_result(response)
    agent = Agent(AgentOptions(
        stream_fn=stream, before_tool_call=before, after_tool_call=after, finish_turn=finish,
        initial_state=AgentInitialState(
            model=make_model(), messages=[UserMessage(content=[TextContent(text="seed")], timestamp=1)],
            tools=[AgentTool(name="echo", description="Echo", parameters={}, label="Echo", execute=execute)],
        ),
    ))
    await agent.continue_()

    assert seen == [{"nested": {"value": "raw"}}]
    request = stream.requests[-1].context.messages
    assert request[1].content == [TextContent(text="seed")]
    assert request[2].content[0].arguments == {"nested": {"value": "raw"}}
    assert request[3].details == {"nested": ["original"]}


async def test_standalone_loop_keeps_open_application_messages_and_wide_details() -> None:
    class ApplicationMessage:
        role = "application"

        def __init__(self, resource):
            self.resource = resource

    resource = object()
    application = ApplicationMessage(resource)
    stream = RecordingStreamFn()

    def response():
        message = text_message("done")
        if stream.calls == 1:
            message.content = [ToolCall(id="call", name="echo", arguments={})]
            message.stop_reason = "toolUse"
        return message

    stream.set_result(response)
    result = await run_agent_loop(
        [application, UserMessage(content="go", timestamp=1)],
        AgentContext(messages=[], tools=[AgentTool(
            name="echo", description="Echo", parameters={}, label="Echo",
            execute=lambda *args: AgentToolResult(content=[TextContent(text="result")], details=resource),
        )]),
        AgentLoopConfig(model=make_model()), lambda event: None, stream_fn=stream,
    )
    assert application in result
    assert application.resource is resource
    assert next(message for message in result if isinstance(message, ToolResultMessage)).details is resource
    assert all(message.role != "application" for message in stream.requests[0].context.messages)


async def test_cyclic_extensions_are_rejected_and_bad_model_messages_do_not_commit() -> None:
    cyclic = []
    cyclic.append(cyclic)
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    with pytest.raises(ValueError, match="cycle"):
        await agent.prompt(CustomAgentMessage(custom_type="bad", content="bad", details=cyclic))

    def invalid_response():
        message = text_message("bad")
        message.content = [ToolCall(id="call", name="echo", arguments={"bad": object()})]
        return message

    stream.set_result(invalid_response)
    await agent.prompt("go")
    assert "JSON" in agent.state.error_message
    messages = [entry.message for entry in agent.history.entries if entry.type == "message"]
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].stop_reason == "error"
    assert messages[-1].content == [TextContent(text="")]


async def test_argument_preparation_cannot_rewrite_recorded_raw_arguments() -> None:
    stream = RecordingStreamFn()
    received = []

    def response():
        message = text_message("done")
        if stream.calls == 1:
            message.content = [ToolCall(id="call", name="echo", arguments={"value": "raw"})]
            message.stop_reason = "toolUse"
        return message

    def prepare(args):
        args["value"] = "prepared"
        return args

    def execute(call_id, args, signal, on_update):
        received.append(args)
        return AgentToolResult(content=[TextContent(text="result")])

    stream.set_result(response)
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[AgentTool(
            name="echo", description="Echo", parameters={}, label="Echo",
            execute=execute, prepare_arguments=prepare,
        )]),
    ))
    await agent.prompt("go")

    assert received == [{"value": "prepared"}]
    call = next(message for message in stream.requests[-1].context.messages if message.role == "assistant")
    assert call.content[0].arguments == {"value": "raw"}
    recorded = next(entry.message for entry in agent.history.entries if entry.type == "message" and entry.message.role == "assistant")
    assert recorded == call
