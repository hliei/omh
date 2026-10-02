"""Decoded history restoration through the public Agent boundary."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from omh.agent import (
    Agent,
    AgentHistory,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CompactionHistoryEntry,
    CompactionSummaryMessage,
    ContextEditHistoryEntry,
    ContextEditReplacement,
    CustomAgentMessage,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
    ModelChangeHistoryEntry,
    ThinkingLevelChangeHistoryEntry,
    validate_history,
)
from omh.llm.types import (
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.transcript import get_current_system_message, get_current_tools
from tests.agent.test_agent_conversation import (
    RecordingStreamFn,
    make_model,
    text_message,
)

SAVED_AT = datetime(2026, 10, 2, tzinfo=UTC)


def saved_history(entries, leaf_id=None):
    return AgentHistory(
        conversation_id="saved", created_at=SAVED_AT, entries=tuple(entries),
        leaf_id=entries[-1].id if leaf_id is None and entries else leaf_id,
    )


async def test_restore_preserves_records_and_identity_without_starting_work() -> None:
    stream = RecordingStreamFn()
    original = Agent(AgentOptions(
        stream_fn=stream, conversation_id="saved-conversation",
        initial_state=AgentInitialState(model=make_model(), messages=[
            UserMessage(content="saved input", timestamp=1),
        ]),
    ))
    original.steer(UserMessage(content="unsaved queue", timestamp=2))
    history = original.history
    settings = validate_history(history)
    assert (settings.provider, settings.model_id) == ("test", "test-model")
    assert settings.thinking_level == "off"
    assert settings.has_thinking_level

    restored = Agent.from_history(history, AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
    ))
    assert restored.history == history
    assert restored.state.messages == original.state.messages
    assert not restored.state.is_streaming
    assert restored.state.streaming_message is None
    assert restored.state.error_message is None
    assert restored.state.pending_tool_calls == frozenset()
    assert restored.signal is None
    assert not restored.has_queued_messages()
    await restored.wait_for_idle()
    assert stream.calls == 0

    await restored.continue_()
    assert stream.requests[0].context.messages == [UserMessage(content="saved input", timestamp=1)]
    assert restored.history.entries[:-1] == history.entries
    assert restored.history.entries[-1].parent_id == history.leaf_id
    assert restored.history.conversation_id == history.conversation_id


async def test_latest_compaction_checkpoint_tail_edits_and_new_messages_replay_once() -> None:
    checkpoint = SystemMessage(
        content="base\n\nappended", timestamp=0, sections={"keep": "checkpoint", "remove": "old"},
        tools_added=[Tool(name="old", description="Historical", parameters={})],
    )
    entries = [
        MessageHistoryEntry(id="root", parent_id=None, timestamp=SAVED_AT,
                            message=SystemMessage(content="obsolete", timestamp=0)),
        MessageHistoryEntry(id="old", parent_id="root", timestamp=SAVED_AT,
                            message=UserMessage(content="summarized", timestamp=1)),
        MessageHistoryEntry(id="tail", parent_id="old", timestamp=SAVED_AT,
                            message=UserMessage(content="kept", timestamp=2)),
        CompactionHistoryEntry(id="compact1", parent_id="tail", timestamp=SAVED_AT,
                               summary="old summary", first_kept_entry_id="tail", tokens_before=100,
                               system_message=SystemMessage(content="old checkpoint", timestamp=0)),
        MessageHistoryEntry(id="failed", parent_id="compact1", timestamp=SAVED_AT,
                            message=text_message("failed", "error", "transient")),
        ContextEditHistoryEntry(id="omit", parent_id="failed", timestamp=SAVED_AT,
                                target_id="failed", replacement=None),
        CompactionHistoryEntry(id="compact2", parent_id="omit", timestamp=SAVED_AT,
                               summary="latest summary", first_kept_entry_id="tail", tokens_before=200,
                               system_message=checkpoint, details={"read_files": ["a.py"]}),
        ContextEditHistoryEntry(id="replace1", parent_id="compact2", timestamp=SAVED_AT,
                                target_id="tail", replacement=ContextEditReplacement(content="first edit")),
        ContextEditHistoryEntry(id="replace2", parent_id="replace1", timestamp=SAVED_AT,
                                target_id="tail", replacement=ContextEditReplacement(content="last edit")),
        MessageHistoryEntry(id="delta", parent_id="replace2", timestamp=SAVED_AT,
                            message=SystemMessage(content="after", timestamp=3,
                                                  sections={"keep": "new", "remove": None},
                                                  tools_removed=[ToolReference(name="old")])),
        CustomMessageHistoryEntry(id="custom", parent_id="delta", timestamp=SAVED_AT,
                                  custom_type="note", content="custom context", display=False,
                                  details={"private": True}),
        MessageHistoryEntry(id="new", parent_id="custom", timestamp=SAVED_AT,
                            message=UserMessage(content="after compaction", timestamp=4)),
    ]
    history = saved_history(entries)
    stream = RecordingStreamFn()
    restored = Agent.from_history(history, AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
    ))
    messages = restored.state.messages
    assert [message.role for message in messages] == [
        "system", "compactionSummary", "user", "system", "custom", "user",
    ]
    assert messages[0] == checkpoint
    assert messages[1] == CompactionSummaryMessage(
        summary="latest summary", tokens_before=200, timestamp=int(SAVED_AT.timestamp() * 1000),
    )
    assert messages[2] == UserMessage(content="last edit", timestamp=2)
    assert get_current_system_message(messages) == SystemMessage(
        content="base\n\nappended\n\nafter", timestamp=0, sections={"keep": "new"},
    )
    assert restored.history == history

    await restored.continue_()
    request = stream.requests[0].context.messages
    assert [message.role for message in request] == ["system", "user", "user", "system", "user", "user"]
    assert "latest summary" in request[1].content[0].text
    assert "old summary" not in request[1].content[0].text
    assert request[4].content == "custom context"
    assert restored.history.entries[:-1] == history.entries


@pytest.mark.parametrize("kind, relation", [
    ("duplicate", "duplicate"), ("missing_parent", "parent_id"), ("cycle", "parent_id"),
    ("missing_leaf", "leaf_id"), ("missing_kept", "first_kept_entry_id"),
    ("sibling_kept", "first_kept_entry_id"), ("self_kept", "first_kept_entry_id"),
    ("missing_target", "target_id"), ("sibling_target", "target_id"),
    ("future_target", "target_id"), ("system_target", "target_id"),
    ("metadata_target", "target_id"),
])
def test_bad_relationships_are_rejected_by_validate_and_restore(kind, relation) -> None:
    root = MessageHistoryEntry(id="root", parent_id=None, timestamp=SAVED_AT,
                               message=SystemMessage(content="base", timestamp=0))
    user = MessageHistoryEntry(id="user", parent_id="root", timestamp=SAVED_AT,
                               message=UserMessage(content="input", timestamp=1))
    sibling = MessageHistoryEntry(id="sibling", parent_id="root", timestamp=SAVED_AT,
                                  message=UserMessage(content="other path", timestamp=2))
    meta = ThinkingLevelChangeHistoryEntry(id="meta", parent_id="user", timestamp=SAVED_AT,
                                          thinking_level="off")
    compact = CompactionHistoryEntry(id="compact", parent_id="meta", timestamp=SAVED_AT,
                                    summary="summary", first_kept_entry_id="user", tokens_before=10,
                                    system_message=None)
    edit = ContextEditHistoryEntry(id="edit", parent_id="compact", timestamp=SAVED_AT,
                                  target_id="user", replacement=None)
    entries = [root, user, sibling, meta, compact, edit]
    leaf = "edit"
    if kind == "duplicate":
        entries.append(user)
    elif kind == "missing_parent":
        entries[1] = replace(user, parent_id="missing")
    elif kind == "cycle":
        entries[0] = replace(root, parent_id="user")
    elif kind == "missing_leaf":
        leaf = "missing"
    elif kind.endswith("kept"):
        target = {"missing_kept": "missing", "sibling_kept": "sibling", "self_kept": "compact"}[kind]
        entries[4] = replace(compact, first_kept_entry_id=target)
    else:
        target = {
            "missing_target": "missing", "sibling_target": "sibling", "future_target": "future",
            "system_target": "root", "metadata_target": "meta",
        }[kind]
        entries[5] = replace(edit, target_id=target)
        if kind == "future_target":
            entries.append(replace(user, id="future", parent_id="edit"))
    history = saved_history(entries, leaf)
    for operation in (
        lambda: validate_history(history),
        lambda: Agent.from_history(history, AgentOptions(stream_fn=RecordingStreamFn())),
    ):
        with pytest.raises(ValueError, match=relation):
            operation()


@pytest.mark.parametrize("entry, field", [
    (MessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                         message=UserMessage(content=42, timestamp=1)), "content"),
    (MessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                         message=UserMessage(content=[TextContent(text=42)], timestamp=1)), "text"),
    (MessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                         message=SystemMessage(content="", timestamp=0, sections={"s": 42})), "sections"),
    (MessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                         message=UserMessage(content="", timestamp=True)), "timestamp"),
    (MessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                         message={"role": "user", "content": "raw dict"}), "message"),
    (CustomMessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                               custom_type="note", content="", display="yes"), "display"),
    (CustomMessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                               custom_type="note", content="", display=True, details={"x": object()}), "details"),
    (ModelChangeHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                             provider="test", model_id=42), "model_id"),
    (ThinkingLevelChangeHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT,
                                     thinking_level="unsupported"), "thinking_level"),
    (MessageHistoryEntry(id="bad", parent_id=None, timestamp=datetime(2026, 10, 2),
                         message=UserMessage(content="", timestamp=0)), "timestamp"),
])
def test_bad_payload_types_are_located_to_record_and_field(entry, field) -> None:
    history = saved_history([entry])
    for operation in (
        lambda: validate_history(history),
        lambda: Agent.from_history(history, AgentOptions(stream_fn=RecordingStreamFn())),
    ):
        with pytest.raises(ValueError, match=f"bad.*{field}"):
            operation()


@pytest.mark.parametrize("kind, field", [
    ("summary", "summary"), ("tokens", "tokens_before"), ("usage", "usage"),
    ("checkpoint", "system_message"), ("details", "details"),
    ("replacement", "replacement"), ("content", "content"),
])
def test_compaction_and_edit_payloads_are_strictly_validated(kind, field) -> None:
    root = MessageHistoryEntry(id="root", parent_id=None, timestamp=SAVED_AT,
                               message=UserMessage(content="input", timestamp=1))
    compact = CompactionHistoryEntry(id="bad", parent_id="root", timestamp=SAVED_AT,
                                    summary="summary", first_kept_entry_id="root", tokens_before=1,
                                    system_message=None)
    if kind == "summary":
        entry = replace(compact, summary=42)
    elif kind == "tokens":
        entry = replace(compact, tokens_before=-1)
    elif kind == "usage":
        entry = replace(compact, usage={"input": 1})
    elif kind == "checkpoint":
        entry = replace(compact, system_message=UserMessage(content="wrong role", timestamp=0))
    elif kind == "details":
        entry = replace(compact, details=float("nan"))
    else:
        entry = ContextEditHistoryEntry(
            id="bad", parent_id="root", timestamp=SAVED_AT, target_id="root",
            replacement={"content": "dict"} if kind == "replacement" else
            ContextEditReplacement(content=[text_message("wrong content block")]),
        )
    with pytest.raises(ValueError, match=f"bad.*{field}"):
        validate_history(saved_history([root, entry]))


def test_edit_replacement_must_match_target_content_type() -> None:
    assistant = MessageHistoryEntry(id="assistant", parent_id=None, timestamp=SAVED_AT,
                                    message=text_message("original"))
    edit = ContextEditHistoryEntry(id="bad", parent_id="assistant", timestamp=SAVED_AT,
                                  target_id="assistant", replacement=ContextEditReplacement(content=[
                                      ImageContent(data="image", mime_type="image/png"),
                                  ]))
    with pytest.raises(ValueError, match="bad.*replacement.*content"):
        validate_history(saved_history([assistant, edit]))


@pytest.mark.parametrize("explicit, expected", [(None, "high"), ("off", "off"), ("low", "low")])
def test_saved_settings_are_resolved_separately_from_explicit_execution_options(explicit, expected) -> None:
    selected = text_message("selected assistant")
    selected.model = "assistant-model"
    selected.provider = "assistant-provider"
    selected.thinking_level = "max"
    entries = [
        ModelChangeHistoryEntry(id="model", parent_id=None, timestamp=SAVED_AT,
                                provider="old-provider", model_id="old-model"),
        ThinkingLevelChangeHistoryEntry(id="thinking", parent_id="model", timestamp=SAVED_AT,
                                        thinking_level="high"),
        MessageHistoryEntry(id="assistant", parent_id="thinking", timestamp=SAVED_AT, message=selected),
        ModelChangeHistoryEntry(id="other-branch", parent_id="model", timestamp=SAVED_AT,
                                provider="unused-provider", model_id="unused-model"),
    ]
    history = saved_history(entries, "assistant")
    settings = validate_history(history)
    assert (settings.provider, settings.model_id) == ("assistant-provider", "assistant-model")
    assert settings.thinking_level == "high"
    assert settings.has_thinking_level
    model = replace(make_model(), reasoning=True)
    agent = Agent.from_history(history, AgentOptions(
        stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=model, thinking_level=explicit),
    ))
    assert agent.state.model == model
    assert agent.state.thinking_level == expected
    assert agent.history == history
    assert [entry.id for entry in agent.history.entries] == ["model", "thinking", "assistant", "other-branch"]


def test_restored_thinking_is_clamped_without_adding_a_metadata_record() -> None:
    history = saved_history([ThinkingLevelChangeHistoryEntry(
        id="thinking", parent_id=None, timestamp=SAVED_AT, thinking_level="high",
    )])
    agent = Agent.from_history(history, AgentOptions(
        stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model()),
    ))
    assert agent.state.thinking_level == "off"
    assert agent.history == history


def test_absent_thinking_selection_is_distinct_from_a_saved_off() -> None:
    assistant = text_message("done")
    assistant.thinking_level = "high"
    history = saved_history([MessageHistoryEntry(
        id="assistant", parent_id=None, timestamp=SAVED_AT, message=assistant,
    )])
    settings = validate_history(history)
    assert settings.thinking_level == "off"
    assert not settings.has_thinking_level
    assert validate_history(saved_history([])).provider is None
    restored = Agent.from_history(history, AgentOptions(stream_fn=RecordingStreamFn()))
    assert restored.state.thinking_level == "off"
    assert restored.history == history


@pytest.mark.parametrize("initial, identity, error", [
    (AgentInitialState(messages=[]), None, "messages"),
    (AgentInitialState(messages=[UserMessage(content="seed", timestamp=1)]), None, "messages"),
    (None, "another-conversation", "conversation_id"),
])
def test_restore_rejects_message_seeds_and_conflicting_identity(initial, identity, error) -> None:
    with pytest.raises(ValueError, match=error):
        Agent.from_history(saved_history([]), AgentOptions(
            stream_fn=RecordingStreamFn(), initial_state=initial, conversation_id=identity,
        ))


async def test_restored_payloads_are_isolated_and_only_current_host_tools_execute(tmp_path) -> None:
    past = text_message("", "toolUse")
    past.content = [ThinkingContent(thinking="consider", thinking_signature="sig"),
                    ToolCall(id="past-call", name="historical", arguments={"nested": [1]})]
    past.usage.input = 123
    past.usage.cost.input = 0.5
    seed = Agent(AgentOptions(
        stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model(), messages=[
            SystemMessage(content="base", timestamp=0, tools_added=[
                Tool(name="historical", description="Old tool", parameters={"type": "object"}),
            ]),
            UserMessage(content=[TextContent(text="input"), ImageContent(data="aW1n", mime_type="image/png")],
                        timestamp=1),
            past,
            ToolResultMessage(tool_call_id="past-call", tool_name="historical", timestamp=2,
                              content=[TextContent(text="past result")], details={"nested": ["original"]}),
            CustomAgentMessage(custom_type="note", content="more context", timestamp=3,
                               details={"nested": ["private"]}),
        ]),
    ))
    saved = seed.history
    stream = RecordingStreamFn()
    resource = tmp_path / "effect.txt"
    calls = []

    def execute(call_id, args, signal, on_update):
        calls.append(call_id)
        resource.write_text("current host tool")
        return AgentToolResult(content=[TextContent(text=resource.read_text())], details={"saved": True})

    def respond():
        if stream.calls == 1:
            response = text_message("", "toolUse")
            response.content = [ToolCall(id="new-call", name="current", arguments={})]
            return response
        return text_message("done")

    stream.set_result(respond)
    host_tool = AgentTool(name="current", label="Current", description="Current tool",
                          parameters={"type": "object"}, execute=execute)
    restored = Agent.from_history(saved, AgentOptions(
        stream_fn=stream, conversation_id=saved.conversation_id,
        initial_state=AgentInitialState(model=make_model(), tools=[host_tool]),
    ))
    assert restored.history == saved
    assert not resource.exists()
    assert stream.calls == 0
    assert calls == []
    # Mutate nested input and both public read views after restoration.
    past_entry = next(entry for entry in saved.entries if entry.type == "message" and entry.message.role == "assistant")
    past_entry.message.content[1].arguments["nested"].append(99)
    past_entry.message.usage.input = 999
    restored.history.entries[-1].details["nested"].append("changed history snapshot")
    restored.state.messages[3].details["nested"].append("changed state snapshot")
    host_tool.parameters["extra"] = "changed host declaration"
    await restored.continue_()

    assert calls == ["new-call"]
    assert resource.read_text() == "current host tool"
    assert stream.calls == 2
    request = stream.requests[0].context.messages
    assert [tool.name for tool in get_current_tools(request)] == ["current"]
    assert get_current_tools(request)[0].parameters == {"type": "object"}
    assert request[2].content == [
        ThinkingContent(thinking="consider", thinking_signature="sig"),
        ToolCall(id="past-call", name="historical", arguments={"nested": [1]}),
    ]
    assert request[2].usage.input == 123
    assert request[2].usage.cost.input == 0.5
    assert request[3].details == {"nested": ["original"]}
    assert request[1].content[1] == ImageContent(data="aW1n", mime_type="image/png")
    assert restored.history.entries[:len(seed.history.entries)] == seed.history.entries


async def test_replacement_changes_only_content_and_omissions_preserve_original_results() -> None:
    assistant = text_message("original", "error", "transient")
    result = ToolResultMessage(tool_call_id="call", tool_name="old", timestamp=9,
                               content=[TextContent(text="result")], details={"retained": True})
    history = saved_history([
        MessageHistoryEntry(id="assistant", parent_id=None, timestamp=SAVED_AT, message=assistant),
        MessageHistoryEntry(id="result", parent_id="assistant", timestamp=SAVED_AT, message=result),
        ContextEditHistoryEntry(id="edit", parent_id="result", timestamp=SAVED_AT, target_id="assistant",
                                replacement=ContextEditReplacement(content="replacement")),
        ContextEditHistoryEntry(id="omit", parent_id="edit", timestamp=SAVED_AT, target_id="result",
                                replacement=None),
    ])
    restored = Agent.from_history(history, AgentOptions(stream_fn=RecordingStreamFn()))
    assert restored.state.messages == (replace(assistant, content=[TextContent(text="replacement")]),)
    assert restored.history == history


@pytest.mark.parametrize("history, field", [
    (AgentHistory("", SAVED_AT, (), None), "conversation_id"),
    (AgentHistory("saved", datetime(2026, 10, 2), (), None), "created_at"),
    (AgentHistory("saved", SAVED_AT, [], None), "entries"),
    (AgentHistory("saved", SAVED_AT, (), 123), "leaf_id"),
    (AgentHistory("saved", SAVED_AT, (object(),), None), "entries"),
])
def test_bad_history_header_or_unknown_records_are_rejected(history, field) -> None:
    with pytest.raises(ValueError, match=field):
        validate_history(history)


def test_decoded_usage_fields_and_non_json_arguments_are_rejected() -> None:
    assistant = text_message("invalid usage")
    assistant.usage.cost.total = float("inf")
    entry = MessageHistoryEntry(id="bad", parent_id=None, timestamp=SAVED_AT, message=assistant)
    with pytest.raises(ValueError, match="bad.*usage.cost.total"):
        validate_history(saved_history([entry]))
    assistant.usage = empty_usage()
    assistant.content = [ToolCall(id="call", name="tool", arguments={"nested": object()})]
    with pytest.raises(ValueError, match="bad.*arguments.nested"):
        validate_history(saved_history([entry]))


@pytest.mark.parametrize("timestamp", [1001, 1007, 1700000000123])
def test_custom_timestamp_round_trip_keeps_the_exact_millisecond(timestamp) -> None:
    original = Agent(AgentOptions(stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(messages=[
        CustomAgentMessage(custom_type="note", content="context", timestamp=timestamp),
    ])))
    restored = Agent.from_history(original.history, AgentOptions(stream_fn=RecordingStreamFn()))
    assert restored.state.messages == original.state.messages
    assert restored.history == original.history
