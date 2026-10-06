from datetime import UTC, datetime

import pytest
from omh.agent import Agent, AgentHistory, AgentOptions, MessageHistoryEntry
from omh.llm.types import ImageContent, TextContent, UserMessage

from coding_agent import decode_history, encode_history


def test_history_roundtrip_preserves_sdk_data_and_projection():
    history = AgentHistory(
        conversation_id="conversation", created_at=datetime(2026, 10, 4, tzinfo=UTC),
        entries=(MessageHistoryEntry(
            id="user", parent_id=None, timestamp=datetime(2026, 10, 4, tzinfo=UTC),
            message=UserMessage(content=[TextContent(text="hello", text_signature="sig"),
                                         ImageContent(data="YWJj", mime_type="image/png")],
                                timestamp=123),
        ),), leaf_id="user",
    )
    decoded = decode_history(encode_history(history, cwd="/work", display_name="demo"))
    assert decoded.history == history
    assert (decoded.cwd, decoded.display_name) == ("/work", "demo")
    # Restoration is a public SDK seam; the provider is never invoked here.
    def unused(*args):
        raise AssertionError("restore must not request a model")
    restored = Agent.from_history(decoded.history, AgentOptions(stream_fn=unused))
    assert restored.state.messages == (history.entries[0].message,)


def test_legacy_usage_without_provenance_stays_unknown():
    import json

    history = complete_history()
    records = [json.loads(line) for line in encode_history(history, cwd="/work").splitlines()]
    for record in records:
        for envelope in (record, record.get("message", {})):
            if isinstance(envelope.get("usage"), dict):
                envelope["usage"].pop("reported", None)
    legacy = "\n".join(json.dumps(record) for record in records) + "\n"
    assert decode_history(legacy).history == history


def complete_history(reported=None):
    from omh.agent import (
        CompactionHistoryEntry,
        ContextEditHistoryEntry,
        ContextEditReplacement,
        CustomMessageHistoryEntry,
        ModelChangeHistoryEntry,
        ThinkingLevelChangeHistoryEntry,
    )
    from omh.llm.types import (
        AssistantMessage,
        SystemMessage,
        ThinkingContent,
        Tool,
        ToolCall,
        ToolReference,
        ToolResultMessage,
        Usage,
        UsageCost,
    )
    saved_at = datetime(2026, 10, 4, 1, 2, 3, 456000, tzinfo=UTC)
    usage = Usage(input=123, output=45, cache_read=6, cache_write=7, total_tokens=181, reported=reported,
                  cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1.0),
                  reasoning=9, cache_write_1h=2)
    checkpoint = SystemMessage(content=[TextContent(text="checkpoint", text_signature="system_sig")],
                               timestamp=1, sections={"base": "instruction", "deleted": "old"},
                               tools_added=[Tool(name="old", description="old tool", parameters={"snake_key": {"type": "string"}})])
    entries = []

    def add(cls, id, **kwargs):
        entries.append(cls(id=id, parent_id=entries[-1].id if entries else None,
                           timestamp=saved_at, **kwargs))

    add(ModelChangeHistoryEntry, "model", provider="test", model_id="offline")
    add(ThinkingLevelChangeHistoryEntry, "thinking", thinking_level="high")
    add(MessageHistoryEntry, "system", message=checkpoint)
    add(MessageHistoryEntry, "original", message=UserMessage(content="summarized original", timestamp=2))
    add(MessageHistoryEntry, "kept", message=UserMessage(content=[ImageContent(data="abc", mime_type="image/png"), TextContent(text="kept")], timestamp=3))
    add(MessageHistoryEntry, "assistant", message=AssistantMessage(
        api="custom-api", provider="test", model="physical", timestamp=4, usage=usage,
        stop_reason="toolUse", response_model="served", response_id="response", thinking_level="high",
        provider_thinking_level="provider-high", raw_stop_reason="raw", error_message=None,
        content=[ThinkingContent(thinking="reasoning", thinking_signature="thinking_sig", redacted=False),
                 TextContent(text="thinking done", text_signature="text_sig"),
                 ToolCall(id="call", name="read", arguments={"snake_key": {"camelKey": 1}},
                          thought_signature="call_sig", namespace="ns")],
    ))
    add(MessageHistoryEntry, "result", message=ToolResultMessage(
        tool_call_id="call", tool_name="read", timestamp=5, is_error=True, usage=usage,
        content=[TextContent(text="result"), ImageContent(data="def", mime_type="image/jpeg")],
        details={"user_key": [None, True, 1.5, {"inner_key": "value"}]},
    ))
    add(CompactionHistoryEntry, "compact1", summary="old summary", first_kept_entry_id="kept",
        tokens_before=1000, system_message=checkpoint, usage=usage, details={"read_files": ["a.py"]})
    add(CustomMessageHistoryEntry, "custom", custom_type="note", content=[TextContent(text="note"), ImageContent(data="ghi", mime_type="image/png")],
        display=False, details={"hidden_key": "not model text"})
    add(ContextEditHistoryEntry, "omit", target_id="assistant", replacement=None)
    add(CompactionHistoryEntry, "compact2", summary="latest summary", first_kept_entry_id="kept",
        tokens_before=2000, system_message=checkpoint, usage=None, details={"modified_files": ["b.py"]})
    add(ContextEditHistoryEntry, "replace", target_id="kept", replacement=ContextEditReplacement(content="replacement"))
    add(MessageHistoryEntry, "delta", message=SystemMessage(content="added", timestamp=6,
        sections={"base": "new", "deleted": None}, tools_removed=[ToolReference(name="old")],
        tools_added=[Tool(name="new", description="new tool", parameters={"outer_key": {"inner_key": "unchanged"}})]))
    add(MessageHistoryEntry, "last", message=UserMessage(content="after checkpoint", timestamp=7))
    # Keep a different branch and a selected leaf earlier than the physical tail.
    entries.append(MessageHistoryEntry(id="inactive", parent_id="original", timestamp=saved_at,
                                      message=UserMessage(content="inactive branch", timestamp=8)))
    return AgentHistory(conversation_id="rich", created_at=saved_at, entries=tuple(entries), leaf_id="last")


@pytest.mark.parametrize("reported", [True, False, None])
async def test_all_records_roundtrip_and_continue_from_latest_projection(tmp_path, reported):
    from support import OfflineStream, model

    from coding_agent import AgentSessionRuntime, CodingAgentOptions

    history = complete_history(reported)
    text = encode_history(history, cwd=str(tmp_path), display_name="all records")
    assert '"parentId"' in text and '"firstKeptEntryId"' in text
    assert '"snake_key"' in text and '"hidden_key"' in text
    assert '"snakeKey"' not in text and '"hiddenKey"' not in text
    assert decode_history(text).history == history
    path = tmp_path / "all.jsonl"
    path.write_text(text)
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, available_models=(model("physical"),), stream_fn=stream, tools=()))
    session = await runtime.open_session(path)
    assert session.agent.history == history
    assert session.agent.state.thinking_level == "high"
    assert not stream.requests
    assert [item.role for item in session.agent.state.messages] == [
        "system", "compactionSummary", "user", "toolResult", "custom", "system", "user",
    ]
    assert session.agent.state.messages[1].summary == "latest summary"
    assert session.agent.state.messages[2].content == "replacement"
    await session.continue_()
    projection = str(stream.requests[0][1])
    assert "latest summary" in projection
    assert "old summary" not in projection
    assert "summarized original" not in projection
    assert "inactive branch" not in projection
    assert "hidden_key" not in projection
    await session.close()
    reopened = await AgentSessionRuntime(runtime.options).open_session(path)
    assert reopened.agent.history == session.agent.history
    assert history.entries[-1] in reopened.agent.history.entries


def test_jsonl_only_splits_physical_newlines_and_rejects_semantic_bad_data():
    import json

    import pytest

    history = complete_history()
    text = encode_history(history, cwd="/work", display_name="unicode\u2028paragraph\u2029")
    assert decode_history(text.encode()).display_name == "unicode\u2028paragraph\u2029"
    lines = text.split("\n")
    invalid = json.loads(lines[1])
    invalid["type"] = "unknown_record"
    lines[1] = json.dumps(invalid)
    with pytest.raises(ValueError, match="entry model"):
        decode_history("\n".join(lines))


def test_append_does_not_hide_an_invalid_snapshot_leaf():
    import json

    import pytest

    history = complete_history()
    text = encode_history(history, cwd="/work")
    records = [json.loads(line) for line in text.splitlines()]
    records[0]["leafId"] = "missing-snapshot-leaf"
    records.append({
        "type": "message", "id": "appended", "parentId": "last",
        "timestamp": history.created_at.isoformat(),
        "message": {"role": "user", "content": "later", "timestamp": 9},
    })
    with pytest.raises(ValueError, match="leaf_id"):
        decode_history("\n".join(json.dumps(record) for record in records))
