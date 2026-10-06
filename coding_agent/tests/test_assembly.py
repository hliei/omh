from dataclasses import replace
from datetime import UTC, datetime

import pytest
from omh.agent import AgentHistory, MessageHistoryEntry, ThinkingLevelChangeHistoryEntry
from omh.llm.types import AssistantMessage, TextContent, empty_usage
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions, encode_history


def write_history(path, *, thinking="high"):
    stamp = datetime(2026, 10, 4, tzinfo=UTC)
    history = AgentHistory(conversation_id="saved", created_at=stamp, leaf_id="thinking", entries=(
        MessageHistoryEntry(id="assistant", parent_id=None, timestamp=stamp,
                            message=AssistantMessage(api="openai-completions", provider="test", model="saved-model",
                                timestamp=1, content=[TextContent(text="earlier response")], usage=empty_usage(), stop_reason="stop")),
        ThinkingLevelChangeHistoryEntry(id="thinking", parent_id="assistant", timestamp=stamp, thinking_level=thinking),
    ))
    path.write_text(encode_history(history, cwd=str(path.parent), display_name="saved"))
    return history


@pytest.mark.parametrize("selection", ["explicit", "history", "fallback"])
async def test_model_selection_and_thinking_restore_use_current_host(tmp_path, selection):
    path = tmp_path / "history.jsonl"
    history = write_history(path)
    current_model = model("explicit" if selection == "explicit" else "saved-model" if selection == "history" else "fallback")
    stream = OfflineStream()
    options = CodingAgentOptions(cwd=tmp_path, stream_fn=stream, tools=("write",),
        model=current_model if selection == "explicit" else None,
        available_models=(current_model,) if selection == "history" else (),
        fallback_model=current_model if selection == "fallback" else None,
        thinking_level="off" if selection == "explicit" else None)
    session = await AgentSessionRuntime(options).open_session(path)
    assert session.agent.state.model == current_model
    assert session.agent.history == history
    assert session.agent.state.thinking_level == ("off" if selection == "explicit" else "high")
    assert [tool.name for tool in session.agent.state.tools] == ["write"]
    assert not session.agent.has_queued_messages()
    assert not stream.requests
    if selection == "fallback":
        assert "test/saved-model" in session.model_fallback_message
        assert "test/fallback" in session.model_fallback_message
    else:
        assert session.model_fallback_message is None
    await session.prompt("next")
    assert stream.requests[0][0] == current_model


async def test_no_model_and_failed_destination_leave_current_session_usable(tmp_path):
    with pytest.raises(ValueError, match="model"):
        await AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, stream_fn=OfflineStream())).new_session()
    runtime = AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=()))
    current = await runtime.new_session()
    with pytest.raises(FileNotFoundError):
        await runtime.open_session(tmp_path / "missing")
    assert runtime.current_session is current
    assert not current.agent.state.is_closed
    await current.prompt("still usable")
    await current.agent.close()
    replacement = await runtime.new_session()
    assert replacement.agent.history.conversation_id != current.agent.history.conversation_id


@pytest.mark.parametrize("tools", [("read", "read"), ("unknown",)])
async def test_invalid_tool_selection_rejected_before_execution(tmp_path, tools):
    with pytest.raises(ValueError, match="subset"):
        await AgentSessionRuntime(CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=tools)).new_session()


async def test_history_cwd_used_unless_host_explicitly_overrides(tmp_path):
    path = tmp_path / "history.jsonl"
    write_history(path)
    options = CodingAgentOptions(model=model(), stream_fn=OfflineStream(), tools=())
    stored = await AgentSessionRuntime(options).open_session(path)
    assert stored.cwd == tmp_path.resolve()
    await stored.close()
    overridden = await AgentSessionRuntime(replace(options, cwd=tmp_path / "host")).open_session(path)
    assert overridden.cwd == (tmp_path / "host").resolve()
