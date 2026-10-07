"""The public session admission boundary used before a host shell starts."""

import asyncio

import pytest
from omh.agent import (
    AgentOptions,
    CompactionSettings,
    CustomAgentMessage,
    create_bash_tool,
)
from omh.llm.types import AbortController, AssistantMessage, ErrorEvent, empty_usage
from omh.llm.utils.event_stream import create_assistant_message_event_stream
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions


async def test_manual_compaction_refuses_custom_shell_submission(tmp_path):
    started = asyncio.Event()

    def summary_stream(selected, context, options):
        stream = create_assistant_message_event_stream()
        started.set()

        def abort():
            output = AssistantMessage(
                api=selected.api, provider=selected.provider, model=selected.id,
                usage=empty_usage(), stop_reason="aborted", timestamp=1000,
                error_message="summary aborted",
            )
            stream.push(ErrorEvent(reason="aborted", error=output))

        options.signal.add_callback(abort)
        return stream

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), tools=(), stream_fn=summary_stream,
        load_context_files=False,
        agent_options=AgentOptions(compaction=CompactionSettings(enabled=False, keep_recent_tokens=0)),
    ))
    session = await runtime.new_session()
    await runtime.submit_custom_message(CustomAgentMessage(custom_type="note", content="work to summarize"))
    await runtime.submit_custom_message(CustomAgentMessage(custom_type="note", content="current work"))
    compact = asyncio.create_task(runtime.compact())
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert session.agent.state.activity_kind == "manual_compaction"
        history = session.agent.history
        with pytest.raises(RuntimeError, match="manual compaction"):
            await runtime.submit_custom_message(CustomAgentMessage(
                custom_type="user_shell_hidden", content="", details={"command": "never accepted"},
            ))
        assert session.agent.history == history
    finally:
        session.agent.abort()
        await asyncio.gather(compact, return_exceptions=True)
        await runtime.close()


async def test_admitted_shell_records_its_terminal_outcome_after_a_later_save_failure(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text("occupied")
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), tools=(), stream_fn=OfflineStream(),
        session_file=path, load_context_files=False,
    ))
    session = await runtime.new_session()
    ready = asyncio.Event()
    cancellation = AbortController()

    def progress(result):
        if "ready" in result.content[0].text:
            ready.set()

    session.ensure_can_accept_work()
    execution = asyncio.create_task(create_bash_tool(tmp_path).execute(
        "admitted", {"command": "printf ready; sleep 60"}, cancellation.signal, progress,
    ))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        with pytest.raises(FileExistsError):
            await runtime.submit_custom_message(CustomAgentMessage(custom_type="note", content="accepted"))
        assert session.save_state == "unsaved"
        cancellation.abort()
        with pytest.raises(Exception, match="Command aborted") as error:
            await execution
        # The host admitted the shell before the failure. Its final record must
        # bypass admission and still be retained when the saving listener fails.
        with pytest.raises(FileExistsError):
            await session.agent.submit_custom_message(CustomAgentMessage(
                custom_type="user_shell_hidden", content="",
                details={"command": "printf ready; sleep 60", "status": "cancelled", "output": str(error.value)},
            ))
        await session.save()
        assert session.save_state == "saved"
        reopened = await runtime.open_session(path)
        final = reopened.agent.state.messages[-1]
        assert final.custom_type == "user_shell_hidden"
        assert final.details["status"] == "cancelled"
        assert "ready" in final.details["output"]
        assert "Command aborted" in final.details["output"]
    finally:
        cancellation.abort()
        await asyncio.gather(execution, return_exceptions=True)
        await runtime.close()
