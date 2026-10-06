"""Offline provider and public hooks installed as sitecustomize by CLI tests."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from omh.agent import AfterToolCallResult, CompactionSettings, RetryPolicy
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageCost,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream

import coding_agent.cli as cli

SCENARIO = os.environ.get("JSON_SCENARIO", "rich")
CALLS = 0
SUMMARY_CALLS = 0

#: Test-controlled synchronization gate. Signal tests wait for ``ready_<name>``
#: before sending a process signal so cancellation timing never relies on sleep.
GATE = Path(os.environ["JSON_GATE"]) if os.environ.get("JSON_GATE") else None

#: Force ``tempfile`` to a real filesystem boundary for the rescue-failure case.
if os.environ.get("JSON_TMPDIR"):
    tempfile.tempdir = os.environ["JSON_TMPDIR"]


def gate_ready(name):
    if GATE is not None:
        GATE.mkdir(parents=True, exist_ok=True)
        (GATE / f"ready_{name}").write_text("")


async def gate_wait(name):
    if GATE is None:
        return
    gate_ready(name)
    target = GATE / f"go_{name}"
    while not target.exists():
        await asyncio.sleep(0.02)


def gate_command(name):
    release = GATE / f"go_{name}"
    return f"touch '{GATE}/ready_{name}'; while [ ! -f '{release}' ]; do sleep 0.02; done"


def counted(kind):
    marker = Path(os.environ["JSON_SENDS"])
    with marker.open("a") as file:
        file.write(kind + "\n")


def usage():
    return Usage(input=7, output=5, cache_read=2, cache_write=3, total_tokens=17,
                 cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
                 cache_write_1h=1, reasoning=4, reported=True)


def message(model, **values):
    return AssistantMessage(api=model.api, provider=model.provider, model=model.id,
                            usage=empty_usage(), stop_reason="pending", timestamp=1000, **values)


def stream_fn(model, context, options):
    if SCENARIO == "gate_model":
        return _gated_response("model", model, context, options)
    if SCENARIO == "signal_summary" and _is_summary(context):
        return _gated_response("summary", model, context, options)
    return _respond(model, context, options)


async def _gated_response(name, model, context, options):
    await gate_wait(name)
    return _respond(model, context, options)


def _is_summary(context):
    return any("Summarize" in str(item.content) or "summarize" in str(item.content)
               for item in context.messages if item.role == "user")


def _respond(model, context, options):
    global CALLS, SUMMARY_CALLS
    stream = create_assistant_message_event_stream()
    summary = _is_summary(context)
    counted("summary" if summary else "dialogue")
    CALLS += 1
    if summary:
        SUMMARY_CALLS += 1
    output = message(model)
    if SCENARIO == "runtime_failure":
        raise RuntimeError("controlled runtime failure")
    if SCENARIO in {"error", "aborted"} or (
        SCENARIO in {"retry", "intent_abort", "intent_listener", "intent_save", "signal_retry"}
        and CALLS == 1
    ) or (SCENARIO in {"compact", "compact_error"} and summary and SUMMARY_CALLS == 1):
        output.stop_reason = "aborted" if SCENARIO == "aborted" else "error"
        output.error_message = "controlled aborted" if SCENARIO == "aborted" else ("401 bad key" if SCENARIO == "error" else "503 service unavailable")
        stream.push(ErrorEvent(reason=output.stop_reason, error=output))
        return stream
    stream.push(StartEvent(partial=output))
    if SCENARIO in {"rich", "gate_tool", "signal_close"} and CALLS == 1:
        thought = ThinkingContent(thinking="", thinking_signature="replay-token", redacted=False)
        output.content.append(thought)
        stream.push(ThinkingStartEvent(content_index=0, partial=output))
        thought.thinking = "consider"
        stream.push(ThinkingDeltaEvent(content_index=0, delta="consider", partial=output))
        stream.push(ThinkingEndEvent(content_index=0, content="consider", partial=output))
        call = ToolCall(id="call-1", name="bash", arguments={},
                        thought_signature="thought-token", namespace="shell")
        output.content.append(call)
        stream.push(ToolCallStartEvent(content_index=1, partial=output))
        command = gate_command("tool") if SCENARIO in {"gate_tool", "signal_close"} else "printf progress"
        args = json.dumps(
            {"command": command, "snake_key": {"inner_key": 1}}, separators=(",", ":"),
        )
        call.arguments = json.loads(args)
        stream.push(ToolCallDeltaEvent(content_index=1, delta=args, partial=output))
        stream.push(ToolCallEndEvent(content_index=1, tool_call=call, partial=output))
        output.stop_reason = "toolUse"
    else:
        text = TextContent(text="", text_signature="text-token")
        output.content.append(text)
        stream.push(TextStartEvent(content_index=0, partial=output))
        text.text = "summary" if summary else "answer"
        stream.push(TextDeltaEvent(content_index=0, delta=text.text, partial=output))
        stream.push(TextEndEvent(content_index=0, content=text.text, partial=output))
        output.stop_reason = "error" if SCENARIO == "compact_error" and summary else "stop"
        if output.stop_reason == "error":
            output.error_message = "permanent summary failure"
    output.usage = usage()
    output.response_model = "concrete-model"
    output.response_id = "response-1"
    output.provider_thinking_level = "high"
    output.raw_stop_reason = "tool_calls" if output.stop_reason == "toolUse" else "stop"
    if output.stop_reason == "error":
        stream.push(ErrorEvent(reason="error", error=output))
    else:
        stream.push(DoneEvent(reason=output.stop_reason, message=output))
    return stream


def after_tool(context, signal):
    return AfterToolCallResult(content=[TextContent(text="tool answer", text_signature="tool-text"),
                                        ImageContent(data="AQID", mime_type="image/png")],
                               details={"snake_key": {"inner_key": "opaque"}, "null_key": None},
                               usage=usage(), terminate=False)


class ControlledHost(cli.CodingAgentHost):
    def build_options(self, selection, **kwargs):
        options = super().build_options(selection, **kwargs)
        options.stream_fn = stream_fn
        options.agent_options.after_tool_call = after_tool
        options.agent_options.retry = RetryPolicy(
            base_delay_ms=3000 if SCENARIO == "signal_retry" else 0,
        )
        if SCENARIO in {"compact", "compact_error", "signal_summary"}:
            options.agent_options.compaction = CompactionSettings(reserve_tokens=999999, keep_recent_tokens=0)
        return options


cli.CodingAgentHost = ControlledHost

# Gate the first user-history commit at the real SessionManager boundary.
if SCENARIO == "signal_save":
    from omh.agent import MessageHistoryEntry

    import coding_agent.session_manager as session_manager_module

    _real_commit = session_manager_module.SessionManager.commit
    _save_gate = {"opened": False}

    async def _gated_commit(self, history, entries):
        if not _save_gate["opened"] and any(
            isinstance(entry, MessageHistoryEntry) and entry.message.role == "user"
            for entry in entries
        ):
            _save_gate["opened"] = True
            await gate_wait("save")
        return await _real_commit(self, history, entries)

    session_manager_module.SessionManager.commit = _gated_commit

# Interleave after the runner has observed the public boundary event.
if SCENARIO in {"intent_abort", "intent_listener", "intent_save", "close_failure", "notification_failure", "prompt_failure", "signal_retry", "signal_close"}:
    import inspect

    import coding_agent.print_runner as runner
    from coding_agent.agent_session_runtime import AgentSessionRuntime

    class ControlledRuntime(AgentSessionRuntime):
        async def prompt(self, *args, **kwargs):
            if SCENARIO == "prompt_failure":
                raise RuntimeError("controlled prompt failure")
            return await super().prompt(*args, **kwargs)

        async def close(self):
            if SCENARIO == "signal_close":
                await gate_wait("close")
            await super().close()
            if SCENARIO == "close_failure":
                raise RuntimeError("controlled close failure")

        def subscribe(self, listener):
            async def observed(event, signal):
                result = listener(event, signal)
                if inspect.isawaitable(result):
                    await result
                if event.type == "retry_start" and SCENARIO == "signal_retry":
                    gate_ready("retry")
                if event.type == "agent_end" and event.will_retry:
                    if SCENARIO == "intent_abort":
                        self.current_session.agent.abort()
                    elif SCENARIO == "intent_listener":
                        raise RuntimeError("controlled listener failure")
                    elif SCENARIO == "intent_save":
                        # The next omission commit hits an unwritable real path.
                        target = next(Path(os.environ["JSON_SESSIONS"]).glob("*.jsonl"))
                        target.unlink()
                        target.mkdir()
                if event.type == "turn_end" and SCENARIO == "notification_failure":
                    raise RuntimeError("controlled notification failure")
            return super().subscribe(observed)

    runner.AgentSessionRuntime = ControlledRuntime
