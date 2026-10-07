"""Offline stream installed as sitecustomize for interactive PTY tests."""

import json
import os
from dataclasses import asdict

from omh.agent import CompactionSettings, RetryPolicy
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
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
    ToolCallEndEvent,
    ToolCallStartEvent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream

import coding_agent.cli as cli

SCENARIO = os.environ.get("INTERACTIVE_SCENARIO", "echo")
CALLS = 0
SUMMARY_CALLS = 0
FINAL = "# Fixed the value\n\n```python\nvalue = 2\n```\n"
THINKING = "checked the assertion"


def counted(kind: str) -> None:
    path = os.environ.get("INTERACTIVE_SENDS")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(kind + "\n")


def message(model):
    return AssistantMessage(
        api=model.api, provider=model.provider, model=model.id, usage=empty_usage(),
        stop_reason="pending", timestamp=1000,
    )


def user_text(item) -> str:
    content = item.content
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))


def last_user(context) -> str:
    found = ""
    for item in context.messages:
        if getattr(item, "role", None) == "user":
            found = user_text(item)
    return found


def last_user_images(context) -> list:
    """Image blocks on the most recent user message, in content order."""
    found: list = []
    for item in context.messages:
        if getattr(item, "role", None) == "user":
            content = item.content
            found = (
                [block for block in content if getattr(block, "type", None) == "image"]
                if isinstance(content, list) else []
            )
    return found


def record_request(user: str) -> None:
    """Record the model request's last user text for boundary assertions."""
    path = os.environ.get("INTERACTIVE_REQUESTS")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(user.replace("\n", "\\n") + "\n")


def record_images(images) -> None:
    path = os.environ.get("INTERACTIVE_IMAGES")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for image in images:
            handle.write(f"{image.mime_type} {image.data}\n")


def latest_tool(context) -> str | None:
    name = None
    for item in context.messages:
        if getattr(item, "role", None) == "toolResult":
            name = item.tool_name
    return name


def is_summary(context) -> bool:
    return any(
        getattr(item, "role", None) == "user" and "ummarize" in user_text(item)
        for item in context.messages
    )


def emit_text(stream, output, text: str, *, thinking: str | None = None) -> None:
    stream.push(StartEvent(partial=output))
    index = 0
    if thinking:
        thought = ThinkingContent(thinking="")
        output.content.append(thought)
        stream.push(ThinkingStartEvent(content_index=0, partial=output))
        thought.thinking = thinking
        stream.push(ThinkingDeltaEvent(content_index=0, delta=thinking, partial=output))
        stream.push(ThinkingEndEvent(content_index=0, content=thinking, partial=output))
        index = 1
    block = TextContent(text="")
    output.content.append(block)
    stream.push(TextStartEvent(content_index=index, partial=output))
    block.text = text
    stream.push(TextDeltaEvent(content_index=index, delta=text, partial=output))
    stream.push(TextEndEvent(content_index=index, content=text, partial=output))
    output.stop_reason = "stop"
    stream.push(DoneEvent(reason="stop", message=output))


def emit_error(stream, output, text: str, error: str, *, aborted: bool = False) -> None:
    block = TextContent(text=text)
    output.content.append(block)
    output.stop_reason = "aborted" if aborted else "error"
    output.error_message = error
    stream.push(StartEvent(partial=output))
    stream.push(ErrorEvent(reason=output.stop_reason, error=output))


def emit_tool(stream, output, name: str, arguments: dict, tool_id: str) -> None:
    call = ToolCall(id=tool_id, name=name, arguments=arguments)
    output.content.append(call)
    stream.push(StartEvent(partial=output))
    stream.push(ToolCallStartEvent(content_index=0, partial=output))
    stream.push(ToolCallEndEvent(content_index=0, tool_call=call, partial=output))
    output.stop_reason = "toolUse"
    stream.push(DoneEvent(reason="toolUse", message=output))


def emit_tools(stream, output, calls) -> None:
    """One assistant message that requests several tool calls in one batch."""
    output.content.extend(calls)
    stream.push(StartEvent(partial=output))
    for index, call in enumerate(calls):
        stream.push(ToolCallStartEvent(content_index=index, partial=output))
        stream.push(ToolCallEndEvent(content_index=index, tool_call=call, partial=output))
    output.stop_reason = "toolUse"
    stream.push(DoneEvent(reason="toolUse", message=output))


def stream_fn(model, context, options):
    global CALLS, SUMMARY_CALLS
    CALLS += 1
    summary = is_summary(context)
    selection_path = os.environ.get("INTERACTIVE_SELECTIONS")
    if selection_path:
        from omh.llm.utils.transcript import get_current_tools
        with open(selection_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "model": f"{model.provider}/{model.id}",
                "thinking": options.reasoning,
                "tools": [tool.name for tool in get_current_tools(context.messages)],
            }) + "\n")
    payload_path = os.environ.get("INTERACTIVE_PAYLOADS")
    if payload_path:
        with open(payload_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps([asdict(item) for item in context.messages]) + "\n")
    counted("summary" if summary else "dialogue")
    stream = create_assistant_message_event_stream()
    output = message(model)
    if summary:
        SUMMARY_CALLS += 1
        if SCENARIO == "compact-cancel" and SUMMARY_CALLS == 1:
            stream.push(StartEvent(partial=output))

            def finish_summary() -> None:
                counted("summary-aborted")
                output.stop_reason = "aborted"
                output.error_message = "summary aborted by caller"
                stream.push(ErrorEvent(reason="aborted", error=output))

            signal = None if options is None else getattr(options, "signal", None)
            if signal is not None:
                signal.add_callback(finish_summary)
            return stream
        emit_text(stream, output, "summary of the conversation")
        return stream
    if SCENARIO == "settings-retry" and CALLS <= 2:
        emit_error(stream, output, "kept visible", "503 service unavailable")
        return stream
    if SCENARIO == "controls":
        emit_text(stream, output, "before\x1b]52;c;dGVzdA==\x07after\x1b[2J")
        return stream
    if SCENARIO == "hang":
        block = TextContent(text="")
        output.content.append(block)
        stream.push(StartEvent(partial=output))
        block.text = "partial answer"
        stream.push(TextDeltaEvent(content_index=0, delta="partial answer", partial=output))

        def finish() -> None:
            output.stop_reason = "aborted"
            output.error_message = "aborted by caller"
            stream.push(ErrorEvent(reason="aborted", error=output))

        signal = None if options is None else getattr(options, "signal", None)
        if signal is not None:
            signal.add_callback(finish)
        return stream
    user = last_user(context).strip()
    tool = latest_tool(context)
    record_request(user)
    if SCENARIO == "settings-batch":
        if CALLS == 1:
            emit_tools(stream, output, [
                ToolCall(id="call-a", name="bash", arguments={"command": (
                    "printf settings-ready; while [ ! -f release-settings ]; do sleep 0.02; done; "
                    "printf a > first.txt"
                )}),
                ToolCall(id="call-b", name="bash", arguments={"command": "printf b > second.txt"}),
            ])
        else:
            emit_text(stream, output, "settings batch done")
        return stream
    if SCENARIO == "settings-request" and CALLS == 1:
        import asyncio
        stream.push(StartEvent(partial=output))
        async def finish_later():
            from pathlib import Path
            while not Path("release-settings").exists():
                await asyncio.sleep(0.02)
            emit_tool(stream, output, "write", {"path": "first.txt", "content": "old snapshot"}, "old-request")
        asyncio.create_task(finish_later())
        return stream
    if SCENARIO == "shell-tools":
        if tool is None and CALLS == 1:
            emit_tool(stream, output, "bash", {"command": (
                "printf model-ready; while [ ! -f release-model ]; do sleep 0.02; done; "
                "printf model-done > model-done.txt"
            )}, "parallel-model")
        else:
            emit_text(stream, output, "parallel done")
        return stream
    if SCENARIO in {"steer", "queued-end"}:
        if tool is None and CALLS == 1:
            emit_tools(stream, output, [
                ToolCall(id="call-a", name="bash",
                         arguments={"command": "sleep 2; printf a > first.txt"}),
                ToolCall(id="call-b", name="bash",
                         arguments={"command": "sleep 2; printf b > second.txt"}),
            ])
            return stream
        if user != "go":
            emit_text(stream, output, f"steered:{user}")
        else:
            emit_text(stream, output, "batch-done")
        return stream
    if SCENARIO == "follow":
        if tool is None and CALLS == 1:
            emit_tool(stream, output, "bash",
                      {"command": "sleep 2; printf mark > marker.txt"}, "call-mark")
            return stream
        if user != "go":
            emit_text(stream, output, f"followed:{user}")
        else:
            emit_text(stream, output, "turn-done")
        return stream
    if SCENARIO == "cancel-side":
        if tool is None and CALLS == 1:
            emit_tool(stream, output, "bash", {"command": "printf kept > kept.txt"}, "call-kept")
            return stream
        if CALLS == 2:
            block = TextContent(text="")
            output.content.append(block)
            stream.push(StartEvent(partial=output))
            block.text = "partial answer"
            stream.push(TextDeltaEvent(content_index=0, delta="partial answer", partial=output))

            def finish() -> None:
                output.stop_reason = "aborted"
                output.error_message = "aborted by caller"
                stream.push(ErrorEvent(reason="aborted", error=output))

            gate = None if options is None else getattr(options, "signal", None)
            if gate is not None:
                gate.add_callback(finish)
            return stream
        emit_text(stream, output, f"reply:{user}")
        return stream
    if SCENARIO == "error":
        if CALLS == 1:
            emit_error(stream, output, "kept visible", "provider rejected the turn")
        else:
            emit_text(stream, output, f"reply:{user}")
        return stream
    if SCENARIO == "retry-cancel":
        if CALLS == 1:
            emit_error(stream, output, "kept visible", "503 service unavailable")
        else:
            emit_text(stream, output, f"reply:{user}")
        return stream
    if SCENARIO == "retry":
        if CALLS == 1:
            emit_error(stream, output, "", "503 service unavailable")
        else:
            emit_text(stream, output, "retried")
        return stream
    if SCENARIO == "compact":
        emit_text(stream, output, "compacted")
        return stream
    if SCENARIO == "tool-error":
        if user == "again":
            emit_text(stream, output, "reply:again")
        elif tool is None:
            emit_tool(stream, output, "read", {"path": "missing.py"}, "call-read")
        else:
            emit_text(stream, output, "still here")
        return stream
    if SCENARIO == "bounded":
        if tool is None:
            emit_tool(
                stream, output, "bash",
                {"command": "i=0; while [ \"$i\" -lt 2500 ]; do echo line; i=$((i+1)); done"},
                "call-bash",
            )
        else:
            emit_text(stream, output, "bounded done")
        return stream
    if SCENARIO == "tools":
        if user == "continue":
            emit_text(stream, output, "reply:continue")
            return stream
        if tool is None:
            emit_tool(stream, output, "read", {"path": "app.py"}, "call-read")
        elif tool == "read":
            emit_tool(stream, output, "edit", {
                "path": "app.py",
                "edits": [{"oldText": "value = 1", "newText": "value = 2"}],
            }, "call-edit")
        elif tool == "edit":
            emit_tool(stream, output, "write", {"path": "note.txt", "content": "noted"}, "call-write")
        elif tool == "write":
            emit_tool(stream, output, "bash", {"command": "printf coded"}, "call-bash")
        else:
            emit_text(stream, output, FINAL, thinking=THINKING)
        return stream
    if SCENARIO == "vision":
        images = last_user_images(context)
        record_images(images)
        emit_text(stream, output, f"seen-image:{len(images)}")
        return stream
    if user == "mark":
        if tool == "write":
            emit_text(stream, output, "marked")
        else:
            emit_tool(stream, output, "write", {"path": "marker.txt", "content": "marked"}, "call-mark")
        return stream
    emit_text(stream, output, f"reply:{user}")
    return stream


class ControlledHost(cli.CodingAgentHost):
    async def readiness(self, selection):
        self.readiness_calls = getattr(self, "readiness_calls", 0) + 1
        if SCENARIO == "switch-prepare" and self.readiness_calls == 2:
            import asyncio
            print("selection preparation", flush=True)
            await asyncio.Event().wait()
        if SCENARIO == "startup":
            import asyncio
            from pathlib import Path
            Path(os.environ["INTERACTIVE_READY"]).write_text("ready")
            await asyncio.Event().wait()
        return await super().readiness(selection)

    def build_options(self, selection, *, session_file=None, agent_options=None):
        options = super().build_options(
            selection, session_file=session_file, agent_options=agent_options,
        )
        options.stream_fn = stream_fn
        if SCENARIO == "queued-end":
            options.agent_options.finish_turn = lambda context, signal: "end"
        # The retry-cancel scenario holds the retry wait open long enough to
        # cancel it deliberately; the rest retry without a timer.
        if SCENARIO not in {"settings-retry", "settings-compaction"}:
            options.agent_options.retry = RetryPolicy(
                base_delay_ms=8000 if SCENARIO == "retry-cancel" else 0,
            )
        if SCENARIO in {"compact", "compact-cancel"}:
            options.agent_options.compaction = CompactionSettings(
                reserve_tokens=999_999, keep_recent_tokens=0,
            )
        return options


cli.CodingAgentHost = ControlledHost

if SCENARIO == "rename-failure":
    def fail_replace(source, destination):
        raise OSError("controlled rename failure")

    os.replace = fail_replace


if SCENARIO in {"handoff", "handoff-error"}:
    import asyncio
    from pathlib import Path

    from omh.agent import AgentSettledEvent

    import coding_agent.interactive as interactive

    class ControlledRuntime(interactive.AgentSessionRuntime):
        """An external host activity races an idle UI's handoff, at public APIs.

        The old terminal notification is held until the PTY driver releases it;
        cancellation can therefore be observed without guessing scheduler time.
        """

        async def new_session(self, *, display_name=None):
            old = self.current_session
            if old is None:
                return await super().new_session(display_name=display_name)
            settled = asyncio.Event()

            async def hold(event, signal):
                if isinstance(event, AgentSettledEvent):
                    settled.set()
                    signal.add_callback(lambda: print("handoff closing", flush=True))
                    release = Path(os.environ["HOME"]) / "release-handoff"
                    while not release.exists():
                        await asyncio.sleep(0.01)
                    if SCENARIO == "handoff-error":
                        raise LookupError("terminal notification failed")

            old.agent.subscribe(hold)
            prompt = asyncio.create_task(old.prompt("external host activity"))
            await settled.wait()
            try:
                return await super().new_session(display_name=display_name)
            finally:
                # Cancelling the waiter must leave Runtime's owned handoff
                # untouched. Retrieve the independently owned activity error.
                prompt.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)

    interactive.AgentSessionRuntime = ControlledRuntime


if SCENARIO == "settings-write-failure":
    real_replace = os.replace
    def fail_settings_replace(source, destination):
        real_replace(source, destination)
        if str(destination).endswith("settings.json"):
            raise OSError("controlled failure after replacement")
    os.replace = fail_settings_replace
