"""Offline stream installed as sitecustomize for interactive PTY tests."""

import os

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


def stream_fn(model, context, options):
    global CALLS
    CALLS += 1
    summary = is_summary(context)
    counted("summary" if summary else "dialogue")
    stream = create_assistant_message_event_stream()
    output = message(model)
    if summary:
        emit_text(stream, output, "summary of the conversation")
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
    if SCENARIO == "error":
        if CALLS == 1:
            emit_error(stream, output, "kept visible", "provider rejected the turn")
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
    if user == "mark":
        if tool == "write":
            emit_text(stream, output, "marked")
        else:
            emit_tool(stream, output, "write", {"path": "marker.txt", "content": "marked"}, "call-mark")
        return stream
    emit_text(stream, output, f"reply:{user}")
    return stream


class ControlledHost(cli.CodingAgentHost):
    def build_options(self, selection, *, session_file=None, agent_options=None):
        options = super().build_options(
            selection, session_file=session_file, agent_options=agent_options,
        )
        options.stream_fn = stream_fn
        options.agent_options.retry = RetryPolicy(base_delay_ms=0)
        if SCENARIO == "compact":
            options.agent_options.compaction = CompactionSettings(
                reserve_tokens=999_999, keep_recent_tokens=0,
            )
        return options


cli.CodingAgentHost = ControlledHost
