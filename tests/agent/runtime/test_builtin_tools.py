from __future__ import annotations

from pathlib import Path

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    BranchScan,
    MemorySessionRepo,
    SessionCreateOptions,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from omh.agent.env import LocalExecutionEnv
from omh.agent.tools import ExecutionToolContext
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


class ScriptedModels:
    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self.tool_calls = tool_calls
        self.contexts: list[LlmContext] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return MODEL if (provider, model_id) == (MODEL.provider, MODEL.id) else None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is MODEL
        index = len(self.contexts)
        self.contexts.append(context)
        if index < len(self.tool_calls):
            return _stream(
                AssistantMessage(
                    api=MODEL.api,
                    provider=MODEL.provider,
                    model=MODEL.id,
                    usage=USAGE,
                    stop_reason="toolUse",
                    timestamp=NOW + index + 1,
                    content=[self.tool_calls[index]],
                )
            )
        return _stream(
            AssistantMessage(
                api=MODEL.api,
                provider=MODEL.provider,
                model=MODEL.id,
                usage=USAGE,
                stop_reason="stop",
                timestamp=NOW + index + 1,
                content=[TextContent(text="done")],
            )
        )


async def test_builtin_tools_run_through_the_harness(tmp_path: Path) -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    env = LocalExecutionEnv(str(tmp_path))
    models = ScriptedModels(
        [
            ToolCall(
                id="write-1",
                name="write",
                arguments={"path": "note.txt", "content": "hello\n"},
            ),
            ToolCall(
                id="edit-1",
                name="edit",
                arguments={
                    "path": "note.txt",
                    "edits": [{"oldText": "hello", "newText": "goodbye"}],
                },
            ),
            ToolCall(
                id="read-1", name="read", arguments={"path": "note.txt"}
            ),
            ToolCall(
                id="bash-1", name="bash", arguments={"command": "wc -c < note.txt"}
            ),
        ]
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            tools=(
                create_write_tool(),
                create_edit_tool(),
                create_read_tool(),
                create_bash_tool(),
            ),
            tool_context=ExecutionToolContext(env),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("update the note", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.status == "completed"
    assert (tmp_path / "note.txt").read_text() == "goodbye\n"

    tool_results = [
        message
        for message in models.contexts[-1].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [message.tool_call_id for message in tool_results] == [
        "write-1",
        "edit-1",
        "read-1",
        "bash-1",
    ]
    assert tool_results[0].content == [
        TextContent(text="Successfully wrote to note.txt")
    ]
    assert tool_results[1].content == [
        TextContent(text="Successfully replaced 1 block(s) in note.txt.")
    ]
    assert tool_results[2].content == [TextContent(text="goodbye\n")]
    bash_text = tool_results[3].content[0]
    assert isinstance(bash_text, TextContent)
    assert "8" in bash_text.text

    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [entry.message.role for entry in history if entry.type == "message"] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
        "toolResult",
        "assistant",
        "toolResult",
        "assistant",
        "toolResult",
        "assistant",
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_tool_context_provider_is_resolved_per_turn(tmp_path: Path) -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    env = LocalExecutionEnv(str(tmp_path))
    resolved: list[str] = []

    def provider(context: object) -> ExecutionToolContext:
        del context
        resolved.append("called")
        return ExecutionToolContext(env)

    models = ScriptedModels(
        [
            ToolCall(
                id="write-1",
                name="write",
                arguments={"path": "file.txt", "content": "x"},
            )
        ]
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            tools=(create_write_tool(),),
            tool_context=provider,
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("write a file", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert (tmp_path / "file.txt").read_text() == "x"
    assert resolved == ["called"]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
