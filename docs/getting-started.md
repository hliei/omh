# Getting started

omh provides durable agent execution with streaming, tools, and conversation history. Start with a DeepSeek agent below, then use the Session examples to explore storage without model credentials. For execution and recovery contracts, see the [harness overview](harness.md) and [public surface](harness/public-api.md).

## Install from source

Use standard CPython 3.14 on macOS or Linux. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The Python distribution and import name are both `omh`. For development dependencies and checks, see [Development](../README.md#development).

## Run a minimal agent

The agent examples call DeepSeek and require an API key in the process environment:

```bash
export DEEPSEEK_API_KEY="your-api-key"
python examples/minimal_agent.py
```

The SDK reads the environment variable; it does not automatically load `.env` files. These commands run from the repository root after the source installation above. Each script is self-contained and can also be copied into your application. The examples select `deepseek-flash` from the built-in model registry.

The built-in provider sends requests to `https://api.deepseek.com`. An HTTP 401 authentication error means that endpoint rejected the credential; check that `DEEPSEEK_API_KEY` is valid for the official DeepSeek API, then rerun the script.

[Minimal agent source](../examples/minimal_agent.py):

```python
import asyncio

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    BranchScan,
    MemorySessionRepo,
    SessionCreateOptions,
)
from omh.llm import content_text, create_models, deepseek_provider


async def main() -> None:
    ctx = BACKGROUND_CONTEXT
    models = create_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None

    repo = MemorySessionRepo()
    try:
        session = await repo.create(SessionCreateOptions(), ctx)
        created = await AgentHarness.create(
            AgentHarnessOptions(session=session, models=models, model=model), ctx
        )
        harness = created.harness
        try:
            lane = await harness.lane("main", ctx)
            result = await lane.prompt("What is 2 + 2? Answer briefly.", ctx)
            if not result.ok:
                raise RuntimeError(result.error)
            if result.value.status != "completed":
                raise RuntimeError(result.value.error or result.value.status)

            entries = await lane.find_entries(BranchScan(limit=1), ctx)
            for entry in entries:
                if entry.type == "message" and entry.message.role == "assistant":
                    print(content_text(list(entry.message.content)))
        finally:
            await harness.close(ctx)
    finally:
        await repo.close(ctx)


if __name__ == "__main__":
    asyncio.run(main())
```

A Session owns history, the harness drives model and tool calls, and a named lane selects a conversation. `lane.prompt` waits for the run to settle. Call it again on the same lane to continue with its existing history.

Check both result layers: `result.ok` reports whether the interface call succeeded, while `result.value.status` reports whether the operation completed, failed, or was aborted. The operation result contains IDs and status; read the assistant message from lane history to get its text. The default `BranchScan` order is newest first.

Close the harness before the repository. These examples use process-local Memory storage; use SQLite when the conversation must survive process restarts.

## Stream the answer

```bash
python examples/streaming_agent.py
```

The [streaming example](../examples/streaming_agent.py) subscribes before prompting and prints each text delta:

```python
def print_delta(event: HarnessEvent, context: Context) -> None:
    if event.type == "message_update" and event.lane == "main":
        if event.event.type == "text_delta":
            print(event.event.delta, end="", flush=True)

unsubscribe = harness.events.on("message_update", print_delta)
```

Import `HarnessEvent` and `Context` from `omh.agent`. Events arrive while `lane.prompt` is awaiting completion. Call `unsubscribe()` when finished, and still check the terminal operation status: streamed text may belong to an attempt that later fails or retries. For an initial snapshot plus subsequent events, use `lane.watch()`.

## Give the agent a tool

```bash
python examples/tool_agent.py
```

The [tool example](../examples/tool_agent.py) registers an `add` tool with a JSON Schema, an async implementation, and `replay="safe"`. It asks the model to calculate `137 + 289` using that tool. A successful tool invocation prints:

```text
Tool: add(137, 289) = 426
```

The harness validates the model's arguments, calls the function, stores the tool result, and asks the model to continue. The script then prints the model's final answer. Tool selection is made by the model; the `Tool:` line confirms that it actually invoked the function.

The callable takes `(tool_call_id, arguments, on_update, tool_context, invocation, context)` and returns `AgentToolResult`. The example only needs `arguments`. Pure addition is safe to replay after interruption; choose a replay policy based on the effects of your own tool. The default is `"never"`. See [tools and execution environments](harness/public-api.md#tools-and-execution-environments) for progress, tool context, and built-in tools.

## Set a system prompt

Register a context transformation before prompting:

```python
from omh.agent import Context, TransformContextResult


def assistant_instructions(event: object, context: Context) -> TransformContextResult:
    return TransformContextResult(system_prompt="You are a concise assistant.")


harness.hooks.on("transform_context", assistant_instructions)
```

Hooks are process-local configuration; register them again when constructing a harness for a reopened Session.

## In-memory history

The following storage examples run offline and do not require an API key.

A Session stores history; a named Branch selects a path through it. Save this example as a Python file and run it in the environment above:

```python
import asyncio

from omh.agent import BACKGROUND_CONTEXT, MemorySessionRepo, SessionCreateOptions
from omh.llm import UserMessage


async def main() -> None:
    repo = MemorySessionRepo()
    try:
        session = await repo.create(SessionCreateOptions(), BACKGROUND_CONTEXT)
        branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
        await branch.append_message(
            UserMessage(content="hello", timestamp=0), BACKGROUND_CONTEXT
        )
        history = await branch.find_entries(None, BACKGROUND_CONTEXT)
        print(history)
    finally:
        await repo.close(BACKGROUND_CONTEXT)


asyncio.run(main())
```

Memory storage is process-local. Use SQLite when history must survive closing the repository and restarting the process.

## SQLite persistence

This example uses a temporary directory so it can be rerun without conflicting with an existing Session. An application should supply a persistent directory instead.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.agent import BACKGROUND_CONTEXT, SessionCreateOptions
from omh.llm import UserMessage
from omh.session_backends.sqlite import SqliteSessionRepo


async def main() -> None:
    with TemporaryDirectory() as directory:
        repo = SqliteSessionRepo(Path(directory))
        try:
            session = await repo.create(
                SessionCreateOptions(id="chat"), BACKGROUND_CONTEXT
            )
            branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
            await branch.append_message(
                UserMessage(content="hello", timestamp=0), BACKGROUND_CONTEXT
            )
            metadata = session.metadata
        finally:
            await repo.close(BACKGROUND_CONTEXT)

        repo = SqliteSessionRepo(Path(directory))
        try:
            session = await repo.open(metadata, BACKGROUND_CONTEXT)
            print(session.metadata)
        finally:
            await repo.close(BACKGROUND_CONTEXT)


asyncio.run(main())
```

Reopening accesses the same stored Session. It does not automatically run an agent or resume an operation. Across processes, retain the Session metadata or discover it through the repository's listing API. The host must ensure a single writable owner; see the [storage contract](harness/storage.md).
