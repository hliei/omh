# Getting started

omh provides durable agent execution and a Session API that can also be used directly. These examples introduce Session history and persistence without calling a model or requiring credentials. For execution, tools, and recovery, continue with the [harness overview](harness.md) and [public surface](harness/public-api.md).

## Install from source

Use standard CPython 3.14 on macOS or Linux. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The Python distribution and import name are both `omh`. For development dependencies and checks, see [Development](../README.md#development).

## In-memory history

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
