# oh-my-harness

Python SDK imported as `omh`. The llm layer configures DeepSeek through Models/Provider and exposes unified text, thinking, and tool-call streams. The agent layer provides in-memory durable Sessions with named Branch history, current values/lists, and an append-only usage ledger.

The supported platforms are macOS and Linux, using standard CPython 3.14 and asyncio. Ubuntu 24.04 x86_64 is the Linux CI baseline; other Linux distributions and architectures are not separately validated. Offline pytest is the implementation check; it does not call a live provider.

```python
from omh.agent import BACKGROUND_CONTEXT, MemorySessionRepo, SessionCreateOptions
from omh.llm import UserMessage

repo = MemorySessionRepo()
session = await repo.create(SessionCreateOptions(), BACKGROUND_CONTEXT)
main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
await main.append_message(UserMessage(content="hello", timestamp=0), BACKGROUND_CONTEXT)
history = await main.find_entries(None, BACKGROUND_CONTEXT)
```

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
mypy
pytest
```

Before pushing code changes, run Ruff, mypy, and pytest as shown above. CI repeats these checks on macOS and Ubuntu 24.04. See [the Git workflow](docs/agents/git-workflow.md) for local checks and handling CI failures.
