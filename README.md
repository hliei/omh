# oh-my-harness

Python SDK imported as `omh`. First delivery is the llm layer: configure DeepSeek through Models/Provider and consume unified text, thinking, and tool-call streams.

The supported platforms are macOS and Linux, using standard CPython 3.14 and asyncio. Ubuntu 24.04 x86_64 is the Linux CI baseline; other Linux distributions and architectures are not separately validated. Offline pytest is the implementation check; it does not call a live provider.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
mypy
pytest
```

Before pushing code changes, run Ruff, mypy, and pytest as shown above. CI repeats these checks on macOS and Ubuntu 24.04. See [the Git workflow](docs/agents/git-workflow.md) for local checks and handling CI failures.
