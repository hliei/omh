# oh-my-harness

Python SDK imported as `omh`. First delivery is the llm layer: configure DeepSeek through Models/Provider and consume unified text, thinking, and tool-call streams.

Formal support is macOS, CPython 3.14, and asyncio. Offline pytest is the implementation check; it does not call a live provider.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
mypy
```
