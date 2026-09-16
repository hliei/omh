from __future__ import annotations

import importlib
import sys


def test_omh_is_importable() -> None:
    module = importlib.import_module("omh")
    assert module.__name__ == "omh"


def test_llm_does_not_import_agent() -> None:
    for name in list(sys.modules):
        if name == "omh.agent" or name.startswith("omh.agent."):
            del sys.modules[name]

    importlib.import_module("omh.llm")
    assert "omh.agent" not in sys.modules
    assert not any(name.startswith("omh.agent.") for name in sys.modules)


def test_agent_does_not_import_session_backends() -> None:
    importlib.import_module("omh.session_backends.sqlite")

    for name in list(sys.modules):
        if name.startswith(("omh.agent", "omh.session_backends")):
            del sys.modules[name]

    importlib.import_module("omh.agent")
    assert "omh.agent" in sys.modules
    assert not any(name.startswith("omh.session_backends") for name in sys.modules)
