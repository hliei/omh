from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest


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


def test_agent_is_traditional_entry_and_does_not_import_durable() -> None:
    for name in list(sys.modules):
        if name.startswith(("omh.agent", "omh.durable")):
            del sys.modules[name]

    agent = importlib.import_module("omh.agent")
    assert hasattr(agent, "Agent")
    assert not hasattr(agent, "AgentHarness")
    assert not any(name.startswith("omh.durable") for name in sys.modules)


def test_durable_namespace_is_explicitly_importable() -> None:
    durable = importlib.import_module("omh.durable")
    assert hasattr(durable, "AgentHarness")
    assert hasattr(durable, "create_agent_harness")


@pytest.mark.parametrize("module", ["omh.durable", "omh.session_backends.sqlite"])
def test_durable_imports_do_not_load_agent(module: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}\n"
            "import sys\n"
            "assert not any(name == 'omh.agent' or name.startswith('omh.agent.') "
            "for name in sys.modules)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_old_durable_namespace_is_removed() -> None:
    assert importlib.util.find_spec("omh.agent.durable") is None


def test_sdk_does_not_import_application() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import omh.agent, omh.durable, omh.session_backends.sqlite\n"
            "import sys\n"
            "assert not any(name == 'coding_agent' or name.startswith('coding_agent.') "
            "for name in sys.modules)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
