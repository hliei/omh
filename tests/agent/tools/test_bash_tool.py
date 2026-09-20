from __future__ import annotations

from pathlib import Path

import pytest

from omh.agent import BACKGROUND_CONTEXT
from omh.agent.env import LocalExecutionEnv
from omh.agent.tools import BashExecution, BashToolOptions, create_bash_tool
from omh.llm.types import TextContent
from tests.agent.tools.support import run_tool


async def test_bash_returns_combined_output(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    result, updates = await run_tool(
        create_bash_tool(), env, {"command": "echo out; echo err >&2"}, BACKGROUND_CONTEXT
    )
    assert isinstance(result.content[0], TextContent)
    assert "out" in result.content[0].text
    assert "err" in result.content[0].text
    assert updates


async def test_bash_reports_nonzero_exit(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="Command exited with code 7"):
        await run_tool(
            create_bash_tool(), env, {"command": "exit 7"}, BACKGROUND_CONTEXT
        )


async def test_bash_reports_timeout(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="Command timed out after 0.2 seconds"):
        await run_tool(
            create_bash_tool(),
            env,
            {"command": "sleep 30", "timeout": 0.2},
            BACKGROUND_CONTEXT,
        )


async def test_bash_rejects_invalid_timeout(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="Invalid timeout"):
        await run_tool(
            create_bash_tool(),
            env,
            {"command": "true", "timeout": -1},
            BACKGROUND_CONTEXT,
        )


async def test_bash_truncates_output_and_spills_full_log(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_bash_tool(),
        env,
        {"command": "seq 1 3000"},
        BACKGROUND_CONTEXT,
    )
    assert isinstance(result.content[0], TextContent)
    assert "[Showing lines" in result.content[0].text
    assert isinstance(result.details, dict)
    assert result.details["truncation"]["truncated"] is True
    spill_path = result.details["fullOutputPath"]
    assert isinstance(spill_path, str)
    spilled = Path(spill_path).read_text()
    assert spilled.startswith("1\n")
    assert spilled.rstrip().endswith("3000")


async def test_bash_applies_command_prefix_and_prepare(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    seen: list[BashExecution] = []

    async def prepare(
        execution: BashExecution, tool_context: object, context: object
    ) -> None:
        del tool_context, context
        seen.append(execution)

    result, _ = await run_tool(
        create_bash_tool(
            BashToolOptions(command_prefix="echo prefix", prepare=prepare)
        ),
        env,
        {"command": "echo body"},
        BACKGROUND_CONTEXT,
    )
    assert seen and seen[0].cwd == str(tmp_path)
    assert isinstance(result.content[0], TextContent)
    assert "prefix" in result.content[0].text
    assert "body" in result.content[0].text


async def test_bash_requires_command(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="command must be a string"):
        await run_tool(create_bash_tool(), env, {}, BACKGROUND_CONTEXT)
