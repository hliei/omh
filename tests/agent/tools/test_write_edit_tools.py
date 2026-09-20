from __future__ import annotations

from pathlib import Path

import pytest

from omh.agent import BACKGROUND_CONTEXT
from omh.agent.env import LocalExecutionEnv
from omh.agent.tools import create_edit_tool, create_write_tool
from omh.llm.types import TextContent
from tests.agent.tools.support import run_tool


async def test_write_creates_parent_directories(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_write_tool(),
        env,
        {"path": "nested/deep/file.txt", "content": "hello"},
        BACKGROUND_CONTEXT,
    )
    assert result.content == [TextContent(text="Successfully wrote to nested/deep/file.txt")]
    assert (tmp_path / "nested" / "deep" / "file.txt").read_text() == "hello"


async def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("old")
    env = LocalExecutionEnv(str(tmp_path))
    await run_tool(
        create_write_tool(),
        env,
        {"path": "file.txt", "content": "new"},
        BACKGROUND_CONTEXT,
    )
    assert (tmp_path / "file.txt").read_text() == "new"


async def test_edit_replaces_unique_text_and_reports_diff(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("one\ntwo\nthree\n")
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_edit_tool(),
        env,
        {
            "path": "file.txt",
            "edits": [{"oldText": "two", "newText": "TWO"}],
        },
        BACKGROUND_CONTEXT,
    )
    assert (tmp_path / "file.txt").read_text() == "one\nTWO\nthree\n"
    assert result.content == [
        TextContent(text="Successfully replaced 1 block(s) in file.txt.")
    ]
    assert isinstance(result.details, dict)
    assert result.details["firstChangedLine"] == 2
    assert "+2 TWO" in result.details["diff"]
    assert "--- file.txt" in result.details["patch"]


async def test_edit_applies_multiple_disjoint_edits(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("a\nb\nc\nd\n")
    env = LocalExecutionEnv(str(tmp_path))
    await run_tool(
        create_edit_tool(),
        env,
        {
            "path": "file.txt",
            "edits": [
                {"oldText": "a", "newText": "A"},
                {"oldText": "d", "newText": "D"},
            ],
        },
        BACKGROUND_CONTEXT,
    )
    assert (tmp_path / "file.txt").read_text() == "A\nb\nc\nD\n"


async def test_edit_preserves_crlf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes(b"one\r\ntwo\r\n")
    env = LocalExecutionEnv(str(tmp_path))
    await run_tool(
        create_edit_tool(),
        env,
        {"path": "file.txt", "edits": [{"oldText": "two", "newText": "TWO"}]},
        BACKGROUND_CONTEXT,
    )
    assert target.read_bytes() == b"one\r\nTWO\r\n"


async def test_edit_preserves_bom(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes("\ufeffhello".encode("utf-8"))
    env = LocalExecutionEnv(str(tmp_path))
    await run_tool(
        create_edit_tool(),
        env,
        {"path": "file.txt", "edits": [{"oldText": "hello", "newText": "bye"}]},
        BACKGROUND_CONTEXT,
    )
    assert target.read_text(encoding="utf-8") == "\ufeffbye"


async def test_edit_fuzzy_matches_trailing_whitespace(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("alpha   \nbeta\n")
    env = LocalExecutionEnv(str(tmp_path))
    await run_tool(
        create_edit_tool(),
        env,
        {"path": "file.txt", "edits": [{"oldText": "alpha\nbeta", "newText": "gamma"}]},
        BACKGROUND_CONTEXT,
    )
    assert (tmp_path / "file.txt").read_text() == "gamma\n"


async def test_edit_rejects_missing_text(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hello\n")
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="Could not find the exact text"):
        await run_tool(
            create_edit_tool(),
            env,
            {"path": "file.txt", "edits": [{"oldText": "absent", "newText": "x"}]},
            BACKGROUND_CONTEXT,
        )


async def test_edit_rejects_duplicate_text(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("same\nsame\n")
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="must be unique"):
        await run_tool(
            create_edit_tool(),
            env,
            {"path": "file.txt", "edits": [{"oldText": "same", "newText": "x"}]},
            BACKGROUND_CONTEXT,
        )


async def test_edit_rejects_overlapping_edits(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("abcdef\n")
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="overlap"):
        await run_tool(
            create_edit_tool(),
            env,
            {
                "path": "file.txt",
                "edits": [
                    {"oldText": "abcd", "newText": "x"},
                    {"oldText": "cdef", "newText": "y"},
                ],
            },
            BACKGROUND_CONTEXT,
        )


async def test_edit_rejects_empty_edits(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hello\n")
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="at least one replacement"):
        await run_tool(
            create_edit_tool(),
            env,
            {"path": "file.txt", "edits": []},
            BACKGROUND_CONTEXT,
        )


async def test_edit_rejects_directory(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError, match="Path is not a file"):
        await run_tool(
            create_edit_tool(),
            env,
            {"path": "dir", "edits": [{"oldText": "a", "newText": "b"}]},
            BACKGROUND_CONTEXT,
        )
