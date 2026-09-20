from __future__ import annotations

import struct
from pathlib import Path

import pytest

from omh.agent import BACKGROUND_CONTEXT
from omh.agent.env import LocalExecutionEnv
from omh.agent.execution_env import FileError
from omh.agent.tools import (
    ReadImageProcessorFailure,
    ReadImageProcessorOptions,
    ReadImageProcessorSuccess,
    ReadToolOptions,
    create_read_tool,
)
from omh.llm.types import ImageContent, TextContent
from tests.agent.tools.support import run_tool


def _png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + b"\x00" * 13
        + b"\x00" * 4
    )


async def test_read_text_file(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha\nbeta\n")
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_read_tool(), env, {"path": "note.txt"}, BACKGROUND_CONTEXT
    )
    assert result.content == [TextContent(text="alpha\nbeta\n")]
    assert result.details is None


async def test_read_offset_and_limit_reports_remaining(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("a\nb\nc\nd\n")
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_read_tool(),
        env,
        {"path": "note.txt", "offset": 2, "limit": 1},
        BACKGROUND_CONTEXT,
    )
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text.startswith("b")
    assert "Use offset=3 to continue" in result.content[0].text


async def test_read_beyond_end_raises(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("a\n")
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(ValueError):
        await run_tool(
            create_read_tool(),
            env,
            {"path": "note.txt", "offset": 10},
            BACKGROUND_CONTEXT,
        )


async def test_read_missing_file_raises_file_error(tmp_path: Path) -> None:
    env = LocalExecutionEnv(str(tmp_path))
    with pytest.raises(FileError):
        await run_tool(
            create_read_tool(), env, {"path": "missing.txt"}, BACKGROUND_CONTEXT
        )


async def test_read_truncates_and_reports_details(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(3000)))
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_read_tool(), env, {"path": "big.txt"}, BACKGROUND_CONTEXT
    )
    assert isinstance(result.content[0], TextContent)
    assert "Use offset=2001 to continue" in result.content[0].text
    assert isinstance(result.details, dict)
    truncation = result.details["truncation"]
    assert truncation["truncated"] is True
    assert truncation["truncatedBy"] == "lines"


async def test_read_png_returns_image_content(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(_png())
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_read_tool(), env, {"path": "image.png"}, BACKGROUND_CONTEXT
    )
    assert result.content[0] == TextContent(text="Read image file [image/png]")
    image = result.content[1]
    assert isinstance(image, ImageContent)
    assert image.mime_type == "image/png"
    assert image.data


async def test_read_bmp_without_processor_omits_image(tmp_path: Path) -> None:
    (tmp_path / "image.bmp").write_bytes(_bmp())
    env = LocalExecutionEnv(str(tmp_path))
    result, _ = await run_tool(
        create_read_tool(), env, {"path": "image.bmp"}, BACKGROUND_CONTEXT
    )
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert "Image omitted" in result.content[0].text


async def test_read_bmp_with_processor_converts(tmp_path: Path) -> None:
    (tmp_path / "image.bmp").write_bytes(_bmp())
    env = LocalExecutionEnv(str(tmp_path))
    seen: list[ReadImageProcessorOptions] = []

    async def processor(
        data: bytes, mime_type: str, options: ReadImageProcessorOptions, context: object
    ) -> object:
        del data, context
        seen.append(options)
        assert mime_type == "image/bmp"
        return ReadImageProcessorSuccess(
            data="converted", mime_type="image/png", hints=["resized"]
        )

    result, _ = await run_tool(
        create_read_tool(ReadToolOptions(image_processor=processor)),
        env,
        {"path": "image.bmp"},
        BACKGROUND_CONTEXT,
    )
    assert seen == [ReadImageProcessorOptions(auto_resize_images=True)]
    assert result.content[0] == TextContent(
        text="Read image file [image/png]\nresized"
    )
    image = result.content[1]
    assert isinstance(image, ImageContent)
    assert image.data == "converted"


async def test_read_processor_failure_returns_message(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(_png())
    env = LocalExecutionEnv(str(tmp_path))

    async def processor(
        data: bytes, mime_type: str, options: ReadImageProcessorOptions, context: object
    ) -> object:
        del data, mime_type, options, context
        return ReadImageProcessorFailure(message="cannot convert")

    result, _ = await run_tool(
        create_read_tool(ReadToolOptions(image_processor=processor)),
        env,
        {"path": "image.png"},
        BACKGROUND_CONTEXT,
    )
    assert result.content == [
        TextContent(text="Read image file [image/png]\ncannot convert")
    ]


def _bmp() -> bytes:
    # 26 bytes with a 12-byte DIB header, one color plane, 24 bits per pixel.
    header = bytearray(26)
    header[0:2] = b"BM"
    header[2:6] = (30).to_bytes(4, "little")
    header[10:14] = (26).to_bytes(4, "little")
    header[14:18] = (12).to_bytes(4, "little")
    header[22:24] = (1).to_bytes(2, "little")
    header[24:26] = (24).to_bytes(2, "little")
    return bytes(header)
