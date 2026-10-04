"""File reading through the in-process AgentTool contract."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from omh.agent.execution._async import call_with_signal
from omh.agent.execution.tools import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
)
from omh.agent.tools.arguments import required_string
from omh.agent.tools.file_io import run_file_io
from omh.agent.tools.image import detect_supported_image_mime_type, encode_base64
from omh.agent.tools.path_utils import resolve_read_tool_path
from omh.agent.tools.read_text import read_text
from omh.agent.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from omh.llm.types import AbortSignal, ImageContent, TextContent


@dataclass(frozen=True, slots=True)
class ReadImageProcessorOptions:
    auto_resize_images: bool


@dataclass(frozen=True, slots=True)
class ReadImageProcessorSuccess:
    data: str
    mime_type: str
    hints: list[str]
    ok: bool = True


@dataclass(frozen=True, slots=True)
class ReadImageProcessorFailure:
    message: str
    ok: bool = False


type ReadImageProcessorResult = ReadImageProcessorSuccess | ReadImageProcessorFailure

type ReadImageProcessor = Callable[
    [bytes, str, ReadImageProcessorOptions, AbortSignal | None],
    Awaitable[ReadImageProcessorResult],
]


@dataclass(frozen=True, slots=True)
class ReadToolOptions:
    """Optional read-tool behavior; no image conversion is built in."""

    auto_resize_images: bool = True
    image_processor: ReadImageProcessor | None = None


def create_read_tool(cwd: str | Path, options: ReadToolOptions | None = None) -> AgentTool:
    """Create a read tool with an explicit working directory."""
    directory = Path(cwd).expanduser().absolute()
    resolved_options = options or ReadToolOptions()

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        if signal is not None:
            signal.throw_if_aborted()
        user_path = required_string(arguments, "path")
        offset = _optional_line_count(arguments, "offset")
        limit = _optional_line_count(arguments, "limit")
        path = await run_file_io(lambda: resolve_read_tool_path(directory, user_path))
        if signal is not None:
            signal.throw_if_aborted()
        data = await run_file_io(path.read_bytes)
        if signal is not None:
            signal.throw_if_aborted()
        mime_type = detect_supported_image_mime_type(data)
        if mime_type is not None:
            return await _read_image(data, mime_type, resolved_options, signal)
        content, details = read_text(data, user_path, offset, limit)
        return AgentToolResult(content=[TextContent(text=content)], details=details)

    return AgentTool(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text and images (jpg, png, gif, webp, bmp). "
            f"Text output is truncated to {DEFAULT_MAX_LINES} lines or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "Use offset/limit for large files and continue with offset until complete."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1, "description": "Starting line (1-indexed)"},
                "limit": {"type": "integer", "minimum": 1, "description": "Maximum number of lines"},
            },
            "required": ["path"],
        },
        execute=execute,
    )


def _optional_line_count(arguments: dict[str, object], name: str) -> int | None:
    if name not in arguments:
        return None
    value = arguments[name]
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 1 or value % 1 != 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


async def _read_image(
    data: bytes,
    mime_type: str,
    options: ReadToolOptions,
    signal: AbortSignal | None,
) -> AgentToolResult:
    processor = options.image_processor
    if processor is not None:
        processed: ReadImageProcessorResult = await call_with_signal(
            lambda: processor(
                data, mime_type,
                ReadImageProcessorOptions(auto_resize_images=options.auto_resize_images), signal,
            ), signal,
        )
        if signal is not None:
            signal.throw_if_aborted()
        if isinstance(processed, ReadImageProcessorFailure):
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Read image file [{mime_type}]\n{processed.message}"
                    )
                ]
            )
        hints = (
            "\n" + "\n".join(processed.hints) if processed.hints else ""
        )
        return AgentToolResult(
            content=[
                TextContent(
                    text=f"Read image file [{processed.mime_type}]{hints}"
                ),
                ImageContent(
                    data=processed.data, mime_type=processed.mime_type
                ),
            ]
        )
    if mime_type == "image/bmp":
        return AgentToolResult(
            content=[
                TextContent(
                    text=(
                        "Read image file [image/bmp]\n[Image omitted: configure "
                        "an image_processor to convert BMP images.]"
                    )
                )
            ]
        )
    return AgentToolResult(
        content=[
            TextContent(text=f"Read image file [{mime_type}]"),
            ImageContent(data=encode_base64(data), mime_type=mime_type),
        ]
    )
