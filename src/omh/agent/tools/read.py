from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from omh.agent.agent_harness import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentToolResult,
)
from omh.agent.context import Context
from omh.agent.execution_env import get_or_throw
from omh.agent.tools.arguments import optional_int, required_string
from omh.agent.tools.image import detect_supported_image_mime_type, encode_base64
from omh.agent.tools.path_utils import resolve_read_tool_path
from omh.agent.tools.tool_context import execution_env
from omh.agent.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
    truncation_json,
    utf8_byte_length,
)
from omh.llm.types import ImageContent, JsonValue, TextContent

_READ_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "number",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of lines to read",
        },
    },
    "required": ["path"],
}


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
    [bytes, str, ReadImageProcessorOptions, Context],
    Awaitable[ReadImageProcessorResult],
]


@dataclass(frozen=True, slots=True)
class ReadToolOptions:
    """Optional read-tool behavior; no image conversion is built in."""

    auto_resize_images: bool = True
    image_processor: ReadImageProcessor | None = None


def create_read_tool(options: ReadToolOptions | None = None) -> AgentHarnessTool:
    """Create the ``read`` tool. ``tool_context`` must be an ``ExecutionToolContext``."""
    resolved_options = options or ReadToolOptions()

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        on_update: AgentHarnessToolUpdateCallback,
        tool_context: object,
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, on_update, invocation
        env = execution_env(tool_context)
        path = required_string(arguments, "path")
        offset = optional_int(arguments, "offset")
        limit = optional_int(arguments, "limit")
        absolute_path = await resolve_read_tool_path(env, path, context)
        data = get_or_throw(await env.read_binary_file(absolute_path, context))
        mime_type = detect_supported_image_mime_type(data)
        if mime_type is not None:
            return await _read_image(
                data, mime_type, resolved_options, context
            )
        return _read_text(data, path, offset, limit)

    return AgentHarnessTool(
        name="read",
        description=(
            "Read the contents of a file. Supports text files and images "
            "(jpg, png, gif, webp, bmp). Images are sent as attachments. For "
            f"text files, output is truncated to {DEFAULT_MAX_LINES} lines or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use "
            "offset/limit for large files. When you need the full file, "
            "continue with offset until complete."
        ),
        parameters=_READ_SCHEMA,
        execute=execute,
    )


async def _read_image(
    data: bytes,
    mime_type: str,
    options: ReadToolOptions,
    context: Context,
) -> AgentToolResult:
    if options.image_processor is not None:
        processed = await options.image_processor(
            data,
            mime_type,
            ReadImageProcessorOptions(auto_resize_images=options.auto_resize_images),
            context,
        )
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
                        "an imageProcessor to convert BMP images.]"
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


def _read_text(
    data: bytes, path: str, offset: int | None, limit: int | None
) -> AgentToolResult:
    text_content = data.decode("utf-8", errors="replace")
    all_lines = text_content.split("\n")
    total_file_lines = len(all_lines)
    start_line = max(0, offset - 1) if offset else 0
    start_line_display = start_line + 1
    if start_line >= len(all_lines):
        raise ValueError(
            f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
        )

    if limit is not None:
        end_line = min(start_line + limit, len(all_lines))
        selected_content = "\n".join(all_lines[start_line:end_line])
        user_limited_lines: int | None = end_line - start_line
    else:
        selected_content = "\n".join(all_lines[start_line:])
        user_limited_lines = None

    truncation = truncate_head(selected_content)
    details: JsonValue = None
    if truncation.first_line_exceeds_limit:
        first_line_size = format_size(
            utf8_byte_length(all_lines[start_line])
        )
        output_text = (
            f"[Line {start_line_display} is {first_line_size}, exceeds "
            f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n "
            f"'{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )
        details = cast(JsonValue, {"truncation": truncation_json(truncation.without_content())})
    elif truncation.truncated:
        end_line_display = start_line_display + truncation.output_lines - 1
        next_offset = end_line_display + 1
        output_text = truncation.content
        if truncation.truncated_by == "lines":
            output_text += (
                f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                f"{total_file_lines}. Use offset={next_offset} to continue.]"
            )
        else:
            output_text += (
                f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                f"{total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                f"Use offset={next_offset} to continue.]"
            )
        details = cast(JsonValue, {"truncation": truncation_json(truncation.without_content())})
    elif (
        user_limited_lines is not None
        and start_line + user_limited_lines < len(all_lines)
    ):
        remaining = len(all_lines) - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        output_text = (
            f"{truncation.content}\n\n[{remaining} more lines in file. Use "
            f"offset={next_offset} to continue.]"
        )
    else:
        output_text = truncation.content

    return AgentToolResult(
        content=[TextContent(text=output_text)], details=details
    )
