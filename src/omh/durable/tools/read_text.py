"""Bounded text-file views for durable tools."""

from omh.durable.utils.truncate import (
    DEFAULT_MAX_BYTES,
    format_size,
    truncate_head,
    truncation_json,
    utf8_byte_length,
)


def read_text(
    data: bytes, path: str, offset: int | None, limit: int | None
) -> tuple[str, dict[str, object] | None]:
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
    details: dict[str, object] | None = None
    if truncation.first_line_exceeds_limit:
        first_line_size = format_size(
            utf8_byte_length(all_lines[start_line])
        )
        output_text = (
            f"[Line {start_line_display} is {first_line_size}, exceeds "
            f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n "
            f"'{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )
        details = {"truncation": truncation_json(truncation.without_content())}
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
        details = {"truncation": truncation_json(truncation.without_content())}
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

    return output_text, details
