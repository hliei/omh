from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024

type TruncatedBy = Literal["lines", "bytes"]


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """The bounded view of one tool output plus the numbers that describe it."""

    content: str
    truncated: bool
    truncated_by: TruncatedBy | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def without_content(self) -> ShellOutputTruncation:
        """Project this result to the metadata shape carried alongside shell output."""
        return ShellOutputTruncation(
            truncated=self.truncated,
            truncated_by=self.truncated_by,
            total_lines=self.total_lines,
            total_bytes=self.total_bytes,
            output_lines=self.output_lines,
            output_bytes=self.output_bytes,
            last_line_partial=self.last_line_partial,
            first_line_exceeds_limit=self.first_line_exceeds_limit,
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
        )


@dataclass(frozen=True, slots=True)
class ShellOutputTruncation:
    """Truncation metadata without a duplicate copy of the retained text."""

    truncated: bool
    truncated_by: TruncatedBy | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def utf8_byte_length(content: str) -> int:
    """Return the UTF-8 byte length of ``content``, replacing unpaired surrogates.

    JavaScript strings are UTF-16 and ``Buffer.byteLength`` substitutes lone
    surrogates with U+FFFD; Python strings can also hold lone surrogates, so
    encode with replacement instead of raising.
    """
    if content.isascii():
        return len(content)
    return len(content.encode("utf-8", errors="surrogatepass"))


def format_size(byte_count: int) -> str:
    """Format bytes as a human-readable size."""
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def _split_lines_for_counting(content: str) -> list[str]:
    if content == "":
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _replace_unpaired_surrogates(content: str) -> str:
    return content.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="replace"
    )


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes)."""
    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line = lines[0] if lines else ""
    if utf8_byte_length(first_line) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: TruncatedBy = "lines"
    for index, line in enumerate(lines):
        if index >= max_lines:
            break
        line_bytes = utf8_byte_length(line) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes

    if len(output_lines) >= max_lines and output_bytes <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes).

    May return a partial first line when the last line exceeds the byte limit.
    """
    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: TruncatedBy = "lines"
    last_line_partial = False
    for line in reversed(lines):
        if len(output_lines) >= max_lines:
            break
        line_bytes = utf8_byte_length(line) + (1 if output_lines else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes = utf8_byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    if len(output_lines) >= max_lines and output_bytes <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """Truncate a string to fit within a byte limit, counting from the end."""
    if max_bytes <= 0:
        return ""
    output_bytes = 0
    start = len(text)
    needs_replacement = False
    for index in range(len(text), 0, -1):
        character = text[index - 1]
        character_bytes = utf8_byte_length(character)
        if output_bytes + character_bytes > max_bytes:
            break
        output_bytes += character_bytes
        start = index - 1
        needs_replacement = needs_replacement or "\ud800" <= character <= "\udfff"
    output = text[start:]
    return _replace_unpaired_surrogates(output) if needs_replacement else output


def truncation_json(truncation: ShellOutputTruncation) -> dict[str, object]:
    """Return the model- and application-facing JSON shape of a truncation."""
    return {
        "truncated": truncation.truncated,
        "truncatedBy": truncation.truncated_by,
        "totalLines": truncation.total_lines,
        "totalBytes": truncation.total_bytes,
        "outputLines": truncation.output_lines,
        "outputBytes": truncation.output_bytes,
        "lastLinePartial": truncation.last_line_partial,
        "firstLineExceedsLimit": truncation.first_line_exceeds_limit,
        "maxLines": truncation.max_lines,
        "maxBytes": truncation.max_bytes,
    }

