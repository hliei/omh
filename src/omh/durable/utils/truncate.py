"""Durable imports for shared pure tool mechanisms."""

from omh._tool_utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    ShellOutputTruncation,
    TruncatedBy,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_tail,
    truncation_json,
    utf8_byte_length,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "ShellOutputTruncation",
    "TruncatedBy",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_tail",
    "truncation_json",
    "utf8_byte_length",
]
