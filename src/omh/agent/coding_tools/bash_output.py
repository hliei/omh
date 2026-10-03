"""Bounded bash text snapshots and lazily retained complete output."""

import codecs
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

from omh._tool_utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_tail,
    truncation_json,
    utf8_byte_length,
)
from omh.agent.coding_tools.file_io import run_file_io
from omh.agent.tools import AgentToolResult
from omh.llm.types import TextContent


class BashOutput:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._text = ""
        self._prefix = bytearray()
        self._total_bytes = 0
        self._newlines = 0
        self._ends_with_newline = True
        self._last_line_bytes = 0
        self._current_line_bytes = 0
        self._spill: BinaryIO | None = None
        self._spill_path: str | None = None

    async def append(self, chunk: bytes) -> None:
        self._append_text(self._decoder.decode(chunk))
        if self._spill is not None:
            await run_file_io(lambda: self._spill_write(chunk))
        else:
            self._prefix.extend(chunk)
            if self._truncated:
                await run_file_io(self._start_spill)

    async def finish(self) -> None:
        self._append_text(self._decoder.decode(b"", final=True))
        if self._truncated and self._spill is None:
            await run_file_io(self._start_spill)

    async def close(self) -> None:
        if self._spill is not None:
            await run_file_io(self._spill.close)

    def _spill_write(self, chunk: bytes) -> None:
        assert self._spill is not None
        self._spill.write(chunk)
        self._spill.flush()

    def _start_spill(self) -> None:
        descriptor, path = tempfile.mkstemp(prefix="omh-bash-", suffix=".log")
        self._spill = os.fdopen(descriptor, "wb")
        self._spill_path = str(Path(path).absolute())
        self._spill_write(bytes(self._prefix))
        self._prefix.clear()

    @property
    def _total_lines(self) -> int:
        return self._newlines + (0 if self._ends_with_newline else 1)

    @property
    def _truncated(self) -> bool:
        return self._total_bytes > DEFAULT_MAX_BYTES or self._total_lines > DEFAULT_MAX_LINES

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self._total_bytes += utf8_byte_length(text)
        self._newlines += text.count("\n")
        self._ends_with_newline = text.endswith("\n")
        parts = text.split("\n")
        if len(parts) > 1:
            self._last_line_bytes = (
                self._current_line_bytes + utf8_byte_length(parts[0])
                if len(parts) == 2 else utf8_byte_length(parts[-2])
            )
            self._current_line_bytes = utf8_byte_length(parts[-1])
        else:
            self._current_line_bytes += utf8_byte_length(text)
        if not self._ends_with_newline:
            self._last_line_bytes = self._current_line_bytes
        # Keep extra context so truncation can choose whole lines, even when
        # the byte window starts partway through a line.
        self._text = (self._text + text).encode("utf-8")[-2 * DEFAULT_MAX_BYTES:].decode(
            "utf-8", errors="ignore"
        )

    def snapshot(self, *, final: bool = False) -> AgentToolResult:
        tail = truncate_tail(self._text)
        metadata = replace(
            tail.without_content(), truncated=self._truncated,
            total_lines=self._total_lines, total_bytes=self._total_bytes,
        )
        details = None
        text = tail.content
        if metadata.truncated:
            details = {"truncation": truncation_json(metadata), "fullOutputPath": self._spill_path}
            if final:
                start = metadata.total_lines - metadata.output_lines + 1
                end = metadata.total_lines
                if metadata.last_line_partial:
                    description = (
                        f"last {format_size(metadata.output_bytes)} of line {end} "
                        f"(line is {format_size(self._last_line_bytes)})"
                    )
                else:
                    description = f"lines {start}-{end} of {metadata.total_lines}"
                    if metadata.truncated_by == "bytes":
                        description += f" ({format_size(DEFAULT_MAX_BYTES)} limit)"
                text += f"\n\n[Showing {description}. Full output: {self._spill_path}]"
        return AgentToolResult(content=[TextContent(text=text or ("(no output)" if final else ""))], details=details)
