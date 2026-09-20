"""Bounded shell output capture with adaptive publication.

Corresponds to ``packages/agent/src/harness/utils/output-capture.ts``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from omh.agent.context import Context
from omh.agent.execution_env import (
    ShellOutputAppend,
    ShellOutputCaptureOptions,
    ShellOutputMetadata,
    ShellOutputMetadataUpdate,
    ShellOutputReplace,
    ShellOutputSlide,
    ShellOutputUpdate,
    ShellOutputView,
)
from omh.agent.utils.adaptive_publisher import (
    AdaptivePublisher,
    AdaptivePublisherOptions,
)
from omh.agent.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    ShellOutputTruncation,
    truncate_head,
    truncate_tail,
    utf8_byte_length,
)

OUTPUT_MIN_EMIT_INTERVAL_MS = 100
OUTPUT_TARGET_BYTES_PER_SECOND = 100 * 1024

_INVALID_SHELL_OUTPUT = re.compile(r"[\x00-\x08\x0b-\x1f\ufff9-\ufffb]")


@dataclass(frozen=True, slots=True)
class OutputCaptureHandlers:
    on_update: Callable[[ShellOutputUpdate, Context], None] | None = None
    on_error: Callable[[BaseException], None] | None = None


class OutputCapture:
    """Maintain and publish one bounded shell-output view.

    Writes received while publication is rate-limited collapse into the latest
    view. The first update after idle and an explicit final flush are immediate.
    """

    def __init__(
        self,
        options: ShellOutputCaptureOptions | None,
        context: Context,
        handlers: OutputCaptureHandlers,
    ) -> None:
        self._max_bytes = (
            options.limits.max_bytes if options is not None else DEFAULT_MAX_BYTES
        )
        self._max_lines = (
            options.limits.max_lines if options is not None else DEFAULT_MAX_LINES
        )
        self._retain = options.limits.retain if options is not None else "tail"
        self._context = context
        self._on_update = handlers.on_update
        if self._max_bytes <= 0:
            raise TypeError("Output maxBytes must be a positive finite number")
        if self._max_lines <= 0:
            raise TypeError("Output maxLines must be a positive integer")

        self._decoder = _IncrementalUtf8Decoder()
        self._buffer = ""
        self._buffer_bytes = 0
        self._total_bytes = 0
        self._newlines = 0
        self._ends_with_newline = True
        self._current_line_bytes = 0
        self._spill_path: str | None = None
        self._disposed = False
        self._publisher = AdaptivePublisher[ShellOutputView, ShellOutputUpdate](
            AdaptivePublisherOptions(
                snapshot=self.snapshot,
                update=_update_from,
                measure=lambda update: utf8_byte_length(_encode_update(update)),
                publish=self._publish,
                on_error=handlers.on_error or _ignore_error,
                min_interval_ms=OUTPUT_MIN_EMIT_INTERVAL_MS,
                target_bytes_per_second=OUTPUT_TARGET_BYTES_PER_SECOND,
            )
        )

    @property
    def truncated(self) -> bool:
        return self._total_bytes > self._max_bytes or self._total_lines() > self._max_lines

    def push(self, chunk: str | bytes) -> None:
        if self._disposed:
            return
        if isinstance(chunk, str):
            self._append_text(self._decoder.decode())
            self._append_text(chunk)
            return
        self._append_text(self._decoder.decode(chunk))

    def finish(self) -> None:
        if self._disposed:
            return
        self._append_text(self._decoder.decode())

    def set_spill_path(self, path: str) -> None:
        if self._disposed or self._spill_path == path:
            return
        self._spill_path = path
        self._publisher.mark_dirty()
        self.flush()

    def snapshot(self) -> ShellOutputView:
        if self._retain == "head":
            retained = truncate_head(
                self._buffer, max_bytes=self._max_bytes, max_lines=self._max_lines
            )
        else:
            retained = truncate_tail(
                self._buffer, max_bytes=self._max_bytes, max_lines=self._max_lines
            )
        total_lines = self._total_lines()
        truncated = self.truncated
        truncation = ShellOutputTruncation(
            truncated=truncated,
            truncated_by=(
                ("lines" if total_lines > self._max_lines else "bytes")
                if truncated
                else None
            ),
            total_lines=total_lines,
            total_bytes=self._total_bytes,
            output_lines=retained.output_lines,
            output_bytes=retained.output_bytes,
            last_line_partial=retained.last_line_partial,
            first_line_exceeds_limit=retained.first_line_exceeds_limit,
            max_lines=retained.max_lines,
            max_bytes=retained.max_bytes,
        )
        return ShellOutputView(
            text=sanitize_shell_output(retained.content),
            truncation=truncation,
            spill_path=self._spill_path,
            last_line_bytes=(
                self._current_line_bytes if retained.last_line_partial else None
            ),
        )

    def flush(self) -> None:
        if self._disposed:
            return
        self._publisher.flush(force=True)

    def dispose(self) -> None:
        self._publisher.dispose()
        self._disposed = True

    def _publish(self, update: ShellOutputUpdate) -> None:
        if self._on_update is not None:
            self._on_update(update, self._context)

    def _append_text(self, text: str) -> None:
        if text == "":
            return
        text_bytes = utf8_byte_length(text)
        self._total_bytes += text_bytes
        self._newlines += text.count("\n")
        self._ends_with_newline = text.endswith("\n")
        last_newline = text.rfind("\n")
        self._current_line_bytes = (
            self._current_line_bytes + text_bytes
            if last_newline == -1
            else utf8_byte_length(text[last_newline + 1 :])
        )
        self._buffer += text
        self._buffer_bytes += text_bytes

        guard = self._max_bytes * 2
        if self._buffer_bytes > guard * 2:
            self._buffer = (
                _trim_to_last_utf8_bytes(self._buffer, guard)
                if self._retain == "tail"
                else _trim_to_first_utf8_bytes(self._buffer, guard)
            )
            self._buffer_bytes = utf8_byte_length(self._buffer)
        self._publisher.mark_dirty()

    def _total_lines(self) -> int:
        return self._newlines + (
            0 if self._ends_with_newline or self._total_bytes == 0 else 1
        )


def apply_shell_output_update(
    current: ShellOutputView | None,
    update: ShellOutputUpdate,
) -> ShellOutputView:
    """Fold one incremental update into a complete bounded view."""
    if isinstance(update, ShellOutputReplace):
        return update.output
    if isinstance(update, ShellOutputAppend):
        return ShellOutputView(
            text=f"{current.text if current is not None else ''}{update.text}",
            truncation=update.metadata.truncation,
            spill_path=update.metadata.spill_path,
            last_line_bytes=update.metadata.last_line_bytes,
        )
    if isinstance(update, ShellOutputSlide):
        retained = current.text[update.drop :] if current is not None else ""
        return ShellOutputView(
            text=f"{retained}{update.text}",
            truncation=update.metadata.truncation,
            spill_path=update.metadata.spill_path,
            last_line_bytes=update.metadata.last_line_bytes,
        )
    return ShellOutputView(
        text=current.text if current is not None else "",
        truncation=update.metadata.truncation,
        spill_path=update.metadata.spill_path,
        last_line_bytes=update.metadata.last_line_bytes,
    )


def sanitize_shell_output(text: str) -> str:
    """Remove control characters that cannot be represented in text content."""
    return _INVALID_SHELL_OUTPUT.sub("", text)


def _update_from(
    previous: ShellOutputView | None, current: ShellOutputView
) -> ShellOutputUpdate | None:
    if previous is None:
        return ShellOutputReplace(output=current)
    metadata = ShellOutputMetadata(
        truncation=current.truncation,
        spill_path=current.spill_path,
        last_line_bytes=current.last_line_bytes,
    )
    if current.text == previous.text:
        return ShellOutputMetadataUpdate(metadata=metadata)
    if current.text.startswith(previous.text):
        return ShellOutputAppend(
            text=current.text[len(previous.text) :], metadata=metadata
        )
    shared = _suffix_prefix_overlap(
        previous.text,
        current.text,
        min(
            len(previous.text),
            len(current.text),
            current.truncation.max_bytes * 2,
        ),
    )
    if shared > 0:
        return ShellOutputSlide(
            drop=len(previous.text) - shared,
            text=current.text[shared:],
            metadata=metadata,
        )
    return ShellOutputReplace(output=current)


def _suffix_prefix_overlap(before: str, after: str, scan: int) -> int:
    if before == "" or after == "" or scan == 0:
        return 0
    tail = before[-scan:] if len(before) > scan else before
    for probe_length in (min(64, len(after)), 1):
        probe = after[:probe_length]
        candidates = 0
        index = tail.find(probe)
        while index != -1:
            candidates += 1
            if candidates > 8:
                break
            overlap_length = len(tail) - index
            if overlap_length <= len(after) and tail[index:] == after[:overlap_length]:
                return overlap_length
            index = tail.find(probe, index + 1)
        if probe_length == 1:
            break
    return 0


def _encode_update(update: ShellOutputUpdate) -> str:
    if isinstance(update, ShellOutputReplace):
        return update.output.text
    if isinstance(update, ShellOutputMetadataUpdate):
        return ""
    return update.text


class _IncrementalUtf8Decoder:
    def __init__(self) -> None:
        self._pending = b""

    def decode(self, chunk: bytes | None = None) -> str:
        data = self._pending if chunk is None else self._pending + chunk
        self._pending = b""
        if data == b"":
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data":
                self._pending = data[error.start :]
                return data[: error.start].decode("utf-8", errors="replace")
            return data.decode("utf-8", errors="replace")


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="surrogatepass")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8", errors="replace")


def _trim_to_first_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="surrogatepass")
    if len(encoded) <= max_bytes:
        return text
    end = max_bytes
    while end > 0 and (encoded[end] & 0xC0) == 0x80:
        end -= 1
    return encoded[:end].decode("utf-8", errors="replace")


def _ignore_error(error: BaseException) -> None:
    del error
