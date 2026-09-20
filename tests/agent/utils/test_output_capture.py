from __future__ import annotations

from omh.agent.context import BACKGROUND_CONTEXT
from omh.agent.execution_env import (
    ShellOutputAppend,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputMetadata,
    ShellOutputMetadataUpdate,
    ShellOutputReplace,
    ShellOutputSlide,
    ShellOutputView,
)
from omh.agent.utils.output_capture import (
    OutputCapture,
    OutputCaptureHandlers,
    apply_shell_output_update,
    sanitize_shell_output,
)
from omh.agent.utils.truncate import truncate_tail


def _metadata(text: str = "") -> ShellOutputMetadata:
    return ShellOutputMetadata(truncation=truncate_tail(text).without_content())


async def test_capture_collects_updates_and_final_snapshot() -> None:
    updates: list[object] = []
    capture = OutputCapture(
        None,
        BACKGROUND_CONTEXT,
        OutputCaptureHandlers(on_update=lambda update, _context: updates.append(update)),
    )
    capture.push(b"hello ")
    capture.push("world\n")
    capture.finish()
    capture.flush()
    snapshot = capture.snapshot()
    assert snapshot.text == "hello world\n"
    assert updates
    capture.dispose()


async def test_capture_decodes_split_multibyte_sequences() -> None:
    capture = OutputCapture(None, BACKGROUND_CONTEXT, OutputCaptureHandlers())
    encoded = "你好".encode("utf-8")
    capture.push(encoded[:1])
    capture.push(encoded[1:4])
    capture.push(encoded[4:])
    capture.finish()
    assert capture.snapshot().text == "你好"
    capture.dispose()


async def test_capture_truncates_and_reports_metadata() -> None:
    capture = OutputCapture(
        ShellOutputCaptureOptions(
            limits=ShellOutputLimits(max_bytes=1024, max_lines=2), spill=True
        ),
        BACKGROUND_CONTEXT,
        OutputCaptureHandlers(),
    )
    capture.push("one\ntwo\nthree\nfour\n")
    capture.finish()
    snapshot = capture.snapshot()
    assert snapshot.text == "three\nfour"
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "lines"
    assert snapshot.truncation.total_lines == 4
    capture.dispose()


async def test_capture_sanitizes_control_characters() -> None:
    capture = OutputCapture(None, BACKGROUND_CONTEXT, OutputCaptureHandlers())
    capture.push("a\x00b\x1fc")
    capture.finish()
    assert capture.snapshot().text == "abc"
    capture.dispose()


async def test_capture_spill_path_is_reported() -> None:
    capture = OutputCapture(None, BACKGROUND_CONTEXT, OutputCaptureHandlers())
    capture.set_spill_path("/tmp/output.log")
    assert capture.snapshot().spill_path == "/tmp/output.log"
    capture.dispose()


async def test_apply_shell_output_update_folds_each_variant() -> None:
    metadata = _metadata("abc")
    assert apply_shell_output_update(None, ShellOutputReplace(
        output=ShellOutputView(text="abc", truncation=metadata.truncation)
    )).text == "abc"
    appended = apply_shell_output_update(
        ShellOutputView(text="abc", truncation=metadata.truncation),
        ShellOutputAppend(text="def", metadata=metadata),
    )
    assert appended.text == "abcdef"
    slid = apply_shell_output_update(
        ShellOutputView(text="abcdef", truncation=metadata.truncation),
        ShellOutputSlide(drop=3, text="xyz", metadata=metadata),
    )
    assert slid.text == "defxyz"
    unchanged = apply_shell_output_update(
        ShellOutputView(text="abcdef", truncation=metadata.truncation),
        ShellOutputMetadataUpdate(metadata=metadata),
    )
    assert unchanged.text == "abcdef"


async def test_sanitize_shell_output_removes_invalid_controls() -> None:
    assert sanitize_shell_output("a\x0bb\ufff9c") == "abc"
