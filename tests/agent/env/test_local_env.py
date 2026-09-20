from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

from omh.agent.context import BACKGROUND_CONTEXT, with_cancel
from omh.agent.env import LocalExecutionEnv
from omh.agent.execution_env import (
    CreateDirOptions,
    CreateTempFileOptions,
    ReadTextLinesOptions,
    RemoveOptions,
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputUpdate,
    ShellOutputView,
)
from omh.agent.utils.output_capture import apply_shell_output_update

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="LocalExecutionEnv supports macOS and Linux only"
)


def _env(tmp_path: Path) -> LocalExecutionEnv:
    return LocalExecutionEnv(str(tmp_path))


async def test_absolute_path_resolves_relative_and_home(tmp_path: Path) -> None:
    env = _env(tmp_path)
    resolved = await env.absolute_path("sub/file.txt", BACKGROUND_CONTEXT)
    assert resolved.ok
    assert resolved.value == str(tmp_path / "sub" / "file.txt")
    home = await env.absolute_path("~/x", BACKGROUND_CONTEXT)
    assert home.ok
    assert home.value == os.path.expanduser("~/x")


async def test_join_path_uses_platform_separator(tmp_path: Path) -> None:
    env = _env(tmp_path)
    joined = await env.join_path(["a", "b"], BACKGROUND_CONTEXT)
    assert joined.ok
    assert joined.value == os.path.join("a", "b")


async def test_write_read_append_and_rename(tmp_path: Path) -> None:
    env = _env(tmp_path)
    written = await env.write_file("nested/a.txt", "one", BACKGROUND_CONTEXT)
    assert written.ok
    assert (tmp_path / "nested" / "a.txt").read_text() == "one"

    appended = await env.append_file("nested/a.txt", "\ntwo", BACKGROUND_CONTEXT)
    assert appended.ok
    read = await env.read_text_file("nested/a.txt", BACKGROUND_CONTEXT)
    assert read.ok
    assert read.value == "one\ntwo"

    renamed = await env.rename_file("nested/a.txt", "nested/b.txt", BACKGROUND_CONTEXT)
    assert renamed.ok
    assert (tmp_path / "nested" / "b.txt").exists()

    binary = await env.read_binary_file("nested/b.txt", BACKGROUND_CONTEXT)
    assert binary.ok
    assert binary.value == b"one\ntwo"


async def test_file_info_list_dir_and_canonical_path(tmp_path: Path) -> None:
    env = _env(tmp_path)
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "file.txt").write_text("x")

    info = await env.file_info("dir/file.txt", BACKGROUND_CONTEXT)
    assert info.ok
    assert info.value.kind == "file"
    assert info.value.name == "file.txt"
    assert info.value.size == 1

    listing = await env.list_dir("dir", BACKGROUND_CONTEXT)
    assert listing.ok
    assert [entry.name for entry in listing.value] == ["file.txt"]

    canonical = await env.canonical_path("dir/../dir/file.txt", BACKGROUND_CONTEXT)
    assert canonical.ok
    assert canonical.value == str((tmp_path / "dir" / "file.txt").resolve())

    missing = await env.canonical_path("nope", BACKGROUND_CONTEXT)
    assert not missing.ok
    assert missing.error.code == "not_found"


async def test_exists_distinguishes_missing_from_errors(tmp_path: Path) -> None:
    env = _env(tmp_path)
    (tmp_path / "present").write_text("x")
    assert (await env.exists("present", BACKGROUND_CONTEXT)).value is True
    assert (await env.exists("absent", BACKGROUND_CONTEXT)).value is False


async def test_read_text_lines_reports_unterminated_final_line(tmp_path: Path) -> None:
    env = _env(tmp_path)
    (tmp_path / "lines.txt").write_text("a\nb\nc")
    lines = await env.read_text_lines("lines.txt", ReadTextLinesOptions(), BACKGROUND_CONTEXT)
    assert lines.ok
    assert lines.value == ["a", "b", "c"]

    limited = await env.read_text_lines(
        "lines.txt", ReadTextLinesOptions(max_lines=2), BACKGROUND_CONTEXT
    )
    assert limited.ok
    assert limited.value == ["a", "b"]

    empty = await env.read_text_lines(
        "lines.txt", ReadTextLinesOptions(max_lines=0), BACKGROUND_CONTEXT
    )
    assert empty.ok
    assert empty.value == []


async def test_create_dir_and_remove(tmp_path: Path) -> None:
    env = _env(tmp_path)
    created = await env.create_dir("a/b/c", None, BACKGROUND_CONTEXT)
    assert created.ok
    assert (tmp_path / "a" / "b" / "c").is_dir()

    non_recursive = await env.create_dir(
        "d/e", CreateDirOptions(recursive=False), BACKGROUND_CONTEXT
    )
    assert not non_recursive.ok
    assert non_recursive.error.code == "not_found"

    (tmp_path / "a" / "b" / "c" / "file").write_text("x")
    refused = await env.remove("a", None, BACKGROUND_CONTEXT)
    assert not refused.ok
    assert refused.error.code == "is_directory"

    removed = await env.remove("a", RemoveOptions(recursive=True), BACKGROUND_CONTEXT)
    assert removed.ok
    assert not (tmp_path / "a").exists()

    forced = await env.remove("absent", RemoveOptions(force=True), BACKGROUND_CONTEXT)
    assert forced.ok


async def test_temp_dir_and_file_are_created(tmp_path: Path) -> None:
    env = _env(tmp_path)
    temp_dir = await env.create_temp_dir("omh-test-", BACKGROUND_CONTEXT)
    assert temp_dir.ok
    assert os.path.isdir(temp_dir.value)
    assert os.path.basename(temp_dir.value).startswith("omh-test-")

    temp_file = await env.create_temp_file(
        CreateTempFileOptions(prefix="out-", suffix=".log"), BACKGROUND_CONTEXT
    )
    assert temp_file.ok
    assert os.path.isfile(temp_file.value)
    assert os.path.basename(temp_file.value).startswith("out-")
    assert temp_file.value.endswith(".log")


async def test_permission_denied_maps_to_file_error(tmp_path: Path) -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("permission checks do not apply to root")
    env = _env(tmp_path)
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    os.chmod(target, 0)
    try:
        result = await env.read_text_file("secret.txt", BACKGROUND_CONTEXT)
        assert not result.ok
        assert result.error.code == "permission_denied"
    finally:
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)


class _Capture:
    def __init__(self) -> None:
        self.view: ShellOutputView | None = None

    def __call__(self, update: ShellOutputUpdate, context: object) -> None:
        del context
        self.view = apply_shell_output_update(self.view, update)

    @property
    def text(self) -> str:
        return self.view.text if self.view is not None else ""


async def test_exec_runs_command_and_combines_output(tmp_path: Path) -> None:
    env = _env(tmp_path)
    capture = _Capture()
    result = await env.exec(
        "echo out; echo err >&2; exit 3",
        ShellExecOptions(on_update=capture),
        BACKGROUND_CONTEXT,
    )
    assert result.ok
    assert result.value.exit_code == 3
    assert set(capture.text.split()) >= {"out", "err"}


async def test_exec_uses_cwd_and_explicit_environment(tmp_path: Path) -> None:
    env = _env(tmp_path)
    (tmp_path / "sub").mkdir()
    capture = _Capture()
    result = await env.exec(
        "pwd; echo $OMH_TEST",
        ShellExecOptions(
            cwd="sub",
            env={"OMH_TEST": "value"},
            inherit_env=False,
            on_update=capture,
        ),
        BACKGROUND_CONTEXT,
    )
    assert result.ok
    assert "sub" in capture.text
    assert "value" in capture.text


async def test_exec_timeout_reports_timeout_error(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.exec(
        "sleep 30", ShellExecOptions(timeout=0.2), BACKGROUND_CONTEXT
    )
    assert not result.ok
    assert result.error.code == "timeout"


async def test_exec_rejects_invalid_timeout(tmp_path: Path) -> None:
    env = _env(tmp_path)
    for timeout in (0, -1, float("inf")):
        result = await env.exec(
            "true", ShellExecOptions(timeout=timeout), BACKGROUND_CONTEXT
        )
        assert not result.ok
        assert result.error.code == "timeout"


async def test_exec_spills_full_output_and_reports_truncation(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.exec(
        "for i in $(seq 1 200); do echo line$i; done",
        ShellExecOptions(
            capture=ShellOutputCaptureOptions(
                limits=ShellOutputLimits(max_bytes=1024, max_lines=10), spill=True
            )
        ),
        BACKGROUND_CONTEXT,
    )
    assert result.ok
    assert result.value.truncation.truncated is True
    assert result.value.spill_path is not None
    spilled = Path(result.value.spill_path).read_text()
    assert "line1" in spilled
    assert "line200" in spilled


async def test_context_cancellation_kills_the_process(tmp_path: Path) -> None:
    env = _env(tmp_path)
    token = f"omh-cancel-{uuid.uuid4().hex}"
    scope = with_cancel(BACKGROUND_CONTEXT)
    task = asyncio.create_task(
        env.exec(
            f"echo started; sleep 30 # {token}",
            ShellExecOptions(),
            scope.context,
        )
    )
    await asyncio.sleep(0.3)
    scope.cancel()
    result = await task
    assert not result.ok
    assert result.error.code == "aborted"
    await asyncio.sleep(0.2)
    assert _processes_matching(token) == []


async def test_cleanup_kills_active_children(tmp_path: Path) -> None:
    env = _env(tmp_path)
    token = f"omh-cleanup-{uuid.uuid4().hex}"
    task = asyncio.create_task(
        env.exec(
            f"echo started; sleep 30 # {token}",
            ShellExecOptions(),
            BACKGROUND_CONTEXT,
        )
    )
    await asyncio.sleep(0.3)
    await env.cleanup(BACKGROUND_CONTEXT)
    result = await task
    assert result.ok
    assert result.value.exit_code != 0
    await asyncio.sleep(0.2)
    assert _processes_matching(token) == []


def _processes_matching(token: str) -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-f", token], capture_output=True, text=True, check=False
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]
