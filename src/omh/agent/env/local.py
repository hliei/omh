"""Local macOS/Linux execution environment for the built-in tools.

Corresponds to ``packages/agent/src/harness/env/nodejs.ts``. The Node baseline
also handles Windows and WSL bash discovery; this first version supports the
declared macOS/Linux platforms only.

Differences from the baseline are recorded in ``docs/agent-session-upstream.md``:
filesystem calls run inline on the event loop instead of through asynchronous
Node APIs, and shell output spilling writes synchronously once truncation is
reached.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import shutil
import stat as stat_module
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from typing import BinaryIO

from omh.agent.context import Context, wait_for_cancellation
from omh.agent.execution_env import (
    CreateDirOptions,
    CreateTempFileOptions,
    ExecutionError,
    FileError,
    FileErrorCode,
    FileInfo,
    FileKind,
    ReadTextLinesOptions,
    RemoveOptions,
    ShellExecOptions,
    ShellExecResult,
    ShellOutputCaptureOptions,
    TextLine,
    TextLineReader,
)
from omh.agent.result import Result, err, ok
from omh.agent.utils.output_capture import OutputCapture, OutputCaptureHandlers

_MAX_TIMEOUT_MS = 2_147_483_647
_MAX_TIMEOUT_SECONDS = _MAX_TIMEOUT_MS / 1000
_READ_CHUNK_BYTES = 64 * 1024
_KILL_GRACE_SECONDS = 5.0

_ERROR_CODES: dict[int, FileErrorCode] = {
    errno.ENOENT: "not_found",
    errno.EACCES: "permission_denied",
    errno.EPERM: "permission_denied",
    errno.ENOTDIR: "not_directory",
    errno.EISDIR: "is_directory",
    errno.EINVAL: "invalid",
}


def _aborted(context: Context) -> bool:
    return context.cancelled


def _resolve_path(cwd: str, path: str) -> str:
    normalized = path
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/"):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    elif normalized.startswith("file://"):
        parsed = urllib.parse.urlparse(normalized)
        if parsed.scheme == "file":
            normalized = urllib.request.url2pathname(parsed.path)
    if not os.path.isabs(normalized):
        normalized = os.path.join(cwd, normalized)
    return os.path.abspath(normalized)


def _to_file_error(error: BaseException, fallback_path: str | None = None) -> FileError:
    if isinstance(error, FileError):
        return error
    if isinstance(error, asyncio.CancelledError):
        return FileError("aborted", "aborted", fallback_path)
    if isinstance(error, OSError):
        code = _ERROR_CODES.get(error.errno or 0, "unknown")
        path = error.filename if isinstance(error.filename, str) else fallback_path
        return FileError(code, str(error), path, error)
    return FileError("unknown", str(error), fallback_path, error)


def _file_kind(mode: int) -> FileKind | None:
    if stat_module.S_ISREG(mode):
        return "file"
    if stat_module.S_ISDIR(mode):
        return "directory"
    if stat_module.S_ISLNK(mode):
        return "symlink"
    return None


class _LocalTextLineReader:
    """Strict LF reader that reports whether the final line was terminated."""

    def __init__(self, file: BinaryIO, path: str) -> None:
        self._file = file
        self._path = path
        self._buffer = ""
        self._pending = b""
        self._ended = False
        self._closed = False

    async def read_line(self, context: Context) -> Result[TextLine | None, FileError]:
        if _aborted(context):
            return err(FileError("aborted", "aborted", self._path))
        if self._closed:
            return err(FileError("invalid", "Text line reader is closed", self._path))
        try:
            while True:
                newline = self._buffer.find("\n")
                if newline != -1:
                    text = self._buffer[:newline]
                    self._buffer = self._buffer[newline + 1 :]
                    return ok(TextLine(text, True))
                if self._ended:
                    if self._buffer == "":
                        return ok(None)
                    text = self._buffer
                    self._buffer = ""
                    return ok(TextLine(text, False))
                data = self._file.read(_READ_CHUNK_BYTES)
                self._buffer += self._decode(data)
                if not data:
                    self._ended = True
        except OSError as error:
            return err(_to_file_error(error, self._path))

    async def close(self, context: Context) -> None:
        del context
        if self._closed:
            return
        self._closed = True
        self._buffer = ""
        with contextlib.suppress(OSError):
            self._file.close()

    def _decode(self, data: bytes) -> str:
        if not data:
            return ""
        payload = self._pending + data
        self._pending = b""
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data":
                self._pending = payload[error.start :]
                return payload[: error.start].decode("utf-8", errors="replace")
            return payload.decode("utf-8", errors="replace")


class LocalExecutionEnv:
    """Filesystem and process execution environment for macOS and Linux."""

    def __init__(
        self,
        cwd: str,
        *,
        shell_path: str | None = None,
        shell_env: Mapping[str, str] | None = None,
    ) -> None:
        self.cwd = cwd
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._active_pids: set[int] = set()

    async def absolute_path(self, path: str, context: Context) -> Result[str, FileError]:
        del context
        return ok(_resolve_path(self.cwd, path))

    async def join_path(
        self, parts: list[str], context: Context
    ) -> Result[str, FileError]:
        del context
        return ok(os.path.join(*parts))

    async def exec(
        self, command: str, options: ShellExecOptions | None, context: Context
    ) -> Result[ShellExecResult, ExecutionError]:
        if _aborted(context):
            return err(ExecutionError("aborted", "aborted"))
        timeout = None if options is None else options.timeout
        if timeout is not None:
            timeout_error = _validate_timeout(timeout)
            if timeout_error is not None:
                return err(timeout_error)
        cwd = (
            _resolve_path(self.cwd, options.cwd)
            if options is not None and options.cwd is not None
            else self.cwd
        )
        shell = self._shell_config()
        if shell is None:
            return err(ExecutionError("shell_unavailable", "No bash or sh shell found"))
        if not os.path.isdir(cwd):
            return err(
                ExecutionError(
                    "spawn_error",
                    f"Working directory does not exist: {cwd}\nCannot execute bash commands.",
                )
            )

        capture_options = None if options is None else options.capture
        callback_errors: list[BaseException] = []

        def on_error(error: BaseException) -> None:
            if not callback_errors:
                callback_errors.append(error)

        try:
            capture = OutputCapture(
                capture_options,
                context,
                OutputCaptureHandlers(
                    on_update=None if options is None else options.on_update,
                    on_error=on_error,
                ),
            )
        except TypeError as error:
            return err(ExecutionError("unknown", str(error), error))

        environment = _shell_environment(
            self._shell_env,
            None if options is None else options.env,
            True if options is None else options.inherit_env,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *shell,
                "-c",
                command,
                cwd=cwd,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            capture.dispose()
            return err(ExecutionError("spawn_error", str(error), error))

        assert process.stdout is not None
        assert process.stderr is not None
        self._active_pids.add(process.pid)
        spill = _SpillBuffer(self, capture, context) if _spill_enabled(capture_options) else None
        timed_out = False
        pump_tasks: list[asyncio.Task[None]] = []

        async def pump(stream: asyncio.StreamReader) -> None:
            while True:
                chunk = await stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    return
                was_truncated = capture.truncated
                try:
                    capture.push(chunk)
                except BaseException as error:  # noqa: BLE001 - delivered as callback_error
                    on_error(error)
                    return
                if spill is not None:
                    await spill.write(chunk, was_truncated)

        try:
            pump_tasks = [
                asyncio.create_task(pump(process.stdout)),
                asyncio.create_task(pump(process.stderr)),
            ]
            wait_task = asyncio.create_task(process.wait())
            cancellation_task = asyncio.create_task(wait_for_cancellation(context))
            try:
                done, _ = await asyncio.wait(
                    {wait_task, cancellation_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout,
                )
                if wait_task not in done:
                    if cancellation_task not in done:
                        timed_out = True
                    await self._kill_and_wait(process)
            except asyncio.CancelledError:
                await self._kill_and_wait(process)
                raise
            finally:
                for task in (wait_task, cancellation_task):
                    if not task.done():
                        task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(*pump_tasks, return_exceptions=True)
        finally:
            for task in pump_tasks:
                if not task.done():
                    task.cancel()
            if process.returncode is None:
                self._kill_process_tree(process.pid)
            self._active_pids.discard(process.pid)
            if spill is not None:
                spill.close()
            try:
                capture.finish()
                capture.flush()
            except BaseException as error:  # noqa: BLE001 - delivered as callback_error
                if not callback_errors:
                    callback_errors.append(error)
            capture.dispose()

        if callback_errors:
            capture.dispose()
            first = callback_errors[0]
            return err(ExecutionError("callback_error", str(first), first))
        if timed_out:
            capture.dispose()
            return err(ExecutionError("timeout", f"timeout:{timeout}"))
        if _aborted(context):
            capture.dispose()
            return err(ExecutionError("aborted", "aborted"))
        if spill is not None and spill.error is not None:
            capture.dispose()
            return err(ExecutionError("unknown", str(spill.error), spill.error))

        output = capture.snapshot()
        capture.dispose()
        returncode = process.returncode
        exit_code = (
            returncode
            if returncode is not None and returncode >= 0
            else 128 + (-returncode if returncode is not None else 0)
        )
        return ok(
            ShellExecResult(
                exit_code=exit_code,
                truncation=output.truncation,
                spill_path=output.spill_path,
                last_line_bytes=output.last_line_bytes,
            )
        )

    async def _kill_and_wait(self, process: asyncio.subprocess.Process) -> None:
        self._kill_process_tree(process.pid)
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(process.wait(), _KILL_GRACE_SECONDS)

    async def open_text_line_reader(
        self, path: str, context: Context
    ) -> Result[TextLineReader, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        try:
            file = open(resolved, "rb")
        except OSError as error:
            return err(_to_file_error(error, resolved))
        if _aborted(context):
            file.close()
            return err(FileError("aborted", "aborted", resolved))
        return ok(_LocalTextLineReader(file, resolved))

    async def read_text_file(self, path: str, context: Context) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        try:
            with open(resolved, "rb") as file:
                data = file.read()
        except OSError as error:
            return err(_to_file_error(error, resolved))
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        return ok(data.decode("utf-8", errors="replace"))

    async def read_text_lines(
        self,
        path: str,
        options: ReadTextLinesOptions | None,
        context: Context,
    ) -> Result[list[str], FileError]:
        max_lines = None if options is None else options.max_lines
        if max_lines is not None and max_lines <= 0:
            return ok([])
        opened = await self.open_text_line_reader(path, context)
        if not opened.ok:
            return opened
        reader = opened.value
        lines: list[str] = []
        try:
            while max_lines is None or len(lines) < max_lines:
                line = await reader.read_line(context)
                if not line.ok:
                    return line
                if line.value is None:
                    break
                lines.append(line.value.text)
            return ok(lines)
        finally:
            await reader.close(context)

    async def read_binary_file(
        self, path: str, context: Context
    ) -> Result[bytes, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        try:
            with open(resolved, "rb") as file:
                data = file.read()
        except OSError as error:
            return err(_to_file_error(error, resolved))
        return ok(data)

    async def write_file(
        self, path: str, content: str | bytes, context: Context
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        try:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            if _aborted(context):
                return err(FileError("aborted", "aborted", resolved))
            payload = content.encode("utf-8") if isinstance(content, str) else content
            with open(resolved, "wb") as file:
                file.write(payload)
        except OSError as error:
            return err(_to_file_error(error, resolved))
        return ok(None)

    async def append_file(
        self, path: str, content: str | bytes, context: Context
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        try:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            payload = content.encode("utf-8") if isinstance(content, str) else content
            with open(resolved, "ab") as file:
                file.write(payload)
        except OSError as error:
            return err(_to_file_error(error, resolved))
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        return ok(None)

    async def rename_file(
        self, source_path: str, destination_path: str, context: Context
    ) -> Result[None, FileError]:
        source = _resolve_path(self.cwd, source_path)
        destination = _resolve_path(self.cwd, destination_path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", destination))
        try:
            os.replace(source, destination)
        except OSError as error:
            return err(_to_file_error(error, source))
        return ok(None)

    async def file_info(self, path: str, context: Context) -> Result[FileInfo, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        try:
            return _file_info_from_stat(resolved, os.lstat(resolved))
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def list_dir(
        self, path: str, context: Context
    ) -> Result[list[FileInfo], FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        infos: list[FileInfo] = []
        try:
            with os.scandir(resolved) as entries:
                for entry in entries:
                    if _aborted(context):
                        return err(FileError("aborted", "aborted", resolved))
                    entry_path = os.path.join(resolved, entry.name)
                    try:
                        info = _file_info_from_stat(
                            entry_path, entry.stat(follow_symlinks=False)
                        )
                    except OSError as error:
                        return err(_to_file_error(error, entry_path))
                    if info.ok:
                        infos.append(info.value)
        except OSError as error:
            return err(_to_file_error(error, resolved))
        return ok(infos)

    async def canonical_path(
        self, path: str, context: Context
    ) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        if not os.path.lexists(resolved):
            return err(FileError("not_found", "not found", resolved))
        return ok(os.path.realpath(resolved))

    async def exists(self, path: str, context: Context) -> Result[bool, FileError]:
        result = await self.file_info(path, context)
        if result.ok:
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    async def create_dir(
        self, path: str, options: CreateDirOptions | None, context: Context
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        recursive = True if options is None else options.recursive
        try:
            if recursive:
                os.makedirs(resolved, exist_ok=True)
            else:
                os.mkdir(resolved)
        except OSError as error:
            return err(_to_file_error(error, resolved))
        return ok(None)

    async def remove(
        self, path: str, options: RemoveOptions | None, context: Context
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if _aborted(context):
            return err(FileError("aborted", "aborted", resolved))
        recursive = False if options is None else options.recursive
        force = False if options is None else options.force
        try:
            if os.path.isdir(resolved) and not os.path.islink(resolved):
                if not recursive:
                    return err(FileError("is_directory", "is a directory", resolved))
                shutil.rmtree(resolved, ignore_errors=force)
            else:
                try:
                    os.remove(resolved)
                except FileNotFoundError:
                    if not force:
                        raise
        except OSError as error:
            return err(_to_file_error(error, resolved))
        return ok(None)

    async def create_temp_dir(
        self, prefix: str | None, context: Context
    ) -> Result[str, FileError]:
        if _aborted(context):
            return err(FileError("aborted", "aborted"))
        try:
            return ok(tempfile.mkdtemp(prefix=prefix or "tmp-"))
        except OSError as error:
            return err(_to_file_error(error))

    async def create_temp_file(
        self, options: CreateTempFileOptions | None, context: Context
    ) -> Result[str, FileError]:
        created = await self.create_temp_dir("tmp-", context)
        if not created.ok:
            return created
        prefix = "" if options is None else options.prefix
        suffix = "" if options is None else options.suffix
        path = os.path.join(created.value, f"{prefix}{uuid.uuid4()}{suffix}")
        try:
            with open(path, "wb"):
                pass
        except OSError as error:
            return err(_to_file_error(error, path))
        return ok(path)

    async def cleanup(self, context: Context) -> None:
        del context
        for pid in list(self._active_pids):
            self._kill_process_tree(pid)
        self._active_pids.clear()

    def _shell_config(self) -> tuple[str, ...] | None:
        if self._shell_path is not None:
            return (self._shell_path,) if os.path.exists(self._shell_path) else None
        if os.path.exists("/bin/bash"):
            return ("/bin/bash",)
        bash_on_path = shutil.which("bash")
        if bash_on_path is not None:
            return (bash_on_path,)
        if shutil.which("sh") is not None:
            return ("sh",)
        return None

    def _kill_process_tree(self, pid: int) -> None:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(pid), 9)
            return
        with contextlib.suppress(OSError):
            os.kill(pid, 9)


class _SpillBuffer:
    """Preserve complete shell output in a temporary file after truncation."""

    def __init__(
        self, env: LocalExecutionEnv, capture: OutputCapture, context: Context
    ) -> None:
        self._env = env
        self._capture = capture
        self._context = context
        self._file: BinaryIO | None = None
        self._buffered: list[bytes] = []
        self.error: BaseException | None = None

    async def write(self, chunk: bytes, was_truncated: bool) -> None:
        if self._file is None and (self._capture.truncated or was_truncated):
            await self._start()
        if self._file is None:
            self._buffered.append(chunk)
            return
        self._write(chunk)

    async def _start(self) -> None:
        created = await self._env.create_temp_file(
            CreateTempFileOptions(prefix="omh-output-", suffix=".log"), self._context
        )
        if not created.ok:
            self.error = created.error
            return
        try:
            file = open(created.value, "ab")
        except OSError as error:
            self.error = error
            return
        self._file = file
        self._capture.set_spill_path(created.value)
        for buffered in self._buffered:
            self._write(buffered)
        self._buffered.clear()

    def _write(self, chunk: bytes) -> None:
        if self._file is None or self.error is not None:
            return
        try:
            self._file.write(chunk)
        except OSError as error:
            self.error = error

    def close(self) -> None:
        if self._file is None:
            return
        with contextlib.suppress(OSError):
            self._file.close()


def _file_info_from_stat(path: str, stat: os.stat_result) -> Result[FileInfo, FileError]:
    kind = _file_kind(stat.st_mode)
    if kind is None:
        return err(FileError("invalid", "Unsupported file type", path))
    return ok(
        FileInfo(
            name=os.path.basename(path),
            path=path,
            kind=kind,
            size=stat.st_size,
            mtime_ms=stat.st_mtime * 1000,
        )
    )


def _valid_timeout(timeout: float | None) -> bool:
    if timeout is None:
        return True
    if not isinstance(timeout, int | float) or isinstance(timeout, bool):
        return False
    value = float(timeout)
    if value != value or value in (float("inf"), float("-inf")):
        return False
    return value > 0 and value * 1000 <= _MAX_TIMEOUT_MS


def _validate_timeout(timeout: float) -> ExecutionError | None:
    if _valid_timeout(timeout):
        return None
    if isinstance(timeout, int | float) and timeout * 1000 > _MAX_TIMEOUT_MS:
        return ExecutionError(
            "timeout", f"Invalid timeout: maximum is {_MAX_TIMEOUT_SECONDS} seconds"
        )
    return ExecutionError(
        "timeout", "Invalid timeout: must be a finite number of seconds"
    )


def _spill_enabled(options: ShellOutputCaptureOptions | None) -> bool:
    return options is not None and options.spill


def _shell_environment(
    base: Mapping[str, str] | None,
    extra: Mapping[str, str] | None,
    inherit: bool,
) -> dict[str, str]:
    if not inherit:
        return dict(extra or {})
    return {**os.environ, **(base or {}), **(extra or {})}
