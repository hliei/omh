from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from omh.agent.context import Context
from omh.agent.result import Ok, Result
from omh.agent.utils.truncate import ShellOutputTruncation

type FileKind = Literal["file", "directory", "symlink"]
type FileErrorCode = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]
type ExecutionErrorCode = Literal[
    "aborted",
    "timeout",
    "shell_unavailable",
    "spawn_error",
    "callback_error",
    "unknown",
]
type ShellOutputRetention = Literal["head", "tail"]


class FileError(Exception):
    """Backend-independent filesystem failure returned by :class:`FileSystem`."""

    def __init__(
        self,
        code: FileErrorCode,
        message: str,
        path: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        if cause is not None:
            self.__cause__ = cause


class ExecutionError(Exception):
    """Backend-independent process failure returned by :class:`Shell`."""

    def __init__(
        self,
        code: ExecutionErrorCode,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        if cause is not None:
            self.__cause__ = cause


def get_or_throw[TValue](result: Result[TValue, FileError]) -> TValue:
    """Return the success value or raise the failure error."""
    if isinstance(result, Ok):
        return result.value
    raise result.error


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Metadata for one filesystem object. Symlinks are not followed."""

    name: str
    path: str
    kind: FileKind
    size: int
    mtime_ms: float


@dataclass(frozen=True, slots=True)
class TextLine:
    """One UTF-8 line read from a text file."""

    text: str
    terminated: bool


class TextLineReader(Protocol):
    """Pull-based UTF-8 line reader that preserves final-line termination."""

    async def read_line(self, context: Context) -> Result[TextLine | None, FileError]: ...

    async def close(self, context: Context) -> None: ...


@dataclass(frozen=True, slots=True)
class ReadTextLinesOptions:
    max_lines: int | None = None


@dataclass(frozen=True, slots=True)
class CreateDirOptions:
    recursive: bool = True


@dataclass(frozen=True, slots=True)
class RemoveOptions:
    recursive: bool = False
    force: bool = False


@dataclass(frozen=True, slots=True)
class CreateTempFileOptions:
    prefix: str = ""
    suffix: str = ""


class FileSystem(Protocol):
    """Filesystem capability used by the harness.

    Paths passed to methods may be absolute or relative to :attr:`cwd`. Paths
    returned by file operations are addressed paths but are not canonicalized
    through symlinks unless returned by :meth:`canonical_path`.
    """

    cwd: str

    async def absolute_path(
        self, path: str, context: Context
    ) -> Result[str, FileError]: ...

    async def join_path(
        self, parts: Sequence[str], context: Context
    ) -> Result[str, FileError]: ...

    async def read_text_file(
        self, path: str, context: Context
    ) -> Result[str, FileError]: ...

    async def open_text_line_reader(
        self, path: str, context: Context
    ) -> Result[TextLineReader, FileError]: ...

    async def read_text_lines(
        self,
        path: str,
        options: ReadTextLinesOptions | None,
        context: Context,
    ) -> Result[list[str], FileError]: ...

    async def read_binary_file(
        self, path: str, context: Context
    ) -> Result[bytes, FileError]: ...

    async def write_file(
        self, path: str, content: str | bytes, context: Context
    ) -> Result[None, FileError]: ...

    async def append_file(
        self, path: str, content: str | bytes, context: Context
    ) -> Result[None, FileError]: ...

    async def rename_file(
        self, source_path: str, destination_path: str, context: Context
    ) -> Result[None, FileError]: ...

    async def file_info(
        self, path: str, context: Context
    ) -> Result[FileInfo, FileError]: ...

    async def list_dir(
        self, path: str, context: Context
    ) -> Result[list[FileInfo], FileError]: ...

    async def canonical_path(
        self, path: str, context: Context
    ) -> Result[str, FileError]: ...

    async def exists(self, path: str, context: Context) -> Result[bool, FileError]: ...

    async def create_dir(
        self, path: str, options: CreateDirOptions | None, context: Context
    ) -> Result[None, FileError]: ...

    async def remove(
        self, path: str, options: RemoveOptions | None, context: Context
    ) -> Result[None, FileError]: ...

    async def create_temp_dir(
        self, prefix: str | None, context: Context
    ) -> Result[str, FileError]: ...

    async def create_temp_file(
        self, options: CreateTempFileOptions | None, context: Context
    ) -> Result[str, FileError]: ...

    async def cleanup(self, context: Context) -> None: ...


@dataclass(frozen=True, slots=True)
class ShellOutputLimits:
    """Source-side limits for one combined shell output view."""

    max_bytes: int
    max_lines: int
    retain: ShellOutputRetention = "tail"


@dataclass(frozen=True, slots=True)
class ShellOutputCaptureOptions:
    """Bounded shell capture requested by the caller."""

    limits: ShellOutputLimits
    spill: bool = False


@dataclass(frozen=True, slots=True)
class ShellOutputMetadata:
    """Metadata accompanying a bounded shell output view."""

    truncation: ShellOutputTruncation
    spill_path: str | None = None
    last_line_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ShellOutputView(ShellOutputMetadata):
    """Complete bounded shell output view."""

    text: str = ""


@dataclass(frozen=True, slots=True)
class ShellOutputReplace:
    output: ShellOutputView
    kind: Literal["replace"] = "replace"


@dataclass(frozen=True, slots=True)
class ShellOutputAppend:
    text: str
    metadata: ShellOutputMetadata
    kind: Literal["append"] = "append"


@dataclass(frozen=True, slots=True)
class ShellOutputSlide:
    drop: int
    text: str
    metadata: ShellOutputMetadata
    kind: Literal["slide"] = "slide"


@dataclass(frozen=True, slots=True)
class ShellOutputMetadataUpdate:
    metadata: ShellOutputMetadata
    kind: Literal["metadata"] = "metadata"


type ShellOutputUpdate = (
    ShellOutputReplace
    | ShellOutputAppend
    | ShellOutputSlide
    | ShellOutputMetadataUpdate
)


@dataclass(frozen=True, slots=True)
class ShellExecResult:
    """Bounded shell completion. Output text is delivered through ``on_update``."""

    exit_code: int
    truncation: ShellOutputTruncation
    spill_path: str | None = None
    last_line_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ShellExecOptions:
    """Options for :meth:`Shell.exec`."""

    cwd: str | None = None
    env: Mapping[str, str] | None = None
    inherit_env: bool = True
    timeout: float | None = None
    capture: ShellOutputCaptureOptions | None = None
    on_update: Callable[[ShellOutputUpdate, Context], None] | None = None


class Shell(Protocol):
    """Shell execution capability used by the harness."""

    async def exec(
        self, command: str, options: ShellExecOptions | None, context: Context
    ) -> Result[ShellExecResult, ExecutionError]: ...

    async def cleanup(self, context: Context) -> None: ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    """Filesystem and process execution environment used by the harness."""
