"""Shared acceptance of ``@file`` text and image attachments before a task.

Attachment text keeps the original file content and an explicit path boundary;
images become real image content. Files are resolved against the session cwd,
read bytes stay untouched, and a file that cannot be read or decoded is
explained instead of being silently dropped. The first-task composition places
piped stdin, the attachments in argument order, and the first prompt into blank
line separated parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omh.agent import detect_supported_image_mime_type
from omh.llm.types import ImageContent

from coding_agent.images import (
    ImageInputError,
    ImageLimits,
    ProcessedImage,
    process_image,
)


class AttachmentError(Exception):
    """A file attachment could not be read at all, with an explanatory message."""


@dataclass(frozen=True, slots=True)
class FileAttachment:
    """One accepted ``@file`` argument and the text boundary it contributes."""

    path: Path
    text: str
    image: ProcessedImage | None = None


def read_file_attachment(
    argument: str, *, cwd: str | Path, limits: ImageLimits | None = None, auto_resize: bool = True,
) -> FileAttachment:
    """Resolve and read one attachment argument against the session directory.

    Images are converted and bounded; an image that cannot be accepted keeps an
    explanatory boundary text. A missing path or content that is neither a
    supported image nor UTF-8 text raises :class:`AttachmentError`.
    """
    path = resolve_attachment_path(argument, cwd)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AttachmentError(f"cannot read file attachment {path}: {error}") from error

    if detect_supported_image_mime_type(data) is not None:
        try:
            processed = process_image(data, limits=limits, auto_resize=auto_resize)
        except ImageInputError as error:
            return FileAttachment(path=path, text=str(error))
        return FileAttachment(path=path, text="\n".join(processed.hints), image=processed)

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AttachmentError(
            f"{path} is not a supported image or UTF-8 text file: {error}"
        ) from error
    return FileAttachment(path=path, text=text)


def render_attachment(attachment: FileAttachment) -> str:
    """Render one attachment boundary; the file text itself is never trimmed."""
    if not attachment.text:
        return f'<file name="{attachment.path}"></file>'
    return f'<file name="{attachment.path}">\n{attachment.text}\n</file>'


def attachment_images(attachments: list[FileAttachment]) -> list[ImageContent]:
    """Collect the accepted image content in attachment order."""
    return [attachment.image.content for attachment in attachments if attachment.image is not None]


def compose_first_task(
    *, stdin_text: str | None = None, attachments: list[FileAttachment] | None = None,
    first_prompt: str | None = None,
) -> str:
    """Combine the first task as stdin, attachment boundaries, then the first prompt.

    Each part is preserved exactly and the next part starts after a blank line,
    so the piped source, every file boundary and the prompt cannot run into each
    other. Empty parts are omitted; a part that already ends with a newline gains
    only the newlines needed for one separating blank line.
    """
    parts = [stdin_text] if stdin_text else []
    parts.extend(render_attachment(attachment) for attachment in attachments or [])
    if first_prompt:
        parts.append(first_prompt)
    composed = ""
    for part in parts:
        if composed:
            composed += _separator(composed)
        composed += part
    return composed


def _separator(previous: str) -> str:
    if previous.endswith("\n\n"):
        return ""
    if previous.endswith("\n"):
        return "\n"
    return "\n\n"


def resolve_attachment_path(argument: str, cwd: str | Path) -> Path:
    """Resolve one attachment argument against the session directory."""
    candidate = Path(argument).expanduser()
    if not candidate.is_absolute():
        candidate = Path(cwd).expanduser() / candidate
    return candidate.resolve()
