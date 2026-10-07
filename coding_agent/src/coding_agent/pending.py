"""Pending interactive image attachments kept only in the conversation draft.

An attachment accepted by ``/attach`` or a screenshot paste stays in process
memory until the user submits it with a prompt. Each entry keeps a stable
identity, the display name, where it came from, the processed send
representation and its processing status, so later draft management can remove
or reorder entries without losing an image's identity. The draft is never saved
to history before submission.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from omh.llm.types import ImageContent

from coding_agent.attachments import resolve_attachment_path
from coding_agent.images import ImageLimits, ProcessedImage, read_image

#: Where a pending attachment's bytes came from.
AttachmentOrigin = Literal["file", "clipboard", "history"]

#: Origin value for an entry read from an explicit image path.
FILE_ORIGIN: AttachmentOrigin = "file"
HISTORY_ORIGIN: AttachmentOrigin = "history"
#: Origin value for an entry read from the desktop clipboard.
CLIPBOARD_ORIGIN: AttachmentOrigin = "clipboard"


@dataclass(frozen=True, slots=True)
class PendingAttachment:
    """One accepted image waiting for the prompt that will submit it."""

    identity: str
    name: str
    origin: AttachmentOrigin
    source: str | None
    image: ProcessedImage

    @property
    def content(self) -> ImageContent:
        """The real image content submitted with the prompt."""
        return self.image.content

    @property
    def status(self) -> str:
        """Processing status: dimensions, sent type and any conversion or resize."""
        tokens = [f"{self.image.width}x{self.image.height}", self.image.mime_type, "ready"]
        if self.image.converted_from is not None:
            tokens.append(f"converted-from={self.image.converted_from}")
        if self.image.was_resized:
            tokens.append(
                f"resized-from={self.image.original_width}x{self.image.original_height}"
            )
        return " ".join(tokens)

    @property
    def location(self) -> str:
        """Where the bytes were read from, for management output."""
        return self.source if self.source is not None else CLIPBOARD_ORIGIN

    def describe(self, index: int) -> str:
        """One management line: index, name, status, origin and source."""
        return f"#{index} {self.name} {self.status} origin={self.origin} source={self.location}"

    @property
    def label(self) -> str:
        """Short conversation label used when the attachment is submitted."""
        return f"{self.name} {self.image.width}x{self.image.height} {self.image.mime_type}"


@dataclass(slots=True)
class PendingAttachments:
    """The ordered draft list for one conversation, with stable identities."""

    _items: list[PendingAttachment] = field(default_factory=list)
    _counter: int = 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[PendingAttachment]:
        return iter(self._items)

    @property
    def items(self) -> tuple[PendingAttachment, ...]:
        return tuple(self._items)

    def add(
        self, *, name: str, origin: AttachmentOrigin, source: str | None, image: ProcessedImage,
    ) -> PendingAttachment:
        """Append a processed image and give it the next stable identity."""
        self._counter += 1
        attachment = PendingAttachment(
            identity=f"attachment-{self._counter}", name=name, origin=origin,
            source=source, image=image,
        )
        self._items.append(attachment)
        return attachment

    def remove(self, identity: str) -> PendingAttachment | None:
        """Remove one attachment by stable identity."""
        for attachment in self._items:
            if attachment.identity == identity:
                self._items.remove(attachment)
                return attachment
        return None

    def remove_index(self, position: int) -> PendingAttachment | None:
        """Remove one attachment by its 1-based management index."""
        if position < 1 or position > len(self._items):
            return None
        return self._items.pop(position - 1)

    def clear(self) -> None:
        self._items.clear()

    def pop_all(self) -> tuple[PendingAttachment, ...]:
        """Take the complete list for submission, keeping identities intact."""
        items = tuple(self._items)
        self._items.clear()
        return items

    def restore(self, attachments: tuple[PendingAttachment, ...]) -> None:
        """Put submitted attachments back at the front after a rejected send."""
        self._items[0:0] = list(attachments)

    def describe_lines(self) -> list[str]:
        """Management lines for ``/attach`` without arguments."""
        if not self._items:
            return ["attachments none pending"]
        lines = [f"attachments {len(self._items)} pending"]
        lines.extend(
            attachment.describe(index) for index, attachment in enumerate(self._items, start=1)
        )
        return lines


def read_pending_image(
    argument: str, *, cwd: str | Path, limits: ImageLimits | None = None,
) -> tuple[str, str, ProcessedImage]:
    """Read and process one explicit image path for the draft.

    Returns the display name, the resolved source path text and the processed
    image. Every failure raises :class:`coding_agent.images.ImageInputError` with
    a message that names the file.
    """
    path = resolve_attachment_path(argument, cwd)
    image = read_image(path, limits=limits)
    return path.name, str(path), image
