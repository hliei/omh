"""Local image acceptance for user attachments and the read tool.

The product owns image conversion and resizing; the SDK only injects an
``image_processor`` into its read tool. Processing never mutates the source
file: it produces the base64 representation that is sent and saved. Supported
static PNG, JPEG and WebP content passes through unchanged while it already
fits the limits; GIF contributes its first frame and BMP converts to PNG.
Resizing keeps the aspect ratio inside a square bound and re-encodes until the
base64 payload is strictly below the byte limit. Every failure carries an
explanatory message instead of silently dropping the attachment.
"""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path

from omh.agent import (
    AgentHistory,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
    ReadImageProcessor,
    ReadImageProcessorFailure,
    ReadImageProcessorOptions,
    ReadImageProcessorResult,
    ReadImageProcessorSuccess,
)
from omh.agent.tools.image import detect_supported_image_mime_type
from omh.llm.types import AbortSignal, ImageContent
from PIL import Image, ImageOps, UnidentifiedImageError

#: Default maximum width and height of the sent representation.
IMAGE_MAX_DIMENSION = 2000
#: Default maximum base64 payload per image: strictly below 4.5 MiB.
IMAGE_MAX_BASE64_BYTES = int(4.5 * 1024 * 1024)
#: JPEG qualities tried, in order, when PNG does not fit the byte limit.
JPEG_QUALITIES = (80, 85, 70, 55, 40)
#: Dimensions shrink by this factor when no encoding fits the byte limit.
SHRINK_FACTOR = 0.75

_SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp")
_CONVERSION_TARGETS = {"image/gif": "image/png", "image/bmp": "image/png"}
_PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}


class ImageInputError(Exception):
    """An image could not be accepted, with a message that explains why."""


@dataclass(frozen=True, slots=True)
class ImageLimits:
    """The send limits applied to one image; a service may pass stricter values."""

    max_dimension: int = IMAGE_MAX_DIMENSION
    max_base64_bytes: int = IMAGE_MAX_BASE64_BYTES

    def __post_init__(self) -> None:
        if self.max_dimension < 1 or self.max_base64_bytes < 1:
            raise ValueError("image limits must be positive")


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    """The sendable representation of one accepted image."""

    data: str
    mime_type: str
    width: int
    height: int
    original_width: int
    original_height: int
    was_resized: bool
    converted_from: str | None = None
    hints: tuple[str, ...] = ()

    @property
    def content(self) -> ImageContent:
        return ImageContent(data=self.data, mime_type=self.mime_type)


def detect_image_mime_type(data: bytes) -> str | None:
    """Return the supported MIME type for ``data`` using the SDK read contract."""
    return detect_supported_image_mime_type(data)


def process_image(
    data: bytes, *, mime_type: str | None = None, limits: ImageLimits | None = None,
    auto_resize: bool = True,
) -> ProcessedImage:
    """Convert and bound ``data`` for sending, explaining any failure.

    ``mime_type`` is the caller's already-detected type (the read tool passes
    it); the byte signature remains authoritative and must match a supported
    format.
    """
    del mime_type
    effective = limits or ImageLimits()
    detected = detect_image_mime_type(data)
    if detected is None:
        raise ImageInputError(_unsupported_message(data))
    if detected not in _SUPPORTED_MIME_TYPES:
        raise ImageInputError(f"{detected} is not a supported image format; supported: {_supported_list()}")

    image = _decode(data, detected)
    original_width, original_height = image.size
    conversion = _CONVERSION_TARGETS.get(detected)

    if conversion is None and (
        not auto_resize or _within_limits(len(data), original_width, original_height, effective)
    ):
        return ProcessedImage(
            data=base64.b64encode(data).decode("ascii"), mime_type=detected,
            width=original_width, height=original_height,
            original_width=original_width, original_height=original_height, was_resized=False,
        )

    if auto_resize:
        encoded = _encode_under_limit(image, _fit(original_width, original_height, effective.max_dimension), effective)
        if encoded is None:
            raise ImageInputError(
                "image could not be reduced below the inline image size limit "
                f"of {effective.max_base64_bytes} base64 bytes"
            )
        data_text, final_mime, width, height = encoded
    else:
        data_text, final_mime, width, height = _encode_png(image), "image/png", original_width, original_height

    hints: list[str] = []
    if conversion is not None:
        hints.append(f"[Image converted from {detected} to {final_mime}.]")
    if (width, height) != (original_width, original_height):
        scale = original_width / width
        hints.append(
            f"[Image: original {original_width}x{original_height}, displayed at {width}x{height}. "
            f"Multiply coordinates by {scale:.2f} to map to original image.]"
        )
    return ProcessedImage(
        data=data_text, mime_type=final_mime, width=width, height=height,
        original_width=original_width, original_height=original_height,
        was_resized=(width, height) != (original_width, original_height),
        converted_from=detected if conversion is not None else None,
        hints=tuple(hints),
    )


def read_image(
    path: str | Path, *, limits: ImageLimits | None = None, auto_resize: bool = True,
) -> ProcessedImage:
    """Read and process an image file, keeping the source bytes untouched."""
    source = Path(path).expanduser()
    try:
        data = source.read_bytes()
    except OSError as error:
        raise ImageInputError(f"cannot read image file {source}: {error}") from error
    try:
        return process_image(data, limits=limits, auto_resize=auto_resize)
    except ImageInputError as error:
        raise ImageInputError(f"{source}: {error}") from error


def create_read_image_processor(limits: ImageLimits | None = None) -> ReadImageProcessor:
    """Build the SDK read-tool processor backed by this product's conversion."""
    effective = limits or ImageLimits()

    async def processor(
        data: bytes, mime_type: str, options: ReadImageProcessorOptions, signal: AbortSignal | None,
    ) -> ReadImageProcessorResult:
        del mime_type
        if signal is not None:
            signal.throw_if_aborted()
        try:
            processed = await asyncio.to_thread(
                process_image, data, limits=effective, auto_resize=options.auto_resize_images,
            )
        except ImageInputError as error:
            return ReadImageProcessorFailure(message=str(error))
        if signal is not None:
            signal.throw_if_aborted()
        return ReadImageProcessorSuccess(
            data=processed.data, mime_type=processed.mime_type, hints=list(processed.hints),
        )

    return processor


def count_history_images(history: AgentHistory) -> int:
    """Count image blocks anywhere in a saved conversation."""
    total = 0
    for entry in history.entries:
        if isinstance(entry, MessageHistoryEntry):
            content = entry.message.content
        elif isinstance(entry, CustomMessageHistoryEntry):
            content = entry.content
        else:
            continue
        if isinstance(content, list):
            total += sum(1 for block in content if isinstance(block, ImageContent))
    return total


def degradation_notice(model_provider: str, model_id: str, count: int) -> str:
    """Explain that saved images are projected as placeholders for a text-only model."""
    return (
        f"{count} saved image(s) will be replaced by a text placeholder when sending to "
        f"{model_provider}/{model_id}; the complete original history is kept. "
        "Select a vision-capable model to send them."
    )


def _decode(data: bytes, detected: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            source_format = opened.format
            if source_format == "GIF" and getattr(opened, "n_frames", 1) > 1:
                opened.seek(0)
                opened.load()
            image = ImageOps.exif_transpose(opened)
            image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
        raise ImageInputError(f"image could not be decoded as {detected}: {error}") from error
    return image


def _within_limits(byte_length: int, width: int, height: int, limits: ImageLimits) -> bool:
    return (
        width <= limits.max_dimension and height <= limits.max_dimension
        and _base64_size(byte_length) < limits.max_base64_bytes
    )


def _fit(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    if width > max_dimension:
        height = max(1, round(height * max_dimension / width))
        width = max_dimension
    if height > max_dimension:
        width = max(1, round(width * max_dimension / height))
        height = max_dimension
    return width, height


def _encode_under_limit(
    image: Image.Image, target: tuple[int, int], limits: ImageLimits,
) -> tuple[str, str, int, int] | None:
    width, height = target
    while True:
        for data_text, mime_type in _encode_candidates(image, (width, height)):
            if _base64_size_from_text(data_text) < limits.max_base64_bytes:
                return data_text, mime_type, width, height
        if (width, height) == (1, 1):
            return None
        next_width = 1 if width == 1 else max(1, int(width * SHRINK_FACTOR))
        next_height = 1 if height == 1 else max(1, int(height * SHRINK_FACTOR))
        if (next_width, next_height) == (width, height):
            return None
        width, height = next_width, next_height


def _encode_candidates(image: Image.Image, size: tuple[int, int]) -> list[tuple[str, str]]:
    resized = image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)
    candidates: list[tuple[str, str]] = [
        (base64.b64encode(_encode_png_bytes(resized)).decode("ascii"), "image/png"),
    ]
    rgb = _to_rgb(resized)
    for quality in dict.fromkeys(JPEG_QUALITIES):
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
        candidates.append((base64.b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg"))
    return candidates


def _encode_png(image: Image.Image) -> str:
    return base64.b64encode(_encode_png_bytes(image)).decode("ascii")


def _encode_png_bytes(image: Image.Image) -> bytes:
    prepared = image if image.mode in _PNG_MODES else image.convert("RGBA")
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA", "PA", "P"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def _base64_size(byte_length: int) -> int:
    return (byte_length + 2) // 3 * 4


def _base64_size_from_text(data: str) -> int:
    return len(data)


def _supported_list() -> str:
    return ", ".join(_SUPPORTED_MIME_TYPES)


def _unsupported_message(data: bytes) -> str:
    if not data:
        return f"image is empty; supported formats: {_supported_list()}"
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image_format = opened.format
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return (
            "image could not be decoded and is not a supported image format; "
            f"supported: {_supported_list()}"
        )
    if image_format is None:
        return f"image could not be decoded; supported: {_supported_list()}"
    return f"{image_format} is not a supported image format; supported: {_supported_list()}"
