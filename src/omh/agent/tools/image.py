"""Image content detection for the read tool.

Corresponds to ``packages/agent/src/harness/tools/image.ts``.
"""

from __future__ import annotations

import base64

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def detect_supported_image_mime_type(data: bytes) -> str | None:
    """Return the supported image MIME type for ``data``, or ``None``."""
    if data.startswith(b"\xff\xd8\xff"):
        return None if len(data) > 3 and data[3] == 0xF7 else "image/jpeg"
    if data.startswith(_PNG_SIGNATURE):
        return "image/png" if _is_png(data) and not _is_animated_png(data) else None
    if _starts_with_ascii(data, 0, "GIF"):
        return "image/gif"
    if _starts_with_ascii(data, 0, "RIFF") and _starts_with_ascii(data, 8, "WEBP"):
        return "image/webp"
    if _starts_with_ascii(data, 0, "BM") and _is_bmp(data):
        return "image/bmp"
    return None


def encode_base64(data: bytes) -> str:
    """Encode bytes as standard base64 text."""
    return base64.b64encode(data).decode("ascii")


def _is_png(data: bytes) -> bool:
    return (
        len(data) >= 16
        and _read_uint32_be(data, len(_PNG_SIGNATURE)) == 13
        and _starts_with_ascii(data, 12, "IHDR")
    )


def _is_animated_png(data: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        chunk_length = _read_uint32_be(data, offset)
        chunk_type_offset = offset + 4
        if _starts_with_ascii(data, chunk_type_offset, "acTL"):
            return True
        if _starts_with_ascii(data, chunk_type_offset, "IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(data):
            return False
        offset = next_offset
    return False


def _is_bmp(data: bytes) -> bool:
    if len(data) < 26:
        return False
    declared_file_size = _read_uint32_le(data, 2)
    pixel_data_offset = _read_uint32_le(data, 10)
    dib_header_size = _read_uint32_le(data, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        color_planes = _read_uint16_le(data, 22)
        bits_per_pixel = _read_uint16_le(data, 24)
    elif 40 <= dib_header_size <= 124:
        if len(data) < 30:
            return False
        color_planes = _read_uint16_le(data, 26)
        bits_per_pixel = _read_uint16_le(data, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def _read_uint16_le(data: bytes, offset: int) -> int:
    return data[offset] + (data[offset + 1] << 8)


def _read_uint32_be(data: bytes, offset: int) -> int:
    return (
        data[offset] * 0x1000000
        + (data[offset + 1] << 16)
        + (data[offset + 2] << 8)
        + data[offset + 3]
    )


def _read_uint32_le(data: bytes, offset: int) -> int:
    return (
        data[offset]
        + (data[offset + 1] << 8)
        + (data[offset + 2] << 16)
        + data[offset + 3] * 0x1000000
    )


def _starts_with_ascii(data: bytes, offset: int, text: str) -> bool:
    if len(data) < offset + len(text):
        return False
    return data[offset : offset + len(text)] == text.encode("ascii")
