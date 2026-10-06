"""Image acceptance: formats, limits, conversion, source preservation and diagnostics.

The tests use small deterministic images built with Pillow and exercise the
public product image contract only; they never send a model request.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from omh.llm.types import ImageContent
from PIL import Image

from coding_agent.images import (
    IMAGE_MAX_BASE64_BYTES,
    IMAGE_MAX_DIMENSION,
    ImageInputError,
    ImageLimits,
    create_read_image_processor,
    process_image,
    read_image,
)


def image_bytes(size: tuple[int, int] = (8, 4), fmt: str = "PNG", *, color=(255, 0, 0), noise: bool = False) -> bytes:
    image = Image.new("RGB", size, color)
    if noise:
        pixels = image.load()
        assert pixels is not None
        for x in range(size[0]):
            for y in range(size[1]):
                pixels[x, y] = ((x * 37 + y * 11) % 256, (x * 17 + y * 53) % 256, (x * 91 + y * 7) % 256)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def animated_gif_bytes() -> bytes:
    first = Image.new("RGB", (6, 6), (255, 0, 0))
    second = Image.new("RGB", (6, 6), (0, 0, 255))
    buffer = io.BytesIO()
    first.save(buffer, format="GIF", save_all=True, append_images=[second], duration=100, loop=0)
    return buffer.getvalue()


def decode(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


def test_static_supported_images_pass_through_without_reencoding() -> None:
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        raw = image_bytes((8, 4), fmt)
        processed = process_image(raw)
        assert processed.mime_type == mime
        assert processed.data == base64.b64encode(raw).decode("ascii")
        assert processed.was_resized is False
        assert processed.converted_from is None
        assert processed.hints == ()
        assert (processed.width, processed.height) == (8, 4)


def test_gif_uses_the_first_frame_and_bmp_converts_to_png() -> None:
    gif = process_image(animated_gif_bytes())
    assert gif.mime_type == "image/png"
    assert gif.converted_from == "image/gif"
    assert any("image/gif" in hint for hint in gif.hints)
    frame = decode(gif.data).convert("RGB")
    assert frame.getpixel((0, 0)) == (255, 0, 0)

    bmp = process_image(image_bytes((8, 4), "BMP"))
    assert bmp.mime_type == "image/png"
    assert bmp.converted_from == "image/bmp"
    assert decode(bmp.data).size == (8, 4)


def test_large_images_resize_to_the_default_bounds_and_keep_their_ratio() -> None:
    processed = process_image(image_bytes((3000, 1500), noise=True))
    assert (processed.width, processed.height) == (IMAGE_MAX_DIMENSION, IMAGE_MAX_DIMENSION // 2)
    assert (processed.original_width, processed.original_height) == (3000, 1500)
    assert processed.was_resized is True
    assert decode(processed.data).size == (2000, 1000)
    assert any("3000x1500" in hint and "2000x1000" in hint for hint in processed.hints)
    assert len(processed.data) < IMAGE_MAX_BASE64_BYTES


def test_stricter_service_limits_are_honored() -> None:
    processed = process_image(image_bytes((1000, 500), noise=True), limits=ImageLimits(max_dimension=100))
    assert (processed.width, processed.height) == (100, 50)
    assert decode(processed.data).size == (100, 50)


def test_encoded_output_stays_below_the_strict_base64_limit_or_is_explained() -> None:
    limits = ImageLimits(max_base64_bytes=2048)
    assert limits.max_dimension == IMAGE_MAX_DIMENSION  # Only the byte limit can drive the shrink.
    processed = process_image(image_bytes((512, 512), noise=True), limits=limits)
    assert len(processed.data) < limits.max_base64_bytes
    assert processed.width < 512 and processed.height < 512
    assert processed.was_resized is True

    with pytest.raises(ImageInputError, match="inline image size limit"):
        process_image(image_bytes((64, 64), noise=True), limits=ImageLimits(max_base64_bytes=1))


def test_auto_resize_off_still_announces_gif_and_bmp_conversion() -> None:
    bmp = process_image(image_bytes((8, 4), "BMP"), auto_resize=False)
    assert bmp.mime_type == "image/png"
    assert bmp.converted_from == "image/bmp"
    assert any("image/bmp" in hint for hint in bmp.hints)

    raw = image_bytes((8, 4), "JPEG")
    passthrough = process_image(raw, auto_resize=False)
    assert passthrough.mime_type == "image/jpeg"
    assert passthrough.data == base64.b64encode(raw).decode("ascii")


def test_corrupt_unsupported_and_unreadable_sources_are_explained(tmp_path: Path) -> None:
    corrupt = image_bytes((4, 4))[:20]
    with pytest.raises(ImageInputError, match="could not be decoded"):
        process_image(corrupt)

    tiff = io.BytesIO()
    Image.new("RGB", (4, 4)).save(tiff, format="TIFF")
    with pytest.raises(ImageInputError, match="supported"):
        process_image(tiff.getvalue())

    with pytest.raises(ImageInputError, match="supported"):
        process_image(b"")

    apng = io.BytesIO()
    first = Image.new("RGB", (4, 4), (1, 2, 3))
    first.save(apng, format="PNG", save_all=True, append_images=[Image.new("RGB", (4, 4), (4, 5, 6))])
    with pytest.raises(ImageInputError, match="animated PNG"):
        process_image(apng.getvalue())

    missing = tmp_path / "missing.png"
    with pytest.raises(ImageInputError, match="missing.png"):
        read_image(missing)

    broken = tmp_path / "broken.png"
    broken.write_bytes(corrupt)
    with pytest.raises(ImageInputError, match="broken.png"):
        read_image(broken)


def test_read_image_keeps_the_source_file_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "picture.png"
    raw = image_bytes((8, 4))
    source.write_bytes(raw)
    processed = read_image(source)
    assert source.read_bytes() == raw
    assert processed.mime_type == "image/png"


async def test_read_processor_reports_failure_text_and_success_hints() -> None:
    from omh.agent import (
        ReadImageProcessorFailure,
        ReadImageProcessorOptions,
        ReadImageProcessorSuccess,
    )

    processor = create_read_image_processor(ImageLimits(max_base64_bytes=1))
    options = ReadImageProcessorOptions(auto_resize_images=True)
    result = await processor(image_bytes((64, 64), noise=True), "image/png", options, None)
    assert isinstance(result, ReadImageProcessorFailure)
    assert "inline image size limit" in result.message

    processor = create_read_image_processor()
    result = await processor(image_bytes((8, 4), "BMP"), "image/bmp", options, None)
    assert isinstance(result, ReadImageProcessorSuccess)
    assert result.mime_type == "image/png"
    assert result.hints
    assert base64.b64decode(result.data)


async def test_read_processor_honors_auto_resize_off() -> None:
    from omh.agent import ReadImageProcessorOptions, ReadImageProcessorSuccess

    processor = create_read_image_processor()
    result = await processor(
        image_bytes((3000, 100), noise=True), "image/png",
        ReadImageProcessorOptions(auto_resize_images=False), None,
    )
    assert isinstance(result, ReadImageProcessorSuccess)
    assert decode(result.data).size == (3000, 100)


def test_image_content_carries_the_processed_base64_payload() -> None:
    processed = process_image(image_bytes((8, 4), "JPEG"))
    content = processed.content
    assert isinstance(content, ImageContent)
    assert content.mime_type == "image/jpeg"
    assert base64.b64decode(content.data)
