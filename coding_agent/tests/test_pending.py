"""Draft pending attachment identity, management and processing status.

The list and its entries are pure product state; the tests use real temporary
image files and never touch the terminal or a model request.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from coding_agent.images import ImageInputError, ProcessedImage, process_image
from coding_agent.pending import (
    CLIPBOARD_ORIGIN,
    FILE_ORIGIN,
    PendingAttachments,
    read_pending_image,
)


def write_image(path: Path, size: tuple[int, int] = (6, 4), fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format=fmt)
    path.write_bytes(buffer.getvalue())
    return path.read_bytes()


def processed(size: tuple[int, int] = (6, 4)) -> ProcessedImage:
    buffer = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buffer, format="PNG")
    return process_image(buffer.getvalue())


def test_add_keeps_stable_identity_across_removals() -> None:
    pending = PendingAttachments()
    first = pending.add(name="a.png", origin=FILE_ORIGIN, source="/a.png", image=processed())
    second = pending.add(name="b.png", origin=CLIPBOARD_ORIGIN, source=None, image=processed())

    assert first.identity == "attachment-1"
    assert second.identity == "attachment-2"
    assert pending.remove(first.identity) is first
    assert len(pending) == 1

    third = pending.add(name="c.png", origin=FILE_ORIGIN, source="/c.png", image=processed())
    assert third.identity == "attachment-3"
    assert pending.items == (second, third)
    assert second.identity == "attachment-2"


def test_remove_by_one_based_index_and_clear() -> None:
    pending = PendingAttachments()
    for name in ("a.png", "b.png", "c.png"):
        pending.add(name=name, origin=FILE_ORIGIN, source=f"/{name}", image=processed())

    assert pending.remove_index(0) is None
    assert pending.remove_index(4) is None
    assert pending.remove_index(2) is not None
    assert [item.name for item in pending] == ["a.png", "c.png"]
    pending.clear()
    assert len(pending) == 0


def test_pop_all_keeps_identities_for_a_later_restore() -> None:
    pending = PendingAttachments()
    kept = pending.add(name="a.png", origin=FILE_ORIGIN, source="/a.png", image=processed())
    pending.add(name="b.png", origin=CLIPBOARD_ORIGIN, source=None, image=processed())

    taken = pending.pop_all()
    assert len(pending) == 0
    assert [item.name for item in taken] == ["a.png", "b.png"]

    pending.restore((taken[0],))
    assert pending.items == (kept,)
    assert kept.identity == "attachment-1"


def test_images_follow_list_order_and_keep_real_content() -> None:
    pending = PendingAttachments()
    first = processed((6, 4))
    second = processed((3, 2))
    pending.add(name="a.png", origin=FILE_ORIGIN, source="/a.png", image=first)
    pending.add(name="b.png", origin=CLIPBOARD_ORIGIN, source=None, image=second)

    images = pending.images()
    assert [image.data for image in images] == [first.content.data, second.content.data]
    assert images == [first.content, second.content]


def test_status_reports_conversion_and_resize() -> None:
    pending = PendingAttachments()
    pending.add(name="a.png", origin=FILE_ORIGIN, source="/a.png", image=processed((6, 4)))
    converted = process_image(_bmp_bytes((8, 5)))
    pending.add(name="b.bmp", origin=FILE_ORIGIN, source="/b.bmp", image=converted)
    resized = process_image(_big_png_bytes((2001, 10)))
    pending.add(name="c.png", origin=CLIPBOARD_ORIGIN, source=None, image=resized)

    lines = pending.describe_lines()
    assert lines[0] == "attachments 3 pending"
    assert lines[1] == "#1 a.png 6x4 image/png ready origin=file source=/a.png"
    assert "converted-from=image/bmp" in lines[2]
    assert "resized-from=2001x10" in lines[3]
    assert "origin=clipboard source=clipboard" in lines[3]
    assert pending.items[2].label == "c.png 2000x10 image/png"


def test_empty_list_still_prints() -> None:
    assert PendingAttachments().describe_lines() == ["attachments none pending"]


def test_read_pending_image_resolves_against_cwd_and_names_the_source(tmp_path: Path) -> None:
    raw = write_image(tmp_path / "picture.png", (6, 4))

    name, source, image = read_pending_image("picture.png", cwd=tmp_path)

    assert name == "picture.png"
    assert source == str((tmp_path / "picture.png").resolve())
    assert image.content.mime_type == "image/png"
    assert (tmp_path / "picture.png").read_bytes() == raw
    assert image.content.data != ""


def test_read_pending_image_explains_a_corrupt_file_with_its_name(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + b"\x00" * 8)

    with pytest.raises(ImageInputError, match="broken.png"):
        read_pending_image("broken.png", cwd=tmp_path)


def test_read_pending_image_rejects_a_text_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageInputError, match="notes.txt"):
        read_pending_image("notes.txt", cwd=tmp_path)


def _bmp_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (9, 9, 9)).save(buffer, format="BMP")
    return buffer.getvalue()


def _big_png_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (7, 7, 7)).save(buffer, format="PNG")
    return buffer.getvalue()
