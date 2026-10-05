"""Shared text and image attachment acceptance and first-task composition.

The tests exercise the public product attachment contract with real temporary
files; they never send a model request.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from omh.llm.types import ImageContent
from PIL import Image

from coding_agent.attachments import (
    AttachmentError,
    attachment_images,
    compose_first_task,
    read_file_attachment,
    render_attachment,
)


def write_image(path: Path, fmt: str = "PNG") -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), (10, 20, 30)).save(buffer, format=fmt)
    path.write_bytes(buffer.getvalue())


def test_text_attachment_preserves_the_original_text_and_file_boundary(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("  first line\n\nsecond line  \n", encoding="utf-8")

    attachment = read_file_attachment("notes.txt", cwd=tmp_path)

    assert attachment is not None
    assert attachment.text == "  first line\n\nsecond line  \n"
    assert attachment.image is None
    rendered = render_attachment(attachment)
    assert rendered == f'<file name="{source}">\n  first line\n\nsecond line  \n\n</file>'


def test_first_task_composes_stdin_attachments_and_first_prompt_in_order(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    attachments = [
        read_file_attachment("a.txt", cwd=tmp_path),
        read_file_attachment("b.txt", cwd=tmp_path),
    ]

    task = compose_first_task(stdin_text="piped code\n", attachments=attachments, first_prompt="explain")

    assert task == (
        "piped code\n\n"
        f'<file name="{tmp_path / "a.txt"}">\nalpha\n</file>\n\n'
        f'<file name="{tmp_path / "b.txt"}">\nbeta\n</file>\n\n'
        "explain"
    )


def test_image_attachment_contributes_a_boundary_and_real_image_content(tmp_path: Path) -> None:
    write_image(tmp_path / "picture.png")
    attachment = read_file_attachment("picture.png", cwd=tmp_path)
    assert attachment is not None
    assert attachment.image is not None
    assert attachment.text == ""
    assert render_attachment(attachment) == f'<file name="{tmp_path / "picture.png"}"></file>'

    images = attachment_images([attachment])
    assert len(images) == 1
    assert isinstance(images[0], ImageContent)
    assert images[0].mime_type == "image/png"
    assert base64.b64decode(images[0].data) == (tmp_path / "picture.png").read_bytes()


def test_converted_image_reports_hints_in_the_boundary(tmp_path: Path) -> None:
    write_image(tmp_path / "picture.bmp", "BMP")
    attachment = read_file_attachment("picture.bmp", cwd=tmp_path)
    assert attachment is not None and attachment.image is not None
    assert "image/bmp" in attachment.text
    assert attachment.image.mime_type == "image/png"
    assert "image/bmp" in render_attachment(attachment)


def test_corrupt_image_is_explained_in_the_boundary_instead_of_dropped(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    # A valid PNG signature and IHDR prefix, but no decodable image data.
    source.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + b"\x00" * 8)
    attachment = read_file_attachment("broken.png", cwd=tmp_path)
    assert attachment is not None
    assert attachment.image is None
    assert "decoded" in attachment.text
    assert attachment_images([attachment]) == []
    assert str(source) in render_attachment(attachment)


def test_missing_and_non_text_files_are_rejected_before_the_request(tmp_path: Path) -> None:
    with pytest.raises(AttachmentError, match="missing.txt"):
        read_file_attachment("missing.txt", cwd=tmp_path)

    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(AttachmentError, match="UTF-8"):
        read_file_attachment("binary.dat", cwd=tmp_path)


def test_empty_files_keep_their_path_boundary(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    attachment = read_file_attachment("empty.txt", cwd=tmp_path)
    assert attachment is not None
    assert attachment.text == ""
    assert "empty.txt" in render_attachment(attachment)


def test_path_like_prompt_text_is_never_interpreted_as_an_attachment() -> None:
    prompt = "look at /tmp/picture.png and src/main.py"
    assert compose_first_task(first_prompt=prompt) == prompt
    assert attachment_images([]) == []


def test_relative_arguments_resolve_against_the_session_cwd(tmp_path: Path) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "file.txt").write_text("nested", encoding="utf-8")
    attachment = read_file_attachment("sub/file.txt", cwd=tmp_path)
    assert attachment is not None
    assert attachment.text == "nested"
    assert attachment.path == (tmp_path / "sub" / "file.txt").resolve()
