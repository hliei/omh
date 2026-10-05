"""Session-level image input: acceptance, read tool, modality and reopen.

These tests use the public ``AgentSessionRuntime``/``AgentSession`` input
boundary, the SDK read tool and the real provider transport with a controlled
HTTP boundary. They never send a real request.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from omh.agent import AgentHistory, MessageHistoryEntry
from omh.llm.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    UserMessage,
    empty_usage,
)
from PIL import Image
from support import (
    OfflineStream,
    RecordingFetch,
    SequencedFetch,
    model,
    text_stream,
    tool_call_stream,
)

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentHost,
    CodingAgentOptions,
    UnsupportedImageModelError,
    decode_history,
    encode_history,
)
from coding_agent.images import ImageLimits

STAMP = datetime(2026, 10, 6, tzinfo=UTC)


def image_file(path: Path, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), (200, 30, 90)).save(buffer, format=fmt)
    path.write_bytes(buffer.getvalue())
    return path.read_bytes()


def non_vision_model(id: str = "text-only"):
    return replace(model(id=id), input=("text",))


def user_image(data: str = "QUJD") -> ImageContent:
    return ImageContent(data=data, mime_type="image/png")


def history_with_image(path: Path, *, cwd: Path, image: ImageContent) -> None:
    history = AgentHistory(
        conversation_id="conversation", created_at=STAMP, leaf_id="assistant", entries=(
            MessageHistoryEntry(
                id="user", parent_id=None, timestamp=STAMP,
                message=UserMessage(content=[TextContent(text="look"), image], timestamp=1),
            ),
            MessageHistoryEntry(
                id="assistant", parent_id="user", timestamp=STAMP,
                message=AssistantMessage(
                    api="openai-completions", provider="opencode-go", model="glm-5.3",
                    timestamp=2, content=[TextContent(text="seen")], usage=empty_usage(),
                    stop_reason="stop",
                ),
            ),
        ),
    )
    path.write_text(encode_history(history, cwd=str(cwd), display_name="images"))


async def test_vision_model_receives_the_real_image_content(tmp_path: Path) -> None:
    stream = OfflineStream()
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
    )).new_session()

    await session.prompt("describe", images=[user_image("cGl4ZWxz")])

    messages = stream.requests[0][1].messages
    assert isinstance(messages[-1], UserMessage)
    images = [block for block in messages[-1].content if isinstance(block, ImageContent)]
    assert images == [user_image("cGl4ZWxz")]
    assert session.supports_images is True


async def test_text_only_model_rejects_new_images_with_manual_model_guidance(tmp_path: Path) -> None:
    stream = OfflineStream()
    vision = model(id="vision")
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=non_vision_model(), stream_fn=stream, tools=(),
        available_models=(non_vision_model(), vision),
    )).new_session()
    initial = session.agent.history

    with pytest.raises(UnsupportedImageModelError) as failure:
        await session.prompt("describe", images=[user_image()])
    assert "text-only" in str(failure.value)
    assert "test/vision" in str(failure.value)
    assert "does not switch provider automatically" in str(failure.value)

    with pytest.raises(UnsupportedImageModelError):
        session.steer("describe", images=[user_image()])
    with pytest.raises(UnsupportedImageModelError):
        session.follow_up("describe", images=[user_image()])

    assert stream.requests == []
    assert session.agent.history == initial
    assert session.supports_images is False


async def test_read_tool_sends_a_real_image_and_converts_bmp(tmp_path: Path) -> None:
    original = image_file(tmp_path / "picture.bmp", "BMP")
    stream = OfflineStream([[ToolCall(id="r", name="read", arguments={"path": "picture.bmp"})]])
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("read",),
    )).new_session()

    await session.prompt("read the picture")

    results = [
        entry.message for entry in session.agent.history.entries
        if entry.type == "message" and entry.message.role == "toolResult"
    ]
    assert len(results) == 1
    blocks = results[0].content
    assert blocks[0].type == "text" and "image/png" in blocks[0].text
    image = next(block for block in blocks if isinstance(block, ImageContent))
    assert image.mime_type == "image/png"
    assert base64.b64decode(image.data) != original
    assert (tmp_path / "picture.bmp").read_bytes() == original


async def test_read_tool_reports_a_tool_diagnostic_for_an_unreadable_image(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + b"\x00" * 8)
    stream = OfflineStream([[ToolCall(id="r", name="read", arguments={"path": "broken.png"})]])
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("read",),
    )).new_session()

    await session.prompt("read the picture")

    results = [
        entry.message for entry in session.agent.history.entries
        if entry.type == "message" and entry.message.role == "toolResult"
    ]
    blocks = results[0].content
    assert not any(isinstance(block, ImageContent) for block in blocks)
    assert any(isinstance(block, TextContent) and "could not be decoded" in block.text for block in blocks)


async def test_read_tool_honors_stricter_session_image_limits(tmp_path: Path) -> None:
    image_file(tmp_path / "wide.png")
    stream = OfflineStream([[ToolCall(id="r", name="read", arguments={"path": "wide.png"})]])
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("read",),
        image_limits=ImageLimits(max_dimension=10),
    )).new_session()

    await session.prompt("read the picture")

    result = next(
        entry.message for entry in session.agent.history.entries
        if entry.type == "message" and entry.message.role == "toolResult"
    )
    image = next(block for block in result.content if isinstance(block, ImageContent))
    assert Image.open(io.BytesIO(base64.b64decode(image.data))).size == (10, 5)


async def test_submitted_images_survive_save_reopen_after_the_source_disappears(tmp_path: Path) -> None:
    source = tmp_path / "attach.png"
    raw = image_file(source)
    path = tmp_path / "history.jsonl"
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), session_file=path)
    session = await AgentSessionRuntime(options).new_session()
    await session.prompt("mixed text", images=[ImageContent(
        data=base64.b64encode(raw).decode("ascii"), mime_type="image/png",
    )])
    assert session.save_state == "saved"
    source.unlink()

    reopened = await AgentSessionRuntime(options).open_session(path)
    entries = [entry for entry in reopened.agent.history.entries if entry.type == "message"]
    user = next(entry.message for entry in entries if entry.message.role == "user")
    assert user.content[0] == TextContent(text="mixed text")
    assert user.content[1] == ImageContent(data=base64.b64encode(raw).decode("ascii"), mime_type="image/png")
    assistant = next(entry.message for entry in entries if entry.message.role == "assistant")
    assert assistant.provider == model().provider and assistant.model == model().id
    assert assistant.usage == empty_usage()
    assert reopened.supports_images is True

    await reopened.prompt("continue")
    assert reopened.save_state == "saved"
    assert len(reopened.agent.history.entries) > len(entries)


async def test_opening_saved_images_with_a_text_only_model_explains_the_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    history_with_image(path, cwd=tmp_path, image=user_image("c2F2ZWQ="))
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")

    selection = host.select_open(path, model="opencode-go/glm-5.3")

    assert selection.ready
    explanation = [d for d in selection.diagnostics if d.reason == "adjusted"]
    assert len(explanation) == 1
    assert "1 saved image" in explanation[0].message
    assert "placeholder" in explanation[0].message
    assert "original history is kept" in explanation[0].message

    vision = host.select_open(path, model="opencode-go/deepseek-v4.1-flash")
    assert not [d for d in vision.diagnostics if d.reason == "adjusted"]


async def test_text_only_request_projects_the_placeholder_and_keeps_saved_images(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    history_with_image(path, cwd=tmp_path, image=user_image("c2F2ZWQ="))
    fetch = SequencedFetch(text_stream("continued"))
    host = CodingAgentHost(
        startup_dir=tmp_path, agent_dir=tmp_path / "agent", api_key="test", fetch=fetch,
    )
    selection = host.select_open(path)
    assert selection.ready
    assert selection.model is not None and selection.model.id == "glm-5.3"

    runtime = AgentSessionRuntime(host.build_options(selection))
    session = await runtime.open_session(path)
    await runtime.prompt("continue")

    body = json.dumps(fetch.bodies[0])
    assert "(image omitted: model does not support images)" in body
    assert "image_url" not in body
    saved = [entry.message for entry in session.agent.history.entries
             if entry.type == "message" and entry.message.role == "user"]
    assert any(
        isinstance(block, ImageContent) and block.data == "c2F2ZWQ="
        for message in saved for block in message.content
    )
    assert decode_history(path.read_bytes()).history.entries  # file remains a valid omh history


async def test_recording_fetch_payload_carries_user_image_content_through_a_vision_model(tmp_path: Path) -> None:
    raw = image_file(tmp_path / "photo.png")
    encoded = base64.b64encode(raw).decode("ascii")
    fetch = RecordingFetch(text_stream("seen"))
    host = CodingAgentHost(
        startup_dir=tmp_path, agent_dir=tmp_path / "agent", api_key="test", fetch=fetch,
    )
    selection = host.select_new(model="opencode-go/deepseek-v4.1-flash")
    assert selection.ready
    runtime = AgentSessionRuntime(host.build_options(selection))
    await runtime.new_session()

    await runtime.prompt("describe", images=[ImageContent(data=encoded, mime_type="image/png")])

    parts = [
        part for message in fetch.requests[0].json_body["messages"] if isinstance(message.get("content"), list)
        for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert [part["image_url"]["url"] for part in parts] == [f"data:image/png;base64,{encoded}"]


#: The five image targets the phase must prove offline before live acceptance.
VISION_TARGETS = (
    "opencode-go/glm-5.3-flash",
    "opencode-go/kimi-k3",
    "opencode-go/kimi-k2.7-code",
    "opencode-go/deepseek-v4.1-flash",
    "deepseek/deepseek-flash",
)


@pytest.mark.parametrize("model_reference", VISION_TARGETS)
async def test_every_vision_target_receives_real_read_image_content(
    tmp_path: Path, model_reference: str,
) -> None:
    raw = image_file(tmp_path / "photo.png")
    encoded = base64.b64encode(raw).decode("ascii")
    fetch = SequencedFetch(
        tool_call_stream("r", "read", {"path": "photo.png"}), text_stream("described"),
    )
    host = CodingAgentHost(
        startup_dir=tmp_path, agent_dir=str(tmp_path / "agent") + model_reference.replace("/", "-"),
        api_key="test", fetch=fetch,
    )
    selection = host.select_new(model=model_reference)
    assert selection.ready, [diagnostic.message for diagnostic in selection.diagnostics]
    runtime = AgentSessionRuntime(host.build_options(selection))
    await runtime.new_session()

    await runtime.prompt("read the picture")

    assert len(fetch.bodies) == 2
    second = json.dumps(fetch.bodies[1])
    assert "Attached image(s) from tool result:" in second
    assert f"data:image/png;base64,{encoded}" in second
