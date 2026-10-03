"""Coding tools exercised through their public factories and the Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    ReadImageProcessorFailure,
    ReadImageProcessorOptions,
    ReadImageProcessorResult,
    ReadImageProcessorSuccess,
    ReadToolOptions,
    create_read_tool,
    create_write_tool,
)
from omh.llm.types import (
    AbortController,
    AbortError,
    AbortSignal,
    ImageContent,
    TextContent,
    ToolResultMessage,
)
from tests.agent.test_agent_tools import (
    ScriptedStreamFn,
    make_model,
    text_message,
    tool_call_message,
)


async def test_agent_writes_parent_directories_and_reads_file(tmp_path: Path) -> None:
    read = create_read_tool(tmp_path)
    write = create_write_tool(tmp_path)
    assert isinstance(read, AgentTool)
    assert isinstance(write, AgentTool)
    stream = ScriptedStreamFn([
        lambda: tool_call_message("write", {"path": "nested/note.txt", "content": "你好\n"}),
        lambda: tool_call_message("read", {"path": "nested/note.txt"}, call_id="c2"),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[read, write]),
        stream_fn=stream,
    ))
    await agent.prompt("Write and read a note")
    results = [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]
    assert [message.is_error for message in results] == [False, False]
    assert results[1].content == [TextContent(text="你好\n")]
    assert json.loads(json.dumps(results[0].details)) is None

    await write.execute("overwrite", {"path": "nested/note.txt", "content": "replaced"}, None, lambda _: None)
    result = await read.execute("read", {"path": str(tmp_path / "nested/note.txt")}, None, lambda _: None)
    assert result.content == [TextContent(text="replaced")]


@pytest.mark.parametrize(("data", "args", "expected", "truncated_by"), [
    (b"a\nb\nc\nd", {"offset": 2, "limit": 1}, "b\n\n[2 more lines in file. Use offset=3 to continue.]", None),
    (b"a\nb\n", {"offset": 2}, "b\n", None),
    (b"", {}, "", None),
    (b"invalid: \xff", {}, "invalid: �", None),
    (b"a\n" * 2001, {}, "\n\n[Showing lines 1-2000 of 2002. Use offset=2001 to continue.]", "lines"),
    (("你" * 100 + "\n").encode() * 200, {}, "\n\n[Showing lines 1-170 of 201 (50.0KB limit). Use offset=171 to continue.]", "bytes"),
    (b"x" * 51201, {}, "[Line 1 is 50.0KB, exceeds 50.0KB limit.", "bytes"),
], ids=["offset-limit", "trailing-newline", "empty", "invalid-utf8", "line-budget", "byte-budget", "long-line"])
async def test_read_line_selection_and_bounded_utf8_head(
    tmp_path: Path, data: bytes, args: dict[str, object], expected: str, truncated_by: str | None
) -> None:
    (tmp_path / "note.txt").write_bytes(data)
    result = await create_read_tool(tmp_path).execute(
        "read", {"path": "note.txt", **args}, None, lambda _: None
    )
    assert isinstance(result.content[0], TextContent)
    output = result.content[0].text
    if truncated_by is None:
        assert output == expected
        assert result.details is None
    else:
        assert expected in output
        metadata = json.loads(json.dumps(result.details))["truncation"]
        assert metadata["truncatedBy"] == truncated_by
        assert metadata["outputBytes"] <= 51200
        assert metadata["outputLines"] <= 2000
        if metadata["firstLineExceedsLimit"]:
            assert "head -c 51200" in output
        else:
            assert output.startswith("a\na" if truncated_by == "lines" else "你" * 100)


@pytest.mark.parametrize("args", [
    {"offset": 10}, {"offset": 0}, {"limit": -1}, {"limit": 1.5},
    {"offset": True}, {"limit": None}, {"offset": float("inf")}, {"limit": float("nan")},
])
async def test_read_rejects_invalid_line_selection(tmp_path: Path, args: dict[str, object]) -> None:
    (tmp_path / "note.txt").write_text("a\nb")
    with pytest.raises(ValueError):
        await create_read_tool(tmp_path).execute("read", {"path": "note.txt", **args}, None, lambda _: None)


# A real one-pixel PNG; image contents survive tool execution without decoding.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWZkAAAAASUVORK5CYII="
)


async def test_read_png_returns_sdk_image_attachment(tmp_path: Path) -> None:
    (tmp_path / "pixel.png").write_bytes(PNG)
    stream = ScriptedStreamFn([
        lambda: tool_call_message("read", {"path": "pixel.png"}),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_read_tool(tmp_path)]),
        stream_fn=stream,
    ))
    await agent.prompt("Read the image")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.content == [
        TextContent(text="Read image file [image/png]"),
        ImageContent(data=base64.b64encode(PNG).decode("ascii"), mime_type="image/png"),
    ]
    assert not result.is_error


@pytest.mark.parametrize("resize", [True, False])
@pytest.mark.parametrize("success", [True, False])
async def test_image_processor_configuration_and_failure_notes(
    tmp_path: Path, resize: bool, success: bool
) -> None:
    (tmp_path / "pixel.png").write_bytes(PNG)

    async def processor(
        data: bytes, mime_type: str, options: ReadImageProcessorOptions, signal: AbortSignal | None
    ) -> ReadImageProcessorResult:
        assert data == PNG
        assert mime_type == "image/png"
        assert options.auto_resize_images is resize
        assert signal is None
        if not success:
            return ReadImageProcessorFailure(message="[Image omitted: cannot resize]")
        return ReadImageProcessorSuccess(
            data=base64.b64encode(PNG).decode("ascii"), mime_type="image/png", hints=["resized to 1x1"]
        )

    result = await create_read_tool(tmp_path, ReadToolOptions(
        auto_resize_images=resize, image_processor=processor
    )).execute("read", {"path": "pixel.png"}, None, lambda _: None)
    if success:
        assert result.content == [
            TextContent(text="Read image file [image/png]\nresized to 1x1"),
            ImageContent(data=base64.b64encode(PNG).decode("ascii"), mime_type="image/png"),
        ]
    else:
        assert result.content == [TextContent(text="Read image file [image/png]\n[Image omitted: cannot resize]")]


async def test_bmp_requires_explicit_conversion(tmp_path: Path) -> None:
    import struct

    bmp = b"BM" + struct.pack("<IHHI", 58, 0, 0, 54)
    bmp += struct.pack("<IiiHHIIiiII", 40, 1, 1, 1, 24, 0, 4, 0, 0, 0, 0) + b"\x00" * 4
    (tmp_path / "pixel.bmp").write_bytes(bmp)
    result = await create_read_tool(tmp_path).execute("read", {"path": "pixel.bmp"}, None, lambda _: None)
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert "Image omitted" in result.content[0].text

    async def processor(
        data: bytes, mime_type: str, options: ReadImageProcessorOptions, signal: AbortSignal | None
    ) -> ReadImageProcessorResult:
        assert data == bmp
        assert mime_type == "image/bmp"
        return ReadImageProcessorSuccess(data=base64.b64encode(PNG).decode("ascii"), mime_type="image/png", hints=[])

    converted = await create_read_tool(tmp_path, ReadToolOptions(image_processor=processor)).execute(
        "read", {"path": "pixel.bmp"}, None, lambda _: None
    )
    assert converted.content[1] == ImageContent(data=base64.b64encode(PNG).decode("ascii"), mime_type="image/png")


@pytest.mark.parametrize("alias_kind", ["file", "directory", "missing"])
@pytest.mark.parametrize("cancel", ["none", "signal", "task"])
async def test_canonical_writes_serialize_until_cancelled_io_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str, cancel: str
) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    target = directory / "note.txt"
    alias = tmp_path / "alias"
    if alias_kind == "file":
        target.write_text("initial")
        alias.symlink_to(target)
        alias_path = alias
    else:
        alias.symlink_to(directory, target_is_directory=True)
        alias_path = alias / "note.txt"
        if alias_kind == "directory":
            target.write_text("initial")

    entered = asyncio.Event()
    release = threading.Event()
    writes: list[str] = []
    original_write = Path.write_text
    loop = asyncio.get_running_loop()

    def controlled_write(path: Path, content: str, **kwargs: object) -> int:
        writes.append(content)
        if content == "first":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "test did not release filesystem write"
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", controlled_write)
    controller = AbortController()
    first = asyncio.create_task(create_write_tool(tmp_path).execute(
        "first", {"path": str(target), "content": "first"}, controller.signal, lambda _: None
    ))
    second: asyncio.Task | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel == "signal":
            controller.abort()
        elif cancel == "task":
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
        second = asyncio.create_task(create_write_tool(directory).execute(
            "second", {"path": str(alias_path), "content": "second"}, None, lambda _: None
        ))
        # Different files must finish while the first write is still blocked.
        await asyncio.wait_for(create_write_tool(tmp_path).execute(
            "other", {"path": "other.txt", "content": "other"}, None, lambda _: None
        ), 2)
        assert not first.done()
        assert "second" not in writes
    finally:
        release.set()
        outcomes = await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    if cancel == "signal":
        assert isinstance(outcomes[0], AbortError)
    elif cancel == "task":
        assert isinstance(outcomes[0], asyncio.CancelledError)
    else:
        assert not isinstance(outcomes[0], BaseException)
    assert writes.index("first") < writes.index("second")
    result = await create_read_tool(tmp_path).execute("read", {"path": str(target)}, None, lambda _: None)
    assert result.content == [TextContent(text="second")]


async def test_write_and_read_normalize_user_path_text(tmp_path: Path) -> None:
    await create_write_tool(tmp_path).execute(
        "write", {"path": "@nested/note\u00a0name.txt", "content": "normalized"}, None, lambda _: None
    )
    result = await create_read_tool(tmp_path).execute(
        "read", {"path": "nested/note name.txt"}, None, lambda _: None
    )
    assert result.content == [TextContent(text="normalized")]


@pytest.mark.parametrize(("actual", "requested"), [
    ("Capture 10.00.00\u202fAM.txt", "Capture 10.00.00 AM.txt"),
    ("Capture d’écran.txt", "Capture d'écran.txt"),
    ("cafe\u0301.txt", "café.txt"),
])
async def test_read_tries_existing_filename_variants(tmp_path: Path, actual: str, requested: str) -> None:
    (tmp_path / actual).write_text("variant")
    result = await create_read_tool(tmp_path).execute("read", {"path": requested}, None, lambda _: None)
    assert result.content == [TextContent(text="variant")]


@pytest.mark.parametrize("injected", [False, True])
async def test_agent_only_executes_injected_tools_and_valid_arguments(tmp_path: Path, injected: bool) -> None:
    stream = ScriptedStreamFn([
        lambda: tool_call_message("write", {"path": "nested/note.txt", "content": {} if injected else "text"}),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_write_tool(tmp_path)] if injected else []),
        stream_fn=stream,
    ))
    await agent.prompt("Try writing")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert not (tmp_path / "nested").exists()


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_filesystem_errors_become_agent_error_results(tmp_path: Path, operation: str) -> None:
    (tmp_path / "directory").mkdir()
    args = {"path": "directory", "content": "cannot overwrite a directory"}
    stream = ScriptedStreamFn([
        lambda: tool_call_message(operation, args),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_read_tool(tmp_path), create_write_tool(tmp_path)]),
        stream_fn=stream,
    ))
    await agent.prompt("Try a directory")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "directory" in result.content[0].text.lower()
    # A failed mutation cannot poison later writes to the same canonical path.
    (tmp_path / "directory").rmdir()
    write = create_write_tool(tmp_path)
    await write.execute("write", {"path": "directory", "content": "after failure"}, None, lambda _: None)
    result = await create_read_tool(tmp_path).execute("read", {"path": "directory"}, None, lambda _: None)
    assert result.content == [TextContent(text="after failure")]


async def test_cancelled_queued_write_does_not_bypass_predecessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = threading.Event()
    original_write = Path.write_text
    loop = asyncio.get_running_loop()

    def controlled_write(path: Path, content: str, **kwargs: object) -> int:
        if content == "first":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", controlled_write)
    tool = create_write_tool(tmp_path)
    first = asyncio.create_task(tool.execute("first", {"path": "note.txt", "content": "first"}, None, lambda _: None))
    tasks = [first]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        middle = asyncio.create_task(tool.execute("middle", {"path": "note.txt", "content": "middle"}, None, lambda _: None))
        tasks.append(middle)
        await asyncio.sleep(0.02)
        middle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middle
        last = asyncio.create_task(tool.execute("last", {"path": "note.txt", "content": "last"}, None, lambda _: None))
        tasks.append(last)
        await asyncio.sleep(0.02)
        assert not last.done()
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    result = await create_read_tool(tmp_path).execute("read", {"path": "note.txt"}, None, lambda _: None)
    assert result.content == [TextContent(text="last")]


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_preaborted_tools_do_not_touch_files(tmp_path: Path, operation: str) -> None:
    controller = AbortController()
    controller.abort()
    tool = create_read_tool(tmp_path) if operation == "read" else create_write_tool(tmp_path)
    with pytest.raises(AbortError):
        await tool.execute("call", {"path": "nested/note.txt", "content": "text"}, controller.signal, lambda _: None)
    assert not (tmp_path / "nested").exists()


async def test_agent_close_waits_for_builtin_write_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = threading.Event()
    original_write = Path.write_text
    loop = asyncio.get_running_loop()

    def controlled_write(path: Path, content: str, **kwargs: object) -> int:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", controlled_write)
    stream = ScriptedStreamFn([lambda: tool_call_message("write", {"path": "note.txt", "content": "committed"})])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_write_tool(tmp_path)]),
        stream_fn=stream,
    ))
    prompt = asyncio.create_task(agent.prompt("Write the note"))
    close: asyncio.Task | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        close = asyncio.create_task(agent.close())
        await asyncio.sleep(0.02)
        assert agent.state.is_busy
        assert not close.done()
    finally:
        release.set()
        await asyncio.gather(prompt, *([close] if close else []))
    assert agent.state.is_closed
    assert stream.calls == 1
    result = await create_read_tool(tmp_path).execute("read", {"path": "note.txt"}, None, lambda _: None)
    assert result.content == [TextContent(text="committed")]


async def test_read_image_processor_abort_awaits_cleanup(tmp_path: Path) -> None:
    (tmp_path / "pixel.png").write_bytes(PNG)
    entered = asyncio.Event()
    cleanup = asyncio.Event()
    release = asyncio.Event()
    controller = AbortController()

    async def processor(
        data: bytes, mime_type: str, options: ReadImageProcessorOptions, signal: AbortSignal | None
    ) -> ReadImageProcessorResult:
        assert signal is controller.signal
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleanup.set()
            await release.wait()
        return ReadImageProcessorFailure(message="aborted")

    tool = create_read_tool(tmp_path, ReadToolOptions(image_processor=processor))
    task = asyncio.create_task(tool.execute("read", {"path": "pixel.png"}, controller.signal, lambda _: None))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        controller.abort()
        await asyncio.wait_for(cleanup.wait(), 2)
        assert not task.done()
    finally:
        release.set()
        outcome = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome[0], AbortError)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(("first_name", "second_name"), [
    ("mixedName.txt", "MIXEDNAME.txt"), ("σ.txt", "ς.txt"),
    ("ss.txt", "ß.txt"), ("s.txt", "ſ.txt"),
])
async def test_case_aliases_share_mutation_coordination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, first_name: str, second_name: str
) -> None:
    # Ask the actual filesystem rather than assuming every macOS volume ignores case.
    target = tmp_path / first_name
    target.write_text("probe")
    case_sensitive = not (tmp_path / second_name).exists()
    target.unlink()
    if existing:
        target.write_text("initial")
    entered = asyncio.Event()
    release = threading.Event()
    original_write = Path.write_text
    loop = asyncio.get_running_loop()
    writes: list[str] = []

    def controlled_write(path: Path, content: str, **kwargs: object) -> int:
        writes.append(content)
        if content == "first":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", controlled_write)
    tool = create_write_tool(tmp_path)
    first = asyncio.create_task(tool.execute("first", {"path": first_name, "content": "first"}, None, lambda _: None))
    tasks = [first]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(tool.execute("second", {"path": second_name, "content": "second"}, None, lambda _: None))
        tasks.append(second)
        await tool.execute("other", {"path": "other.txt", "content": "other"}, None, lambda _: None)
        if case_sensitive:
            await asyncio.wait_for(second, 2)
        else:
            assert "second" not in writes
    finally:
        release.set()
        await asyncio.gather(*tasks)
    result = await create_read_tool(tmp_path).execute("read", {"path": second_name}, None, lambda _: None)
    assert result.content == [TextContent(text="second")]
    if case_sensitive:
        result = await create_read_tool(tmp_path).execute("read", {"path": first_name}, None, lambda _: None)
        assert result.content == [TextContent(text="first")]


async def test_slower_path_resolution_preserves_write_submission_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = threading.Event()
    original_resolve = Path.resolve
    loop = asyncio.get_running_loop()
    resolutions = 0

    def controlled_resolve(path: Path, **kwargs: object) -> Path:
        nonlocal resolutions
        if path.name == "note.txt":
            resolutions += 1
            if resolutions == 1:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
        return original_resolve(path, **kwargs)

    monkeypatch.setattr(Path, "resolve", controlled_resolve)
    tool = create_write_tool(tmp_path)
    first = asyncio.create_task(tool.execute("first", {"path": "note.txt", "content": "first"}, None, lambda _: None))
    tasks = [first]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(tool.execute("second", {"path": "note.txt", "content": "second"}, None, lambda _: None))
        tasks.append(second)
        await asyncio.sleep(0.02)
        assert not second.done()
    finally:
        release.set()
        await asyncio.gather(*tasks)
    result = await create_read_tool(tmp_path).execute("read", {"path": "note.txt"}, None, lambda _: None)
    assert result.content == [TextContent(text="second")]
