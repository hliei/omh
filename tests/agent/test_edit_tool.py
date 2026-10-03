"""Exact file edits through public AgentTool and Agent interfaces."""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    create_edit_tool,
    create_write_tool,
)
from omh.llm.types import (
    AbortController,
    AbortError,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from tests.agent.test_agent_tools import (
    ScriptedStreamFn,
    make_model,
    text_message,
    tool_call_message,
)


async def test_edit_matches_all_replacements_against_original_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("one\ntwo\nthree\n")
    tool = create_edit_tool(tmp_path)
    assert isinstance(tool, AgentTool)
    result = await tool.execute("edit", {
        "path": "note.txt",
        "edits": [
            {"oldText": "three", "newText": "two"},
            {"oldText": "two", "newText": "TWO"},
        ],
    }, None, lambda _: None)
    assert target.read_bytes() == b"one\nTWO\ntwo\n"
    assert result.content == [TextContent(text="Successfully replaced 2 block(s) in note.txt.")]
    assert isinstance(result.details, dict)
    assert result.details["firstChangedLine"] == 2
    assert "-3 three" in result.details["diff"]
    assert "+2 TWO" in result.details["diff"]
    assert result.details["patch"] == (
        "--- note.txt\n+++ note.txt\n@@ -1,3 +1,3 @@\n one\n+TWO\n two\n-three\n"
    )


@pytest.mark.parametrize("arguments", [
    {"edits": {"oldText": "two", "newText": "TWO"}},
    {"edits": '[{"oldText": "two", "newText": "TWO"}]'},
    {"edits": '{"oldText": "two", "newText": "TWO"}'},
    {"oldText": "two", "newText": "TWO"},
    {"edits": [{"oldText": "one", "newText": "ONE"}], "oldText": "two", "newText": "TWO"},
    {"edits": '{"oldText": "one", "newText": "ONE"}', "oldText": "two", "newText": "TWO"},
], ids=["object", "json-list", "json-object", "legacy", "append-legacy", "json-and-legacy"])
async def test_agent_prepares_compatible_arguments_and_retains_raw_history(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("one\ntwo\n")
    raw = {"path": "note.txt", **arguments}
    original_arguments = json.loads(json.dumps(raw))
    stream = ScriptedStreamFn([
        lambda: tool_call_message("edit", raw),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_edit_tool(tmp_path)]),
        stream_fn=stream,
    ))
    await agent.prompt("Edit the note")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert not result.is_error
    assert target.read_bytes() == (b"ONE\nTWO\n" if "oldText" in arguments and "edits" in arguments else b"one\nTWO\n")
    saved = next(
        entry.message for entry in agent.history.entries
        if hasattr(entry, "message") and isinstance(entry.message, AssistantMessage)
    )
    assert isinstance(saved.content[0], ToolCall)
    assert saved.content[0].arguments == original_arguments
    assert raw == original_arguments
    assert json.loads(json.dumps(result.details)) == result.details
    await agent.close()


@pytest.mark.parametrize(("content", "edits", "error"), [
    ("one\ntwo\n", [{"oldText": "one", "newText": "ONE"}, {"oldText": "absent", "newText": "X"}], "Could not find"),
    ("same\nsame\n", [{"oldText": "same", "newText": "X"}], "must be unique"),
    ("abcdef", [{"oldText": "abcd", "newText": "X"}, {"oldText": "cdef", "newText": "Y"}], "overlap"),
    ("abcdef", [{"oldText": "abcdef", "newText": "X"}, {"oldText": "bc", "newText": "Y"}], "overlap"),
    ("hello", [{"oldText": "", "newText": "X"}], "must not be empty"),
    ("hello", [{"oldText": "hello", "newText": "hello"}], "No changes"),
    ("hello", [], "at least one replacement"),
    ("hello", ["bad"], "must be an object"),
    ("hello", [{"oldText": 1, "newText": "X"}], "string oldText and newText"),
    ("hello", [{"oldText": "hello"}], "string oldText and newText"),
], ids=["missing-later", "ambiguous", "overlap", "nested", "empty-old", "no-change", "empty-list", "bad-item", "bad-type", "missing-new"])
async def test_invalid_edits_leave_original_bytes_untouched(
    tmp_path: Path, content: str, edits: object, error: str
) -> None:
    target = tmp_path / "note.txt"
    original = content.encode()
    target.write_bytes(original)
    with pytest.raises(ValueError, match=error):
        await create_edit_tool(tmp_path).execute("edit", {"path": "note.txt", "edits": edits}, None, lambda _: None)
    assert target.read_bytes() == original


@pytest.mark.parametrize(("original", "edits", "expected"), [
    (b"\xef\xbb\xbfhello\r\nworld\r\n", [{"oldText": "hello\nworld", "newText": "Hello\r\nWorld"}], b"\xef\xbb\xbfHello\r\nWorld\r\n"),
    (b"a\nb\n", [{"oldText": "b\r\n", "newText": "B\r\n"}], b"a\nB\n"),
    (b"one\ntwo", [{"oldText": "two", "newText": ""}], b"one\n"),
    ("keep “smart”  \nalpha   \nbeta\nkeep − dash  \n".encode(), [{"oldText": "alpha\nbeta", "newText": "changed"}], "keep “smart”  \nchanged\nkeep − dash  \n".encode()),
    ("keep  \nＡ “quote”\u00a0− x   \n".encode(), [{"oldText": 'A "quote" - x', "newText": "changed"}], b"keep  \nchanged\n"),
    ("ﬁrst  \nＡ   \ntail\n".encode(), [{"oldText": "A", "newText": "a"}, {"oldText": "tail", "newText": "end"}], "ﬁrst  \na\nend\n".encode()),
], ids=["bom-crlf", "lf-with-crlf-args", "delete-no-final-newline", "unchanged-lines", "unicode-fuzzy", "mixed-fuzzy-exact"])
async def test_edit_preserves_bom_endings_and_untouched_fuzzy_lines(
    tmp_path: Path, original: bytes, edits: list[dict[str, str]], expected: bytes
) -> None:
    target = tmp_path / "note.txt"
    target.write_bytes(original)
    await create_edit_tool(tmp_path).execute("edit", {"path": "note.txt", "edits": edits}, None, lambda _: None)
    assert target.read_bytes() == expected


@pytest.mark.parametrize("arguments", [
    {"edits": None}, {"edits": 42}, {"edits": "bad json"}, {"edits": '"hello"'},
    {"edits": {}}, {"edits": [[{"oldText": "hello", "newText": "bye"}]]},
    {"oldText": "hello"}, {"oldText": "hello", "newText": None},
    {"edits": "bad json", "oldText": "hello", "newText": "bye"},
])
async def test_agent_rejects_malformed_compatible_arguments_without_writing(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    target = tmp_path / "note.txt"
    target.write_bytes(b"hello")
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_edit_tool(tmp_path)]),
        stream_fn=ScriptedStreamFn([
            lambda: tool_call_message("edit", {"path": "note.txt", **arguments}),
            lambda: text_message("done"),
        ]),
    ))
    await agent.prompt("Try editing")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert target.read_bytes() == b"hello"
    await agent.close()


@pytest.mark.parametrize(("content", "old_text"), [("aaa", "aa"), ("hello", "   ")])
async def test_edit_rejects_ambiguous_overlapping_occurrences_and_empty_fuzzy_target(
    tmp_path: Path, content: str, old_text: str
) -> None:
    target = tmp_path / "note.txt"
    target.write_text(content)
    with pytest.raises(ValueError, match="must be unique|must not be empty"):
        await create_edit_tool(tmp_path).execute("edit", {
            "path": "note.txt", "edits": [{"oldText": old_text, "newText": "changed"}],
        }, None, lambda _: None)
    assert target.read_text() == content


@pytest.mark.parametrize("first_kind,second_kind", [("edit", "write"), ("write", "edit"), ("edit", "edit")])
@pytest.mark.parametrize("alias_kind", ["file", "directory"])
@pytest.mark.parametrize("cancel", ["none", "signal", "task"])
async def test_edit_and_write_share_alias_coordination_until_cancelled_io_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    first_kind: str, second_kind: str, alias_kind: str, cancel: str,
) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    target = directory / "note.txt"
    target.write_text("initial")
    alias = tmp_path / "alias"
    alias.symlink_to(target if alias_kind == "file" else directory, target_is_directory=alias_kind == "directory")
    alias_path = alias if alias_kind == "file" else alias / "note.txt"
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
    first_tool = create_edit_tool(tmp_path) if first_kind == "edit" else create_write_tool(tmp_path)
    second_tool = create_edit_tool(directory) if second_kind == "edit" else create_write_tool(directory)
    controller = AbortController()
    first = asyncio.create_task(first_tool.execute("first", {
        "path": str(target), "content": "first", "edits": [{"oldText": "initial", "newText": "first"}],
    }, controller.signal, lambda _: None))
    tasks = [first]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel == "signal":
            controller.abort()
        elif cancel == "task":
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
        second = asyncio.create_task(second_tool.execute("second", {
            "path": str(alias_path), "content": "second", "edits": [{"oldText": "first", "newText": "second"}],
        }, None, lambda _: None))
        tasks.append(second)
        # An actual edit of a different file must finish while the first I/O is blocked.
        other = tmp_path / "other.txt"
        original_write(other, "other")
        await asyncio.wait_for(create_edit_tool(tmp_path).execute("other", {
            "path": "other.txt", "edits": [{"oldText": "other", "newText": "OTHER"}],
        }, None, lambda _: None), 2)
        assert other.read_bytes() == b"OTHER"
        assert not first.done()
        assert not second.done()
        assert "second" not in writes
    finally:
        release.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    if cancel == "signal":
        assert isinstance(outcomes[0], AbortError)
    elif cancel == "task":
        assert isinstance(outcomes[0], asyncio.CancelledError)
    else:
        assert not isinstance(outcomes[0], BaseException)
    assert not isinstance(outcomes[1], BaseException)
    assert writes == ["first", "OTHER", "second"]
    assert target.read_bytes() == b"second"


@pytest.mark.parametrize("cancel", ["signal", "task"])
async def test_cancelled_edit_read_holds_coordination_without_starting_its_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: str,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("initial")
    entered = asyncio.Event()
    release = threading.Event()
    original_read = Path.read_bytes
    loop = asyncio.get_running_loop()

    def controlled_read(path: Path) -> bytes:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", controlled_read)
    controller = AbortController()
    first = asyncio.create_task(create_edit_tool(tmp_path).execute("edit", {
        "path": "note.txt", "edits": [{"oldText": "initial", "newText": "first"}],
    }, controller.signal, lambda _: None))
    tasks = [first]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel == "signal":
            controller.abort()
        else:
            first.cancel()
        second = asyncio.create_task(create_write_tool(tmp_path).execute("write", {
            "path": "note.txt", "content": "second",
        }, None, lambda _: None))
        tasks.append(second)
        await asyncio.wait_for(create_write_tool(tmp_path).execute("other", {
            "path": "other.txt", "content": "other",
        }, None, lambda _: None), 2)
        assert not first.done() and not second.done()
        assert target.read_text() == "initial"
    finally:
        release.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(outcomes[0], AbortError if cancel == "signal" else asyncio.CancelledError)
    assert not isinstance(outcomes[1], BaseException)
    assert target.read_text() == "second"


async def test_edit_patch_represents_missing_final_newline(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_bytes(b"one")
    result = await create_edit_tool(tmp_path).execute("edit", {
        "path": "note.txt", "edits": [{"oldText": "one", "newText": "ONE"}],
    }, None, lambda _: None)
    assert result.details["patch"] == (
        "--- note.txt\n+++ note.txt\n@@ -1 +1 @@\n"
        "-one\n\\ No newline at end of file\n+ONE\n\\ No newline at end of file\n"
    )


async def test_adjacent_and_fuzzy_same_line_edits_write_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "note.txt"
    target.write_text("keep  \nＡbcdef  \ntail  \n")
    writes: list[str] = []
    original_write = Path.write_text

    def capture_write(path: Path, content: str, **kwargs: object) -> int:
        writes.append(content)
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", capture_write)
    await create_edit_tool(tmp_path).execute("edit", {
        "path": "@note.txt",
        "edits": [{"oldText": "Abc", "newText": "ABC"}, {"oldText": "def", "newText": "DEF"}],
    }, None, lambda _: None)
    assert writes == ["keep  \nABCDEF\ntail  \n"]
    assert target.read_bytes() == b"keep  \nABCDEF\ntail  \n"


@pytest.mark.parametrize("path", ["missing.txt", "directory"])
async def test_agent_file_errors_are_saveable_results(tmp_path: Path, path: str) -> None:
    (tmp_path / "directory").mkdir()
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_edit_tool(tmp_path)]),
        stream_fn=ScriptedStreamFn([
            lambda: tool_call_message("edit", {"path": path, "edits": [{"oldText": "a", "newText": "b"}]}),
            lambda: text_message("done"),
        ]),
    ))
    await agent.prompt("Try editing")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert path in result.content[0].text
    assert not (tmp_path / "missing.txt").exists()
    assert (tmp_path / "directory").is_dir()
    await agent.close()


async def test_edit_is_only_available_when_host_injects_it(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("initial")
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model()),
        stream_fn=ScriptedStreamFn([
            lambda: tool_call_message("edit", {"path": "note.txt", "edits": [{"oldText": "initial", "newText": "changed"}]}),
            lambda: text_message("done"),
        ]),
    ))
    await agent.prompt("Try editing")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert target.read_text() == "initial"
    await agent.close()


async def test_preaborted_edit_leaves_file_untouched(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("initial")
    controller = AbortController()
    controller.abort()
    with pytest.raises(AbortError):
        await create_edit_tool(tmp_path).execute("edit", {
            "path": "note.txt", "edits": [{"oldText": "initial", "newText": "changed"}],
        }, controller.signal, lambda _: None)
    assert target.read_text() == "initial"


async def test_agent_close_waits_for_edit_write_to_settle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "note.txt"
    target.write_text("initial")
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_write = Path.write_text

    def controlled_write(path: Path, content: str, **kwargs: object) -> int:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", controlled_write)
    stream = ScriptedStreamFn([lambda: tool_call_message("edit", {
        "path": "note.txt", "edits": [{"oldText": "initial", "newText": "committed"}],
    })])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_edit_tool(tmp_path)]),
        stream_fn=stream,
    ))
    prompt = asyncio.create_task(agent.prompt("Edit the note"))
    close: asyncio.Task | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        close = asyncio.create_task(agent.close())
        await asyncio.sleep(0.02)
        assert agent.state.is_busy and not close.done()
    finally:
        release.set()
        await asyncio.gather(prompt, *([close] if close else []))
    assert agent.state.is_closed
    assert stream.calls == 1
    assert target.read_bytes() == b"committed"


@pytest.mark.parametrize("cancel", ["signal", "task"])
async def test_cancelled_queued_edit_never_mutates_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: str,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("initial")
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_write = Path.write_text
    writes: list[str] = []

    def controlled_write(path: Path, content: str, **kwargs: object) -> int:
        writes.append(content)
        if content == "first":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", controlled_write)
    first = asyncio.create_task(create_write_tool(tmp_path).execute("first", {
        "path": "note.txt", "content": "first",
    }, None, lambda _: None))
    tasks = [first]
    controller = AbortController()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        middle = asyncio.create_task(create_edit_tool(tmp_path).execute("middle", {
            "path": "note.txt", "edits": [{"oldText": "first", "newText": "middle"}],
        }, controller.signal, lambda _: None))
        tasks.append(middle)
        await asyncio.sleep(0.02)
        if cancel == "signal":
            controller.abort()
        else:
            middle.cancel()
        last = asyncio.create_task(create_write_tool(tmp_path).execute("last", {
            "path": "note.txt", "content": "last",
        }, None, lambda _: None))
        tasks.append(last)
        await asyncio.wait_for(create_write_tool(tmp_path).execute("other", {
            "path": "other.txt", "content": "other",
        }, None, lambda _: None), 2)
        assert not last.done()
        assert "middle" not in writes and "last" not in writes
    finally:
        release.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(outcomes[1], AbortError if cancel == "signal" else asyncio.CancelledError)
    assert writes == ["first", "other", "last"]
    assert target.read_bytes() == b"last"


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\v", "\f", "\x85"])
async def test_patch_treats_unicode_and_control_separators_as_file_content(tmp_path: Path, separator: str) -> None:
    target = tmp_path / "note.txt"
    target.write_text(f"one{separator}two\n")
    result = await create_edit_tool(tmp_path).execute("edit", {
        "path": "note.txt", "edits": [{"oldText": "two", "newText": "TWO"}],
    }, None, lambda _: None)
    assert target.read_bytes() == f"one{separator}TWO\n".encode()
    assert result.details["patch"] == (
        f"--- note.txt\n+++ note.txt\n@@ -1 +1 @@\n-one{separator}two\n+one{separator}TWO\n"
    )


@pytest.mark.parametrize("ending", ["   ", "\t", "\n", "   \n"])
async def test_fuzzy_edit_preserves_untouched_terminal_whitespace_line(tmp_path: Path, ending: str) -> None:
    target = tmp_path / "note.txt"
    target.write_text(f"Ａ\n{ending}")
    await create_edit_tool(tmp_path).execute("edit", {
        "path": "note.txt", "edits": [{"oldText": "A", "newText": "B"}],
    }, None, lambda _: None)
    assert target.read_bytes() == f"B\n{ending}".encode()
