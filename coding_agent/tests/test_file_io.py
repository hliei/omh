import json

import pytest
from support import OfflineStream, model

from coding_agent import (
    CodingAgentOptions,
    CodingAgentRuntime,
    decode_history,
    encode_history,
)


@pytest.mark.parametrize("tail", [b"", b'\n{broken', b'\n\xff{invalid'])
async def test_unterminated_tail_survives_open_append_reopen(tmp_path, tail):
    path = tmp_path / "history.jsonl"
    stream = OfflineStream()
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=(), session_file=path)
    original = await CodingAgentRuntime(options).new_session()
    await original.prompt("first")
    raw = path.read_bytes().rstrip(b"\n") + tail
    path.write_bytes(raw)
    opened = await CodingAgentRuntime(options).open_session(path)
    assert path.read_bytes() == raw + b"\n"
    assert opened.agent.history == original.agent.history
    await opened.prompt("second")
    assert path.read_bytes().startswith(raw + b"\n")
    reopened = await CodingAgentRuntime(options).open_session(path)
    assert reopened.agent.history == opened.agent.history
    assert len(stream.requests) == 2


async def test_bad_json_lines_are_skipped_but_missing_relations_fail(tmp_path):
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=())
    runtime = CodingAgentRuntime(options)
    original = await runtime.new_session()
    await original.prompt("first")
    text = await original.export()
    path = tmp_path / "bad.jsonl"
    path.write_text('{bad\n\n' + text.replace('\n', '\nnot json\n'))
    reopened = await CodingAgentRuntime(options).open_session(path)
    assert reopened.agent.history == original.agent.history
    lines = text.splitlines()
    # Replace an actual parent record with broken JSON; do not hide the missing relationship.
    lines[2] = "{missing record"
    path.write_text("\n".join(lines))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        await CodingAgentRuntime(options).open_session(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["version", "format", "payload", "record", "parent", "nonfinite", "header"])
async def test_valid_json_semantic_errors_fail_without_repairing_file(tmp_path, change):
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=())
    session = await CodingAgentRuntime(options).new_session()
    await session.prompt("first")
    lines = [json.loads(line) for line in (await session.export()).splitlines()]
    if change == "version":
        lines[0]["version"] = 99
    elif change == "format":
        lines[0]["format"] = "other"
    elif change == "header":
        lines[0]["id"] = 42
    elif change == "payload":
        lines[-1]["message"]["content"] = 42
    elif change == "record":
        lines[-1]["type"] = "future"
    elif change == "parent":
        lines[-1]["parentId"] = "missing"
    elif change == "nonfinite":
        lines[-1]["message"]["usage"]["cost"]["input"] = float("nan")
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        await CodingAgentRuntime(options).open_session(path)
    assert path.read_bytes() == before


async def test_explicit_save_and_export_before_first_prompt(tmp_path):
    from omh.agent import CustomAgentMessage

    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="new.jsonl")
    runtime = CodingAgentRuntime(options)
    session = await runtime.new_session(display_name="before prompt")
    await session.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="context"))
    assert session.save_state == "pending"
    assert not session.path.exists()
    exported = await runtime.export_session("copy.jsonl")
    assert decode_history(exported).history == session.agent.history
    assert session.save_state == "pending"
    await runtime.save_session()
    assert session.save_state == "saved"
    assert decode_history(session.path.read_bytes()).history == session.agent.history
    await session.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="later"))
    assert decode_history(session.path.read_bytes()).history == session.agent.history


async def test_first_save_includes_seed_history_and_custom_records(tmp_path):
    from omh.agent import AgentInitialState, AgentOptions, CustomAgentMessage
    from omh.llm.types import UserMessage

    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(), session_file="new.jsonl",
        agent_options=AgentOptions(initial_state=AgentInitialState(messages=[UserMessage(content="seed", timestamp=1)])))
    session = await CodingAgentRuntime(options).new_session()
    initial = session.agent.history
    assert not session.path.exists()
    await session.agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="new"))
    assert session.save_state == "saved"
    decoded = decode_history(session.path.read_bytes())
    assert decoded.history == session.agent.history
    assert decoded.history.entries[:len(initial.entries)] == initial.entries
    assert encode_history(decoded.history, cwd=str(tmp_path))


@pytest.mark.parametrize("field", ["role", "type"])
async def test_missing_sdk_discriminator_is_not_inferred(tmp_path, field):
    from omh.llm.types import TextContent, UserMessage

    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=())
    session = await CodingAgentRuntime(options).new_session()
    await session.prompt(UserMessage(content=[TextContent(text="input")], timestamp=1))
    records = [json.loads(line) for line in (await session.export()).splitlines()]
    user = next(entry["message"] for entry in records[1:] if entry.get("message", {}).get("role") == "user")
    if field == "role":
        del user["role"]
    else:
        del user["content"][0]["type"]
    with pytest.raises(ValueError):
        decode_history("\n".join(json.dumps(record) for record in records))
