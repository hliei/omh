"""Trust, attachment content and saving work together through the common host."""

from __future__ import annotations

from pathlib import Path

import pytest
from omh.llm.types import UserMessage
from omh.llm.utils.transcript import get_current_system_prompt
from PIL import Image
from support import OfflineStream

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentHost,
    attachment_images,
    count_history_images,
    decode_history,
    read_file_attachment,
)


@pytest.mark.parametrize("trusted", [True, False])
@pytest.mark.parametrize("memory", [True, False])
async def test_host_preserves_resources_and_image_content_across_storage_modes(
    tmp_path: Path, trusted: bool, memory: bool,
) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / ".git").mkdir()
    (cwd / "AGENTS.md").write_text("Keep project instructions")
    project_prompts = cwd / ".omh" / "prompts"
    project_prompts.mkdir(parents=True)
    (project_prompts / "inspect.md").write_text("Project inspect $1")
    (cwd / ".omh" / "SYSTEM.md").write_text("Project base")
    agent_dir = tmp_path / "agent"
    global_prompts = agent_dir / "prompts"
    global_prompts.mkdir(parents=True)
    (global_prompts / "inspect.md").write_text("Global inspect $1")
    (agent_dir / "SYSTEM.md").write_text("Global base")
    source = cwd / "picture.png"
    Image.new("RGB", (4, 2), (10, 20, 30)).save(source)
    images = attachment_images([read_file_attachment("picture.png", cwd=cwd)])
    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home",
        approve=trusted, no_approve=not trusted, no_session=memory,
        api_key="offline-key",
    )
    selection = host.select_new()
    assert selection.ready, selection.diagnostics
    options = host.build_options(selection)
    stream = OfflineStream()
    options.stream_fn = stream
    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session()
    assert session.save_state == "pending"
    assert not (agent_dir / "sessions").exists()

    await session.prompt("/inspect picture", images=images)

    expected_scope = "Project" if trusted else "Global"
    system = get_current_system_prompt(stream.requests[0][1].messages)
    assert f"{expected_scope} base" in system
    assert "Keep project instructions" in system
    submitted = stream.requests[0][1].messages[-1]
    assert isinstance(submitted, UserMessage)
    assert submitted.content[0].text == f"{expected_scope} inspect picture"
    assert submitted.content[1:] == images
    history = session.agent.history
    assert count_history_images(history) == 1
    source.unlink()

    if memory:
        assert session.save_mode == "memory"
        assert session.path is None
        assert session.save_state == "pending"
        assert not (agent_dir / "sessions").exists()
        assert decode_history(await session.export()).history == history
    else:
        assert session.save_mode == "auto"
        assert session.save_state == "saved"
        assert session.path is not None
        assert session.path.parent.parent == agent_dir / "sessions"
        assert decode_history(session.path.read_bytes()).history == history
        reopened = await runtime.open_session(session.path)
        assert reopened.agent.history == history
        await reopened.prompt("Continue without the source file")
        assert count_history_images(reopened.agent.history) == 1
