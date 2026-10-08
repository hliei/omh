"""Resource reload batch, failure retention and conflict diagnostics.

These tests drive the public application boundary (``AgentSessionRuntime`` /
``AgentSession`` resource options) and the installed ``omh`` PTY. They cover
ticket 24's observable behavior: an idle reload prepares AGENTS/system/skills/
templates/tools as one batch, a fatal preparation failure keeps the previous
snapshot and options, recoverable read/parse problems publish a usable set with
source and winner/loser diagnostics, accepted input is not re-expanded and no
inference request is sent.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from omh.agent import PromptTemplateSource, SkillSource
from support import OfflineStream, model
from test_interactive import InteractiveSession, provider_env, sends, settle

from coding_agent import AgentSessionRuntime, CodingAgentOptions
from coding_agent.agent_session import RELOAD_OPTION_FIELDS


def write_skill(path: Path, *, name: str, body: str = "Skill body", hidden: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {name} guidance\n"
        f"disable-model-invocation: {str(hidden).lower()}\n---\n{body}"
    )
    return path


def user_texts(stream: OfflineStream) -> list[str]:
    return [
        message.content if isinstance(message.content, str) else "".join(
            block.text for block in message.content if block.type == "text"
        )
        for message in stream.requests[-1][1].messages if message.role == "user"
    ]


# --------------------------------------------------------------------------- #
# Application boundary: one prepared batch, shared live options
# --------------------------------------------------------------------------- #


async def test_reload_after_clone_accepts_changed_sources_and_keeps_history(tmp_path: Path) -> None:
    """A derived session shares live resource options, so reload reaches it."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "greet.md").write_text("first template $1")
    (second / "greet.md").write_text("second template $1")
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        template_sources=(PromptTemplateSource(first),), session_dir=tmp_path / "sessions",
    ))
    session = await runtime.new_session()
    await session.prompt("/greet one")
    assert user_texts(stream)[-1] == "first template one"

    clone = await runtime.clone_session()
    history = clone.agent.history
    sent = len(stream.requests)

    runtime.options.template_sources = (PromptTemplateSource(second),)
    await runtime.reload_resources()
    assert len(stream.requests) == sent
    assert clone.agent.history == history
    assert clone.resources.templates[0].file_path == str(second / "greet.md")
    await clone.prompt("/greet two")
    assert user_texts(stream)[-1] == "second template two"


async def test_fatal_preparation_keeps_previous_options_and_successful_batch_adopts(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "greet.md").write_text("first $1")
    (second / "greet.md").write_text("second $1")
    stream = OfflineStream()
    options = CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        template_sources=(PromptTemplateSource(first),),
    )
    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session()
    await session.prompt("/greet a")
    snapshot = session.resources
    sections = session.agent.state.system_sections
    sent = len(stream.requests)

    fatal = replace(options, template_sources=(PromptTemplateSource(second),),
                    context_dirs=("invalid\0directory",))
    with pytest.raises(ValueError):
        await session.reload_resources(fatal)
    # Nothing published on a fatal batch: old snapshot, sections and options stay.
    assert session.resources == snapshot
    assert session.agent.state.system_sections == sections
    assert runtime.options.template_sources == (PromptTemplateSource(first),)
    assert runtime.options.context_dirs == ()
    assert len(stream.requests) == sent
    await session.prompt("/greet b")
    assert user_texts(stream)[-1] == "first b"

    recoverable = replace(options, template_sources=(PromptTemplateSource(second),))
    resources = await session.reload_resources(recoverable)
    assert resources.templates[0].file_path == str(second / "greet.md")
    # Adoption is part of the same successful batch.
    assert runtime.options.template_sources == (PromptTemplateSource(second),)
    for name in RELOAD_OPTION_FIELDS:
        assert getattr(runtime.options, name) == getattr(recoverable, name)
    await session.prompt("/greet c")
    assert user_texts(stream)[-1] == "second c"
    assert len(stream.requests) == sent + 2


async def test_reload_reports_recoverable_parse_error_and_publishes_usable_resources(tmp_path: Path) -> None:
    good = write_skill(tmp_path / "skills" / "review" / "SKILL.md", name="review")
    broken = tmp_path / "broken" / "SKILL.md"
    broken.parent.mkdir()
    broken.write_text("---\ninvalid frontmatter line\n---\nbody")
    stream = OfflineStream()
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("read",),
        skill_sources=(SkillSource(good), SkillSource(broken.parent)),
    )).new_session()
    resources = await session.reload_resources()
    assert [skill.name for skill in resources.skills] == ["review"]
    assert any(item.reason == "parse_error" and item.path == str(broken) for item in resources.diagnostics)
    await session.prompt("work")
    assert "review guidance" in str(stream.requests[-1][1].messages)
    assert len(stream.requests) == 1


# --------------------------------------------------------------------------- #
# Installed interactive command: conflicts, busy guards and settings acceptance
# --------------------------------------------------------------------------- #


def test_reload_and_session_show_source_winner_and_loser_conflicts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write_skill(home / ".omh" / "agent" / "skills" / "review" / "SKILL.md", name="review")
    project = tmp_path / "project"
    project.mkdir()
    explicit_skill = write_skill(project / "extra" / "SKILL.md", name="review", body="Explicit review body")
    templates = project / "templates"
    templates.mkdir()
    (templates / "help.md").write_text("template that must not shadow the built-in help")
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline", "--no-session",
        "--skill", str(explicit_skill.parent), "--prompt-template", str(templates),
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/reload\r")
        session.wait_for("resources reloaded; next new prompt; accepted inputs unchanged")
        session.send(b"/session\r")
        session.wait_for("resources context files")
        offset = session.visible().index("current ")
        visible = session.visible()[offset:]
        assert "winner review (explicit)" in visible
        assert "loser review (global-auto)" in visible
        assert "source explicit" in visible
        assert "builtin-conflict" in visible
        assert "resources context files none; skills explicit 1; templates explicit 1" in visible
        # The built-in still wins for its own name; the template is only diagnosed.
        session.send(b"/help help\r")
        session.wait_for("/help: Show the available commands")
        assert sends(home) == []
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_reload_refuses_while_model_or_shell_is_busy_without_aborting(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline", "--no-session",
        home=home, cwd=project, env=provider_env(home, "settings-batch"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"go\r")
        session.wait_for("settings-ready")
        session.send(b"/reload\r")
        session.wait_for("is busy; wait or cancel")
        # The refused command never aborts the running model.
        (project / "release-settings").touch()
        session.wait_for("settings batch done")
        session.wait_for("phase input", after=session.visible().index("settings batch done"))

        session.send(b"!printf shell-gate; while [ ! -f shell-release ]; do sleep 0.02; done; printf done\r")
        session.wait_for("shell running")
        session.send(b"/reload\r")
        session.wait_for("is busy; wait or cancel")
        (project / "shell-release").touch()
        session.wait_for("shell settled")
        session.send(b"/reload\r")
        session.wait_for("resources reloaded; next new prompt; accepted inputs unchanged")
        # Only the two model calls; reload itself never sends a request.
        assert sends(home) == ["dialogue", "dialogue"]
        session.send(b"\x04")
        assert session.finish() == 0
        assert session.restored()
    finally:
        session.close()


def test_reload_accepts_settings_skill_defaults_and_refreshes_completion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    agent = home / ".omh" / "agent"
    first = write_skill(tmp_path / "first" / "alpha" / "SKILL.md", name="alpha")
    second = write_skill(tmp_path / "second" / "beta" / "SKILL.md", name="beta")
    agent.mkdir(parents=True)
    settings = agent / "settings.json"
    settings.write_text(json.dumps({"skills": [str(first.parent)]}))
    project = tmp_path / "project"
    project.mkdir()
    session = InteractiveSession(
        "--no-approve", "--no-context-files", "--api-key", "offline", "--no-session",
        "--no-skills", "--no-prompt-templates",
        home=home, cwd=project, env=provider_env(home, "echo"),
    )
    try:
        session.wait_for("phase input")
        session.send(b"/session\r")
        session.wait_for("skills global-config 1")
        settings.write_text(json.dumps({"skills": [str(second.parent)]}))
        session.send(b"/reload\r")
        session.wait_for("resources reloaded; next new prompt; accepted inputs unchanged")
        session.send(b"/skill:")
        session.send(b"\t")
        session.wait_for("/skill:beta")
        visible = session.visible()
        assert "/skill:alpha" not in visible[visible.index("resources reloaded"):]
        session.send(b"\x1b")
        settle(session)
        session.send(b"\x03")
        settle(session)
        assert sends(home) == []
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait()
        session.close()
