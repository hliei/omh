"""Trusted project resources, discovery tiers and system-input acceptance.

These tests drive the installed host through public application assembly: the
host resolves configuration, trust, resources and system inputs, then a real
``AgentSessionRuntime`` loads them into an offline Agent whose captured request
shows the effective named sections and expanded input.
"""

from __future__ import annotations

import json
from pathlib import Path

from omh.llm.utils.transcript import get_current_system_prompt
from support import OfflineStream

from coding_agent import AgentSessionRuntime, CodingAgentHost
from coding_agent.resources import git_ancestor_directories
from coding_agent.trust import TrustStore


def skill(path: Path, *, name: str, description: str, body: str = "Review carefully", hidden: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n"
        f"disable-model-invocation: {str(hidden).lower()}\n---\n{body}"
    )
    return path


def template(path: Path, *, body: str = "Template body $1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def write_settings(agent_dir: Path, values: object) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text(json.dumps(values))


def mark_git_root(cwd: Path) -> None:
    """Stop automatic ``.agents`` discovery at the project root."""
    (cwd / ".git").mkdir(parents=True, exist_ok=True)


async def assemble(host: CodingAgentHost, *, text: str = "work"):
    """Select, assemble and prompt an offline session; return it with its stream."""
    selection = host.select_new()
    assert selection.ready, selection.diagnostics
    options = host.build_options(selection)
    stream = OfflineStream()
    options.stream_fn = stream
    session = await AgentSessionRuntime(options).new_session()
    await session.prompt(text)
    return session, stream


def prompt_of(stream: OfflineStream) -> str:
    return get_current_system_prompt(stream.requests[-1][1].messages)


def user_texts(stream: OfflineStream) -> list[str]:
    return [
        message.content if isinstance(message.content, str) else "".join(
            block.text for block in message.content if block.type == "text"
        )
        for message in stream.requests[-1][1].messages if message.role == "user"
    ]


# --------------------------------------------------------------------------- #
# Five-tier priority
# --------------------------------------------------------------------------- #


async def test_skill_tiers_order_explicit_project_auto_global_with_first_wins(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    home = tmp_path / "home"
    explicit = skill(tmp_path / "cli" / "review" / "SKILL.md", name="review", description="cli tier")
    project_config = skill(
        tmp_path / "project-config" / "review" / "SKILL.md", name="review", description="project config tier",
    )
    skill(cwd / ".omh" / "skills" / "review" / "SKILL.md", name="review", description="project auto tier")
    global_config = skill(
        tmp_path / "global-config" / "review" / "SKILL.md", name="review", description="global config tier",
    )
    skill(agent_dir / "skills" / "review" / "SKILL.md", name="review", description="global auto tier")
    skill(home / ".agents" / "skills" / "review" / "SKILL.md", name="review", description="user auto tier")
    (cwd / ".omh").mkdir(exist_ok=True)
    (cwd / ".omh" / "settings.json").write_text(json.dumps({"skills": [str(project_config)]}))
    write_settings(agent_dir, {"skills": [str(global_config)]})

    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=True,
        explicit_skills=(str(explicit),),
    )
    session, _ = await assemble(host)

    assert [item.description for item in session.resources.skills] == ["cli tier"]
    assert session.resources.skills[0].source == "explicit"
    collisions = [item for item in session.resources.diagnostics if item.reason == "collision"]
    assert len(collisions) == 4
    assert {item.loser.source for item in collisions if item.loser is not None} == {
        "project-config", "project-auto", "global-auto",
    }
    # The project array replaced the global array; the covered global path never revives.
    assert "global config tier" not in str(session.resources.diagnostics)


async def test_global_config_resource_still_loads_when_no_project_array_replaces_it(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    global_config = skill(
        tmp_path / "global-config" / "review" / "SKILL.md", name="review", description="global config tier",
    )
    write_settings(agent_dir, {"skills": [str(global_config)]})
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home", project_trusted=True)
    session, _ = await assemble(host)
    assert session.resources.skills[0].source == "global-config"
    assert session.resources.skills[0].description == "global config tier"


async def test_project_auto_agents_skills_walk_to_nearest_git_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "packages" / "app"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    agent_dir = tmp_path / "agent"
    skill(root / ".agents" / "skills" / "review" / "SKILL.md", name="review", description="repo scoped")
    skill(nested / ".agents" / "skills" / "review" / "SKILL.md", name="review", description="nested scoped")
    skill(nested / ".omh" / "skills" / "review" / "SKILL.md", name="review", description="project config dir")
    host = CodingAgentHost(
        startup_dir=nested, agent_dir=agent_dir, home=tmp_path / "home", project_trusted=True,
    )
    session, _ = await assemble(host)
    assert session.resources.skills[0].description == "project config dir"
    assert session.resources.skills[0].source == "project-auto"
    losers = [
        item.loser.description for item in session.resources.diagnostics
        if item.reason == "collision" and item.loser is not None
    ]
    assert losers == ["nested scoped", "repo scoped"]


async def test_template_tiers_and_same_name_collisions(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    explicit = template(tmp_path / "cli" / "greet.md", body="cli $1")
    project_config = template(tmp_path / "project-config" / "greet.md", body="project config $1")
    template(cwd / ".omh" / "prompts" / "greet.md", body="project auto $1")
    template(agent_dir / "prompts" / "greet.md", body="global auto $1")
    (cwd / ".omh").mkdir(exist_ok=True)
    (cwd / ".omh" / "settings.json").write_text(json.dumps({"prompts": [str(project_config)]}))

    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home", project_trusted=True,
        explicit_templates=(str(explicit),),
    )
    selection = host.select_new()
    options = host.build_options(selection)
    stream = OfflineStream()
    options.stream_fn = stream
    session = await AgentSessionRuntime(options).new_session()

    assert session.resources.templates[0].source == "explicit"
    assert session.resources.templates[0].name == "greet"
    collisions = [item for item in session.resources.diagnostics if item.reason == "collision"]
    assert [item.loser.source for item in collisions] == ["project-config", "project-auto", "global-auto"]
    await session.prompt('/greet "two words"')
    assert user_texts(stream)[-1] == "cli two words"


# --------------------------------------------------------------------------- #
# Trust gating
# --------------------------------------------------------------------------- #


def build_trusted_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    home = tmp_path / "home"
    (cwd / ".omh").mkdir()
    (cwd / ".omh" / "settings.json").write_text(json.dumps({"defaultModel": "deepseek/deepseek-v4-pro"}))
    (cwd / ".omh" / "SYSTEM.md").write_text("project system")
    (cwd / ".omh" / "APPEND_SYSTEM.md").write_text("project append")
    skill(cwd / ".omh" / "skills" / "review" / "SKILL.md", name="review", description="project skill")
    (cwd / "AGENTS.md").write_text("project instructions")
    write_settings(agent_dir, {"defaultModel": "deepseek/deepseek-flash"})
    (agent_dir / "SYSTEM.md").write_text("global system")
    (agent_dir / "APPEND_SYSTEM.md").write_text("global append")
    skill(agent_dir / "skills" / "review" / "SKILL.md", name="review", description="global skill")
    (agent_dir / "AGENTS.md").write_text("global instructions")
    return cwd, agent_dir, home


async def test_unknown_trust_skips_project_layer_but_keeps_agents_and_global(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=home)
    selection = host.select_new()
    assert selection.ready
    assert selection.model is not None and selection.model.id == "deepseek-flash"
    assert host.needs_trust_decision is True
    assert host.project_trusted is False
    untrusted = next(diagnostic for diagnostic in selection.diagnostics if diagnostic.reason == "untrusted")
    assert not untrusted.blocking
    assert "sandbox" not in untrusted.message and "approval" not in untrusted.message

    session, stream = await assemble(host)
    prompt = prompt_of(stream)
    # AGENTS ancestor inheritance is independent of trust.
    assert "global instructions" in prompt and "project instructions" in prompt
    assert "global system" in prompt and "project system" not in prompt
    assert "global append" in prompt and "project append" not in prompt
    assert [item.description for item in session.resources.skills] == ["global skill"]


async def test_approved_trust_uses_project_configuration_system_and_resources(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=True)
    session, stream = await assemble(host)
    prompt = prompt_of(stream)
    assert "project system" in prompt and "global system" not in prompt
    assert "project append" in prompt and "global append" not in prompt
    assert [item.description for item in session.resources.skills] == ["project skill"]
    assert host.needs_trust_decision is False


async def test_remembered_trust_and_one_run_flags(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=home)
    assert host.select_new().model is not None
    assert host.select_new().model.id == "deepseek-flash"
    host.remember_trust("approved")
    assert (agent_dir / "trust.json").is_file()
    remembered = host.select_new()
    assert remembered.model is not None and remembered.model.id == "deepseek-v4-pro"
    assert host.needs_trust_decision is False

    reopened = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=home)
    assert reopened.select_new().model is not None
    assert reopened.select_new().model.id == "deepseek-v4-pro"

    denied = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=None, no_approve=True,
    )
    assert denied.select_new().model is not None and denied.select_new().model.id == "deepseek-flash"
    host.remember_trust("denied")
    approved = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=None, approve=True,
    )
    assert approved.select_new().model is not None and approved.select_new().model.id == "deepseek-v4-pro"


def test_trust_store_reports_invalid_entries_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    path.write_text("{broken")
    store = TrustStore(path)
    assert store.decision(tmp_path) is None
    assert any(diagnostic.reason == "invalid-json" for diagnostic in store.diagnostics)

    store = TrustStore(tmp_path / "other.json")
    store.remember(tmp_path, "approved")
    assert store.decision(tmp_path) == "approved"
    assert TrustStore(tmp_path / "other.json").decision(tmp_path) == "approved"

    malformed = TrustStore(tmp_path / "bad.json")
    (tmp_path / "bad.json").write_text(json.dumps({"projects": {"/x": "sometimes"}}))
    assert malformed.decision(tmp_path) is None
    assert any(diagnostic.reason == "invalid-value" for diagnostic in malformed.diagnostics)


# --------------------------------------------------------------------------- #
# Discovery switches
# --------------------------------------------------------------------------- #


async def test_no_skills_and_no_prompt_templates_keep_explicit_and_configured(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    skill(agent_dir / "skills" / "auto" / "SKILL.md", name="auto", description="automatic skill")
    template(agent_dir / "prompts" / "auto.md", body="automatic template")
    explicit = skill(tmp_path / "cli" / "explicit" / "SKILL.md", name="explicit", description="explicit skill")
    configured = skill(tmp_path / "config" / "configured" / "SKILL.md", name="configured", description="configured skill")
    configured_template = template(tmp_path / "config" / "configured.md", body="configured template")
    write_settings(agent_dir, {"skills": [str(configured)], "prompts": [str(configured_template.parent)]})

    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home",
        explicit_skills=(str(explicit),), no_skills=True, no_prompt_templates=True,
    )
    session, _ = await assemble(host)
    assert sorted(item.name for item in session.resources.skills) == ["configured", "explicit"]
    assert sorted(item.name for item in session.resources.templates) == ["configured"]


async def test_agents_follow_root_to_cwd_order_independently_of_trust(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "packages" / "app"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "AGENTS.md").write_text("repo instructions")
    (nested / "AGENTS.md").write_text("nested instructions")
    agent_dir = tmp_path / "agent"
    host = CodingAgentHost(startup_dir=nested, agent_dir=agent_dir, home=tmp_path / "home")
    session, stream = await assemble(host)
    assert [Path(item.path).name for item in session.resources.context_files] == ["AGENTS.md", "AGENTS.md"]
    prompt = prompt_of(stream)
    assert prompt.index("repo instructions") < prompt.index("nested instructions")


def test_git_ancestor_directories_stop_at_git_root_or_filesystem_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    walked = git_ancestor_directories(nested)
    assert walked[:3] == (nested, nested.parent, root)
    assert walked[-1] == Path("/")
    (root / ".git").mkdir()
    assert git_ancestor_directories(nested) == (nested, nested.parent, root)


async def test_explicit_resources_load_for_an_untrusted_project(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    explicit = skill(tmp_path / "cli" / "extra" / "SKILL.md", name="extra", description="explicit skill")
    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, explicit_skills=(str(explicit),),
    )
    session, _ = await assemble(host)
    assert "explicit skill" in [item.description for item in session.resources.skills]


async def test_host_composed_input_expands_skill_before_template_and_rereads_skill(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    body = skill(
        agent_dir / "skills" / "review" / "SKILL.md", name="review", description="review skill", body="first body",
    )
    template(agent_dir / "prompts" / "skill:review.md", body="template must not win")
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home")
    selection = host.select_new()
    options = host.build_options(selection)
    stream = OfflineStream()
    options.stream_fn = stream
    session = await AgentSessionRuntime(options).new_session()
    body.write_text(
        "---\nname: review\ndescription: review skill\ndisable-model-invocation: false\n---\nsecond body"
    )
    await session.prompt("/skill:review explain")
    expanded = user_texts(stream)[-1]
    assert "second body" in expanded and "first body" not in expanded
    assert "template must not win" not in expanded
    assert expanded.endswith("explain")
    assert session.resources.skills[0].description == "review skill"


async def test_no_context_files_disables_agents_only(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=True, no_context_files=True,
    )
    session, stream = await assemble(host)
    prompt = prompt_of(stream)
    assert session.resources.context_files == ()
    assert "project instructions" not in prompt and "global instructions" not in prompt
    assert [item.description for item in session.resources.skills] == ["project skill"]
    assert "project system" in prompt


async def test_disable_model_invocation_hides_catalog_but_keeps_explicit_invocation(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    skill(agent_dir / "skills" / "hidden" / "SKILL.md", name="hidden", description="hidden skill", hidden=True)
    skill(agent_dir / "skills" / "visible" / "SKILL.md", name="visible", description="visible skill")
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home")
    session, stream = await assemble(host)
    prompt = prompt_of(stream)
    assert "visible skill" in prompt and "hidden skill" not in prompt
    await session.prompt("/skill:hidden now")
    expanded = user_texts(stream)[-1]
    assert "<skill name=\"hidden\"" in expanded and "Review carefully" in expanded and expanded.endswith("now")


async def test_skill_catalog_requires_an_executable_reader(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    skill(agent_dir / "skills" / "visible" / "SKILL.md", name="visible", description="visible skill")
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home")
    selection = host.select_new(tools=("write",))
    options = host.build_options(selection)
    assert options.tools == ("write",)
    stream = OfflineStream()
    options.stream_fn = stream
    session = await AgentSessionRuntime(options).new_session()
    await session.prompt("work")
    assert "visible skill" not in prompt_of(stream)


# --------------------------------------------------------------------------- #
# System and append inputs
# --------------------------------------------------------------------------- #


async def test_explicit_system_and_ordered_append_win_over_project_and_global(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    first = tmp_path / "append.md"
    first.write_text("file append")
    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=True,
        explicit_system_prompt="explicit base",
        append_system=(("text", "text append"), ("file", str(first)), ("text", "last append")),
    )
    session, stream = await assemble(host)
    prompt = prompt_of(stream)
    assert "explicit base" in prompt and "project system" not in prompt and "global system" not in prompt
    assert "You are a coding assistant" not in prompt and "<rules>" not in prompt
    # A custom base keeps project context, cwd and the eligible skill catalog.
    assert "project instructions" in prompt
    assert cwd.as_posix() in prompt
    assert "project skill" in prompt
    append = prompt[prompt.index("<addendum>"):]
    assert append.index("text append") < append.index("file append") < append.index("last append")
    assert "project append" not in prompt and "global append" not in prompt


async def test_explicit_empty_system_prompt_replaces_the_base(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=True,
        explicit_system_prompt="",
    )
    _, stream = await assemble(host)
    prompt = prompt_of(stream)
    assert "project system" not in prompt
    assert "You are a coding assistant" not in prompt
    assert "project instructions" in prompt


def test_unreadable_project_system_file_falls_back_to_global(tmp_path: Path) -> None:
    cwd, agent_dir, home = build_trusted_project(tmp_path)
    target = cwd / ".omh" / "SYSTEM.md"
    target.unlink()
    target.mkdir()
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=home, project_trusted=True)
    selection = host.select_new()
    assert selection.ready
    assert any(
        diagnostic.reason == "recoverable" and diagnostic.source == "project"
        for diagnostic in selection.diagnostics
    )
    options = host.build_options(selection)
    assert options.custom_prompt == "global system"


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


async def test_template_using_a_builtin_command_name_is_diagnosed(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    mark_git_root(cwd)
    agent_dir = tmp_path / "agent"
    template(agent_dir / "prompts" / "help.md", body="user help text")
    host = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, home=tmp_path / "home")
    session, _ = await assemble(host)
    conflict = next(
        item for item in session.resources.diagnostics if item.reason == "builtin-conflict"
    )
    assert conflict.path.endswith("help.md")
    assert "help" in conflict.message
