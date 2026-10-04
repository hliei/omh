"""Host-selected skills exercised through the public helpers and Agent."""

import os
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    SkillSource,
    create_bash_tool,
    create_read_tool,
    create_write_tool,
    expand_skill_command,
    format_skills_for_prompt,
    load_skills,
    load_skills_from_dir,
)
from omh.llm.types import SystemMessage, TextContent, ToolResultMessage, UserMessage
from omh.llm.utils.transcript import get_current_system_prompt
from tests.agent.test_agent_tools import (
    ScriptedStreamFn,
    make_model,
    text_message,
    tool_call_message,
)


def write_skill(path: Path, metadata: str = "description: Useful instructions", body: str = "Follow these steps.") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{metadata}\n---\n{body}\n", encoding="utf-8")
    return path


def test_root_skill_takes_precedence_and_stops_recursion(tmp_path: Path) -> None:
    root = tmp_path / "root"
    path = write_skill(root / "SKILL.md")
    write_skill(root / "loose.md", "name: loose\ndescription: Loose")
    write_skill(root / "nested" / "SKILL.md")

    result = load_skills_from_dir(root, source="project")

    assert [(s.name, s.description, s.file_path, s.base_dir, s.source) for s in result.skills] == [
        ("root", "Useful instructions", str(path), str(root), "project"),
    ]
    assert not result.diagnostics


def test_ordered_sources_report_collisions_and_deduplicate_real_files(tmp_path: Path) -> None:
    winner = write_skill(tmp_path / "global" / "SKILL.md", "name: same\ndescription: First")
    loser = write_skill(tmp_path / "project" / "SKILL.md", "name: same\ndescription: Second")
    alias = tmp_path / "alias.md"
    alias.symlink_to(winner)
    (tmp_path / "global" / "loop").symlink_to(tmp_path / "global", target_is_directory=True)

    result = load_skills([
        SkillSource("global", "global"), SkillSource(alias), SkillSource("project", "project"),
    ], cwd=tmp_path)

    assert [(s.description, s.source) for s in result.skills] == [("First", "global")]
    assert len(result.diagnostics) == 1
    collision = result.diagnostics[0]
    assert collision.reason == "collision" and collision.source == "project"
    assert collision.winner == result.skills[0]
    assert collision.loser.file_path == str(loser)
    assert collision.path == str(loser)


def test_symlink_directory_cycles_terminate_and_linked_skills_are_discovered(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    target = tmp_path / "external"
    write_skill(target / "SKILL.md")
    (root / "a-link").symlink_to(target, target_is_directory=True)
    (root / "b-link").symlink_to(target, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    (root / "broken").symlink_to(tmp_path / "missing")

    result = load_skills_from_dir(root)

    assert [s.file_path for s in result.skills] == [str(root / "a-link" / "SKILL.md")]
    assert any(d.reason == "read_error" and Path(d.path).name == "broken" for d in result.diagnostics)


@pytest.mark.parametrize("ignore_name", [".gitignore", ".ignore", ".fdignore"])
def test_discovery_obeys_ignore_and_only_loads_loose_markdown_at_root(tmp_path: Path, ignore_name: str) -> None:
    root = tmp_path / "skills"
    write_skill(root / "loose.md", "name: loose\ndescription: Loose")
    write_skill(root / "notes.md", "title: Notes")
    write_skill(root / "kept" / "SKILL.md")
    write_skill(root / "kept" / "loose.md", "name: nested-loose\ndescription: Hidden by skill root")
    write_skill(root / "container" / "loose.md", "name: nested-loose\ndescription: Not root")
    write_skill(root / "container" / "deep" / "SKILL.md")
    for name in [".hidden", "node_modules", "ignored"]:
        write_skill(root / name / "SKILL.md", f"name: {name}\ndescription: Skip")
    write_skill(root / "skip.md", "name: skip\ndescription: Skip")
    (root / ignore_name).write_text("ignored/\n*.md\n!loose.md\n!kept/SKILL.md\n!container/**/SKILL.md\n")

    result = load_skills_from_dir(root)

    assert [s.name for s in result.skills] == ["deep", "kept", "loose"]
    assert not result.diagnostics


@pytest.mark.parametrize("metadata, expected_name, warning", [
    ("name: Bad_Name\ndescription: Useful", "Bad_Name", "invalid characters"),
    ("name: -bad--\ndescription: Useful", "-bad--", "hyphen"),
    (f"name: {'a' * 65}\ndescription: Useful", "a" * 65, "64 characters"),
    (f"description: {'x' * 1025}", "root", "1024 characters"),
])
def test_validation_warnings_keep_usable_skills(tmp_path: Path, metadata: str, expected_name: str, warning: str) -> None:
    root = tmp_path / "root"
    write_skill(root / "SKILL.md", metadata)
    result = load_skills_from_dir(root)
    assert [s.name for s in result.skills] == [expected_name]
    assert all(d.reason == "invalid_metadata" for d in result.diagnostics)
    assert any(warning in d.message for d in result.diagnostics)


@pytest.mark.parametrize("metadata", ["name: missing", 'description: ""', "description: false", "description: 42"])
def test_declared_skill_requires_nonempty_string_description_and_still_stops_recursion(tmp_path: Path, metadata: str) -> None:
    root = tmp_path / "root"
    write_skill(root / "SKILL.md", metadata)
    write_skill(root / "child" / "SKILL.md")
    result = load_skills_from_dir(root)
    assert not result.skills
    assert result.diagnostics[0].reason == "invalid_metadata"
    assert result.diagnostics[0].message == "description is required"


def test_read_and_frontmatter_failures_report_diagnostics_and_keep_other_sources(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid" / "SKILL.md"
    invalid.parent.mkdir()
    invalid.write_bytes(b"\xff")
    malformed = write_skill(tmp_path / "malformed" / "SKILL.md", "invalid frontmatter")
    valid = write_skill(tmp_path / "valid" / "SKILL.md")
    result = load_skills([SkillSource(invalid), SkillSource(malformed), SkillSource(valid), SkillSource(tmp_path / "missing")])
    assert [s.name for s in result.skills] == ["valid"]
    assert [d.reason for d in result.diagnostics] == ["read_error", "parse_error", "read_error"]


def test_root_notes_without_metadata_are_silent_and_bom_block_scalars_are_supported(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "notes.md").write_text("Ordinary notes.")
    (root / "malformed.md").write_text("---\ninvalid frontmatter\n---\nNotes.")
    path = root / "good.md"
    path.write_bytes(b'\xef\xbb\xbf---\r\nname: good\r\ndescription: >-\r\n  Line one\r\n  line two\r\ndisable-model-invocation: "true"\r\n---\r\nBody')
    result = load_skills_from_dir(root)
    assert [(s.name, s.description, s.disable_model_invocation) for s in result.skills] == [("good", "Line one line two", False)]
    assert not result.diagnostics


def test_catalog_requires_executable_read_or_bash_and_escapes_metadata(tmp_path: Path) -> None:
    path = write_skill(tmp_path / 'path<&"' / "SKILL.md", 'name: "name<&>\\\"\'"\ndescription: "description<&>\\\"\'"')
    hidden = write_skill(tmp_path / "hidden" / "SKILL.md", "description: Secret\ndisable-model-invocation: true")
    skills = load_skills([SkillSource(path), SkillSource(hidden)]).skills
    assert format_skills_for_prompt(skills) == ""
    assert format_skills_for_prompt(skills, tools=[create_write_tool(tmp_path)]) == ""
    catalog = format_skills_for_prompt(skills, tools=[create_read_tool(tmp_path)])
    assert "Use the read tool" in catalog
    assert '<name>name&lt;&amp;&gt;&quot;&apos;</name>' in catalog
    assert '<description>description&lt;&amp;&gt;&quot;&apos;</description>' in catalog
    assert 'path&lt;&amp;&quot;/SKILL.md</location>' in catalog
    assert "Secret" not in catalog
    assert len(skills) == 2
    assert "Use bash" in format_skills_for_prompt(skills, tools=[create_bash_tool(tmp_path)])
    assert format_skills_for_prompt([skills[1]], tools=[create_read_tool(tmp_path)]) == ""


def test_explicit_hidden_skill_rereads_body_and_preserves_raw_arguments(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "hidden" / "SKILL.md", "description: Hidden\ndisable-model-invocation: true", "Old body.")
    skills = load_skills_from_dir(tmp_path).skills
    write_skill(path, "description: Changed", "New body with /template $1.")

    result = expand_skill_command('/skill:hidden "quoted words"  $ARGUMENTS\nnext line ', skills)

    assert result.text == (
        f'<skill name="hidden" location="{path}">\n'
        f"References are relative to {path.parent}.\n\n"
        'New body with /template $1.\n</skill>\n\n"quoted words"  $ARGUMENTS\nnext line '
    )
    assert not result.diagnostics
    assert skills[0].description == "Hidden" and skills[0].disable_model_invocation


@pytest.mark.parametrize("text", ["/skill:unknown args", "/other args", "text /skill:hidden", " /skill:hidden", "/skill:"])
def test_unknown_or_non_skill_commands_pass_through(tmp_path: Path, text: str) -> None:
    write_skill(tmp_path / "hidden" / "SKILL.md")
    skills = load_skills_from_dir(tmp_path).skills
    result = expand_skill_command(text, skills)
    assert result.text == text and not result.diagnostics


@pytest.mark.parametrize("failure", ["deleted", "invalid_utf8", "bad_frontmatter", "directory"])
def test_expansion_failure_retains_original_text_and_returns_diagnostic(tmp_path: Path, failure: str) -> None:
    path = write_skill(tmp_path / "chosen" / "SKILL.md")
    skills = load_skills_from_dir(tmp_path, source="project").skills
    if failure == "deleted":
        path.unlink()
    elif failure == "directory":
        path.unlink()
        path.mkdir()
    elif failure == "invalid_utf8":
        path.write_bytes(b"\xff")
    else:
        write_skill(path, "invalid frontmatter")
    text = "/skill:chosen keep  args"
    result = expand_skill_command(text, skills)
    assert result.text == text
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].path == str(path) and result.diagnostics[0].source == "project"
    assert result.diagnostics[0].reason == ("parse_error" if failure == "bad_frontmatter" else "read_error")


def test_nested_ignore_can_reinclude_files_and_rules_do_not_leak_to_siblings(tmp_path: Path) -> None:
    write_skill(tmp_path / "a" / "keep" / "SKILL.md")
    write_skill(tmp_path / "a" / "excluded" / "SKILL.md")
    write_skill(tmp_path / "b" / "keep" / "SKILL.md", "name: other\ndescription: Other")
    (tmp_path / ".gitignore").write_text("*.md\n")
    (tmp_path / "a" / ".ignore").write_text("!**/SKILL.md\nexcluded/\n")
    (tmp_path / "b" / ".fdignore").write_text("!**/SKILL.md\n")
    result = load_skills_from_dir(tmp_path)
    assert [s.name for s in result.skills] == ["keep", "other"]
    assert not result.diagnostics


def test_ignore_supports_anchoring_escaped_prefixes_and_globs(tmp_path: Path) -> None:
    for name in ["anchored", "container/anchored", "#skip", "!skip", "skip1", "keep space", "container/deep/skip2", "ok"]:
        write_skill(tmp_path / name / "SKILL.md")
    (tmp_path / ".gitignore").write_text("/anchored/\n\\#skip/\n\\!skip/\nskip[0-9]/\nkeep\\ space/\n")
    result = load_skills_from_dir(tmp_path)
    assert [s.name for s in result.skills] == ["anchored", "ok"]
    assert not result.diagnostics


def test_unreadable_ignore_file_is_diagnosed_without_losing_skills(tmp_path: Path) -> None:
    write_skill(tmp_path / "good" / "SKILL.md")
    (tmp_path / ".ignore").write_bytes(b"\xff")
    result = load_skills_from_dir(tmp_path, source="project")
    assert [s.name for s in result.skills] == ["good"]
    assert [(Path(d.path).name, d.reason, d.source) for d in result.diagnostics] == [(".ignore", "read_error", "project")]


async def test_host_catalog_and_expanded_input_use_existing_agent_and_read_tool(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "visible" / "SKILL.md", body="Visible body.")
    write_skill(tmp_path / "hidden" / "SKILL.md", "description: Hidden\ndisable-model-invocation: true", "Explicit body.")
    loaded = load_skills_from_dir(tmp_path)
    tools = [create_read_tool(tmp_path)]
    catalog = format_skills_for_prompt(loaded.skills, tools=tools)
    expanded = expand_skill_command("/skill:hidden original arguments", loaded.skills)
    stream = ScriptedStreamFn([
        lambda: tool_call_message("read", {"path": str(path)}),
        lambda: text_message("done"),
        lambda: text_message("raw input accepted"),
    ])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(
            model=make_model(), tools=tools,
            messages=[SystemMessage(content="", sections={"skills": catalog}, timestamp=1000)],
        ),
    ))
    await agent.prompt(expanded.text)
    request = stream.requests[0].context.messages
    assert "<name>visible</name>" in get_current_system_prompt(request)
    assert "<name>hidden</name>" not in get_current_system_prompt(request)
    assert next(m for m in request if isinstance(m, UserMessage)).content == [TextContent(text=expanded.text)]
    assert "Explicit body." in expanded.text and expanded.text.endswith("original arguments")
    result = next(m for m in agent.state.messages if isinstance(m, ToolResultMessage))
    assert not result.is_error and "Visible body." in result.content[0].text
    await agent.prompt("/skill:hidden unexpanded")
    assert [m for m in stream.requests[-1].context.messages if isinstance(m, UserMessage)][-1].content == [TextContent(text="/skill:hidden unexpanded")]
    await agent.close()


def test_plain_body_is_trimmed_on_expansion_and_catalog_stays_a_snapshot(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "skill" / "SKILL.md", "description: Original")
    loaded = load_skills_from_dir(tmp_path)
    path.write_text(" \n Plain body. \r\n")
    result = expand_skill_command("/skill:skill", loaded.skills)
    assert "\n\nPlain body.\n</skill>" in result.text
    assert "Original" in format_skills_for_prompt(loaded.skills, tools=[create_read_tool(tmp_path)])
    assert not load_skills_from_dir(tmp_path).skills


def test_ignored_root_skill_allows_remaining_discovery_and_local_order_is_stable(tmp_path: Path) -> None:
    write_skill(tmp_path / "SKILL.md", "description: Root")
    later = write_skill(tmp_path / "z.md", "name: same\ndescription: Loser")
    first = write_skill(tmp_path / "a.md", "name: same\ndescription: Winner")
    write_skill(tmp_path / "child" / "SKILL.md")
    (tmp_path / ".ignore").write_text("/SKILL.md\n")
    result = load_skills_from_dir(tmp_path)
    assert [s.description for s in result.skills] == ["Winner", "Useful instructions"]
    assert result.diagnostics[0].winner.file_path == str(first)
    assert result.diagnostics[0].loser.file_path == str(later)


def test_non_markdown_sources_and_extensions_are_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    script = tmp_path / "extension.py"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    write_skill(tmp_path / "good" / "SKILL.md")
    result = load_skills([SkillSource(script), SkillSource(tmp_path)])
    assert [s.name for s in result.skills] == ["good"]
    assert result.diagnostics[0].reason == "not_markdown"
    assert not marker.exists()


@pytest.mark.parametrize("target", ["file", "directory", "ignore"])
def test_permission_failures_are_diagnosed_and_other_sources_still_load(tmp_path: Path, target: str) -> None:
    root = tmp_path / "restricted"
    path = write_skill(root / "SKILL.md")
    if target == "directory":
        path = root
    elif target == "ignore":
        path = root / ".ignore"
        path.write_text("*.md\n")
    good = write_skill(tmp_path / "good" / "SKILL.md")
    mode = path.stat().st_mode
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("Test process can bypass file permissions")
        result = load_skills([SkillSource(root, "project"), SkillSource(good)])
        assert "good" in [s.name for s in result.skills]
        assert any(d.path == str(path) and d.reason == "read_error" and d.source == "project" for d in result.diagnostics)
    finally:
        path.chmod(mode)
