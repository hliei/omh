from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    build_system_sections,
    create_read_tool,
    load_project_context_files,
)
from omh.llm.types import SystemMessage
from omh.llm.utils.transcript import (
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
)
from tests.agent.test_agent_live_config import (
    ScriptedStreamFn,
    make_model,
    text_message,
)


def write_context(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_global_then_ancestors_with_priority_and_bom(tmp_path: Path) -> None:
    global_dir = tmp_path / "global"
    project = tmp_path / "project"
    cwd = project / "src"
    cwd.mkdir(parents=True)
    write_context(global_dir, "CLAUDE.md", "global")
    write_context(tmp_path, "AGENTS.md", "ancestor")
    write_context(project, "AGENTS.md", "lower priority")
    write_context(project, "AGENTS.override.md", "\ufeffoverride")
    write_context(cwd, "AGENTS.MD", "cwd")
    write_context(cwd / "child", "AGENTS.md", "not inherited")

    result = load_project_context_files(cwd=cwd, agent_dir=global_dir)

    # Scope assertions to the controlled fixture; real root ancestors are allowed.
    files = [file for file in result.files if Path(file.path).is_relative_to(tmp_path)]
    assert [file.content for file in files] == ["global", "ancestor", "override", "cwd"]
    assert [file.source for file in files] == ["global", "project", "project", "project"]
    assert files[-1].base_dir == str(cwd)
    assert result.diagnostics == []


@pytest.mark.parametrize("name", ["AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"])
def test_each_candidate_is_supported(tmp_path: Path, name: str) -> None:
    path = write_context(tmp_path, name, "instructions")
    result = load_project_context_files(cwd=tmp_path)
    assert Path(result.files[-1].path).samefile(path)


def test_read_failure_and_non_file_fall_back_with_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.override.md").mkdir()
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")
    write_context(tmp_path, "CLAUDE.md", "fallback")
    result = load_project_context_files(cwd=tmp_path)
    assert result.files[-1].content == "fallback"
    assert [(Path(d.path).name, d.reason) for d in result.diagnostics] == [
        ("AGENTS.override.md", "not_file"), ("AGENTS.md", "read_error"),
    ]
    assert all(d.message and d.source == "project" for d in result.diagnostics)


def test_global_and_explicit_symlink_aliases_are_deduplicated(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_context(project, "AGENTS.md", "only once")
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)
    explicit = tmp_path / "chosen"
    write_context(explicit, "CLAUDE.md", "explicit")
    result = load_project_context_files(
        cwd=project, agent_dir=alias, context_dirs=[explicit, alias],
    )
    files = [f for f in result.files if Path(f.path).is_relative_to(tmp_path)]
    assert [(f.content, f.source) for f in files] == [
        ("only once", "global"), ("explicit", "explicit"),
    ]
    assert [d.reason for d in result.diagnostics] == ["duplicate", "duplicate"]


def test_unreadable_ancestor_reports_failure_instead_of_disappearing(tmp_path: Path) -> None:
    # An overlong path reliably fails stat on both supported platforms.
    selected = tmp_path / ("x" * 256)
    result = load_project_context_files(cwd=tmp_path, context_dirs=[selected])
    assert len(result.diagnostics) == 5
    assert all(d.reason == "read_error" and d.source == "explicit" for d in result.diagnostics)


def git(directory: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(directory), *args], check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "fixture")
    return repo


@pytest.mark.parametrize("name", ["AGENTS.md", "CLAUDE.md", "AGENTS.override.md"])
def test_nested_linked_worktree_shadows_same_scope_across_candidate_names(
    tmp_path: Path, repository: Path, name: str,
) -> None:
    write_context(tmp_path, "AGENTS.md", "outer")
    write_context(repository, "AGENTS.md", "main repo")
    write_context(repository / "trees", "AGENTS.md", "intermediate ancestor")
    worktree = repository / "trees" / "feature"
    git(repository, "worktree", "add", "--detach", str(worktree))
    write_context(worktree, name, "worktree")
    cwd = worktree / "src"
    write_context(cwd, "CLAUDE.md", "inner")
    # Resolve metadata written in realpath form even when cwd uses an alias.
    alias = tmp_path / "alias"
    alias.symlink_to(worktree, target_is_directory=True)

    result = load_project_context_files(cwd=alias / "src")
    # Physical worktree ancestry must be preserved through the alias.
    assert [f.content for f in result.files if Path(f.path).is_relative_to(tmp_path)] == [
        "outer", "intermediate ancestor", "worktree", "inner",
    ]
    assert [(d.reason, Path(d.path).parent) for d in result.diagnostics] == [
        ("shadowed", repository),
    ]


@pytest.mark.parametrize("nested, own_context", [(True, False), (False, True)])
def test_worktree_without_own_context_and_sibling_keep_normal_inheritance(
    tmp_path: Path, repository: Path, nested: bool, own_context: bool,
) -> None:
    write_context(tmp_path, "AGENTS.md", "outer")
    write_context(repository, "AGENTS.md", "main repo")
    worktree = (repository if nested else tmp_path) / "feature"
    git(repository, "worktree", "add", "--detach", str(worktree))
    if own_context:
        write_context(worktree, "AGENTS.md", "worktree")
    result = load_project_context_files(cwd=worktree)
    assert [f.content for f in result.files if Path(f.path).is_relative_to(tmp_path)] == (
        ["outer", "main repo"] if nested else ["outer", "worktree"]
    )
    assert not any(d.reason == "shadowed" for d in result.diagnostics)


async def test_custom_sections_reach_model_only_when_host_injects_them(tmp_path: Path) -> None:
    stream = ScriptedStreamFn([lambda: text_message("done")])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model()),
    ))
    original_history = agent.history
    write_context(tmp_path, "AGENTS.md", "Use the project conventions.")
    result = load_project_context_files(cwd=tmp_path)
    sections = build_system_sections(
        cwd=tmp_path, context_files=result.files, custom_prompt="Host instructions.",
        selected_tools=["read"], tool_snippets={"read": "Read files"},
    )
    assert agent.history == original_history
    assert not stream.requests and not agent.state.is_busy
    assert sections["preamble"] == "Host instructions."
    assert "Use the project conventions." in sections["project_context"]
    assert "tools" not in sections

    # Construct a normal SystemMessage and seed a new Agent using existing APIs.
    injected = SystemMessage(content="", sections=dict(sections), timestamp=1000)
    tool = create_read_tool(tmp_path)
    configured = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[tool], messages=[injected]),
    ))
    await configured.prompt("Start.")
    system = get_current_system_message(stream.requests[0].context.messages)
    assert system.sections == sections
    assert [tool.name for tool in get_current_tools(stream.requests[0].context.messages)] == ["read"]
    prompt = get_current_system_prompt(stream.requests[0].context.messages)
    assert "Host instructions." in prompt and "Use the project conventions." in prompt


def test_default_builder_uses_only_explicit_tool_information(tmp_path: Path) -> None:
    sections = build_system_sections(
        cwd=tmp_path,
        selected_tools=["read", "bash"],
        tool_snippets={"read": "Read files", "bash": "Run commands", "write": "Must stay hidden"},
        tool_guidelines={"read": [" Use offsets. "], "bash": ["Use offsets."], "write": ["hidden"]},
        prompt_guidelines=["", "Use offsets.", "Explain changes."],
        append_system_prompt="Host addendum.",
    )
    assert list(sections) == ["preamble", "tools", "rules", "addendum", "cwd"]
    assert sections["tools"] == "<tools>\n- read: Read files\n- bash: Run commands\n</tools>"
    assert sections["rules"].count("Use offsets.") == 1
    assert "Use bash" in sections["rules"] and "Explain changes." in sections["rules"]
    assert "hidden" not in "\n".join(sections.values())
    empty = build_system_sections(cwd=tmp_path)
    assert empty["tools"] == "<tools>\n(none)\n</tools>"
    assert "Use bash" not in empty["rules"]


def test_project_paths_are_escaped_and_empty_custom_prompt_still_appends_context(tmp_path: Path) -> None:
    cwd = tmp_path / 'a"&<b'
    write_context(cwd, "AGENTS.md", "Keep <content> exactly.")
    files = load_project_context_files(cwd=cwd).files
    sections = build_system_sections(cwd=cwd, context_files=files, custom_prompt="")
    assert list(sections) == ["preamble", "project_context", "cwd"]
    assert sections["preamble"] == ""
    assert "a&quot;&amp;&lt;b" in sections["project_context"]
    assert "Keep <content> exactly." in sections["project_context"]


def test_hardlinked_context_is_loaded_once(tmp_path: Path) -> None:
    path = write_context(tmp_path, "AGENTS.md", "once")
    child = tmp_path / "child"
    child.mkdir()
    (child / "CLAUDE.md").hardlink_to(path)
    result = load_project_context_files(cwd=child)
    assert [f.content for f in result.files if Path(f.path).is_relative_to(tmp_path)] == ["once"]
    assert result.diagnostics[-1].reason == "duplicate"


def test_broken_symlink_falls_back_with_read_diagnostic(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.override.md").symlink_to(tmp_path / "missing")
    write_context(tmp_path, "AGENTS.md", "fallback")
    result = load_project_context_files(cwd=tmp_path)
    assert result.files[-1].content == "fallback"
    assert result.diagnostics[-1].reason == "read_error"


def test_normal_repo_keeps_repo_and_outer_ancestors(tmp_path: Path, repository: Path) -> None:
    write_context(tmp_path, "AGENTS.md", "outer")
    write_context(repository, "AGENTS.md", "repo")
    cwd = repository / "src"
    write_context(cwd, "AGENTS.md", "inner")
    result = load_project_context_files(cwd=cwd)
    assert [f.content for f in result.files if Path(f.path).is_relative_to(tmp_path)] == ["outer", "repo", "inner"]
    assert not any(d.reason == "shadowed" for d in result.diagnostics)


def test_permission_failure_is_diagnosed_and_lower_candidate_is_used(tmp_path: Path) -> None:
    path = write_context(tmp_path, "AGENTS.override.md", "unreadable")
    write_context(tmp_path, "CLAUDE.md", "fallback")
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("Test process can bypass file permissions")
        result = load_project_context_files(cwd=tmp_path)
        assert result.files[-1].content == "fallback"
        assert result.diagnostics[-1].reason == "read_error"
    finally:
        path.chmod(0o600)


def test_unreadable_worktree_context_does_not_shadow_main(tmp_path: Path, repository: Path) -> None:
    write_context(repository, "AGENTS.md", "main")
    worktree = repository / "feature"
    git(repository, "worktree", "add", "--detach", str(worktree))
    (worktree / "AGENTS.md").write_bytes(b"\xff")
    result = load_project_context_files(cwd=worktree)
    assert result.files[-1].content == "main"
    assert all(d.reason == "read_error" for d in result.diagnostics)


def test_bare_layout_keeps_container_context(tmp_path: Path, repository: Path) -> None:
    container = tmp_path / "bare-layout"
    container.mkdir()
    bare = container / ".bare"
    git(tmp_path, "clone", "--bare", str(repository), str(bare))
    worktree = container / "main"
    git(bare, "worktree", "add", "--detach", str(worktree))
    write_context(container, "AGENTS.md", "container")
    write_context(worktree, "CLAUDE.md", "worktree")
    result = load_project_context_files(cwd=worktree)
    assert [f.content for f in result.files if Path(f.path).is_relative_to(container)] == ["container", "worktree"]
    assert not any(d.reason == "shadowed" for d in result.diagnostics)


def test_bom_stripping_preserves_original_line_endings(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"\xef\xbb\xbfFirst\r\nSecond\r\n")
    result = load_project_context_files(cwd=tmp_path)
    assert result.files[-1].content == "First\r\nSecond\r\n"
