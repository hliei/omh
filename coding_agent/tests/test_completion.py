"""Slash-command, resource, argument and path completion candidates."""

from __future__ import annotations

from pathlib import Path

from coding_agent.commands import AVAILABLE_COMMANDS, RESERVED_COMMANDS
from coding_agent.completion import CompletionSources, complete


def sources(cwd: Path, *, skills=(), templates=()) -> CompletionSources:
    return CompletionSources(
        cwd=cwd, commands=AVAILABLE_COMMANDS,
        skills=tuple((name, f"{name} skill") for name in skills),
        templates=tuple((name, f"{name} template") for name in templates),
        reserved=RESERVED_COMMANDS,
    )


def values(completion) -> list[str]:
    assert completion is not None
    return [item.value for item in completion.items]


def test_builtin_skill_and_template_commands_are_all_discoverable(tmp_path: Path) -> None:
    completion = complete(
        "/", 1,
        sources(
            tmp_path,
            skills=("review", "tdd"),
            templates=("explain", "release-notes"),
        ),
    )
    found = values(completion)
    assert "/help " in found
    assert "/hotkeys " in found
    assert "/copy " in found
    assert "/skill:review " in found
    assert "/explain " in found


def test_reserved_command_names_are_never_offered_as_templates(tmp_path: Path) -> None:
    completion = complete("/", 1, sources(tmp_path, templates=("new", "resume", "notes")))
    found = values(completion)
    assert "/notes " in found
    assert "/new " not in found
    assert "/resume " not in found


def test_command_prefix_filters_candidates(tmp_path: Path) -> None:
    completion = complete("/he", 3, sources(tmp_path))
    assert values(completion) == ["/help "]
    assert complete("/zz", 3, sources(tmp_path)) is None


def test_skill_prefix_lists_only_skills(tmp_path: Path) -> None:
    completion = complete("/skill:t", 8, sources(tmp_path, skills=("tdd", "review"), templates=("thing",)))
    assert values(completion) == ["/skill:tdd "]


def test_help_argument_completes_available_commands_only(tmp_path: Path) -> None:
    completion = complete("/help he", 8, sources(tmp_path))
    assert values(completion) == ["help"]
    assert complete("/help z", 8, sources(tmp_path)) is None


def test_argument_position_has_no_candidates_for_commands_without_arguments(tmp_path: Path) -> None:
    assert complete("/hotkeys ", 9, sources(tmp_path)) is None


def test_path_completion_reports_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x")
    (tmp_path / "src" / "readme.md").write_text("y")
    (tmp_path / "notes.txt").write_text("z")
    text = "read @src/ap"
    completion = complete(text, len(text), sources(tmp_path))
    assert values(completion) == ["@src/app.py"]
    directories = complete("list @", 6, sources(tmp_path))
    assert "@src/" in values(directories)


def test_hidden_entries_need_an_explicit_dot(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("x")
    assert complete("@.", 2, sources(tmp_path)) is not None
    assert "@.env" in values(complete("@.", 2, sources(tmp_path)))


def test_absolute_root_completion_uses_root_instead_of_cwd(tmp_path: Path) -> None:
    (tmp_path / "only-in-project.txt").write_text("x")
    root_entry = next(entry for entry in Path("/").iterdir() if not entry.name.startswith("."))
    text = f"read @/{root_entry.name}"
    found = values(complete(text, len(text), sources(tmp_path)))
    assert any(item.rstrip("/") == f"@/{root_entry.name}" for item in found)
    text = "read @/only-in-project"
    assert complete(text, len(text), sources(tmp_path)) is None
