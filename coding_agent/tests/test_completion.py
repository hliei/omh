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
    assert found.count("/new ") == 1
    assert found.count("/resume ") == 1


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


def test_attach_discovers_paths_and_management_arguments(tmp_path: Path) -> None:
    (tmp_path / "picture name.png").write_bytes(b"image")
    text = "/attach pic"
    assert values(complete(text, len(text), sources(tmp_path))) == ["picture name.png"]
    text = "/attach cl"
    assert "clear" in values(complete(text, len(text), sources(tmp_path)))
    assert "/attach " in values(complete("/at", 3, sources(tmp_path)))


def test_attach_removal_completes_only_existing_pending_numbers(tmp_path: Path) -> None:
    selected = CompletionSources(
        cwd=tmp_path, commands=AVAILABLE_COMMANDS, pending_count=2,
    )
    text = "/attach remove "
    assert values(complete(text, len(text), selected)) == ["remove 1", "remove 2"]
    text = "/attach remove 3"
    assert complete(text, len(text), selected) is None


def test_attach_directory_completion_refills_without_executing(tmp_path: Path) -> None:
    (tmp_path / "images dir").mkdir()
    text = "/attach images"
    completion = complete(text, len(text), sources(tmp_path))
    assert values(completion) == ["images dir/"]
    assert completion is not None and not completion.submit


def test_attach_completes_spaced_and_quoted_paths(tmp_path: Path) -> None:
    directory = tmp_path / "images dir"
    directory.mkdir()
    (directory / "picture name.png").write_bytes(b"image")
    text = "/attach images dir/pic"
    assert values(complete(text, len(text), sources(tmp_path))) == ["images dir/picture name.png"]
    text = '/attach "images dir/pic'
    assert values(complete(text, len(text), sources(tmp_path))) == ['"images dir/picture name.png"']


def test_session_commands_complete_sort_formats_and_destinations(tmp_path: Path) -> None:
    (tmp_path / "backup.jsonl").touch()
    text = "/resume --sort cr"
    assert values(complete(text, len(text), sources(tmp_path))) == ["--sort created"]
    text = "/export --retained 1 h"
    assert values(complete(text, len(text), sources(tmp_path))) == ["--retained 1 html "]
    text = "/save --retained 1 back"
    assert values(complete(text, len(text), sources(tmp_path))) == ["--retained 1 backup.jsonl"]
