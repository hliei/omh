from __future__ import annotations

from pathlib import Path

from omh.agent import (
    BACKGROUND_CONTEXT,
    LocalExecutionEnv,
    Skill,
    SkillSource,
    format_skill_invocation,
    load_skills,
    load_sourced_skills,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_format_skill_invocation_wraps_content_with_name_and_location() -> None:
    skill = Skill(
        name="greet",
        description="Say hello",
        content="Greet the user.",
        file_path="/skills/greet/SKILL.md",
    )

    assert format_skill_invocation(skill) == (
        '<skill name="greet" location="/skills/greet/SKILL.md">\n'
        "References are relative to /skills/greet.\n\n"
        "Greet the user.\n</skill>"
    )
    assert format_skill_invocation(skill, "Be brief.") == (
        f"{format_skill_invocation(skill)}\n\nBe brief."
    )


async def test_load_skills_reads_skill_md_frontmatter(tmp_path: Path) -> None:
    _write(
        tmp_path / "skills" / "greet" / "SKILL.md",
        "---\n"
        "name: greet\n"
        "description: Say hello\n"
        "disable-model-invocation: true\n"
        "---\n"
        "\n"
        "Greet the user.\n",
    )

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)), str(tmp_path / "skills"), BACKGROUND_CONTEXT
    )

    assert result.diagnostics == ()
    assert result.skills == (
        Skill(
            name="greet",
            description="Say hello",
            content="Greet the user.",
            file_path=str(tmp_path / "skills" / "greet" / "SKILL.md"),
            disable_model_invocation=True,
        ),
    )


async def test_root_markdown_uses_directory_name_and_requires_description(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "skills" / "alpha.md",
        "---\ndescription: Alpha skill\n---\nAlpha body",
    )
    _write(tmp_path / "skills" / "beta.md", "Beta has no frontmatter description")
    _write(tmp_path / "skills" / "notes.txt", "not a skill")

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)), str(tmp_path / "skills"), BACKGROUND_CONTEXT
    )

    assert result.diagnostics == ()
    assert [skill.name for skill in result.skills] == ["skills"]
    assert result.skills[0].content == "Alpha body"


async def test_ignore_files_exclude_matching_entries(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write(root / ".gitignore", "ignored/\n*.md\n!kept/SKILL.md\n")
    _write(
        root / "kept" / "SKILL.md", "---\ndescription: Kept\n---\nkept body"
    )
    _write(
        root / "ignored" / "SKILL.md", "---\ndescription: Ignored\n---\nbody"
    )
    _write(root / "loose.md", "---\ndescription: Loose\n---\nbody")

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)), str(root), BACKGROUND_CONTEXT
    )

    assert [skill.name for skill in result.skills] == ["kept"]


async def test_ignore_file_variants_and_hidden_entries_are_skipped(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    _write(root / ".fdignore", "skipme/\n")
    _write(root / "skipme" / "SKILL.md", "---\ndescription: skip\n---\nbody")
    _write(root / "keep" / "SKILL.md", "---\ndescription: keep\n---\nbody")
    _write(root / ".hidden" / "SKILL.md", "---\ndescription: hidden\n---\nbody")
    _write(
        root / "node_modules" / "pkg" / "SKILL.md",
        "---\ndescription: vendored\n---\nbody",
    )

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)), str(root), BACKGROUND_CONTEXT
    )

    assert [skill.name for skill in result.skills] == ["keep"]


async def test_root_directory_ignore_patterns_match_at_any_depth(
    tmp_path: Path,
) -> None:
    """A root ``/name/`` rule matches at any depth.

    The loader strips the leading ``/`` before matching the pattern, so the
    directory name applies at any level. This differs from Git's root-anchored
    interpretation and is part of the loader's supported ignore behavior.
    """
    root = tmp_path / "skills"
    _write(root / ".gitignore", "/vendored/\n")
    _write(root / "vendored" / "SKILL.md", "---\ndescription: a\n---\nbody")
    _write(
        root / "sub" / "vendored" / "SKILL.md", "---\ndescription: b\n---\nbody"
    )
    _write(root / "keep" / "SKILL.md", "---\ndescription: c\n---\nbody")

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)), str(root), BACKGROUND_CONTEXT
    )

    assert [skill.name for skill in result.skills] == ["keep"]


async def test_declared_skill_diagnostics_are_warnings(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write(
        root / "Bad_Name" / "SKILL.md",
        "---\nname: Bad_Name\ndescription: Bad name\n---\nbody",
    )
    _write(
        root / "broken" / "SKILL.md",
        "---\nthis is not valid yaml\n---\nbody",
    )
    _write(root / "nodesc" / "SKILL.md", "---\nname: nodesc\n---\nbody")
    _write(
        root / "loose.md",
        "---\ninvalid frontmatter here\n---\nbody",
    )

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)), str(root), BACKGROUND_CONTEXT
    )

    assert [skill.name for skill in result.skills] == ["Bad_Name"]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "invalid_metadata",
        "parse_failed",
        "invalid_metadata",
    ]
    assert all(
        diagnostic.type == "warning" for diagnostic in result.diagnostics
    )
    assert result.diagnostics[0].message == (
        "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
    )
    assert result.diagnostics[2].message == "description is required"


async def test_missing_directories_and_non_directories_are_skipped(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "notes.txt", "plain file")

    result = await load_skills(
        LocalExecutionEnv(str(tmp_path)),
        [str(tmp_path / "missing"), str(tmp_path / "notes.txt")],
        BACKGROUND_CONTEXT,
    )

    assert result.skills == ()
    assert result.diagnostics == ()


async def test_load_sourced_skills_tags_source_and_maps(tmp_path: Path) -> None:
    _write(
        tmp_path / "skills" / "greet" / "SKILL.md",
        "---\ndescription: Say hello\n---\nbody",
    )

    def map_skill(skill: Skill, source: str, context: object) -> tuple[str, str]:
        del context
        return (skill.name, source)

    result = await load_sourced_skills(
        LocalExecutionEnv(str(tmp_path)),
        [SkillSource(path=str(tmp_path / "skills"), source="plugin")],
        map_skill,
        BACKGROUND_CONTEXT,
    )

    assert result.skills[0].skill == ("greet", "plugin")
    assert result.skills[0].source == "plugin"
    assert result.diagnostics == ()
