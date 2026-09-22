from __future__ import annotations

from pathlib import Path

from omh.agent import (
    BACKGROUND_CONTEXT,
    LocalExecutionEnv,
    PromptTemplate,
    PromptTemplateSource,
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
    parse_command_args,
    substitute_args,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_parse_command_args_handles_quotes_and_whitespace() -> None:
    assert parse_command_args('one "two three" four') == ["one", "two three", "four"]
    assert parse_command_args("'a b' c") == ["a b", "c"]
    assert parse_command_args("  a   b  ") == ["a", "b"]
    assert parse_command_args('"unclosed') == ["unclosed"]
    assert parse_command_args("") == []


def test_substitute_args_supports_positional_and_slice_placeholders() -> None:
    args = ["a", "b", "c"]
    assert substitute_args("$1-$2", args) == "a-b"
    assert substitute_args("$9", args) == ""
    assert substitute_args("$@", args) == "a b c"
    assert substitute_args("$ARGUMENTS", args) == "a b c"
    assert substitute_args("${@:2}", args) == "b c"
    assert substitute_args("${@:2:1}", args) == "b"
    assert substitute_args("${@:0}", args) == "a b c"
    assert substitute_args("$10", [str(index) for index in range(1, 11)]) == "10"
    # Substitution applies $N before ${@:N}, so a "$N" introduced by an argument is
    # not re-substituted afterwards.
    assert substitute_args("${@:1}", ["$2", "x"]) == "$2 x"


def test_format_prompt_template_invocation_substitutes_template_content() -> None:
    template = PromptTemplate(
        name="review", content="Review $1 against $2", description=None
    )
    assert format_prompt_template_invocation(template, ["a", "b"]) == (
        "Review a against b"
    )
    assert format_prompt_template_invocation(template) == "Review  against "


async def test_load_prompt_templates_from_directory_and_file(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    _write(
        root / "review.md",
        "---\ndescription: Review a change\n---\nReview the change.",
    )
    _write(root / "standup.md", "Summarize yesterday's work.")
    _write(root / "notes.txt", "ignored")

    result = await load_prompt_templates(
        LocalExecutionEnv(str(tmp_path)), str(root), BACKGROUND_CONTEXT
    )

    assert result.diagnostics == ()
    assert result.prompt_templates == (
        PromptTemplate(
            name="review",
            description="Review a change",
            content="Review the change.",
        ),
        PromptTemplate(
            name="standup",
            description="Summarize yesterday's work.",
            content="Summarize yesterday's work.",
        ),
    )

    explicit = await load_prompt_templates(
        LocalExecutionEnv(str(tmp_path)),
        str(root / "review.md"),
        BACKGROUND_CONTEXT,
    )
    assert explicit.prompt_templates == (result.prompt_templates[0],)


async def test_description_falls_back_to_a_truncated_first_line(
    tmp_path: Path,
) -> None:
    long_line = "x" * 80
    _write(tmp_path / "long.md", f"\n{long_line}\n")

    result = await load_prompt_templates(
        LocalExecutionEnv(str(tmp_path)), str(tmp_path / "long.md"), BACKGROUND_CONTEXT
    )

    assert result.prompt_templates[0].description == f"{'x' * 60}..."


async def test_missing_and_non_markdown_paths_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "plain")

    result = await load_prompt_templates(
        LocalExecutionEnv(str(tmp_path)),
        [str(tmp_path / "missing"), str(tmp_path / "notes.txt")],
        BACKGROUND_CONTEXT,
    )

    assert result.prompt_templates == ()
    assert result.diagnostics == ()


async def test_malformed_frontmatter_reports_a_parse_diagnostic(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "broken.md", "---\ninvalid frontmatter here\n---\nbody")

    result = await load_prompt_templates(
        LocalExecutionEnv(str(tmp_path)),
        str(tmp_path / "broken.md"),
        BACKGROUND_CONTEXT,
    )

    assert result.prompt_templates == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "parse_failed"


async def test_load_sourced_prompt_templates_tags_source_and_maps(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "review.md", "Review the change.")

    def map_template(
        template: PromptTemplate, source: str, context: object
    ) -> tuple[str, str]:
        del context
        return (template.name, source)

    result = await load_sourced_prompt_templates(
        LocalExecutionEnv(str(tmp_path)),
        [PromptTemplateSource(path=str(tmp_path / "review.md"), source="plugin")],
        map_template,
        BACKGROUND_CONTEXT,
    )

    assert result.prompt_templates[0].prompt_template == ("review", "plugin")
    assert result.prompt_templates[0].source == "plugin"
    assert result.diagnostics == ()
