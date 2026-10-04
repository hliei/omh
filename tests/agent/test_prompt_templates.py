"""Host-selected templates exercised through public helpers and Agent."""

import os
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    PromptTemplateSource,
    expand_prompt_template,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)
from omh.llm.types import TextContent, UserMessage
from tests.agent.test_agent_tools import ScriptedStreamFn, make_model, text_message


def test_loader_reads_only_direct_markdown_and_preserves_body_and_provenance(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "review.md").write_text(
        "---\nname: ignored\ndescription: Review changes\n---\nCheck $1.\n", encoding="utf-8",
    )
    (root / "plain.md").write_text("\n# First useful line\nBody\n", encoding="utf-8")
    (root / "skip.txt").write_text("Not Markdown")
    (root / "nested").mkdir()
    (root / "nested" / "hidden.md").write_text("Hidden")

    result = load_prompt_templates([PromptTemplateSource("prompts", "project")], cwd=tmp_path)

    assert [(t.name, t.description, t.content, t.file_path, t.base_dir, t.source) for t in result.templates] == [
        ("plain", "# First useful line", "\n# First useful line\nBody\n", str(root / "plain.md"), str(root), "project"),
        ("review", "Review changes", "Check $1.", str(root / "review.md"), str(root), "project"),
    ]
    assert not result.diagnostics


def test_loader_keeps_ordered_duplicates_and_reports_winner_and_loser(tmp_path: Path) -> None:
    for source in ["global", "project"]:
        directory = tmp_path / source
        directory.mkdir()
        (directory / "review.md").write_text(source)

    result = load_prompt_templates([
        PromptTemplateSource("global", "global"), PromptTemplateSource("project/review.md", "project"),
    ], cwd=tmp_path)

    assert [(t.name, t.content, t.source) for t in result.templates] == [
        ("review", "global", "global"), ("review", "project", "project"),
    ]
    assert len(result.diagnostics) == 1
    collision = result.diagnostics[0]
    assert (collision.reason, collision.path, collision.source) == (
        "collision", str(tmp_path / "project" / "review.md"), "project",
    )
    assert collision.winner == result.templates[0]
    assert collision.loser == result.templates[1]


def test_loader_reports_read_and_format_failures_without_losing_valid_resources(tmp_path: Path) -> None:
    (tmp_path / "bad-encoding.md").write_bytes(b"\xff")
    (tmp_path / "bad-format.md").write_text('---\ndescription: "unterminated\n---\nBody')
    (tmp_path / "valid.md").write_bytes(b'\xef\xbb\xbf---\r\ndescription: >-\r\n  First line\r\n  second line\r\n---\r\nBody\r\n')
    result = load_prompt_templates([
        PromptTemplateSource(tmp_path, "project"), PromptTemplateSource(tmp_path / "missing", "explicit"),
    ])
    assert [(t.name, t.description, t.content) for t in result.templates] == [("valid", "First line second line", "Body")]
    assert [(Path(d.path).name, d.source, d.reason) for d in result.diagnostics] == [
        ("bad-encoding.md", "project", "read_error"), ("bad-format.md", "project", "parse_error"),
        ("missing", "explicit", "read_error"),
    ]
    assert all(d.message for d in result.diagnostics)


@pytest.mark.parametrize("metadata, body, description", [
    ("description: false", "\n" + "x" * 61, "x" * 60 + "..."),
    ('description: ""', "\nFirst\nSecond", "First"),
    ("description: 42", "", ""),
])
def test_loader_description_falls_back_to_first_nonempty_body_line(
    tmp_path: Path, metadata: str, body: str, description: str,
) -> None:
    path = tmp_path / "template.md"
    path.write_text(f"---\n{metadata}\n---\n{body}")
    result = load_prompt_templates([PromptTemplateSource(path)])
    assert result.templates[0].description == description
    assert not result.diagnostics


def test_loader_follows_file_and_source_directory_symlinks_without_recursing(tmp_path: Path) -> None:
    directory = tmp_path / "templates"
    directory.mkdir()
    original = tmp_path / "original.md"
    original.write_text("Original")
    (directory / "alias.md").symlink_to(original)
    (directory / "loop.md").symlink_to(directory, target_is_directory=True)
    (directory / "broken.md").symlink_to(tmp_path / "missing")
    source = tmp_path / "source"
    source.symlink_to(directory, target_is_directory=True)
    result = load_prompt_templates([PromptTemplateSource(source)])
    assert [(t.name, t.file_path) for t in result.templates] == [("alias", str(source / "alias.md"))]
    assert [(Path(d.path).name, d.reason) for d in result.diagnostics] == [("broken.md", "read_error")]


def test_substitution_supports_positionals_all_arguments_defaults_and_slices() -> None:
    assert substitute_args(
        "$1|$2|$0|$10|$@|$ARGUMENTS|${2:-fallback}|${4:-fallback}|${@:2}|${@:2:1}|${@:0:2}|${@:9}|${@:1:0}",
        ["one", "two words", "three"],
    ) == "one|two words|||one two words three|one two words three|two words|fallback|two words three|two words|one two words||"
    assert substitute_args("${1:-first}|${@:-all}|${ARGUMENTS:-all}", []) == "first|all|all"
    assert substitute_args("${1:-first}|${@:-all}|${ARGUMENTS:-all}", [""]) == "first|all|all"
    assert substitute_args("${@:-all}|${ARGUMENTS:-all}", ["one", "two"]) == "one two|one two"


@pytest.mark.parametrize("text, expected", [
    ('alpha "two words" \'three words\'', ["alpha", "two words", "three words"]),
    ('pre"two words"post\tlast\nnext', ["pretwo wordspost", "last", "next"]),
    ('"" \'\' \'open quote', ["open quote"]),
    (r'path\name "$HOME $(echo hi) `date`"', [r"path\name", "$HOME $(echo hi) `date`"]),
    ('"one \'two\' three"', ["one 'two' three"]),
    (" \t\n", []),
])
def test_command_arguments_group_quotes_without_shell_interpretation(text: str, expected: list[str]) -> None:
    assert parse_command_args(text) == expected


def test_substitution_never_reprocesses_argument_or_default_placeholders(tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-exist"
    argument = f"$2 $@ $ARGUMENTS ${{@:2}} ${{1:-default}} $(touch {sentinel}) `date`"
    assert substitute_args("$1|$2|$@|$ARGUMENTS|${@:1:1}", [argument, "second"]) == (
        f"{argument}|second|{argument} second|{argument} second|{argument}"
    )
    assert substitute_args("${1:-$2 $@ $ARGUMENTS}|$2", []) == "$2 $@ $ARGUMENTS|"
    assert substitute_args("${@:-$1}|${ARGUMENTS:-$@}", []) == "$1|$@"
    assert not sentinel.exists()


@pytest.mark.parametrize("text", ["$HOME", "${1}", "${@:x}", "${@:1:-1}", "${1-default}", "$$"])
def test_unsupported_placeholder_forms_remain_literal(text: str) -> None:
    assert substitute_args(text, ["value"]) == text


def test_expansion_uses_first_named_template_and_loaded_body_until_explicit_reload(tmp_path: Path) -> None:
    first = tmp_path / "global" / "review.md"
    second = tmp_path / "project" / "review.md"
    for path, body in [(first, "First: $1 / $2"), (second, "Second: $@")]:
        path.parent.mkdir()
        path.write_text(body)
    sources = [PromptTemplateSource(first, "global"), PromptTemplateSource(second, "project")]
    loaded = load_prompt_templates(sources)
    first.write_text("Updated: $1")

    assert expand_prompt_template('/review "two words" \'three words\'', loaded.templates) == "First: two words / three words"
    assert expand_prompt_template("/review\nfirst\tsecond", loaded.templates) == "First: first / second"
    assert expand_prompt_template("/review", loaded.templates) == "First:  / "
    assert expand_prompt_template("/review first", load_prompt_templates(sources).templates) == "Updated: first"


@pytest.mark.parametrize("text", ["ordinary $1", " /review value", "/", "/ review", "/unknown value\n", "/Review value", "/review-extra value"])
def test_unknown_names_and_unmatched_input_pass_through_verbatim(tmp_path: Path, text: str) -> None:
    path = tmp_path / "review.md"
    path.write_text("Known $1")
    loaded = load_prompt_templates([PromptTemplateSource(path)])
    assert expand_prompt_template(text, loaded.templates) == text


async def test_expanded_and_unknown_input_reach_agent_as_ordinary_messages(tmp_path: Path) -> None:
    path = tmp_path / "review.md"
    path.write_text('---\ndescription: Private metadata $2\n---\nReview $1. ${2:-$ARGUMENTS}')
    loaded = load_prompt_templates([PromptTemplateSource(path)])
    stream = ScriptedStreamFn([lambda: text_message("done"), lambda: text_message("raw accepted")])
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    expanded = expand_prompt_template('/review "quoted words $2"', loaded.templates)
    await agent.prompt(expanded)
    await agent.prompt(expand_prompt_template("/unknown untouched", loaded.templates))

    users = [m for m in stream.requests[-1].context.messages if isinstance(m, UserMessage)]
    assert [m.content for m in users] == [
        [TextContent(text="Review quoted words $2. $ARGUMENTS")], [TextContent(text="/unknown untouched")],
    ]
    assert loaded.templates[0].description == "Private metadata $2"
    await agent.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
@pytest.mark.parametrize("deny_directory", [False, True])
def test_unreadable_selected_resources_return_diagnostics(tmp_path: Path, deny_directory: bool) -> None:
    directory = tmp_path / "prompts"
    directory.mkdir()
    path = directory / "review.md"
    path.write_text("Review")
    denied = directory if deny_directory else path
    denied.chmod(0)
    try:
        result = load_prompt_templates([PromptTemplateSource(directory, "project")])
    finally:
        denied.chmod(0o700 if deny_directory else 0o600)
    assert not result.templates
    assert [(d.path, d.source, d.reason) for d in result.diagnostics] == [(str(denied), "project", "read_error")]


def test_empty_template_expands_to_empty_input_without_fallback(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("---\ndescription: Empty\n---\n")
    loaded = load_prompt_templates([PromptTemplateSource(path)])
    assert expand_prompt_template("/empty supplied args", loaded.templates) == ""
