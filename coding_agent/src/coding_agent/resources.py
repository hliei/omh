"""Application resource snapshots assembled through public SDK helpers.

The host composes ordered sources; this module only resolves them into SDK
resource values, optional named system sections and diagnostics. Source labels
name the tier a path came from so winner/loser diagnostics stay explainable.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from omh.agent import (
    AgentTool,
    ProjectContextFile,
    PromptTemplate,
    PromptTemplateDiagnostic,
    PromptTemplateSource,
    ResourceDiagnostic,
    Skill,
    SkillDiagnostic,
    SkillSource,
    build_system_sections,
    format_skills_for_prompt,
    load_project_context_files,
    load_prompt_templates,
    load_skills,
)

from coding_agent.commands import RESERVED_COMMANDS
from coding_agent.config import (
    AGENTS_RESOURCES_DIR,
    CONFIG_DIR_NAME,
    PROMPTS_DIR,
    SKILLS_DIR,
)

if TYPE_CHECKING:
    from coding_agent.agent_session import CodingAgentOptions

ApplicationDiagnostic = ResourceDiagnostic | SkillDiagnostic | PromptTemplateDiagnostic

#: Ordered source tiers used by the installed product.
#:
#: An explicit path is the user's own choice; a configured project path is
#: authorized by trust; automatic project discovery is only meaningful once the
#: project is trusted; configured and automatic global resources are always
#: available. Embedded callers that do not supply tiers keep the historical
#: ``global -> project -> explicit`` ordering.
RESOURCE_TIERS: tuple[str, ...] = (
    "explicit", "project-config", "project-auto", "global-config", "global-auto",
)

_EMBEDDED_PRIORITY = {"global": 0, "project": 1, "explicit": 2}


class _TieredSource(Protocol):
    @property
    def source(self) -> str: ...


_SourceT = TypeVar("_SourceT", bound=_TieredSource)


@dataclass(frozen=True, slots=True)
class ApplicationResources:
    context_files: tuple[ProjectContextFile, ...] = ()
    skills: tuple[Skill, ...] = ()
    templates: tuple[PromptTemplate, ...] = ()
    diagnostics: tuple[ApplicationDiagnostic, ...] = ()


def git_ancestor_directories(cwd: str | Path) -> tuple[Path, ...]:
    """Directories from ``cwd`` through the nearest Git root or filesystem root.

    Git metadata is inspected as a file or directory; no subprocess runs.
    """
    working_dir = Path(cwd).expanduser().resolve()
    directories: list[Path] = []
    for directory in (working_dir, *working_dir.parents):
        directories.append(directory)
        if (directory / ".git").exists():
            break
    return tuple(directories)


def project_skill_directories(cwd: str | Path) -> tuple[Path, ...]:
    """Directories whose automatic skills are trust-controlled for a project.

    That is ``<cwd>/.omh/skills`` and ``.agents/skills`` from the cwd through
    the nearest Git root (or the filesystem root).
    """
    working_dir = Path(cwd).expanduser().resolve()
    return (
        working_dir / CONFIG_DIR_NAME / SKILLS_DIR,
        *(
            directory / AGENTS_RESOURCES_DIR / SKILLS_DIR
            for directory in git_ancestor_directories(working_dir)
        ),
    )


def automatic_skill_sources(
    *, cwd: str | Path, agent_dir: str | Path, home: str | Path, project_trusted: bool,
) -> tuple[SkillSource, ...]:
    """Return the automatically discovered skill directories in priority order.

    Project tiers appear only for a trusted project: ``<cwd>/.omh/skills`` and
    then ``.agents/skills`` from the cwd through the nearest Git root. Global
    tiers are the agent directory's ``skills`` and ``<home>/.agents/skills``.
    Absent directories are omitted so a normal run reports no read error.
    """
    working_dir = Path(cwd).expanduser().resolve()
    global_dir = Path(agent_dir).expanduser()
    home_dir = Path(home).expanduser()
    entries: list[tuple[Path, str]] = []
    if project_trusted:
        entries.extend((path, "project-auto") for path in project_skill_directories(working_dir))
    entries.append((global_dir / SKILLS_DIR, "global-auto"))
    entries.append((home_dir / AGENTS_RESOURCES_DIR / SKILLS_DIR, "global-auto"))
    return tuple(SkillSource(path, tier) for path, tier in entries if path.is_dir())


def automatic_template_sources(
    *, cwd: str | Path, agent_dir: str | Path, project_trusted: bool,
) -> tuple[PromptTemplateSource, ...]:
    """Return the automatically discovered template directories in priority order.

    Only direct Markdown children of ``<cwd>/.omh/prompts`` (trusted project)
    and ``<agent_dir>/prompts`` are discovered; discovery does not recurse.
    Absent directories are omitted so a normal run reports no read error.
    """
    working_dir = Path(cwd).expanduser().resolve()
    global_dir = Path(agent_dir).expanduser()
    entries: list[tuple[Path, str]] = []
    if project_trusted:
        entries.append((working_dir / CONFIG_DIR_NAME / PROMPTS_DIR, "project-auto"))
    entries.append((global_dir / PROMPTS_DIR, "global-auto"))
    return tuple(PromptTemplateSource(path, tier) for path, tier in entries if path.is_dir())


def _ordered(sources: Sequence[_SourceT], tiers: tuple[str, ...] | None) -> list[_SourceT]:
    """Order sources by tier; stable within a tier so input order is first-wins."""
    if tiers is None:
        return sorted(sources, key=lambda source: _EMBEDDED_PRIORITY.get(source.source, 2))
    index = {tier: position for position, tier in enumerate(tiers)}
    return sorted(sources, key=lambda source: index.get(source.source, len(tiers)))


def _builtin_conflicts(templates: Sequence[PromptTemplate]) -> list[PromptTemplateDiagnostic]:
    return [
        PromptTemplateDiagnostic(
            template.file_path, template.source, "builtin-conflict",
            f'Prompt template "{template.name}" uses a built-in command name; '
            "the built-in command wins",
        )
        for template in templates
        if template.name in RESERVED_COMMANDS
    ]


def load_resources(
    options: "CodingAgentOptions", cwd: Path, tools: list[AgentTool],
) -> tuple[ApplicationResources, dict[str, str]]:
    # Paths at this application boundary are relative to the session cwd.
    def path(value: str | Path) -> Path:
        return cwd / Path(value).expanduser()

    context = (
        load_project_context_files(
            cwd=cwd, agent_dir=path(options.agent_dir) if options.agent_dir is not None else None,
            context_dirs=[path(directory) for directory in options.context_dirs],
        )
        if options.load_context_files else None
    )
    skills = load_skills(_ordered(options.skill_sources, options.resource_tiers), cwd=cwd)
    templates = load_prompt_templates(_ordered(options.template_sources, options.resource_tiers), cwd=cwd)
    resources = ApplicationResources(
        tuple(context.files) if context is not None else (),
        tuple(skills.skills), tuple(templates.templates),
        (
            *(context.diagnostics if context is not None else ()),
            *skills.diagnostics, *templates.diagnostics, *_builtin_conflicts(templates.templates),
        ),
    )
    return resources, resource_sections(options, cwd, tools, resources)


def resource_sections(
    options: "CodingAgentOptions", cwd: Path, tools: list[AgentTool], resources: ApplicationResources,
) -> dict[str, str]:
    """Build next-prompt sections from an already accepted resource snapshot."""
    sections = build_system_sections(
        cwd=cwd, context_files=resources.context_files,
        custom_prompt=options.custom_prompt, append_system_prompt=options.append_system_prompt,
        selected_tools=[tool.name for tool in tools],
        tool_snippets={tool.name: tool.description for tool in tools},
    )
    catalog = format_skills_for_prompt(resources.skills, tools=tools)
    if catalog:
        sections["skills"] = catalog
    return sections
