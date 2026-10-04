"""Application resource snapshots assembled through public SDK helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omh.agent import (
    AgentTool,
    ProjectContextFile,
    PromptTemplate,
    PromptTemplateDiagnostic,
    ResourceDiagnostic,
    Skill,
    SkillDiagnostic,
    build_system_sections,
    format_skills_for_prompt,
    load_project_context_files,
    load_prompt_templates,
    load_skills,
)

if TYPE_CHECKING:
    from coding_agent.agent_session import CodingAgentOptions

ApplicationDiagnostic = ResourceDiagnostic | SkillDiagnostic | PromptTemplateDiagnostic


@dataclass(frozen=True, slots=True)
class ApplicationResources:
    context_files: tuple[ProjectContextFile, ...] = ()
    skills: tuple[Skill, ...] = ()
    templates: tuple[PromptTemplate, ...] = ()
    diagnostics: tuple[ApplicationDiagnostic, ...] = ()


def load_resources(
    options: "CodingAgentOptions", cwd: Path, tools: list[AgentTool],
) -> tuple[ApplicationResources, dict[str, str]]:
    # Paths at this application boundary are relative to the session cwd.
    def path(value: str | Path) -> Path:
        return cwd / Path(value).expanduser()

    priority = {"global": 0, "project": 1, "explicit": 2}
    context = load_project_context_files(
        cwd=cwd, agent_dir=path(options.agent_dir) if options.agent_dir is not None else None,
        context_dirs=[path(directory) for directory in options.context_dirs],
    )
    skills = load_skills(
        sorted(options.skill_sources, key=lambda source: priority.get(source.source, 2)), cwd=cwd,
    )
    templates = load_prompt_templates(
        sorted(options.template_sources, key=lambda source: priority.get(source.source, 2)), cwd=cwd,
    )
    sections = build_system_sections(
        cwd=cwd, context_files=context.files, custom_prompt=options.custom_prompt,
        selected_tools=[tool.name for tool in tools],
        tool_snippets={tool.name: tool.description for tool in tools},
    )
    catalog = format_skills_for_prompt(skills.skills, tools=tools)
    if catalog:
        sections["skills"] = catalog
    return ApplicationResources(
        tuple(context.files), tuple(skills.skills), tuple(templates.templates),
        (*context.diagnostics, *skills.diagnostics, *templates.diagnostics),
    ), sections
