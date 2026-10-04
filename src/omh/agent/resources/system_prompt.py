"""Build named system sections from explicit host inputs, without loading files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path

from omh.agent.resources.context_files import ProjectContextFile


def build_system_sections(
    *, cwd: str | Path,
    context_files: Sequence[ProjectContextFile] = (),
    custom_prompt: str | None = None,
    selected_tools: Sequence[str] = (),
    tool_snippets: Mapping[str, str] | None = None,
    tool_guidelines: Mapping[str, Sequence[str]] | None = None,
    prompt_guidelines: Sequence[str] = (),
    append_system_prompt: str | None = None,
) -> dict[str, str]:
    """Return ordered sections ready for ``SystemMessage`` or Agent configuration.

    ``custom_prompt`` replaces the default preamble, tools and rules, including
    when it is empty. Project context, an optional addendum and cwd still follow.
    Tools are prompt information only; the host supplies executable AgentTools
    separately. Only explicitly selected tools contribute snippets or rules.
    """
    sections: dict[str, str] = {}
    if custom_prompt is not None:
        sections["preamble"] = custom_prompt
    else:
        sections["preamble"] = (
            "You are a coding assistant. Help the user understand and change their project."
        )
        snippets = tool_snippets or {}
        tools = [f"- {name}: {snippets[name]}" for name in selected_tools if snippets.get(name)]
        sections["tools"] = "\n".join(tools) if tools else "(none)"
        guidelines = tool_guidelines or {}
        rules: list[str] = []
        if "bash" in selected_tools and not any(name in selected_tools for name in ("grep", "find", "ls")):
            rules.append("Use bash to list and search project files.")
        for name in selected_tools:
            rules.extend(guidelines.get(name, ()))
        rules.extend(prompt_guidelines)
        rules.extend(("Keep responses concise.", "Show file paths clearly."))
        sections["rules"] = "\n".join(
            f"- {rule}" for rule in dict.fromkeys(rule.strip() for rule in rules if rule.strip())
        )
    if append_system_prompt:
        sections["addendum"] = append_system_prompt
    if context_files:
        sections["project_context"] = "\n\n".join([
            "Project instructions:",
            *(f'<project_instructions path="{escape(file.path, quote=True)}">\n'
              f"{file.content}\n</project_instructions>" for file in context_files),
        ])
    sections["cwd"] = str(cwd).replace("\\", "/")
    return {
        name: text if name == "preamble" else f"<{name}>\n{text}\n</{name}>"
        for name, text in sections.items()
    }
