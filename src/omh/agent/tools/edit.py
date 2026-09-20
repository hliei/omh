"""The built-in edit tool.

Corresponds to ``packages/agent/src/harness/tools/edit.ts``. The Python tool
accepts the documented ``edits`` array directly; the upstream legacy
``oldText``/``newText`` and JSON-string argument normalization is not applied
because tool declarations have no prepare-arguments hook in this first version.
"""

from __future__ import annotations

from typing import cast

from omh.agent.agent_harness import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentToolResult,
)
from omh.agent.context import Context
from omh.agent.execution_env import ExecutionEnv, FileError
from omh.agent.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from omh.agent.tools.file_mutation_queue import with_file_mutation_queue
from omh.agent.tools.path_utils import resolve_tool_path
from omh.agent.tools.tool_context import execution_env
from omh.llm.types import JsonValue, TextContent

_REPLACE_EDIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "oldText": {
            "type": "string",
            "description": (
                "Exact text for one targeted replacement. It must be unique in "
                "the original file and must not overlap with any other "
                "edits[].oldText in the same call."
            ),
        },
        "newText": {
            "type": "string",
            "description": "Replacement text for this targeted edit.",
        },
    },
    "required": ["oldText", "newText"],
}

_EDIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit (relative or absolute)",
        },
        "edits": {
            "type": "array",
            "items": _REPLACE_EDIT_SCHEMA,
            "description": (
                "One or more targeted replacements. Each edit is matched "
                "against the original file, not incrementally. Do not include "
                "overlapping or nested edits. If two changes touch the same "
                "block or nearby lines, merge them into one edit instead."
            ),
        },
    },
    "required": ["path", "edits"],
}


def create_edit_tool() -> AgentHarnessTool:
    """Create the ``edit`` tool. ``tool_context`` must be an ``ExecutionToolContext``."""

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        on_update: AgentHarnessToolUpdateCallback,
        tool_context: object,
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, on_update, invocation
        env = execution_env(tool_context)
        path, edits = _validate_edit_input(arguments)
        absolute_path = await resolve_tool_path(env, path, context)

        async def edit() -> AgentToolResult:
            return await _apply_edit(env, absolute_path, path, edits, context)

        return await with_file_mutation_queue(
            env, absolute_path, edit, context
        )

    return AgentHarnessTool(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every "
            "edits[].oldText must match a unique, non-overlapping region of "
            "the original file. If two changes affect the same block or nearby "
            "lines, merge them into one edit instead of emitting overlapping "
            "edits. Do not include large unchanged regions just to connect "
            "distant changes."
        ),
        parameters=_EDIT_SCHEMA,
        execute=execute,
    )


async def _apply_edit(
    env: ExecutionEnv,
    absolute_path: str,
    path: str,
    edits: list[Edit],
    context: Context,
) -> AgentToolResult:
    context.raise_if_cancelled()
    info = await env.file_info(absolute_path, context)
    if not info.ok:
        raise _edit_access_error(path, info.error)
    if info.value.kind not in {"file", "symlink"}:
        raise ValueError(f"Could not edit file: {path}. Path is not a file.")

    read_result = await env.read_text_file(absolute_path, context)
    if not read_result.ok:
        raise _edit_access_error(path, read_result.error)
    context.raise_if_cancelled()

    bom, content = strip_bom(read_result.value)
    original_ending = detect_line_ending(content)
    normalized_content = normalize_to_lf(content)
    applied = apply_edits_to_normalized_content(normalized_content, edits, path)
    context.raise_if_cancelled()

    final_content = bom + restore_line_endings(applied.new_content, original_ending)
    write_result = await env.write_file(absolute_path, final_content, context)
    if not write_result.ok:
        raise _edit_access_error(path, write_result.error)
    context.raise_if_cancelled()

    diff, first_changed_line = generate_diff_string(
        applied.base_content, applied.new_content
    )
    patch = generate_unified_patch(path, applied.base_content, applied.new_content)
    details = cast(
        JsonValue,
        {
            "diff": diff,
            "patch": patch,
            "firstChangedLine": first_changed_line,
        },
    )
    return AgentToolResult(
        content=[
            TextContent(
                text=f"Successfully replaced {len(edits)} block(s) in {path}."
            )
        ],
        details=details,
    )


def _validate_edit_input(
    arguments: dict[str, object],
) -> tuple[str, list[Edit]]:
    path = arguments.get("path")
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    raw_edits = arguments.get("edits")
    if not isinstance(raw_edits, list) or len(raw_edits) == 0:
        raise ValueError(
            "Edit tool input is invalid. edits must contain at least one replacement."
        )
    edits: list[Edit] = []
    for index, raw_edit in enumerate(raw_edits):
        if not isinstance(raw_edit, dict):
            raise ValueError(f"edits[{index}] must be an object")
        record = cast(dict[str, object], raw_edit)
        old_text = record.get("oldText")
        new_text = record.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValueError(
                f"edits[{index}] must have string oldText and newText"
            )
        edits.append(Edit(old_text=old_text, new_text=new_text))
    return path, edits


def _edit_access_error(path: str, error: FileError) -> ValueError:
    return ValueError(
        f"Could not edit file: {path}. Error code: {error.code}."
    )
