"""Original-file text replacements through the in-process AgentTool contract."""

import json
from pathlib import Path
from typing import cast

from omh._tool_utils.arguments import required_string
from omh._tool_utils.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from omh.agent.coding_tools.file_io import run_file_io
from omh.agent.coding_tools.file_mutation_queue import with_file_mutation_queue
from omh.agent.coding_tools.path_utils import resolve_tool_path
from omh.agent.tools import AgentTool, AgentToolResult, AgentToolUpdateCallback
from omh.llm.types import AbortSignal, TextContent


def create_edit_tool(cwd: str | Path) -> AgentTool:
    """Create an edit tool with an explicit working directory."""
    directory = Path(cwd).expanduser().absolute()

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        if signal is not None:
            signal.throw_if_aborted()
        path = required_string(arguments, "path")
        raw_edits = arguments.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ValueError("edits must contain at least one replacement")
        edits: list[Edit] = []
        for index, raw_edit in enumerate(raw_edits):
            if not isinstance(raw_edit, dict):
                raise ValueError(f"edits[{index}] must be an object")
            record = cast(dict[str, object], raw_edit)
            old_text, new_text = record.get("oldText"), record.get("newText")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ValueError(f"edits[{index}] must have string oldText and newText")
            edits.append(Edit(old_text=old_text, new_text=new_text))
        absolute_path = resolve_tool_path(directory, path)

        async def edit() -> AgentToolResult:
            # Read bytes to retain BOM and CRLF instead of universal-newline decoding.
            content = (await run_file_io(absolute_path.read_bytes)).decode("utf-8", errors="replace")
            if signal is not None:
                signal.throw_if_aborted()
            bom, content = strip_bom(content)
            ending = detect_line_ending(content)
            applied = apply_edits_to_normalized_content(normalize_to_lf(content), edits, path)
            if signal is not None:
                signal.throw_if_aborted()
            final = bom + restore_line_endings(applied.new_content, ending)
            await run_file_io(lambda: absolute_path.write_text(final, encoding="utf-8", newline=""))
            if signal is not None:
                signal.throw_if_aborted()
            diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details={
                    "diff": diff,
                    "patch": generate_unified_patch(path, applied.base_content, applied.new_content),
                    "firstChangedLine": first_changed_line,
                },
            )

        return await with_file_mutation_queue(absolute_path, edit, signal)

    return AgentTool(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText "
            "must match a unique, non-overlapping region of the original file. "
            "Merge overlapping or nearby changes into one edit. Do not include "
            "large unchanged regions just to connect distant changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative or absolute)"},
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Replacements matched against the original file, not incrementally.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string", "description": "Unique original text to replace"},
                            "newText": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        execute=execute,
        prepare_arguments=_prepare_edit_arguments,
    )


def _prepare_edit_arguments(arguments: dict[str, object]) -> dict[str, object]:
    prepared = dict(arguments)
    edits = prepared.get("edits")
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except ValueError as error:
            raise ValueError("edits must be a JSON list or replacement object") from error
    if isinstance(edits, dict):
        edits = [edits]
    if "oldText" in prepared or "newText" in prepared:
        old_text = required_string(prepared, "oldText")
        new_text = required_string(prepared, "newText")
        if "edits" not in prepared:
            edits = []
        if not isinstance(edits, list):
            raise ValueError("edits must be a list or replacement object")
        edits = [*edits, {"oldText": old_text, "newText": new_text}]
        del prepared["oldText"], prepared["newText"]
    prepared["edits"] = edits
    return prepared
