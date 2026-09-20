"""The built-in bash tool.

Corresponds to ``packages/agent/src/harness/tools/bash.ts``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from omh.agent.agent_harness import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentHarnessToolUpdateOptions,
    AgentToolResult,
)
from omh.agent.context import Context
from omh.agent.execution_env import (
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputUpdate,
    ShellOutputView,
)
from omh.agent.tools.tool_context import execution_env
from omh.agent.utils.output_capture import apply_shell_output_update
from omh.agent.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncation_json,
)
from omh.llm.types import JsonValue, TextContent

_MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
_BASH_CHECKPOINT_INTERVAL_MS = 2_000

_BASH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds (optional, no default timeout)",
        },
    },
    "required": ["command"],
}


@dataclass(frozen=True, slots=True)
class BashExecution:
    command: str
    cwd: str
    env: dict[str, str]
    inherit_env: bool


type BashPrepare = Callable[
    [BashExecution, object, Context], Awaitable[None] | None
]


@dataclass(frozen=True, slots=True)
class BashToolOptions:
    command_prefix: str | None = None
    prepare: BashPrepare | None = None


def create_bash_tool(options: BashToolOptions | None = None) -> AgentHarnessTool:
    """Create the ``bash`` tool. ``tool_context`` must be an ``ExecutionToolContext``."""
    resolved_options = options or BashToolOptions()

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        on_update: AgentHarnessToolUpdateCallback,
        tool_context: object,
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, invocation
        env = execution_env(tool_context)
        command = _required_string(arguments, "command")
        timeout = _optional_number(arguments, "timeout")
        _validate_timeout(timeout)

        execution = BashExecution(
            command=(
                f"{resolved_options.command_prefix}\n{command}"
                if resolved_options.command_prefix
                else command
            ),
            cwd=env.cwd,
            env={},
            inherit_env=True,
        )
        if resolved_options.prepare is not None:
            prepared = resolved_options.prepare(execution, tool_context, context)
            if prepared is not None:
                await prepared

        view: ShellOutputView | None = None
        last_checkpoint_at = _now_ms()
        last_checkpoint: str | None = None
        accepting_updates = True

        def handle_update(update: ShellOutputUpdate, update_context: Context) -> None:
            nonlocal view, last_checkpoint_at, last_checkpoint
            if not accepting_updates:
                return
            view = apply_shell_output_update(view, update)
            snapshot = AgentToolResult(
                content=[TextContent(text=view.text)],
                details=_bash_details(view),
            )
            now = _now_ms()
            encoded = _encode_snapshot(snapshot)
            checkpoint = (
                now - last_checkpoint_at >= _BASH_CHECKPOINT_INTERVAL_MS
                and encoded != last_checkpoint
            )
            on_update(
                snapshot,
                AgentHarnessToolUpdateOptions(checkpoint=True) if checkpoint else None,
            )
            if checkpoint:
                last_checkpoint_at = now
                last_checkpoint = encoded
            del update_context

        on_update(AgentToolResult(content=[]))
        result = await env.exec(
            execution.command,
            ShellExecOptions(
                cwd=execution.cwd,
                env=execution.env,
                inherit_env=execution.inherit_env,
                timeout=timeout,
                capture=ShellOutputCaptureOptions(
                    limits=ShellOutputLimits(
                        max_bytes=DEFAULT_MAX_BYTES,
                        max_lines=DEFAULT_MAX_LINES,
                        retain="tail",
                    ),
                    spill=True,
                ),
                on_update=handle_update,
            ),
            context,
        )
        accepting_updates = False

        output_text = view.text if view is not None else ""
        if result.ok:
            truncation = result.value.truncation
            spill_path = result.value.spill_path
            last_line_bytes = result.value.last_line_bytes
        else:
            truncation = (
                view.truncation
                if view is not None
                else _empty_view().truncation
            )
            spill_path = view.spill_path if view is not None else None
            last_line_bytes = view.last_line_bytes if view is not None else None

        details: JsonValue = None
        if truncation.truncated:
            details = cast(
                JsonValue,
                {
                    "truncation": truncation_json(truncation),
                    "fullOutputPath": spill_path,
                },
            )
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(
                    last_line_bytes
                    if last_line_bytes is not None
                    else truncation.output_bytes
                )
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of "
                    f"line {end_line} (line is {last_line_size}). Full output: "
                    f"{spill_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of "
                    f"{truncation.total_lines}. Full output: {spill_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of "
                    f"{truncation.total_lines} ({format_size(DEFAULT_MAX_BYTES)} "
                    f"limit). Full output: {spill_path}]"
                )

        if not result.ok:
            error = result.error
            status = (
                f"Command timed out after {timeout} seconds"
                if error.code == "timeout"
                else "Command aborted"
                if error.code == "aborted"
                else str(error)
            )
            raise ValueError(
                f"{output_text}\n\n{status}" if output_text else status
            ) from error
        if result.value.exit_code != 0:
            prefix = f"{output_text}\n\n" if output_text else ""
            raise ValueError(
                f"{prefix}Command exited with code {result.value.exit_code}"
            )
        return AgentToolResult(
            content=[TextContent(text=output_text or "(no output)")],
            details=details,
        )

    return AgentHarnessTool(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns "
            "combined stdout and stderr. Output is truncated to last "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a "
            "temp file. Optionally provide a timeout in seconds."
        ),
        parameters=_BASH_SCHEMA,
        execute=execute,
    )


def _bash_details(view: ShellOutputView) -> JsonValue:
    return cast(
        JsonValue,
        {
            "truncation": (
                truncation_json(view.truncation)
                if view.truncation.truncated
                else None
            ),
            "fullOutputPath": view.spill_path,
        },
    )


def _empty_view() -> ShellOutputView:
    from omh.agent.utils.truncate import truncate_tail

    return ShellOutputView(
        text="", truncation=truncate_tail("").without_content()
    )


def _encode_snapshot(snapshot: AgentToolResult) -> str:
    return json.dumps(
        {
            "text": (
                snapshot.content[0].text
                if snapshot.content and isinstance(snapshot.content[0], TextContent)
                else ""
            ),
            "details": snapshot.details,
        },
        sort_keys=True,
        default=str,
    )


def _now_ms() -> float:
    return time.monotonic() * 1000


def _required_string(arguments: dict[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_number(arguments: dict[str, object], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if timeout != timeout or timeout in (float("inf"), float("-inf")) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > _MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"Invalid timeout: maximum is {_MAX_TIMEOUT_SECONDS} seconds"
        )
