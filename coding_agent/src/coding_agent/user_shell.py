"""Host-owned shell execution and its display/context history contract."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omh.agent import AgentToolResult, CustomAgentMessage, create_bash_tool
from omh.llm.types import AbortSignal, JsonValue, TextContent

SHELL_TYPE = "user_shell"
HIDDEN_SHELL_TYPE = "user_shell_hidden"


@dataclass(slots=True)
class UserShell:
    command: str
    exclude_context: bool
    status: str = "running"
    output: str = ""
    result_details: JsonValue = None

    async def execute(self, cwd: Path, signal: AbortSignal, redraw: Callable[[], None]) -> None:
        def progress(result: AgentToolResult) -> None:
            self.output = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
            self.result_details = cast(JsonValue, result.details)
            redraw()

        try:
            execution = create_bash_tool(cwd).execute(
                "user-shell", {"command": self.command}, signal, progress,
            )
            result = await execution if inspect.isawaitable(execution) else execution
        except Exception as error:
            # The public exception text contains the final bounded output and
            # termination explanation. The last callback retains spill details.
            self.output = str(error)
            self.status = "cancelled" if signal.aborted else "failed"
        else:
            progress(result)
            self.status = "completed"
        redraw()

    def message(self) -> CustomAgentMessage:
        content = f"User shell command: {self.command}\nStatus: {self.status}\n{self.output}"
        return CustomAgentMessage(
            custom_type=HIDDEN_SHELL_TYPE if self.exclude_context else SHELL_TYPE,
            content="" if self.exclude_context else content,
            details={
                "command": self.command, "status": self.status, "output": self.output,
                "result": self.result_details,
            },
        )

    @classmethod
    def from_message(cls, message: CustomAgentMessage) -> UserShell | None:
        if message.custom_type not in (SHELL_TYPE, HIDDEN_SHELL_TYPE):
            return None
        details = message.details
        if not isinstance(details, dict):
            return None
        command, status, output = (details.get(key) for key in ("command", "status", "output"))
        if not isinstance(command, str) or not isinstance(status, str) or not isinstance(output, str):
            return None
        return cls(command, message.custom_type == HIDDEN_SHELL_TYPE, status, output, details.get("result"))
