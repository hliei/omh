"""Executable tool contracts and model-facing declarations."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from omh.llm.types import (
    AbortSignal,
    ImageContent,
    TextContent,
    Tool,
    ToolCall,
    Usage,
)
from omh.llm.utils.transcript import to_tool_declaration as _to_llm_tool_declaration

#: How the tool calls of one assistant message are executed.
#:
#: - ``sequential``: each call is prepared, executed, and finalized before the
#:   next one starts.
#: - ``parallel``: calls are prepared in source order, then the allowed calls
#:   execute concurrently. ``tool_execution_end`` is emitted in completion
#:   order, while result messages keep assistant source order.
ToolExecutionMode = Literal["sequential", "parallel"]

#: A single tool call emitted by an assistant message.
AgentToolCall = ToolCall


@dataclass(slots=True)
class AgentToolResult:
    """Result returned by an :class:`AgentTool` execution."""

    content: list[TextContent | ImageContent]
    details: object | None = None
    usage: Usage | None = None
    terminate: bool | None = None


#: Progress callback for one ``execute`` invocation. Calls made after the tool
#: settles are ignored. The loop settles accepted progress before finalizing.
AgentToolUpdateCallback = Callable[[AgentToolResult], None]

AgentToolExecute = Callable[
    [str, dict[str, object], AbortSignal | None, AgentToolUpdateCallback],
    Awaitable[AgentToolResult] | AgentToolResult,
]

#: Optional compatibility shim applied to raw tool-call arguments before schema
#: conversion and validation. It must return the argument object to validate.
AgentToolPrepareArguments = Callable[[dict[str, object]], dict[str, object]]


@dataclass(slots=True)
class AgentTool:
    """Executable tool held by the Agent.

    The declaration sent to the model is derived from ``name``, ``description``
    and ``parameters`` only; ``label``, ``execute``, ``prepare_arguments`` and
    ``execution_mode`` stay on the execution side.
    """

    name: str
    description: str
    parameters: dict[str, object]
    label: str
    execute: AgentToolExecute
    prepare_arguments: AgentToolPrepareArguments | None = None
    execution_mode: ToolExecutionMode | None = None


def to_tool_declaration(tool: AgentTool) -> Tool:
    """Build the model-facing declaration for an executable tool."""
    return _to_llm_tool_declaration(
        Tool(name=tool.name, description=tool.description, parameters=copy.deepcopy(tool.parameters))
    )
