"""Synchronize executable tools with the declarations carried by the transcript."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from omh.agent.context import AgentContext
from omh.agent.messages import LoopMessage
from omh.agent.tools import to_tool_declaration
from omh.llm.types import SystemMessage, Tool, ToolReference
from omh.llm.utils.transcript import get_current_tools, get_tool_state_changes


def declare_tool_changes(context: AgentContext, pending_messages: list[LoopMessage]) -> list[LoopMessage]:
    """Announce the difference between executable and transcript tools before a request.

    A pending system message has its tool fields treated as intent and replaced
    with the delta between the committed transcript and the executable set, so
    replaying the transcript always yields exactly ``context.tools``. Otherwise a
    new system message is inserted before the first non-system pending message.
    """
    system_index = -1
    for index in range(len(pending_messages) - 1, -1, -1):
        if pending_messages[index].role == "system":
            system_index = index
            break

    pending = (
        cast(SystemMessage, pending_messages[system_index]) if system_index >= 0 else None
    )
    if pending is not None:
        baseline = [
            _with_tool_changes(cast(SystemMessage, message), [], []) if index == system_index else message
            for index, message in enumerate(pending_messages)
        ]
    else:
        baseline = pending_messages

    changes = get_tool_state_changes(
        get_current_tools([*context.messages, *baseline]),
        [to_tool_declaration(tool) for tool in context.tools],
    )
    unchanged = not changes.tools_added and not changes.tools_removed

    if pending is not None:
        if unchanged and not pending.tools_added and not pending.tools_removed:
            return pending_messages
        return [
            _with_tool_changes(cast(SystemMessage, message), changes.tools_added, changes.tools_removed)
            if index == system_index
            else message
            for index, message in enumerate(baseline)
        ]

    if unchanged:
        return pending_messages

    update = _with_tool_changes(SystemMessage(content="", timestamp=0), changes.tools_added, changes.tools_removed)
    insert_index = next(
        (index for index, message in enumerate(pending_messages) if message.role != "system"),
        len(pending_messages),
    )
    return [*pending_messages[:insert_index], update, *pending_messages[insert_index:]]


def _with_tool_changes(
    message: SystemMessage,
    tools_added: Sequence[Tool],
    tools_removed: Sequence[ToolReference],
) -> SystemMessage:
    """Copy a system message with its tool fields replaced; empty lists omit the field."""
    return replace(
        message,
        tools_added=list(tools_added) or None,
        tools_removed=list(tools_removed) or None,
    )
