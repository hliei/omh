"""Initial values and read-only, isolated observations of an in-process Agent."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Literal

from omh.agent.data import snapshot_messages, validate_json
from omh.agent.messages import AgentMessage
from omh.agent.tools import AgentTool, to_tool_declaration
from omh.llm.types import Model, ModelCost
from omh.llm.types import ModelThinkingLevel as ThinkingLevel
from omh.llm.utils.transcript import (
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
)

#: Fallback model used when an Agent is created without an explicit model. Mirrors
#: the baseline's placeholder so callers can inspect identity before configuring one.
DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="openai-completions",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=(),
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=0,
    max_tokens=0,
)


@dataclass(slots=True)
class AgentInitialState:
    """Seed values for a new :class:`~omh.agent.agent.Agent`."""

    system_prompt: str | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    tools: list[AgentTool] | None = None
    messages: list[AgentMessage] | None = None


class AgentState:
    """Read-only observations. Mutable message elements are isolated copies."""

    def __init__(self, initial: AgentInitialState | None = None) -> None:
        initial = initial or AgentInitialState()
        self._tools = snapshot_tools(initial.tools or [])
        self._messages = snapshot_messages(initial.messages or [])
        initial_message = create_initial_system_message(
            initial.system_prompt,
            [to_tool_declaration(tool) for tool in self._tools],
        )
        if (not self._messages or self._messages[0].role != "system") and initial_message is not None:
            self._messages.insert(0, initial_message)
        self._model = copy.deepcopy(initial.model if initial.model is not None else DEFAULT_MODEL)
        self._thinking_level: ThinkingLevel = initial.thinking_level or "off"
        replayed_system = get_current_system_message(self._messages)
        self._system_sections: dict[str, str] = {
            name: value
            for name, value in (replayed_system.sections or {}).items()
            if value is not None
        } if replayed_system is not None else {}
        self._is_streaming = False
        self._is_busy = False
        self._is_closed = False
        self._activity_kind: Literal["dialogue", "manual_compaction"] | None = None
        self._streaming_message: AgentMessage | None = None
        self._error_message: str | None = None
        self._pending_tool_calls: set[str] = set()

    @property
    def system_prompt(self) -> str:
        return get_current_system_prompt(self._messages)

    @property
    def pending_tool_calls(self) -> frozenset[str]:
        """Tool call ids currently executing, tracked from tool execution events."""
        return frozenset(self._pending_tool_calls)

    def _add_pending_tool_call(self, tool_call_id: str) -> None:
        self._pending_tool_calls.add(tool_call_id)

    def _remove_pending_tool_call(self, tool_call_id: str) -> None:
        self._pending_tool_calls.discard(tool_call_id)

    def _clear_pending_tool_calls(self) -> None:
        self._pending_tool_calls.clear()

    @property
    def model(self) -> Model:
        return copy.deepcopy(self._model)

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    @property
    def system_sections(self) -> dict[str, str]:
        """Expected named base sections, isolated from the Agent's configuration."""
        return dict(self._system_sections)

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def is_busy(self) -> bool:
        """Whether an accepted activity, including its listeners, is unsettled."""
        return self._is_busy

    @property
    def is_closed(self) -> bool:
        """Whether permanent closure has finished, including internal cleanup."""
        return self._is_closed

    @property
    def activity_kind(self) -> Literal["dialogue", "manual_compaction"] | None:
        return self._activity_kind

    @property
    def streaming_message(self) -> AgentMessage | None:
        return copy.deepcopy(self._streaming_message)

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(snapshot_tools(self._tools))

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return copy.deepcopy(tuple(self._messages))


def derive_system_sections(messages: list[AgentMessage]) -> dict[str, str]:
    """Replayed named sections of a transcript, ignoring deleted entries."""
    replayed = get_current_system_message(messages)
    if replayed is None:
        return {}
    return {
        name: value for name, value in (replayed.sections or {}).items() if value is not None
    }


def snapshot_tools(tools: list[AgentTool] | tuple[AgentTool, ...]) -> list[AgentTool]:
    """Copy declaration data while retaining host-owned execution callbacks."""
    for tool in tools:
        validate_json(tool.parameters, f"tools.{tool.name}.parameters")
    return [replace(tool, parameters=copy.deepcopy(tool.parameters)) for tool in tools]
