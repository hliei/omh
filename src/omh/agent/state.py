"""Initial values and mutable public state of an in-process Agent."""

from __future__ import annotations

from dataclasses import dataclass

from omh.agent.messages import AgentMessage
from omh.agent.tools import AgentTool, to_tool_declaration
from omh.llm.types import Model, ModelCost
from omh.llm.types import ModelThinkingLevel as ThinkingLevel
from omh.llm.utils.transcript import (
    create_initial_system_message,
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
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] | None = None
    messages: list[AgentMessage] | None = None


class AgentState:
    """Public Agent state with copy-on-assign message and tool lists.

    Assigning :attr:`messages` or :attr:`tools` copies the top-level list; the
    message and tool objects themselves are shared, not deep-copied. The prompt
    is replayed from the transcript's system messages and is read-only here.
    """

    def __init__(self, initial: AgentInitialState | None = None) -> None:
        initial = initial or AgentInitialState()
        self._tools: list[AgentTool] = list(initial.tools or [])
        self._messages: list[AgentMessage] = list(initial.messages or [])
        initial_message = create_initial_system_message(
            initial.system_prompt,
            [to_tool_declaration(tool) for tool in self._tools],
        )
        if (not self._messages or self._messages[0].role != "system") and initial_message is not None:
            self._messages.insert(0, initial_message)
        self.model: Model = initial.model if initial.model is not None else DEFAULT_MODEL
        self.thinking_level: ThinkingLevel = initial.thinking_level
        self.is_streaming: bool = False
        self.streaming_message: AgentMessage | None = None
        self.error_message: str | None = None
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
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)
