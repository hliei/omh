"""Application conversation entry, input expansion and resource assembly."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal

from omh.agent import (
    Agent,
    AgentEvent,
    AgentMessage,
    AgentOptions,
    AgentQueueSnapshot,
    AgentTool,
    CompactionResult,
    HistoryCommitEvent,
    PromptTemplateSource,
    SkillDiagnostic,
    SkillSource,
    StreamFn,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
    expand_prompt_template,
    expand_skill_command,
)
from omh.llm.types import (
    AbortSignal,
    ImageContent,
    Model,
    ModelThinkingLevel,
    TextContent,
    UserMessage,
)

from coding_agent.resources import ApplicationResources, load_resources
from coding_agent.session_manager import SaveMode, SaveState, SessionManager

ToolName = Literal["read", "bash", "edit", "write"]


@dataclass(slots=True)
class CodingAgentOptions:
    cwd: str | Path | None = None
    model: Model | None = None
    stream_fn: StreamFn | None = None
    thinking_level: ModelThinkingLevel | None = None
    available_models: tuple[Model, ...] = ()
    fallback_model: Model | None = None
    tools: tuple[ToolName, ...] = ("read", "bash", "edit", "write")
    session_file: str | Path | None = None
    #: Directory holding automatic per-conversation files; ``None`` keeps the session in memory.
    session_dir: str | Path | None = None
    # Credentials, hooks, policies and provider options remain current host code.
    agent_options: AgentOptions = field(default_factory=AgentOptions)
    agent_dir: str | Path | None = None
    context_dirs: tuple[str | Path, ...] = ()
    skill_sources: tuple[SkillSource, ...] = ()
    template_sources: tuple[PromptTemplateSource, ...] = ()
    custom_prompt: str | None = None


def _create_tools(names: tuple[ToolName, ...], cwd: Path) -> list[AgentTool]:
    factories: dict[str, Callable[[str | Path], AgentTool]] = {
        "read": create_read_tool, "bash": create_bash_tool,
        "edit": create_edit_tool, "write": create_write_tool,
    }
    if len(set(names)) != len(names) or any(name not in factories for name in names):
        raise ValueError("tools must be a distinct subset of read/bash/edit/write")
    return [factories[name](cwd) for name in names]


class AgentSession:
    """One conversation composing the SDK Agent, resources and saving manager."""

    def __init__(
        self, agent: Agent, *, session_manager: SessionManager,
        model_fallback_message: str | None = None,
        resources: ApplicationResources = ApplicationResources(),
        resource_options: CodingAgentOptions | None = None,
    ) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.model_fallback_message = model_fallback_message
        self.resources = resources
        self._resource_options = resource_options
        self._input_diagnostics: list[SkillDiagnostic] = []
        self._retained_queues: AgentQueueSnapshot | None = None
        agent.subscribe(self._on_history_commit)

    @property
    def queued_messages(self) -> AgentQueueSnapshot:
        """Complete isolated queues, frozen at retirement for retained sessions."""
        if self._retained_queues is not None:
            return copy.deepcopy(self._retained_queues)
        return self.agent.get_queued_messages()

    @property
    def cwd(self) -> Path:
        return self.session_manager.cwd

    @cwd.setter
    def cwd(self, value: Path) -> None:
        self.session_manager.cwd = value

    @property
    def path(self) -> Path | None:
        return self.session_manager.path

    @path.setter
    def path(self, value: Path | None) -> None:
        self.session_manager.path = value

    @property
    def display_name(self) -> str | None:
        return self.session_manager.display_name

    @display_name.setter
    def display_name(self, value: str | None) -> None:
        self.session_manager.display_name = value

    @property
    def save_state(self) -> SaveState:
        return self.session_manager.save_state

    @property
    def save_mode(self) -> SaveMode:
        return self.session_manager.save_mode

    @property
    def save_error(self) -> Exception | None:
        return self.session_manager.save_error

    def _ensure_can_accept_work(self) -> None:
        if self.save_state == "unsaved":
            raise RuntimeError("Session has unsaved history; save the complete history before continuing") from self.save_error

    async def prompt(
        self, message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        self._ensure_can_accept_work()
        if isinstance(message, str):
            message = self._expand_input(message)
        await self.agent.prompt(message, images)

    @property
    def input_diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        return tuple(self._input_diagnostics)

    def _expand_input(self, text: str) -> str:
        expanded = expand_skill_command(text, self.resources.skills)
        self._input_diagnostics.extend(expanded.diagnostics)
        return expand_prompt_template(expanded.text, self.resources.templates)

    def _queued_input(self, message: str | AgentMessage, images: list[ImageContent] | None) -> AgentMessage:
        if not isinstance(message, str):
            if images is not None:
                raise ValueError("images require string input")
            return message
        return UserMessage(
            content=[TextContent(text=self._expand_input(message)), *(images or [])],
            timestamp=int(time() * 1000),
        )

    def steer(self, message: str | AgentMessage, images: list[ImageContent] | None = None) -> None:
        self._ensure_can_accept_work()
        self.agent.steer(self._queued_input(message, images))

    def follow_up(self, message: str | AgentMessage, images: list[ImageContent] | None = None) -> None:
        self._ensure_can_accept_work()
        self.agent.follow_up(self._queued_input(message, images))

    async def reload_resources(self) -> ApplicationResources:
        """Load current options before publishing next-prompt sections and live tools."""
        if self._resource_options is None:
            raise ValueError("Resource reload requires application options")
        tools = _create_tools(self._resource_options.tools, self.cwd)
        resources, sections = load_resources(self._resource_options, self.cwd, tools)
        await self.agent.set_system_sections(sections)
        await self.agent.set_tools(tools)
        self.resources = resources
        return resources

    async def continue_(self) -> None:
        self._ensure_can_accept_work()
        await self.agent.continue_()

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        self._ensure_can_accept_work()
        return await self.agent.compact(custom_instructions)

    async def _on_history_commit(self, event: AgentEvent, signal: AbortSignal | None) -> None:
        if isinstance(event, HistoryCommitEvent) and self.path is not None:
            await self.session_manager.commit(self.agent.history, event.entries)

    async def save(self, path: str | Path | None = None) -> Path:
        """Explicitly write the complete history, even before a first prompt."""
        return await self.session_manager.save(self.agent.history, path)

    async def export(self, path: str | Path | None = None) -> str:
        """Return a full JSONL snapshot; an optional separate file receives it."""
        return await self.session_manager.export(self.agent.history, path)
