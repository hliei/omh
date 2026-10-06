"""Application conversation entry, input expansion and resource assembly."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import time

from omh.agent import (
    Agent,
    AgentEvent,
    AgentMessage,
    AgentOptions,
    AgentQueueSnapshot,
    AgentTool,
    CompactionResult,
    CompactionSummaryMessage,
    CustomAgentMessage,
    HistoryCommitEvent,
    PromptTemplateSource,
    ReadToolOptions,
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

from coding_agent.config import ToolName as ToolName
from coding_agent.images import ImageLimits, create_read_image_processor
from coding_agent.resources import ApplicationResources, load_resources
from coding_agent.session_manager import (
    ExportFormat,
    SaveMode,
    SaveState,
    SessionManager,
)


class UnsupportedImageModelError(RuntimeError):
    """A new image attachment was rejected because the selected model is text-only."""


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
    image_limits: ImageLimits | None = None
    #: Directory holding automatic per-conversation files; ``None`` keeps the session in memory.
    session_dir: str | Path | None = None
    # Credentials, hooks, policies and provider options remain current host code.
    agent_options: AgentOptions = field(default_factory=AgentOptions)
    agent_dir: str | Path | None = None
    context_dirs: tuple[str | Path, ...] = ()
    skill_sources: tuple[SkillSource, ...] = ()
    template_sources: tuple[PromptTemplateSource, ...] = ()
    custom_prompt: str | None = None
    #: Optional ordered tiers for resource sources. ``None`` keeps the embedded
    #: ``global -> project -> explicit`` default; the installed host supplies the
    #: product's five-tier order in :data:`coding_agent.resources.RESOURCE_TIERS`.
    resource_tiers: tuple[str, ...] | None = None
    #: Whether project instruction files participate in named sections.
    load_context_files: bool = True
    #: Optional system-prompt addendum assembled by the host.
    append_system_prompt: str | None = None


def _create_tools(names: tuple[ToolName, ...], cwd: Path, *, image_limits: ImageLimits | None = None) -> list[AgentTool]:
    factories: dict[str, Callable[[str | Path], AgentTool]] = {
        "read": create_read_tool, "bash": create_bash_tool,
        "edit": create_edit_tool, "write": create_write_tool,
    }
    if len(set(names)) != len(names) or any(name not in factories for name in names):
        raise ValueError("tools must be a distinct subset of read/bash/edit/write")
    return [
        create_read_tool(cwd, ReadToolOptions(image_processor=create_read_image_processor(image_limits)))
        if name == "read" else factories[name](cwd)
        for name in names
    ]


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
        self._close_task: asyncio.Task[None] | None = None
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

    def ensure_can_accept_work(self) -> None:
        """Check admission before the host starts new work, including a user shell."""
        if self.save_state == "unsaved":
            raise RuntimeError("Session has unsaved history; save the complete history before continuing") from self.save_error
        if self._close_task is not None or self.agent.state.is_closed:
            raise RuntimeError("Session is closing or closed")

    @property
    def supports_images(self) -> bool:
        """Whether the current model declares image input."""
        return "image" in self.agent.state.model.input

    def unsupported_image_message(self) -> str | None:
        """Explain the manual model choice when the current model is text-only.

        The interactive draft warns with this message when an attachment is
        added; submission still rejects the image through
        :meth:`_ensure_images_supported`. ``None`` means the model accepts images.
        """
        if self.supports_images:
            return None
        model = self.agent.state.model
        caption = f"{model.provider}/{model.id}"
        suggestions = self._vision_model_suggestions()
        guidance = f" For example: {', '.join(suggestions)}." if suggestions else ""
        return (
            f"{caption} does not accept image input; select a vision-capable model manually "
            f"with --model, /model or set_model.{guidance} omh does not switch provider automatically."
        )

    def _ensure_images_supported(
        self, message: str | AgentMessage | list[AgentMessage], images: list[ImageContent] | None,
    ) -> None:
        """Reject new image attachments for a text-only model before accepting work.

        The product never switches provider to make an image fit. Saved history
        images are not rejected here; the SDK projects those as placeholders.
        """
        if self.supports_images:
            return
        messages = message if isinstance(message, list) else [message]
        has_images = bool(images) or any(
            not isinstance(item, (str, CompactionSummaryMessage))
            and isinstance(item.content, list)
            and any(isinstance(block, ImageContent) for block in item.content)
            for item in messages
        )
        if not has_images:
            return
        guidance = self.unsupported_image_message()
        if guidance is not None:
            raise UnsupportedImageModelError(guidance)

    def _vision_model_suggestions(self) -> list[str]:
        if self._resource_options is None:
            return []
        return [
            f"{candidate.provider}/{candidate.id}"
            for candidate in self._resource_options.available_models
            if "image" in candidate.input
        ]

    async def prompt(
        self, message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        self.ensure_can_accept_work()
        self._ensure_images_supported(message, images)
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
        self.ensure_can_accept_work()
        self._ensure_images_supported(message, images)
        self.agent.steer(self._queued_input(message, images))

    def follow_up(self, message: str | AgentMessage, images: list[ImageContent] | None = None) -> None:
        self.ensure_can_accept_work()
        self._ensure_images_supported(message, images)
        self.agent.follow_up(self._queued_input(message, images))

    async def reload_resources(self) -> ApplicationResources:
        """Load current options before publishing next-prompt sections and live tools."""
        if self._resource_options is None:
            raise ValueError("Resource reload requires application options")
        tools = _create_tools(self._resource_options.tools, self.cwd, image_limits=self._resource_options.image_limits)
        resources, sections = load_resources(self._resource_options, self.cwd, tools)
        await self.agent.set_system_sections(sections)
        await self.agent.set_tools(tools)
        self.resources = resources
        return resources

    async def continue_(self) -> None:
        self.ensure_can_accept_work()
        await self.agent.continue_()

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        self.ensure_can_accept_work()
        return await self.agent.compact(custom_instructions)

    async def submit_custom_message(self, message: CustomAgentMessage) -> None:
        """Admit a host record using the SDK's safe history commit boundary."""
        self.ensure_can_accept_work()
        await self.agent.submit_custom_message(message)

    async def _on_history_commit(self, event: AgentEvent, signal: AbortSignal | None) -> None:
        if isinstance(event, HistoryCommitEvent) and self.path is not None:
            await self.session_manager.commit(self.agent.history, event.entries)

    async def save(self, path: str | Path | None = None) -> Path:
        """Explicitly write the complete history, even before a first prompt."""
        return await self.session_manager.save(self.agent.history, path)

    async def set_name(self, name: str | None) -> None:
        """Persist a display name; None clears it, failure retains the requested name."""
        await self.session_manager.set_name(self.agent.history, name)

    async def export(self, path: str | Path | None = None, *, format: ExportFormat = "jsonl") -> str:
        """Return a full JSONL snapshot; an optional separate file receives it."""
        return await self.session_manager.export(self.agent.history, path, format=format)

    async def close(self) -> None:
        """Finish Agent-owned cleanup before releasing the application writer."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._closed)
        await asyncio.shield(self._close_task)

    def _closed(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
        if not self.agent.state.is_closed:
            self._close_task = None

    async def _close(self) -> None:
        try:
            await self.agent.close()
        finally:
            if self.agent.state.is_closed:
                await self.session_manager.close()
