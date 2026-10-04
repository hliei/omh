"""Host assembly and normal-path file persistence for application sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import time
from typing import Literal

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentMessage,
    AgentOptions,
    AgentTool,
    CompactionResult,
    HistoryCommitEvent,
    MessageHistoryEntry,
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
    validate_history,
)
from omh.llm.types import (
    AbortSignal,
    ImageContent,
    Model,
    ModelThinkingLevel,
    TextContent,
    UserMessage,
)

from coding_agent.history import (
    DecodedHistory,
    decode_history,
    encode_entries,
    encode_history,
)
from coding_agent.resources import ApplicationResources, load_resources

ToolName = Literal["read", "bash", "edit", "write"]
SaveState = Literal["pending", "saved", "unsaved"]


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
    # Credentials, hooks, policies and provider options remain current host code.
    agent_options: AgentOptions = field(default_factory=AgentOptions)
    agent_dir: str | Path | None = None
    context_dirs: tuple[str | Path, ...] = ()
    skill_sources: tuple[SkillSource, ...] = ()
    template_sources: tuple[PromptTemplateSource, ...] = ()
    custom_prompt: str | None = None


def _path(path: str | Path, cwd: Path) -> Path:
    result = Path(path).expanduser()
    return (result if result.is_absolute() else cwd / result).resolve()


def _create_tools(names: tuple[ToolName, ...], cwd: Path) -> list[AgentTool]:
    factories: dict[str, Callable[[str | Path], AgentTool]] = {
        "read": create_read_tool, "bash": create_bash_tool,
        "edit": create_edit_tool, "write": create_write_tool,
    }
    if len(set(names)) != len(names) or any(name not in factories for name in names):
        raise ValueError("tools must be a distinct subset of read/bash/edit/write")
    return [factories[name](cwd) for name in names]


class ApplicationSession:
    """One Agent plus application file metadata and observable saving status."""

    def __init__(
        self, agent: Agent, *, cwd: Path, path: Path | None,
        display_name: str | None, saved: bool = False,
        model_fallback_message: str | None = None,
        resources: ApplicationResources = ApplicationResources(),
        resource_options: CodingAgentOptions | None = None,
    ) -> None:
        self.agent = agent
        self.cwd = cwd
        self.path = path
        self.display_name = display_name
        self.model_fallback_message = model_fallback_message
        self.resources = resources
        self._resource_options = resource_options
        self._input_diagnostics: list[SkillDiagnostic] = []
        self._save_state: SaveState = "saved" if saved else "pending"
        self._save_error: Exception | None = None
        self._saved_ids = {entry.id for entry in agent.history.entries} if saved else set()
        self._file_lock = asyncio.Lock()
        agent.subscribe(self._on_history_commit)

    @property
    def save_state(self) -> SaveState:
        return self._save_state

    @property
    def save_error(self) -> Exception | None:
        return self._save_error

    async def prompt(
        self, message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
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
        self.agent.steer(self._queued_input(message, images))

    def follow_up(self, message: str | AgentMessage, images: list[ImageContent] | None = None) -> None:
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
        await self.agent.continue_()

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        return await self.agent.compact(custom_instructions)

    async def _on_history_commit(self, event: AgentEvent, signal: AbortSignal | None) -> None:
        if not isinstance(event, HistoryCommitEvent) or self.path is None:
            return
        async with self._file_lock:
            if self._save_error is not None:
                raise self._save_error
            history = self.agent.history
            if self._save_state == "pending" and not any(
                isinstance(entry, MessageHistoryEntry) and entry.message.role in ("user", "assistant")
                for entry in history.entries
            ):
                return
            try:
                if self._save_state == "pending":
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("x", encoding="utf-8", newline="\n") as file:
                        file.write(encode_history(history, cwd=str(self.cwd), display_name=self.display_name))
                    self._saved_ids = {entry.id for entry in history.entries}
                else:
                    entries = tuple(entry for entry in event.entries if entry.id not in self._saved_ids)
                    with self.path.open("a", encoding="utf-8", newline="\n") as file:
                        file.write(encode_entries(entries))
                    self._saved_ids.update(entry.id for entry in entries)
            except Exception as error:
                self._save_state, self._save_error = "unsaved", error
                raise
            self._save_state, self._save_error = "saved", None

    async def save(self, path: str | Path | None = None) -> Path:
        """Explicitly write the complete history, even before a first prompt."""
        async with self._file_lock:
            destination = _path(path, self.cwd) if path is not None else self.path
            if destination is None:
                raise ValueError("save requires a session path")
            history = self.agent.history
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("w", encoding="utf-8", newline="\n") as file:
                    file.write(encode_history(history, cwd=str(self.cwd), display_name=self.display_name))
            except Exception as error:
                self._save_state, self._save_error = "unsaved", error
                raise
            self.path = destination
            self._saved_ids = {entry.id for entry in history.entries}
            self._save_state, self._save_error = "saved", None
            return destination

    async def export(self, path: str | Path | None = None) -> str:
        """Return a full JSONL snapshot; an optional separate file receives it."""
        if path is not None and _path(path, self.cwd) == self.path:
            await self.save()
        text = encode_history(self.agent.history, cwd=str(self.cwd), display_name=self.display_name)
        if path is not None and _path(path, self.cwd) != self.path:
            destination = _path(path, self.cwd)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8", newline="\n")
        return text


class CodingAgentRuntime:
    """Assemble a current session from current host execution dependencies."""

    def __init__(self, options: CodingAgentOptions) -> None:
        self.options = options
        self.current_session: ApplicationSession | None = None

    def _ensure_available(self) -> None:
        if self.current_session is not None and not self.current_session.agent.state.is_closed:
            raise RuntimeError("Close the current session before creating or opening another")

    def _assemble(self, decoded: DecodedHistory | None = None) -> tuple[
        AgentOptions, Path, str | None, ApplicationResources, dict[str, str],
    ]:
        options = self.options
        cwd = Path(options.cwd or (decoded.cwd if decoded else Path.cwd())).expanduser().resolve()
        initial = options.agent_options.initial_state or AgentInitialState()
        selected = options.model or initial.model
        fallback_message = None
        if decoded is not None:
            settings = validate_history(decoded.history)
            if selected is None and settings.provider is not None:
                selected = next((candidate for candidate in options.available_models
                                 if (candidate.provider, candidate.id) == (settings.provider, settings.model_id)), None)
                if selected is None:
                    fallback_message = f"History model {settings.provider}/{settings.model_id} is unavailable"
        selected = selected or options.fallback_model
        if selected is None:
            raise ValueError("No executable model configured; provide model or fallback_model")
        if fallback_message is not None:
            fallback_message += f"; using {selected.provider}/{selected.id}"
        tools = _create_tools(options.tools, cwd)
        assembled = replace(
            options.agent_options,
            stream_fn=options.stream_fn or options.agent_options.stream_fn,
            initial_state=replace(
                initial, model=selected,
                thinking_level=options.thinking_level if options.thinking_level is not None else initial.thinking_level,
                tools=tools,
            ),
        )
        resources, sections = load_resources(options, cwd, tools)
        return assembled, cwd, fallback_message, resources, sections

    async def new_session(self, *, display_name: str | None = None) -> ApplicationSession:
        self._ensure_available()
        options, cwd, _, resources, sections = self._assemble()
        agent = Agent(options)
        await agent.set_system_sections(sections)
        # The model request identity is independently overridable by the host.
        if options.session_id is None:
            agent.session_id = agent.history.conversation_id
        path = _path(self.options.session_file, cwd) if self.options.session_file is not None else None
        session = ApplicationSession(
            agent, cwd=cwd, path=path, display_name=display_name,
            resources=resources, resource_options=self.options,
        )
        self.current_session = session
        return session

    async def open_session(self, path: str | Path) -> ApplicationSession:
        self._ensure_available()
        destination = _path(path, Path(self.options.cwd or Path.cwd()).expanduser().resolve())
        raw = destination.read_bytes()
        decoded = decode_history(raw)
        options, cwd, fallback_message, resources, sections = self._assemble(decoded)
        agent = Agent.from_history(decoded.history, options)
        await agent.set_system_sections(sections)
        if options.session_id is None:
            agent.session_id = agent.history.conversation_id
        # Repair only after semantic validation and assembly have succeeded.
        if raw and not raw.endswith(b"\n"):
            with destination.open("ab") as file:
                file.write(b"\n")
        session = ApplicationSession(
            agent, cwd=cwd, path=destination, display_name=decoded.display_name,
            saved=True, model_fallback_message=fallback_message, resources=resources,
            resource_options=self.options,
        )
        self.current_session = session
        return session

    def _current(self) -> ApplicationSession:
        if self.current_session is None:
            raise RuntimeError("Create or open a session first")
        return self.current_session

    async def prompt(
        self, message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        await self._current().prompt(message, images)

    def steer(self, message: str | AgentMessage, images: list[ImageContent] | None = None) -> None:
        self._current().steer(message, images)

    def follow_up(self, message: str | AgentMessage, images: list[ImageContent] | None = None) -> None:
        self._current().follow_up(message, images)

    async def reload_resources(self) -> ApplicationResources:
        return await self._current().reload_resources()

    async def continue_(self) -> None:
        await self._current().continue_()

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        return await self._current().compact(custom_instructions)

    async def save_session(self, path: str | Path | None = None) -> Path:
        return await self._current().save(path)

    async def export_session(self, path: str | Path | None = None) -> str:
        return await self._current().export(path)
