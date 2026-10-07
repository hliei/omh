"""Current-session assembly, replacement and subscription rebinding."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid7

from omh.agent import (
    Agent,
    AgentEvent,
    AgentHistory,
    AgentInitialState,
    AgentMessage,
    AgentOptions,
    CompactionResult,
    CustomAgentMessage,
    MessageHistoryEntry,
    validate_history,
)
from omh.agent.conversation.history import history_path
from omh.llm.types import AbortSignal, ImageContent

from coding_agent.agent_session import AgentSession, CodingAgentOptions, _create_tools
from coding_agent.history import ConversationSource, DecodedHistory
from coding_agent.resources import ApplicationResources, load_resources
from coding_agent.session_manager import ExportFormat, SaveMode, SessionManager, _path
from coding_agent.session_paths import session_file_path
from coding_agent.user_shell import HIDDEN_SHELL_TYPE

RuntimeListener = Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]


@dataclass(eq=False, slots=True)
class _Subscription:
    listener: RuntimeListener
    unsubscribe: Callable[[], None] = lambda: None


class AgentSessionRuntime:
    """Assemble a current session from current host execution dependencies."""

    def __init__(self, options: CodingAgentOptions) -> None:
        self.options = options
        self.current_session: AgentSession | None = None
        self._preparing = False
        self._switch_task: asyncio.Task[AgentSession] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._subscriptions: list[_Subscription] = []
        self._retained_sessions: list[AgentSession] = []

    @property
    def retained_sessions(self) -> tuple[AgentSession, ...]:
        """Retired sessions in switch order; history remains saveable/exportable."""
        return tuple(self._retained_sessions)

    def subscribe(self, listener: RuntimeListener) -> Callable[[], None]:
        """Follow Agent events across session replacements, in registration order."""
        subscription = _Subscription(listener)
        if (self.current_session is not None and not self.current_session.agent.state.is_closed
                and not self._switching()):
            subscription.unsubscribe = self.current_session.agent.subscribe(listener)
        self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            subscription.unsubscribe()
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)

        return unsubscribe

    def _switching(self) -> bool:
        return self._switch_task is not None and not self._switch_task.done()

    async def wait_for_switch(self) -> AgentSession:
        """Observe the latest handoff; cancellation ends only this wait."""
        task = self._switch_task
        if task is None:
            return self._current()
        if not task.done() and self.current_session is not None:
            # Use the SDK wait boundary to reject an activity awaiting itself.
            idle = asyncio.create_task(self.current_session.agent.wait_for_idle())
            try:
                await asyncio.gather(idle, asyncio.shield(task))
            finally:
                idle.cancel()
            return task.result()
        return await asyncio.shield(task)

    async def _publish(
        self, session: AgentSession, *, previous_history: AgentHistory | None, stable_source: bool = False,
    ) -> AgentSession:
        old = self.current_session
        if (previous_history is not None and old is not None and (stable_source or session.path == old.path)
                and (old.agent.state.is_busy or old.agent.history != previous_history)):
            await session.session_manager.close()
            action = "deriving a conversation" if stable_source else "reopening the current session file"
            raise RuntimeError(f"Wait for stable idle history before {action}")
        try:
            if old is not None:
                await old.agent.close()
        finally:
            # A failed terminal notification still retires the old instance.
            if old is None or old.agent.state.is_closed:
                if old is not None:
                    if session.path is not None and session.path == old.path:
                        session.session_manager._take_writer_from(old.session_manager)
                    else:
                        await old.session_manager.close()
                    old._retained_queues = old.agent.get_queued_messages()
                    self._retained_sessions.append(old)
                self.current_session = session
                for subscription in self._subscriptions:
                    subscription.unsubscribe()
                    subscription.unsubscribe = session.agent.subscribe(subscription.listener)
                if stable_source:
                    await session.session_manager.commit(session.agent.history, session.agent.history.entries)
            else:
                await session.session_manager.close()
        return session

    async def _replace_session(
        self, path: str | Path | None = None, *, display_name: str | None = None,
        prepare: Callable[[], Awaitable[AgentSession]] | None = None,
    ) -> AgentSession:
        if self._preparing or self._switching():
            raise RuntimeError("A session switch is already in progress")
        if self._close_task is not None and not self._close_task.done():
            raise RuntimeError("Runtime is closing")
        self._preparing = True
        session = None
        try:
            previous_history = (self.current_session.agent.history
                                if (path is not None or prepare is not None) and self.current_session is not None else None)
            # Keep cancellation in preparation separate from the owned handoff.
            await asyncio.sleep(0)
            session = await (prepare() if prepare is not None else
                             self._prepare_new_session(display_name=display_name)
                             if path is None else self._prepare_open_session(path))
            await asyncio.sleep(0)
            self._switch_task = asyncio.create_task(self._publish(session, previous_history=previous_history, stable_source=prepare is not None))
            self._switch_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        except BaseException:
            if session is not None:
                await session.session_manager.close()
            raise
        finally:
            self._preparing = False
        return await asyncio.shield(self._switch_task)

    def _assemble(self, decoded: DecodedHistory | None = None, *, options: CodingAgentOptions | None = None) -> tuple[
        AgentOptions, Path, str | None, ApplicationResources, dict[str, str],
    ]:
        options = options or self.options
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
        tools = _create_tools(options.tools, cwd, image_limits=options.image_limits)
        transform = options.agent_options.transform_context

        async def model_context(messages: list[AgentMessage], signal: AbortSignal | None) -> list[AgentMessage]:
            visible = [message for message in messages if not (
                isinstance(message, CustomAgentMessage) and message.custom_type == HIDDEN_SHELL_TYPE
            )]
            if transform is None:
                return visible
            result = transform(visible, signal)
            return await result if inspect.isawaitable(result) else result

        assembled = replace(
            options.agent_options,
            transform_context=model_context,
            stream_fn=options.stream_fn or options.agent_options.stream_fn,
            initial_state=replace(
                initial, model=selected,
                thinking_level=options.thinking_level if options.thinking_level is not None else initial.thinking_level,
                tools=tools,
            ),
        )
        resources, sections = load_resources(options, cwd, tools)
        return assembled, cwd, fallback_message, resources, sections

    async def new_session(self, *, display_name: str | None = None) -> AgentSession:
        """Prepare a fresh identity, close the old Agent, then publish it."""
        return await self._replace_session(display_name=display_name)

    async def _prepare_new_session(self, *, display_name: str | None = None) -> AgentSession:
        options, cwd, _, resources, sections = self._assemble()
        agent = Agent(options)
        await agent.set_system_sections(sections)
        # The model request identity is independently overridable by the host.
        if options.session_id is None:
            agent.session_id = agent.history.conversation_id
        session = AgentSession(
            agent, session_manager=SessionManager(
                cwd=cwd, path=self._new_session_path(agent.history, cwd), display_name=display_name,
            ),
            resources=resources, resource_options=self.options,
        )
        return session

    def _new_session_path(self, history: AgentHistory, cwd: Path) -> Path | None:
        """Resolve an explicit file, then a storage directory, else memory.

        The directory holds this conversation's file directly; under the default
        root the host supplies a per-cwd directory. Nothing is written here; the
        saving manager creates the file on the first real user activity.
        """
        if self.options.session_file is not None:
            return _path(self.options.session_file, cwd)
        if self.options.session_dir is not None:
            return session_file_path(
                _path(self.options.session_dir, cwd), history.conversation_id, history.created_at,
            )
        return None

    async def clone_session(
        self, *, cwd: str | Path | None = None, save_mode: SaveMode | None = None,
        session_dir: str | Path | None = None,
    ) -> AgentSession:
        """Copy the current active path into an independent idle conversation."""
        return await self._derive_session(cwd=cwd, save_mode=save_mode, session_dir=session_dir)

    async def fork_session(
        self, entry_id: str, *, cwd: str | Path | None = None, save_mode: SaveMode | None = None,
        session_dir: str | Path | None = None,
    ) -> AgentSession:
        """Copy ancestors before a user on the active path, excluding that user."""
        return await self._derive_session(entry_id=entry_id, cwd=cwd, save_mode=save_mode, session_dir=session_dir)

    async def _derive_session(
        self, *, entry_id: str | None = None, cwd: str | Path | None,
        save_mode: SaveMode | None, session_dir: str | Path | None,
    ) -> AgentSession:
        old = self._current()
        if old.agent.state.is_busy or old.agent.state.is_closed:
            raise RuntimeError("Derivation requires an open idle Agent")
        if save_mode not in (None, "auto", "memory"):
            raise ValueError("save_mode must be auto or memory")
        if save_mode == "memory" and session_dir is not None:
            raise ValueError("A memory conversation cannot specify a session directory")
        original = old.agent.history
        path = history_path(original)
        leaf = original.leaf_id
        if entry_id is not None:
            selected = next((entry for entry in path if entry.id == entry_id), None)
            if not isinstance(selected, MessageHistoryEntry) or selected.message.role != "user":
                raise ValueError("Fork requires a user entry on the current active path")
            leaf = selected.parent_id
            path = path[:path.index(selected)]
        history = replace(original, entries=tuple(path), leaf_id=leaf,
                          conversation_id=str(uuid7()), created_at=datetime.now(UTC))
        validate_history(history)
        source = ConversationSource("fork" if entry_id is not None else "clone", original.conversation_id,
                                    str(old.path) if old.path is not None else None, leaf, entry_id)
        return await self._replace_session(prepare=lambda: self._prepare_derived_session(
            history, source, cwd=cwd, save_mode=save_mode, session_dir=session_dir,
        ))

    async def _prepare_derived_session(
        self, history: AgentHistory, source: ConversationSource, *, cwd: str | Path | None,
        save_mode: SaveMode | None, session_dir: str | Path | None,
    ) -> AgentSession:
        old = self._current()
        target_cwd = _path(cwd, old.cwd) if cwd is not None else old.cwd
        mode = save_mode or ("auto" if session_dir is not None else old.save_mode)
        directory = (_path(session_dir, target_cwd) if session_dir is not None else
                     old.path.parent if old.path is not None else
                     _path(self.options.session_dir, target_cwd) if self.options.session_dir is not None else None)
        if mode == "auto" and directory is None:
            raise ValueError("Automatic derivation requires a session directory")
        initial = self.options.agent_options.initial_state or AgentInitialState()
        options = replace(
            self.options, cwd=target_cwd, model=None, thinking_level=None,
            available_models=(*self.options.available_models, old.agent.state.model),
            fallback_model=old.agent.state.model,
            agent_options=replace(self.options.agent_options, conversation_id=None, session_id=None,
                                  initial_state=replace(initial, model=None, thinking_level=None, messages=None)),
        )
        assembled, target_cwd, fallback, resources, sections = self._assemble(
            DecodedHistory(history, str(target_cwd), None), options=options,
        )
        if fallback is not None:
            raise ValueError(fallback)
        agent = Agent.from_history(history, assembled)
        await agent.set_system_sections(sections)
        agent.session_id = history.conversation_id
        manager = SessionManager(
            cwd=target_cwd, source=source,
            path=session_file_path(directory, history.conversation_id, history.created_at)
            if mode == "auto" and directory is not None else None,
        )
        return AgentSession(agent, session_manager=manager, resources=resources, resource_options=options)

    async def open_session(self, path: str | Path) -> AgentSession:
        """Prepare a saved identity before closing and replacing the old Agent."""
        return await self._replace_session(path)

    async def _prepare_open_session(self, path: str | Path) -> AgentSession:
        destination = _path(path, Path(self.options.cwd or Path.cwd()).expanduser().resolve())
        old = self.current_session
        owner = old.session_manager if old is not None and destination == old.path else None
        if owner is not None and old is not None and old.agent.state.is_busy:
            raise RuntimeError("Wait for stable idle history before reopening the current session file")
        manager, decoded = SessionManager._load(destination, owner)
        try:
            options, cwd, fallback_message, resources, sections = self._assemble(decoded)
            agent = Agent.from_history(decoded.history, options)
            await agent.set_system_sections(sections)
            if options.session_id is None:
                agent.session_id = agent.history.conversation_id
            manager.cwd = cwd
            # Same-file preparation retains the old writer through retirement.
            await manager.prepare_append()
            return AgentSession(
                agent, session_manager=manager,
                model_fallback_message=fallback_message, resources=resources,
                resource_options=self.options,
            )
        except BaseException:
            await manager.close()
            raise

    async def switch_session(self, path: str | Path) -> AgentSession:
        """Prepare a saved conversation before retiring the current Agent."""
        return await self.open_session(path)

    def _current(self) -> AgentSession:
        if self.current_session is None:
            raise RuntimeError("Create or open a session first")
        return self.current_session

    def ensure_can_accept_work(self) -> None:
        """Check current-session admission before starting host work."""
        if self._preparing or self._switching():
            raise RuntimeError("A session switch is already in progress")
        if self._close_task is not None and not self._close_task.done():
            raise RuntimeError("Runtime is closing")
        self._current().ensure_can_accept_work()

    async def submit_custom_message(self, message: CustomAgentMessage) -> None:
        self.ensure_can_accept_work()
        await self._current().submit_custom_message(message)

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

    async def set_session_name(self, name: str | None) -> None:
        await self._current().set_name(name)

    async def export_session(self, path: str | Path | None = None, *, format: ExportFormat = "jsonl") -> str:
        return await self._current().export(path, format=format)

    async def close(self) -> None:
        """Finish any owned handoff, then close the actual current session."""
        if self._preparing:
            raise RuntimeError("A session switch is being prepared")
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        try:
            if self._switching():
                await self.wait_for_switch()
        finally:
            if self.current_session is not None:
                await self.current_session.close()
