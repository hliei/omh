"""Common host: configuration, credentials and effective session selection.

New and reopened sessions resolve through the same host so print and
interactive use one decision. The host reads the global agent directory and a
trusted project's settings, merges them with explicit CLI values, resolves the
model directory and reports credentials. It never sends a verification
request: a configured key or a Go subscription does not prove an account is
usable, and only an actual user task surfaces authentication failure.

Precedence implemented here:

- cwd: explicit ``--cwd`` -> restored history cwd -> new-session startup cwd;
- model: explicit -> restored history -> configured default -> product default;
- thinking: explicit -> restored history -> configured default -> product
  default, clamped to the selected model's effective levels;
- tools: explicit -> configured default -> the four built-in tools.

A missing or unsupported history selection, a missing cwd or a missing key is
reported as a diagnostic instead of silently switching provider or directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from omh.agent import (
    AgentHistorySettings,
    AgentOptions,
    CompactionSettings,
    PromptTemplateSource,
    RetryPolicy,
    SkillSource,
    StreamFn,
    validate_history,
)
from omh.llm import ModelsError, create_models
from omh.llm.models import (
    CreateModelsOptions,
    ModelsImpl,
    clamp_thinking_level,
    get_supported_thinking_levels,
)
from omh.llm.types import (
    Context,
    FetchFunction,
    Model,
    ModelThinkingLevel,
    SimpleStreamOptions,
    TranscriptContext,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream

from coding_agent.agent_session import CodingAgentOptions, ToolName
from coding_agent.config import (
    APPEND_SYSTEM_FILE,
    AUTH_FILE,
    CONFIG_DIR_NAME,
    DEFAULT_TOOLS,
    PROMPTS_DIR,
    SETTINGS_FILE,
    SKILLS_DIR,
    SYSTEM_FILE,
    TRUST_FILE,
    ConfigDiagnostic,
    ConfigError,
    DiagnosticSource,
    FileCredentialStore,
    load_settings,
    project_settings_path,
    provider_api_key_env,
    resolve_agent_dir,
    settings_compaction,
    settings_model,
    settings_retry,
    settings_strings,
    settings_thinking,
    settings_tools,
    validate_tools,
)
from coding_agent.history import DecodedHistory, decode_history
from coding_agent.model_directory import ModelDirectory
from coding_agent.resources import (
    RESOURCE_TIERS,
    automatic_skill_sources,
    automatic_template_sources,
)
from coding_agent.trust import TrustDecision, TrustStore

#: One explicit system-prompt addendum: literal text or a file path.
AppendKind = Literal["text", "file"]

#: The product's default model when neither history nor configuration selects one.
DEFAULT_MODEL_REFERENCE = ("opencode-go", "deepseek-v4.1-flash")
#: The default thinking level for models that accept a choice.
DEFAULT_THINKING_LEVEL: ModelThinkingLevel = "high"


_TIER_FOR_SCOPE: dict[DiagnosticSource, str] = {
    "cli": "explicit", "project": "project-config", "global": "global-config",
}


@dataclass(frozen=True, slots=True)
class SessionSelection:
    """The resolved host decision for one new or reopened session."""

    cwd: Path | None = None
    model: Model | None = None
    thinking_level: ModelThinkingLevel | None = None
    tools: tuple[ToolName, ...] = DEFAULT_TOOLS
    history: DecodedHistory | None = None
    diagnostics: tuple[ConfigDiagnostic, ...] = ()

    @property
    def ready(self) -> bool:
        """Whether the selection can assemble an executable session."""
        return (
            self.cwd is not None and self.model is not None and self.thinking_level is not None
            and not any(diagnostic.blocking for diagnostic in self.diagnostics)
        )

    @property
    def thinking_mode(self) -> str | None:
        """The effective mode for display, including models with fixed thinking."""
        if self.model is not None and self.model.reasoning and not get_supported_thinking_levels(self.model):
            return "fixed-on"
        return self.thinking_level


class CodingAgentHost:
    """Resolve configuration, credentials and effective selection for a session.

    Project-controlled configuration and resources load only for a trusted
    project. ``approve``/``no_approve`` are one-run explicit decisions, a
    ``TrustStore`` remembers a durable decision, and ``project_trusted`` lets an
    embedding pass an already-made decision. Unknown project resources are
    skipped with a diagnostic in both modes; interactive use can inspect
    :attr:`needs_trust_decision` and call :meth:`remember_trust`.
    """

    def __init__(
        self, *, startup_dir: str | Path, agent_dir: str | Path | None = None,
        home: str | Path | None = None,
        project_trusted: bool | None = None,
        approve: bool = False, no_approve: bool = False,
        trust_store: TrustStore | None = None,
        explicit_settings: Mapping[str, object] | None = None,
        explicit_skills: tuple[str, ...] = (), explicit_templates: tuple[str, ...] = (),
        explicit_system_prompt: str | None = None,
        explicit_system_prompt_file: str | Path | None = None,
        append_system: Sequence[tuple[AppendKind, str]] = (),
        no_skills: bool = False, no_prompt_templates: bool = False,
        no_context_files: bool = False,
        api_key: str | None = None, fetch: FetchFunction | None = None,
    ) -> None:
        self.startup_dir = Path(startup_dir).expanduser().resolve()
        self.home = Path(home).expanduser().resolve() if home is not None else Path.home()
        self.agent_dir = (
            Path(agent_dir).expanduser().resolve()
            if agent_dir is not None else resolve_agent_dir(home=self.home)
        )
        self._project_trusted = project_trusted
        self._approve = approve
        self._no_approve = no_approve
        self._trust_store = trust_store if trust_store is not None else TrustStore(self.agent_dir / TRUST_FILE)
        self._explicit_settings = explicit_settings
        self._explicit_skills = explicit_skills
        self._explicit_templates = explicit_templates
        self._explicit_system_prompt = explicit_system_prompt
        self._explicit_system_prompt_file = explicit_system_prompt_file
        self._append_system = tuple(append_system)
        self._no_skills = no_skills
        self._no_prompt_templates = no_prompt_templates
        self._no_context_files = no_context_files
        self._settings_cwd = self.startup_dir
        self._trusted = False
        self._trust_unknown = False
        self._custom_prompt: str | None = None
        self._append_prompt: str | None = None
        self.directory = ModelDirectory(agent_dir=self.agent_dir)
        self.credentials = FileCredentialStore(self.agent_dir / AUTH_FILE)
        self.api_key = api_key
        self._fetch = fetch
        self._refresh_settings(self.startup_dir, [])
        self._models = self._build_models()

    def _build_models(self) -> ModelsImpl:
        models = create_models(CreateModelsOptions(credentials=self.credentials))
        for provider in self.directory.providers:
            models.set_provider(provider)
        return models

    # ------------------------------------------------------------------ #
    # Observable state
    # ------------------------------------------------------------------ #

    @property
    def diagnostics(self) -> tuple[ConfigDiagnostic, ...]:
        return (
            *self.settings.diagnostics, *self.directory.diagnostics,
            *self._trust_store.diagnostics,
        )

    @property
    def project_trusted(self) -> bool:
        """Whether the current effective project's controlled layer was loaded."""
        return self._trusted

    @property
    def needs_trust_decision(self) -> bool:
        """Whether the effective project has resources awaiting a trust decision.

        Interactive mode can ask and then call :meth:`remember_trust`; print mode
        skips the controlled layer and reports the untrusted diagnostic instead
        of waiting for an answer.
        """
        return self._trust_unknown and self._project_resources(self._settings_cwd)

    def remember_trust(self, decision: TrustDecision, *, cwd: str | Path | None = None) -> None:
        """Persist a project trust decision and refresh the effective layer."""
        target = Path(cwd).expanduser().resolve() if cwd is not None else self._settings_cwd
        self._trust_store.remember(target, decision)
        self._refresh_settings(target, [])

    @property
    def models(self) -> ModelsImpl:
        return self._models

    @property
    def compaction(self) -> CompactionSettings | None:
        return settings_compaction(self.settings)

    @property
    def retry(self) -> RetryPolicy | None:
        return settings_retry(self.settings)

    def stream_fn(self) -> StreamFn:
        """Build the host stream function, applying the injectable HTTP boundary.

        The Agent hands over an already-normalized transcript. Wrapping it in a
        ``Context`` lets the SDK resolve stored or environment credentials
        before forwarding the same transcript to the provider.
        """
        fetch = self._fetch

        def stream(
            model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            wrapped = Context(messages=list(context.messages))
            selected = options
            if fetch is not None:
                selected = replace(options or SimpleStreamOptions(), fetch=fetch)
            return self._models.stream_simple(model, wrapped, selected)

        return cast(StreamFn, stream)

    async def key_source(self, model: Model) -> str | None:
        """Report where a usable key would come from without sending a request.

        The temporary CLI override wins over a stored credential, which wins
        over the provider's environment variable. ``None`` means no key is
        available; that is a readiness fact, not a claim about the account.
        """
        if self.api_key:
            return "cli"
        check = await self._models.check_auth(model.provider)
        return check.source if check is not None else None

    async def readiness(self, selection: SessionSelection) -> tuple[ConfigDiagnostic, ...]:
        """Return every reason a selection cannot run yet, including a missing key.

        This is the pre-request check print should fail non-zero on and
        interactive should surface as a repair step. It sends no request.
        """
        diagnostics = list(selection.diagnostics)
        try:
            source = await self.key_source(selection.model) if selection.ready and selection.model is not None else None
        except (ConfigError, ModelsError) as error:
            diagnostics.append(ConfigDiagnostic(
                str(self.agent_dir / AUTH_FILE), "credentials", "invalid-schema", str(error),
            ))
            return tuple(diagnostics)
        if selection.ready and selection.model is not None and source is None:
            provider = selection.model.provider
            environment = provider_api_key_env(provider)
            repair = f", save one with /login, or set {environment}" if environment else " or save one with /login"
            diagnostics.append(ConfigDiagnostic(
                provider, "credentials", "unavailable",
                f"No API key available for {provider}; pass --api-key{repair}",
            ))
        return tuple(diagnostics)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def select_new(
        self, *, provider: str | None = None, model: str | None = None,
        thinking: ModelThinkingLevel | None = None,
        tools: tuple[ToolName, ...] | None = None, cwd: str | Path | None = None,
    ) -> SessionSelection:
        """Resolve a fresh session from explicit input, configuration then defaults."""
        diagnostics: list[ConfigDiagnostic] = []
        effective_cwd = self._resolve_cwd(cwd, history=None, diagnostics=diagnostics)
        self._refresh_settings(effective_cwd, diagnostics)
        resolved_model = self._resolve_new_model(provider, model, diagnostics)
        effective_thinking = self._resolve_thinking(thinking, history=None, model=resolved_model, diagnostics=diagnostics)
        effective_tools = self._resolve_tools(tools, diagnostics)
        return SessionSelection(
            cwd=effective_cwd, model=resolved_model, thinking_level=effective_thinking,
            tools=effective_tools, diagnostics=(*self.diagnostics, *diagnostics),
        )

    def select_open(
        self, path: str | Path, *, provider: str | None = None, model: str | None = None,
        thinking: ModelThinkingLevel | None = None,
        tools: tuple[ToolName, ...] | None = None, cwd: str | Path | None = None,
    ) -> SessionSelection:
        """Resolve a reopened session, preserving its saved cwd and selection."""
        diagnostics: list[ConfigDiagnostic] = []
        destination = self._resolve_path(path)
        try:
            decoded = decode_history(destination.read_text(encoding="utf-8"))
        except FileNotFoundError:
            diagnostics.append(ConfigDiagnostic(
                str(destination), "history", "invalid-schema", f"Session file does not exist: {destination}",
            ))
            return SessionSelection(diagnostics=tuple(diagnostics))
        except (OSError, ValueError) as error:
            diagnostics.append(ConfigDiagnostic(
                str(destination), "history", "invalid-schema", f"Cannot read session {destination}: {error}",
            ))
            return SessionSelection(diagnostics=tuple(diagnostics))
        settings = validate_history(decoded.history)
        effective_cwd = self._resolve_cwd(cwd, history=decoded, diagnostics=diagnostics)
        self._refresh_settings(effective_cwd, diagnostics)
        resolved_model = self._resolve_history_model(provider, model, settings, diagnostics)
        effective_thinking = self._resolve_thinking(
            thinking, history=settings, model=resolved_model, diagnostics=diagnostics,
        )
        effective_tools = self._resolve_tools(tools, diagnostics)
        return SessionSelection(
            cwd=effective_cwd, model=resolved_model, thinking_level=effective_thinking,
            tools=effective_tools, history=decoded, diagnostics=(*self.diagnostics, *diagnostics),
        )

    def build_options(
        self, selection: SessionSelection, *, session_file: str | Path | None = None,
        agent_options: AgentOptions | None = None,
    ) -> CodingAgentOptions:
        """Assemble host dependencies for a ready selection."""
        if not selection.ready or selection.model is None or selection.cwd is None:
            raise ConfigError("Cannot assemble a session from an unresolved selection")
        base = agent_options or AgentOptions()
        compaction = self.compaction
        retry = self.retry
        assembled = replace(
            base,
            api_key=self.api_key if self.api_key is not None else base.api_key,
            compaction=compaction if compaction is not None else base.compaction,
            retry=retry if retry is not None else base.retry,
        )
        return CodingAgentOptions(
            cwd=selection.cwd,
            model=selection.model,
            stream_fn=self.stream_fn(),
            thinking_level=selection.thinking_level,
            available_models=tuple(entry.model for entry in self.directory.listings()),
            tools=selection.tools,
            session_file=session_file,
            agent_options=assembled,
            agent_dir=self.agent_dir,
            skill_sources=self._skill_sources(),
            template_sources=self._template_sources(),
            custom_prompt=self._custom_prompt,
            append_system_prompt=self._append_prompt,
            resource_tiers=RESOURCE_TIERS,
            load_context_files=not self._no_context_files,
        )

    # ------------------------------------------------------------------ #
    # Resolution helpers
    # ------------------------------------------------------------------ #

    def _refresh_settings(self, cwd: Path | None, diagnostics: list[ConfigDiagnostic]) -> None:
        """Re-resolve project trust, settings and system inputs for the effective cwd.

        Project configuration and resource discovery follow the effective
        session cwd, not the startup directory, so an explicit ``--cwd`` or a
        restored cwd selects its own trust decision, ``.omh/settings.json`` and
        system files.
        """
        resolved_cwd = cwd if cwd is not None else self.startup_dir
        self._settings_cwd = resolved_cwd
        trusted = self._resolve_trust(resolved_cwd, diagnostics)
        self._trusted = trusted
        self.settings = load_settings(
            agent_dir=self.agent_dir, cwd=resolved_cwd,
            project_trusted=trusted, explicit=self._explicit_settings,
        )
        self._custom_prompt, self._append_prompt = self._resolve_system(
            resolved_cwd, trusted, diagnostics,
        )

    def _resolve_trust(self, cwd: Path, diagnostics: list[ConfigDiagnostic]) -> bool:
        """Resolve the project's loading authorization without asking a question.

        Explicit one-run flags win over a remembered decision, which wins over
        the embedding default. With no decision and real controlled project
        resources present, the layer is skipped and reported so print never
        waits for an answer; interactive can ask and then remember one.
        """
        self._trust_unknown = False
        if self._approve:
            return True
        if self._no_approve:
            return False
        if self._project_trusted is not None:
            return self._project_trusted
        decision = self._trust_store.decision(cwd)
        if decision == "approved":
            return True
        if decision == "denied":
            return False
        self._trust_unknown = True
        if self._project_resources(cwd):
            diagnostics.append(ConfigDiagnostic(
                str(cwd), "trust", "untrusted",
                "Project configuration and resources are not trusted and were skipped; "
                "use --approve or remember a decision with /trust",
            ))
        return False

    @staticmethod
    def _project_resources(cwd: Path) -> bool:
        """Report whether a project holds any trust-controlled resource."""
        base = cwd / CONFIG_DIR_NAME
        return (
            (base / SETTINGS_FILE).is_file()
            or (base / SYSTEM_FILE).exists()
            or (base / APPEND_SYSTEM_FILE).exists()
            or (base / SKILLS_DIR).is_dir()
            or (base / PROMPTS_DIR).is_dir()
        )

    def _resolve_system(
        self, cwd: Path, trusted: bool, diagnostics: list[ConfigDiagnostic],
    ) -> tuple[str | None, str | None]:
        """Resolve the base replacement and the single addendum in fallback order.

        SYSTEM and APPEND each use explicit input, then a trusted project file,
        then the global file. A project APPEND replaces the global fallback
        instead of stacking on it; explicit APPEND parts keep command-line order.
        """
        if self._explicit_system_prompt is not None:
            custom: str | None = self._explicit_system_prompt
        elif self._explicit_system_prompt_file is not None:
            custom = self._read_explicit_file(self._explicit_system_prompt_file, diagnostics)
        else:
            custom = self._read_fallback(cwd, trusted, diagnostics, SYSTEM_FILE)
        if self._append_system:
            parts: list[str] = []
            for kind, value in self._append_system:
                text = value if kind == "text" else self._read_explicit_file(value, diagnostics)
                if text is not None:
                    parts.append(text)
            append: str | None = "\n\n".join(parts)
        else:
            append = self._read_fallback(cwd, trusted, diagnostics, APPEND_SYSTEM_FILE)
        return custom, append

    def _read_explicit_file(
        self, value: str | Path, diagnostics: list[ConfigDiagnostic],
    ) -> str | None:
        path = self._resolve_path(value)
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            diagnostics.append(ConfigDiagnostic(
                str(path), "cli", "unavailable", f"Cannot read {path.name}: {error}",
            ))
            return None

    def _read_fallback(
        self, cwd: Path, trusted: bool, diagnostics: list[ConfigDiagnostic], name: str,
    ) -> str | None:
        if trusted:
            project_file = cwd / CONFIG_DIR_NAME / name
            text = self._read_resource(project_file, diagnostics, "project", name)
            if text is not None:
                return text
        return self._read_resource(self.agent_dir / name, diagnostics, "global", name)

    @staticmethod
    def _read_resource(
        path: Path, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, name: str,
    ) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as error:
            diagnostics.append(ConfigDiagnostic(
                str(path), source, "recoverable", f"Cannot read {name}: {error}",
            ))
            return None


    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.startup_dir / candidate
        return candidate.resolve()

    def _resolve_cwd(
        self, explicit: str | Path | None, *, history: DecodedHistory | None,
        diagnostics: list[ConfigDiagnostic],
    ) -> Path | None:
        if explicit is not None:
            candidate = self._resolve_path(explicit)
            if not candidate.is_dir():
                diagnostics.append(ConfigDiagnostic(
                    str(candidate), "cli", "unavailable", f"Working directory does not exist: {candidate}",
                ))
                return None
            return candidate
        if history is not None:
            stored = Path(history.cwd).expanduser()
            if not stored.is_dir():
                diagnostics.append(ConfigDiagnostic(
                    str(stored), "history", "unavailable",
                    f"Saved working directory no longer exists: {stored}; choose a replacement directory",
                ))
                return None
            return stored.resolve()
        return self.startup_dir

    def _configured_default_model(self, diagnostics: list[ConfigDiagnostic]) -> Model | None:
        """Resolve the configured default model, diagnosing an unavailable selection."""
        configured = settings_model(self.settings)
        if configured is None:
            return None
        provider, model_id = configured
        start = len(diagnostics)
        model = self._resolve_reference(
            provider, model_id, diagnostics, source=self.settings.source_of("defaultModel"),
        )
        for index in range(start, len(diagnostics)):
            diagnostics[index] = replace(diagnostics[index], path=str(self._settings_path()))
        return model

    def _settings_path(self) -> Path:
        if self.settings.source_of("defaultModel") == "project":
            return project_settings_path(self._settings_cwd)
        return self.agent_dir / SETTINGS_FILE

    def _configured_or_product_model(self, diagnostics: list[ConfigDiagnostic]) -> Model | None:
        """Apply the configured default, falling back to the product default only when unset."""
        configured = self._configured_default_model(diagnostics)
        if configured is not None or settings_model(self.settings) is not None:
            return configured
        return self._resolve_reference(
            DEFAULT_MODEL_REFERENCE[0], DEFAULT_MODEL_REFERENCE[1], diagnostics, source="settings",
        )

    def _resolve_new_model(
        self, provider: str | None, model: str | None, diagnostics: list[ConfigDiagnostic],
    ) -> Model | None:
        if model is not None or provider is not None:
            return self._resolve_reference(provider, model, diagnostics, source="cli")
        return self._configured_or_product_model(diagnostics)

    def _resolve_history_model(
        self, provider: str | None, model: str | None, settings: AgentHistorySettings,
        diagnostics: list[ConfigDiagnostic],
    ) -> Model | None:
        if model is not None or provider is not None:
            return self._resolve_reference(provider, model, diagnostics, source="cli")
        if settings.provider is not None and settings.model_id is not None:
            matches = self.directory.find_models(settings.provider, settings.model_id)
            if not matches:
                diagnostics.append(ConfigDiagnostic(
                    f"{settings.provider}/{settings.model_id}", "history", "unavailable",
                    f"Saved model {settings.provider}/{settings.model_id} is not available; "
                    "select a model explicitly",
                ))
                return None
            return matches[0]
        return self._configured_or_product_model(diagnostics)

    def _resolve_reference(
        self, provider: str | None, reference: str | None, diagnostics: list[ConfigDiagnostic],
        *, source: DiagnosticSource,
    ) -> Model | None:
        if reference is None:
            if provider is None:
                return None
            diagnostics.append(ConfigDiagnostic(
                provider, source, "invalid-value", f"Provider {provider!r} was given without a model",
            ))
            return None
        prefix, separator, model_id = reference.partition("/")
        if separator:
            if not prefix or not model_id:
                diagnostics.append(ConfigDiagnostic(
                    reference, source, "invalid-value",
                    f"Invalid model {reference!r}; expected <provider>/<model> or <model>",
                ))
                return None
            if provider is not None and provider != prefix:
                diagnostics.append(ConfigDiagnostic(
                    reference, source, "invalid-value",
                    f"Provider {provider} does not match model provider {prefix}",
                ))
                return None
            provider = prefix
        else:
            model_id = reference
        matches = self.directory.find_models(provider, model_id)
        if not matches:
            diagnostics.append(ConfigDiagnostic(
                reference, source, "unavailable", f"Unknown model {reference!r}",
            ))
            return None
        if len(matches) > 1:
            diagnostics.append(ConfigDiagnostic(
                reference, source, "unavailable",
                f"Ambiguous model {reference!r}; use <provider>/<model>",
            ))
            return None
        return matches[0]

    def _resolve_thinking(
        self, explicit: ModelThinkingLevel | None, *, history: AgentHistorySettings | None,
        model: Model | None, diagnostics: list[ConfigDiagnostic],
    ) -> ModelThinkingLevel | None:
        if model is None:
            return None
        if explicit is not None:
            supported = get_supported_thinking_levels(model)
            if explicit not in supported:
                diagnostics.append(ConfigDiagnostic(
                    f"{model.provider}/{model.id}", "cli", "invalid-value",
                    f"{model.provider}/{model.id} has a fixed thinking mode with no adjustable level"
                    if model.reasoning and not supported else
                    f"Thinking level {explicit!r} is not supported by {model.provider}/{model.id}; "
                    f"valid levels: {', '.join(supported)}",
                ))
                return None
            return explicit
        requested: ModelThinkingLevel | None = None
        origin: DiagnosticSource = "settings"
        if history is not None and history.has_thinking_level:
            requested = history.thinking_level
            origin = "history"
        if requested is None:
            configured = settings_thinking(self.settings)
            if configured is not None:
                requested = configured
                origin = self.settings.source_of("defaultThinkingLevel")
        if requested is None:
            requested = DEFAULT_THINKING_LEVEL
            origin = "settings"
        clamped = clamp_thinking_level(model, requested)
        if clamped != requested:
            fixed = model.reasoning and not get_supported_thinking_levels(model)
            if not fixed or history is not None or settings_thinking(self.settings) is not None:
                mode = "fixed-on" if fixed else clamped
                diagnostics.append(ConfigDiagnostic(
                    f"{model.provider}/{model.id}", origin, "adjusted",
                    f"Thinking level {requested!r} is not supported by {model.provider}/{model.id}; using {mode!r}",
                ))
        return clamped

    def _resolve_tools(
        self, explicit: tuple[ToolName, ...] | None, diagnostics: list[ConfigDiagnostic],
    ) -> tuple[ToolName, ...]:
        if explicit is not None:
            validated = validate_tools(explicit)
            if validated is None:
                diagnostics.append(ConfigDiagnostic(
                    "--tools", "cli", "invalid-value",
                    f"--tools must be a distinct subset of {', '.join(DEFAULT_TOOLS)}",
                ))
                return DEFAULT_TOOLS
            return validated
        configured = settings_tools(self.settings)
        if configured is not None:
            return configured
        return DEFAULT_TOOLS

    def _skill_sources(self) -> tuple[SkillSource, ...]:
        sources = [
            SkillSource(self._resolve_path(value), "explicit")
            for value in self._explicit_skills
        ]
        sources.extend(
            SkillSource(value, self._configured_tier("skills"))
            for value in settings_strings(self.settings, "skills")
        )
        if not self._no_skills:
            sources.extend(automatic_skill_sources(
                cwd=self._settings_cwd, agent_dir=self.agent_dir, home=self.home,
                project_trusted=self._trusted,
            ))
        return tuple(sources)

    def _template_sources(self) -> tuple[PromptTemplateSource, ...]:
        sources = [
            PromptTemplateSource(self._resolve_path(value), "explicit")
            for value in self._explicit_templates
        ]
        sources.extend(
            PromptTemplateSource(value, self._configured_tier("prompts"))
            for value in settings_strings(self.settings, "prompts")
        )
        if not self._no_prompt_templates:
            sources.extend(automatic_template_sources(
                cwd=self._settings_cwd, agent_dir=self.agent_dir,
                project_trusted=self._trusted,
            ))
        return tuple(sources)

    def _configured_tier(self, key: str) -> str:
        """Map the effective settings scope for a resource array onto its tier."""
        return _TIER_FOR_SCOPE.get(self.settings.source_of(key), "global-config")
