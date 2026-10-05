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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

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
from omh.llm import create_models
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
    AUTH_FILE,
    DEFAULT_TOOLS,
    SETTINGS_FILE,
    ConfigDiagnostic,
    ConfigError,
    DiagnosticSource,
    FileCredentialStore,
    SettingsSnapshot,
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

#: The product's default model when neither history nor configuration selects one.
DEFAULT_MODEL_REFERENCE = ("opencode-go", "deepseek-v4.1-flash")
#: The default thinking level for models that accept a choice.
DEFAULT_THINKING_LEVEL: ModelThinkingLevel = "high"


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
        return self.cwd is not None and self.model is not None and self.thinking_level is not None


class CodingAgentHost:
    """Resolve configuration, credentials and effective selection for a session."""

    def __init__(
        self, *, startup_dir: str | Path, agent_dir: str | Path | None = None,
        project_trusted: bool = False, explicit_settings: Mapping[str, object] | None = None,
        explicit_skills: tuple[str, ...] = (), explicit_templates: tuple[str, ...] = (),
        api_key: str | None = None, fetch: FetchFunction | None = None,
    ) -> None:
        self.startup_dir = Path(startup_dir).expanduser().resolve()
        self.agent_dir = (
            Path(agent_dir).expanduser().resolve()
            if agent_dir is not None else resolve_agent_dir()
        )
        self.settings: SettingsSnapshot = load_settings(
            agent_dir=self.agent_dir, cwd=self.startup_dir,
            project_trusted=project_trusted, explicit=explicit_settings,
        )
        self._project_trusted = project_trusted
        self._explicit_settings = explicit_settings
        self._settings_cwd = self.startup_dir
        self.directory = ModelDirectory(agent_dir=self.agent_dir)
        self.credentials = FileCredentialStore(self.agent_dir / AUTH_FILE)
        self.api_key = api_key
        self._fetch = fetch
        self._explicit_skills = explicit_skills
        self._explicit_templates = explicit_templates
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
        return (*self.settings.diagnostics, *self.directory.diagnostics)

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
        if selection.ready and selection.model is not None and await self.key_source(selection.model) is None:
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
        self._refresh_settings(effective_cwd)
        resolved_model = self._resolve_new_model(provider, model, diagnostics)
        effective_thinking = self._resolve_thinking(thinking, history=None, model=resolved_model, diagnostics=diagnostics)
        effective_tools = self._resolve_tools(tools, diagnostics)
        return SessionSelection(
            cwd=effective_cwd, model=resolved_model, thinking_level=effective_thinking,
            tools=effective_tools, diagnostics=tuple(diagnostics),
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
        self._refresh_settings(effective_cwd)
        resolved_model = self._resolve_history_model(provider, model, settings, diagnostics)
        effective_thinking = self._resolve_thinking(
            thinking, history=settings, model=resolved_model, diagnostics=diagnostics,
        )
        effective_tools = self._resolve_tools(tools, diagnostics)
        return SessionSelection(
            cwd=effective_cwd, model=resolved_model, thinking_level=effective_thinking,
            tools=effective_tools, history=decoded, diagnostics=tuple(diagnostics),
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
        )

    # ------------------------------------------------------------------ #
    # Resolution helpers
    # ------------------------------------------------------------------ #

    def _refresh_settings(self, cwd: Path | None) -> None:
        """Re-resolve project settings for the effective cwd.

        Project configuration and resource discovery follow the effective
        session cwd, not the startup directory, so an explicit ``--cwd`` or a
        restored cwd selects its own ``.omh/settings.json``.
        """
        resolved_cwd = cwd if cwd is not None else self.startup_dir
        self._settings_cwd = resolved_cwd
        self.settings = load_settings(
            agent_dir=self.agent_dir, cwd=resolved_cwd,
            project_trusted=self._project_trusted, explicit=self._explicit_settings,
        )

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
        matches = self.directory.find_models(provider, model_id)
        if len(matches) == 1:
            return matches[0]
        message = (
            f"Configured default model {provider}/{model_id} is ambiguous; use <provider>/<model>"
            if matches else
            f"Configured default model {provider}/{model_id} is not available"
        )
        diagnostics.append(ConfigDiagnostic(str(self._settings_path()), "settings", "unavailable", message))
        return None

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
            diagnostics.append(ConfigDiagnostic(
                f"{model.provider}/{model.id}", origin, "invalid-value",
                f"Thinking level {requested!r} is not supported by {model.provider}/{model.id}; using {clamped!r}",
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
        source = self.settings.source_of("skills")
        sources.extend(
            SkillSource(value, source) for value in settings_strings(self.settings, "skills")
        )
        return tuple(sources)

    def _template_sources(self) -> tuple[PromptTemplateSource, ...]:
        sources = [
            PromptTemplateSource(self._resolve_path(value), "explicit")
            for value in self._explicit_templates
        ]
        source = self.settings.source_of("prompts")
        sources.extend(
            PromptTemplateSource(value, source) for value in settings_strings(self.settings, "prompts")
        )
        return tuple(sources)
