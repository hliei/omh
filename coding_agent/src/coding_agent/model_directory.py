"""Built-in model directory with explicit global ``models.json`` overrides.

The directory is credential-blind metadata. It registers the providers whose
runtime support ships with this product so ``--list-models`` and explicit
``--provider``/``--model`` validation stay offline and never resolve
credentials. A global ``models.json`` may override metadata for, or add models
to, those already-supported providers through the existing Completions
protocol; it cannot introduce another provider, protocol or route. Missing
required fields are diagnosed rather than inferred from a similar ID.

Startup and reopen never fetch a catalog over the network or replace a
history selection. A model's presence here is a metadata claim, not proof that
a live route has been verified.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from omh.llm import (
    create_models,
    deepseek_provider,
    get_supported_thinking_levels,
    opencode_go_provider,
)
from omh.llm.models import Models, Provider
from omh.llm.types import (
    Api,
    Model,
    ModelCost,
    ModelThinkingLevel,
    OpenAICompletionsCompat,
    OpenAICompletionsOptions,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingLevelMap,
    TranscriptContext,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream

from coding_agent.config import MODELS_FILE, ConfigDiagnostic, DiagnosticSource

#: Marks model metadata that ships with the product rather than user config.
SOURCE_BUILTIN = "builtin"
#: Marks metadata supplied or overridden by the global ``models.json``.
SOURCE_USER = "user"
#: Source date for the built-in Completions catalog snapshots.
BUILTIN_CATALOG_DATE = "2026-10-05"
#: The only protocol this product's model directory supports.
SUPPORTED_API = "openai-completions"

ProviderFactory = Callable[[], Provider]

#: Providers whose models this product can actually run. Registering one here
#: makes its models appear in ``--list-models`` and valid for exact selection.
_BUILTIN_PROVIDERS: tuple[ProviderFactory, ...] = (
    deepseek_provider,
    opencode_go_provider,
)

MODEL_DEFINITION_FIELDS = frozenset({
    "id", "name", "api", "reasoning", "thinkingLevelMap", "input", "cost",
    "contextWindow", "maxTokens", "samplingParams", "headers", "compat",
})
MODEL_OVERRIDE_FIELDS = frozenset({
    "name", "reasoning", "thinkingLevelMap", "input", "cost",
    "contextWindow", "maxTokens", "samplingParams", "headers", "compat",
})
_PROVIDER_FIELDS = frozenset({"models", "modelOverrides"})
_REQUIRED_DEFINITION_FIELDS = ("id", "api", "reasoning", "input", "cost", "contextWindow", "maxTokens")
_COMPAT_FIELDS = {
    "supportsStore": "supports_store",
    "supportsDeveloperRole": "supports_developer_role",
    "supportsReasoningEffort": "supports_reasoning_effort",
    "supportsUsageInStreaming": "supports_usage_in_streaming",
    "supportsFinishReason": "supports_finish_reason",
    "requiresToolResultName": "requires_tool_result_name",
    "requiresAssistantAfterToolResult": "requires_assistant_after_tool_result",
    "requiresThinkingAsText": "requires_thinking_as_text",
    "requiresReasoningContentOnAssistantMessages": "requires_reasoning_content_on_assistant_messages",
    "supportsStrictMode": "supports_strict_mode",
    "supportsMidConvoSystemMessages": "supports_mid_convo_system_messages",
}
_COMPAT_ENUMS = {"maxTokensField": "max_tokens_field", "thinkingFormat": "thinking_format"}


def builtin_models() -> Models:
    """Assemble the registered providers without reading credentials."""
    models = create_models()
    for factory in _BUILTIN_PROVIDERS:
        models.set_provider(factory())
    return models


def builtin_provider_ids() -> tuple[str, ...]:
    """Provider IDs whose runtime support ships with this product."""
    return tuple(dict.fromkeys(model.provider for model in _builtin_catalog()))


def _builtin_catalog() -> tuple[Model, ...]:
    catalog: list[Model] = []
    for factory in _BUILTIN_PROVIDERS:
        catalog.extend(factory().get_models())
    return tuple(catalog)


@dataclass(frozen=True, slots=True)
class ModelListing:
    """One provider/model entry with the effective options this product exposes."""

    provider: str
    model: Model
    thinking_levels: tuple[str, ...]
    source: str = SOURCE_BUILTIN
    source_date: str = BUILTIN_CATALOG_DATE


@dataclass(slots=True)
class _ProviderConfig:
    models: list[dict[str, object]] = field(default_factory=list)
    overrides: dict[str, dict[str, object]] = field(default_factory=dict)


class ModelDirectory:
    """Read-only view over the registered providers and their models."""

    def __init__(
        self, models: Models | None = None, *,
        agent_dir: str | Path | None = None, models_path: str | Path | None = None,
    ) -> None:
        self._meta: dict[tuple[str, str], tuple[str, str]] = {}
        if models is not None:
            self._models = models
            self.diagnostics: tuple[ConfigDiagnostic, ...] = ()
            return
        path = Path(models_path) if models_path is not None else (
            Path(agent_dir) / MODELS_FILE if agent_dir is not None else None
        )
        diagnostics: list[ConfigDiagnostic] = []
        config = _load_models_config(path, diagnostics)
        user_date = _user_source_date(path)
        directory = create_models()
        for factory in _BUILTIN_PROVIDERS:
            base = factory()
            provider_config = config.get(base.id, _ProviderConfig())
            directory.set_provider(_merge_provider(
                base, provider_config, path, diagnostics, self._meta, user_date,
            ))
        self.diagnostics = tuple(diagnostics)
        self._models = directory

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(provider.id for provider in self._models.get_providers())

    @property
    def providers(self) -> tuple[Provider, ...]:
        """The effective providers, including any merged ``models.json`` metadata."""
        return tuple(self._models.get_providers())

    def listings(self, search: str | None = None) -> tuple[ModelListing, ...]:
        """Return every effective model, ordered by provider and ID, optionally searched."""
        entries = []
        for model in self._models.get_models():
            source, source_date = self._meta.get(
                (model.provider, model.id), (SOURCE_BUILTIN, BUILTIN_CATALOG_DATE),
            )
            entries.append(ModelListing(
                provider=model.provider,
                model=model,
                thinking_levels=self.thinking_levels(model),
                source=source,
                source_date=source_date,
            ))
        entries.sort(key=lambda entry: (entry.provider, entry.model.id))
        if search is not None and search.strip():
            entries = [entry for entry in entries if _matches(entry, search.strip())]
        return tuple(entries)

    def find_models(self, provider: str | None, model_id: str) -> tuple[Model, ...]:
        """Find exact model IDs; an omitted provider searches all providers."""
        if provider is not None:
            found = self._models.get_model(provider, model_id)
            return (found,) if found is not None else ()
        return tuple(
            model
            for model in self._models.get_models()
            if model.id == model_id
        )

    @staticmethod
    def thinking_levels(model: Model) -> tuple[str, ...]:
        return tuple(get_supported_thinking_levels(model))


def _matches(entry: ModelListing, search: str) -> bool:
    needle = search.casefold()
    haystacks = (
        entry.provider.casefold(),
        entry.model.id.casefold(),
        entry.model.name.casefold(),
        f"{entry.provider}/{entry.model.id}".casefold(),
    )
    if any(needle in haystack for haystack in haystacks):
        return True
    return _is_subsequence(needle, f"{entry.provider}/{entry.model.id}".casefold())


def _is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(character in iterator for character in needle)


# --------------------------------------------------------------------------- #
# models.json
# --------------------------------------------------------------------------- #


def _user_source_date(path: Path | None) -> str:
    """Report when user-supplied metadata was configured, not the built-in date."""
    if path is None:
        return BUILTIN_CATALOG_DATE
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date().isoformat()
    except OSError:
        return BUILTIN_CATALOG_DATE


def _load_models_config(
    path: Path | None, diagnostics: list[ConfigDiagnostic],
) -> dict[str, _ProviderConfig]:
    if path is None:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as error:
        diagnostics.append(ConfigDiagnostic(
            str(path), "models", "invalid-json", f"Cannot read models.json: {error}",
        ))
        return {}
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except ValueError as error:
        diagnostics.append(ConfigDiagnostic(
            str(path), "models", "invalid-json", f"Invalid JSON in models.json: {error}",
        ))
        return {}
    if not isinstance(parsed, dict):
        diagnostics.append(ConfigDiagnostic(
            str(path), "models", "invalid-schema", "models.json must contain a JSON object",
        ))
        return {}
    providers = parsed.get("providers")
    if not isinstance(providers, dict):
        diagnostics.append(ConfigDiagnostic(
            str(path), "models", "invalid-schema", "models.json requires a providers object",
        ))
        return {}

    supported = set(builtin_provider_ids())
    config: dict[str, _ProviderConfig] = {}
    for provider_id, raw in providers.items():
        if provider_id not in supported:
            diagnostics.append(ConfigDiagnostic(
                str(path), "models", "invalid-value",
                f"models.json provider {provider_id!r} is not a supported provider",
            ))
            continue
        if not isinstance(raw, dict):
            diagnostics.append(ConfigDiagnostic(
                str(path), "models", "invalid-type", f"models.json provider {provider_id} must be an object",
            ))
            continue
        unknown = set(raw) - _PROVIDER_FIELDS
        if unknown:
            diagnostics.append(ConfigDiagnostic(
                str(path), "models", "unknown-key",
                f"models.json provider {provider_id} has unknown keys {', '.join(sorted(unknown))}",
            ))
        entry = _ProviderConfig()
        definitions = raw.get("models", [])
        if not isinstance(definitions, list):
            diagnostics.append(ConfigDiagnostic(
                str(path), "models", "invalid-type", f"models.json {provider_id}.models must be an array",
            ))
        else:
            for index, definition in enumerate(definitions):
                if not isinstance(definition, dict):
                    diagnostics.append(ConfigDiagnostic(
                        str(path), "models", "invalid-type",
                        f"models.json {provider_id}.models[{index}] must be an object",
                    ))
                    continue
                entry.models.append(definition)
        overrides = raw.get("modelOverrides", {})
        if not isinstance(overrides, dict):
            diagnostics.append(ConfigDiagnostic(
                str(path), "models", "invalid-type",
                f"models.json {provider_id}.modelOverrides must be an object",
            ))
        else:
            for model_id, override in overrides.items():
                if not isinstance(override, dict):
                    diagnostics.append(ConfigDiagnostic(
                        str(path), "models", "invalid-type",
                        f"models.json {provider_id}.modelOverrides.{model_id} must be an object",
                    ))
                    continue
                entry.overrides[model_id] = override
        config[provider_id] = entry
    return config


def _merge_provider(
    base: Provider, config: _ProviderConfig, path: Path | None,
    diagnostics: list[ConfigDiagnostic], meta: dict[tuple[str, str], tuple[str, str]],
    user_date: str,
) -> Provider:
    models: list[Model] = []
    seen: set[str] = {model.id for model in base.get_models()}
    for model in base.get_models():
        override = config.overrides.get(model.id)
        if override is None:
            models.append(model)
            continue
        models.append(_apply_override(path, base.id, model, override, diagnostics))
        meta[(base.id, model.id)] = (SOURCE_USER, user_date)
    for definition in config.models:
        definition_model = _definition_model(path, base, definition, diagnostics)
        if definition_model is None:
            continue
        if definition_model.id in seen:
            diagnostics.append(ConfigDiagnostic(
                _source(path), "models", "invalid-value",
                f"models.json model {definition_model.id!r} duplicates an existing model; "
                "use modelOverrides instead",
            ))
            continue
        models.append(definition_model)
        meta[(base.id, definition_model.id)] = (SOURCE_USER, user_date)
        seen.add(definition_model.id)
    return _WrappedProvider(base, models)


class _WrappedProvider:
    """A registered provider whose catalog is replaced by user-merged metadata."""

    def __init__(self, base: Provider, models: Sequence[Model]) -> None:
        self._base = base
        self.id = base.id
        self.name = base.name
        self.base_url = base.base_url
        self.headers = base.headers
        self.auth = base.auth
        self._models = tuple(models)

    def get_models(self) -> Sequence[Model]:
        return self._models

    def stream(
        self, model: Model, context: TranscriptContext,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._base.stream(model, context, options)

    def stream_simple(
        self, model: Model, context: TranscriptContext,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._base.stream_simple(model, context, options)


def _definition_model(
    path: Path | None, base: Provider, definition: Mapping[str, object],
    diagnostics: list[ConfigDiagnostic],
) -> Model | None:
    source = _source(path)
    provider_id = base.id
    unknown = set(definition) - MODEL_DEFINITION_FIELDS
    if unknown:
        diagnostics.append(ConfigDiagnostic(
            source, "models", "unknown-key",
            f"models.json {provider_id}.models has unknown keys {', '.join(sorted(unknown))}",
        ))
    missing = [name for name in _REQUIRED_DEFINITION_FIELDS if definition.get(name) is None]
    model_id = definition.get("id")
    label = model_id if isinstance(model_id, str) else "<missing id>"
    if missing:
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-schema",
            f"models.json model {label!r} is missing {', '.join(missing)}",
        ))
        return None
    if not isinstance(model_id, str) or not model_id:
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type",
            f"models.json {provider_id}.models id must be a non-empty string",
        ))
        return None
    if definition.get("api") != SUPPORTED_API:
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-value",
            f"models.json model {model_id!r} must use the {SUPPORTED_API} protocol",
        ))
        return None
    input_modalities = _input(definition.get("input"), diagnostics, source, model_id)
    if input_modalities is None:
        return None
    cost = _cost(definition.get("cost"), diagnostics, source, model_id)
    if cost is None:
        return None
    reasoning = definition.get("reasoning")
    if not isinstance(reasoning, bool):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type",
            f"models.json model {model_id!r} reasoning must be a boolean",
        ))
        return None
    context_window = _positive_int(definition.get("contextWindow"), "contextWindow", diagnostics, source, model_id)
    max_tokens = _positive_int(definition.get("maxTokens"), "maxTokens", diagnostics, source, model_id)
    if context_window is None or max_tokens is None:
        return None
    name = definition.get("name")
    thinking_map = _thinking_map(definition.get("thinkingLevelMap"), diagnostics, source, model_id)
    headers = _headers(definition.get("headers"), diagnostics, source, model_id)
    sampling = _sampling_params(definition.get("samplingParams"), diagnostics, source, model_id)
    return Model(
        id=model_id,
        name=name if isinstance(name, str) and name else model_id,
        api=cast("Api", SUPPORTED_API),
        provider=provider_id,
        base_url=base.base_url or "",
        reasoning=reasoning,
        input=input_modalities,
        cost=cost,
        context_window=context_window,
        max_tokens=max_tokens,
        thinking_level_map=thinking_map,
        sampling_params=sampling,
        headers=headers,
        compat=_compat(definition.get("compat"), diagnostics, source, model_id),
    )


def _apply_override(
    path: Path | None, provider_id: str, model: Model, override: Mapping[str, object],
    diagnostics: list[ConfigDiagnostic],
) -> Model:
    source = _source(path)
    unknown = set(override) - MODEL_OVERRIDE_FIELDS
    if unknown:
        diagnostics.append(ConfigDiagnostic(
            source, "models", "unknown-key",
            f"models.json {provider_id}.modelOverrides.{model.id} has unknown keys "
            f"{', '.join(sorted(unknown))}",
        ))
    changes: dict[str, object] = {}
    if isinstance(override.get("name"), str) and override["name"]:
        changes["name"] = override["name"]
    if "reasoning" in override:
        if isinstance(override["reasoning"], bool):
            changes["reasoning"] = override["reasoning"]
        else:
            diagnostics.append(ConfigDiagnostic(
                source, "models", "invalid-type",
                f"models.json override {provider_id}/{model.id} reasoning must be a boolean",
            ))
    if "input" in override:
        modalities = _input(override["input"], diagnostics, source, model.id)
        if modalities is not None:
            changes["input"] = modalities
    if "cost" in override:
        cost = _cost(override["cost"], diagnostics, source, model.id, base=model.cost)
        if cost is not None:
            changes["cost"] = cost
    for key, field_name in (("contextWindow", "context_window"), ("maxTokens", "max_tokens")):
        if key not in override:
            continue
        value = _positive_int(override[key], key, diagnostics, source, model.id)
        if value is not None:
            changes[field_name] = value
    if "thinkingLevelMap" in override:
        changes["thinking_level_map"] = _thinking_map(override["thinkingLevelMap"], diagnostics, source, model.id)
    if "samplingParams" in override:
        changes["sampling_params"] = _sampling_params(override["samplingParams"], diagnostics, source, model.id)
    if "headers" in override:
        changes["headers"] = _headers(override["headers"], diagnostics, source, model.id)
    if "compat" in override:
        changes["compat"] = _compat(override["compat"], diagnostics, source, model.id)
    return replace(model, **changes)  # type: ignore[arg-type]


def _source(path: Path | None) -> DiagnosticSource:
    return cast(DiagnosticSource, str(path)) if path is not None else "models"


def _input(
    value: object, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, model_id: str,
) -> tuple[Literal["text", "image"], ...] | None:
    if not isinstance(value, list) or not value or not all(entry in ("text", "image") for entry in value):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-value",
            f"models.json model {model_id!r} input must be a non-empty array of text/image",
        ))
        return None
    return tuple(cast("list[Literal['text', 'image']]", value))


def _positive_int(
    value: object, label: str, diagnostics: list[ConfigDiagnostic],
    source: DiagnosticSource, model_id: str,
) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    diagnostics.append(ConfigDiagnostic(
        source, "models", "invalid-value",
        f"models.json model {model_id!r} {label} must be a positive integer",
    ))
    return None


def _cost(
    value: object, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, model_id: str, *,
    base: ModelCost | None = None,
) -> ModelCost | None:
    if not isinstance(value, dict):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type",
            f"models.json model {model_id!r} cost must be an object",
        ))
        return None
    resolved: dict[str, float] = {}
    for key, field_name in (
        ("input", "input"), ("output", "output"), ("cacheRead", "cache_read"), ("cacheWrite", "cache_write"),
    ):
        entry = value.get(key, getattr(base, field_name) if base is not None else None)
        if not isinstance(entry, int | float) or isinstance(entry, bool) or entry < 0:
            diagnostics.append(ConfigDiagnostic(
                source, "models", "invalid-value",
                f"models.json model {model_id!r} cost.{key} must be a non-negative number",
            ))
            return None
        resolved[field_name] = float(entry)
    return ModelCost(**resolved)


def _compat(
    value: object, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, model_id: str,
) -> OpenAICompletionsCompat | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type", f"models.json model {model_id!r} compat must be an object",
        ))
        return None
    options: dict[str, object] = {}
    for key, field_name in _COMPAT_FIELDS.items():
        if key in value:
            if isinstance(value[key], bool):
                options[field_name] = value[key]
            else:
                diagnostics.append(ConfigDiagnostic(
                    source, "models", "invalid-type",
                    f"models.json model {model_id!r} compat.{key} must be a boolean",
                ))
                return None
    for key, field_name in _COMPAT_ENUMS.items():
        if key in value:
            if isinstance(value[key], str):
                options[field_name] = value[key]
            else:
                diagnostics.append(ConfigDiagnostic(
                    source, "models", "invalid-type",
                    f"models.json model {model_id!r} compat.{key} must be a string",
                ))
                return None
    return OpenAICompletionsCompat(**options)  # type: ignore[arg-type]


def _thinking_map(
    value: object, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, model_id: str,
) -> ThinkingLevelMap | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type",
            f"models.json model {model_id!r} thinkingLevelMap must be an object",
        ))
        return None
    allowed = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
    result: ThinkingLevelMap = {}
    for key, entry in value.items():
        if key not in allowed or not (entry is None or isinstance(entry, str)):
            diagnostics.append(ConfigDiagnostic(
                source, "models", "invalid-value",
                f"models.json model {model_id!r} thinkingLevelMap {key!r} must be a string or null",
            ))
            continue
        result[cast(ModelThinkingLevel, key)] = entry
    return result or None


def _headers(
    value: object, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, model_id: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(entry, str) for name, entry in value.items()
    ):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type",
            f"models.json model {model_id!r} headers must be an object of strings",
        ))
        return None
    return {name: entry for name, entry in value.items()}


def _sampling_params(
    value: object, diagnostics: list[ConfigDiagnostic], source: DiagnosticSource, model_id: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        diagnostics.append(ConfigDiagnostic(
            source, "models", "invalid-type",
            f"models.json model {model_id!r} samplingParams must be an object",
        ))
        return None
    return dict(value)
