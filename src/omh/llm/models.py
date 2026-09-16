from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from omh.llm.auth.context import default_provider_auth_context
from omh.llm.auth.credential_store import InMemoryCredentialStore
from omh.llm.auth.resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
from omh.llm.auth.types import (
    AuthCheck,
    AuthContext,
    AuthOperationOptions,
    AuthResult,
    CredentialStore,
    ProviderAuth,
)
from omh.llm.types import (
    Api,
    AssistantMessage,
    Context,
    Model,
    ModelCostRates,
    ModelThinkingLevel,
    OpenAICompletionsOptions,
    ProviderHeaders,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
    UsageCost,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream
from omh.llm.utils.lazy import lazy_stream


class ProviderStreams(Protocol):
    def stream(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...


class Provider(Protocol):
    id: str
    name: str
    base_url: str | None
    headers: ProviderHeaders | None
    auth: ProviderAuth

    def get_models(self) -> Sequence[Model]: ...

    def stream(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...


class Models(Protocol):
    def get_providers(self) -> Sequence[Provider]: ...

    def get_provider(self, provider_id: str) -> Provider | None: ...

    def get_models(self, provider: str | None = None) -> Sequence[Model]: ...

    def get_model(self, provider: str, model_id: str) -> Model | None: ...

    def stream(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def complete(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> Awaitable[AssistantMessage]: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> Awaitable[AssistantMessage]: ...


class MutableModels(Models, Protocol):
    def set_provider(self, provider: Provider) -> None: ...

    def delete_provider(self, provider_id: str) -> None: ...

    def clear_providers(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CreateModelsOptions:
    credentials: CredentialStore | None = None
    auth_context: AuthContext | None = None


def _merge_headers(
    base: ProviderHeaders | None,
    override: ProviderHeaders | None,
) -> ProviderHeaders | None:
    if not base and not override:
        return None
    merged = dict(base or {})
    for name, value in (override or {}).items():
        lower_name = name.lower()
        for existing in list(merged):
            if existing.lower() == lower_name:
                del merged[existing]
        merged[name] = value
    return merged


class ModelsImpl:
    def __init__(self, options: CreateModelsOptions | None = None) -> None:
        self._providers: dict[str, Provider] = {}
        self._credentials = options.credentials if options and options.credentials else InMemoryCredentialStore()
        self._auth_context = (
            options.auth_context if options and options.auth_context else default_provider_auth_context()
        )

    def set_provider(self, provider: Provider) -> None:
        self._providers[provider.id] = provider

    def delete_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_providers(self) -> Sequence[Provider]:
        return tuple(self._providers.values())

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def get_models(self, provider: str | None = None) -> Sequence[Model]:
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None:
                return ()
            try:
                return tuple(entry.get_models())
            except Exception:
                return ()
        models: list[Model] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except Exception:
                continue
        return tuple(models)

    def get_model(self, provider: str, model_id: str) -> Model | None:
        for model in self.get_models(provider):
            if model.id == model_id:
                return model
        return None

    def _require_provider(self, model: Model) -> Provider:
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        result = await resolve_provider_auth(provider, self._credentials, self._auth_context, overrides)
        if result is None or isinstance(provider_or_model, str) or provider_or_model.headers is None:
            return result if isinstance(result, AuthResult) or result is None else None
        return AuthResult(
            auth=replace(
                result.auth,
                headers=_merge_headers(result.auth.headers, dict(provider_or_model.headers)),
            ),
            env=result.env,
            source=result.source,
        )

    async def check_auth(self, provider_id: str, options: AuthOperationOptions | None = None) -> AuthCheck | None:
        from omh.llm.utils.abort import operation_signal, race_with_abort_signal

        signal = operation_signal(options.signal if options else None)

        async def check() -> AuthCheck | None:
            signal.throw_if_aborted()
            provider = self._providers.get(provider_id)
            if provider is None:
                return None
            resolution = await resolve_provider_auth(
                provider,
                self._credentials,
                self._auth_context,
                AuthResolutionOverrides(signal=signal),
            )
            if resolution is None:
                return None
            return AuthCheck(type="api_key", source=resolution.source)

        return await race_with_abort_signal(check(), signal)

    async def _apply_auth[T: StreamOptions](self, model: Model, options: T | None) -> tuple[Model, T]:
        self._require_provider(model)
        resolution = await self.get_auth(
            model,
            AuthResolutionOverrides(
                api_key=options.api_key if options else None,
                env=options.env if options else None,
                signal=options.signal if options else None,
            ),
        )
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        api_key = (options.api_key if options else None) or resolution.auth.api_key
        headers = _merge_headers(resolution.auth.headers, options.headers if options else None)
        env = None
        if resolution.env or (options and options.env):
            env = {**(resolution.env or {}), **((options.env if options else None) or {})}
        request_model = replace(model, base_url=resolution.auth.base_url) if resolution.auth.base_url else model
        request_options = replace(options) if options is not None else StreamOptions()
        request_options.api_key = api_key
        request_options.headers = headers
        request_options.env = env
        return request_model, request_options  # type: ignore[return-value]

    def stream(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def setup() -> AssistantMessageEventStream:
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(model, options)
            return provider.stream(request_model, context, request_options)

        return lazy_stream(model, setup)

    async def complete(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessage:
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def setup() -> AssistantMessageEventStream:
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(model, options or SimpleStreamOptions())
            return provider.stream_simple(request_model, context, request_options)

        return lazy_stream(model, setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        return await self.stream_simple(model, context, options).result()


def create_models(options: CreateModelsOptions | None = None) -> ModelsImpl:
    return ModelsImpl(options)


@dataclass(frozen=True, slots=True)
class CreateProviderOptions:
    id: str
    auth: ProviderAuth
    models: Sequence[Model]
    api: ProviderStreams
    name: str | None = None
    base_url: str | None = None
    headers: ProviderHeaders | None = None


class CreatedProvider:
    def __init__(self, options: CreateProviderOptions) -> None:
        self.id = options.id
        self.name = options.name or options.id
        self.base_url = options.base_url
        self.headers = options.headers
        self.auth = options.auth
        self._models = tuple(options.models)
        self._api = options.api

    def get_models(self) -> Sequence[Model]:
        return self._models

    def stream(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._api.stream(model, context, options)

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._api.stream_simple(model, context, options)


def create_provider(options: CreateProviderOptions) -> CreatedProvider:
    return CreatedProvider(options)


def calculate_cost(model: Model, usage: Usage) -> UsageCost:
    rates: ModelCostRates = model.cost
    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = (rates.input / 1_000_000) * usage.input
    usage.cost.output = (rates.output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


EXTENDED_THINKING_LEVELS: tuple[ModelThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    if not model.reasoning:
        return ["off"]
    available: list[ModelThinkingLevel] = []
    for level in EXTENDED_THINKING_LEVELS:
        mapped = None if model.thinking_level_map is None else model.thinking_level_map.get(level)
        if mapped is None and model.thinking_level_map is not None and level in model.thinking_level_map:
            continue
        if level in {"xhigh", "max"} and (model.thinking_level_map is None or level not in model.thinking_level_map):
            continue
        available.append(level)
    return available


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    try:
        requested_index = EXTENDED_THINKING_LEVELS.index(level)
    except ValueError:
        return available[0] if available else "off"
    for candidate in EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    for candidate in reversed(EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def models_are_equal(left: Model | None, right: Model | None) -> bool:
    if left is None or right is None:
        return False
    return left.id == right.id and left.provider == right.provider


def has_api(model: Model, api: Api) -> bool:
    return model.api == api
