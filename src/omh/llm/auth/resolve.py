from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass

from omh.llm.auth.types import (
    ApiKeyCredential,
    AuthContext,
    AuthResult,
    Credential,
    CredentialStore,
    ProviderAuth,
)
from omh.llm.types import AbortSignal, ProviderEnv
from omh.llm.utils.abort import operation_signal, race_with_abort_signal


class ModelsError(Exception):
    def __init__(self, code: str, message: str, cause: object | None = None) -> None:
        detail = _cause_detail(cause)
        super().__init__(f"{message}: {detail}" if detail and detail not in message else message)
        self.code = code
        self.cause = cause


def _cause_detail(cause: object | None) -> str:
    if cause is None:
        return ""
    return str(cause).strip()


@dataclass(frozen=True, slots=True)
class AuthResolutionOverrides:
    api_key: str | None = None
    env: ProviderEnv | None = None
    signal: AbortSignal | None = None


def resolve_provider_auth(
    provider: object,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> Awaitable[AuthResult | None]:
    signal = operation_signal(overrides.signal if overrides else None)
    return race_with_abort_signal(
        _resolve(provider, credentials, auth_context, overrides, signal),
        signal,
    )


async def _resolve(
    provider: object,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None,
    signal: AbortSignal,
) -> AuthResult | None:
    signal.throw_if_aborted()
    request_context = _overlay_env(auth_context, overrides.env) if overrides and overrides.env else auth_context
    auth = getattr(provider, "auth")
    api_key_auth = auth.api_key if isinstance(auth, ProviderAuth) else getattr(auth, "api_key", None)
    provider_id = getattr(provider, "id")
    if overrides is not None and overrides.api_key is not None and api_key_auth is not None:
        return await api_key_auth.resolve(
            ctx=request_context,
            credential=ApiKeyCredential(key=overrides.api_key, env=overrides.env),
            signal=signal,
        )
    stored = await _read_credential(credentials, provider_id, signal)
    if stored is not None:
        if stored.type == "api_key" and api_key_auth is not None:
            credential = stored
            if overrides and overrides.env:
                credential = ApiKeyCredential(key=stored.key, env={**(stored.env or {}), **overrides.env})
            return await api_key_auth.resolve(ctx=request_context, credential=credential, signal=signal)
        return None
    if api_key_auth is None:
        return None
    return await api_key_auth.resolve(ctx=request_context, credential=None, signal=signal)


def _overlay_env(base: AuthContext, env: ProviderEnv) -> AuthContext:
    class Overlay:
        async def env(self, name: str) -> str | None:
            return env.get(name) or await base.env(name)

        async def file_exists(self, path: str) -> bool:
            return await base.file_exists(path)

    return Overlay()


async def _read_credential(credentials: CredentialStore, provider_id: str, signal: AbortSignal) -> Credential | None:
    try:
        from omh.llm.auth.types import AuthOperationOptions

        return await credentials.read(provider_id, AuthOperationOptions(signal=signal))
    except Exception as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error
