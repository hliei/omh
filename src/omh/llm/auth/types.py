from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from omh.llm.types import AbortSignal, ProviderEnv, ProviderHeaders


@dataclass(slots=True)
class ModelAuth:
    api_key: str | None = None
    headers: ProviderHeaders | None = None
    base_url: str | None = None


@dataclass(slots=True)
class ApiKeyCredential:
    type: Literal["api_key"] = "api_key"
    key: str | None = None
    env: ProviderEnv | None = None


Credential = ApiKeyCredential


@dataclass(frozen=True, slots=True)
class CredentialInfo:
    provider_id: str
    type: Literal["api_key"]


@dataclass(frozen=True, slots=True)
class AuthOperationOptions:
    signal: AbortSignal | None = None


class CredentialStore(Protocol):
    def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[Credential | None]: ...

    def list(self, options: AuthOperationOptions | None = None) -> Awaitable[Sequence[CredentialInfo]]: ...

    def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Awaitable[Credential | None]: ...

    def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[None]: ...


class AuthContext(Protocol):
    def env(self, name: str) -> Awaitable[str | None]: ...

    def file_exists(self, path: str) -> Awaitable[bool]: ...


@dataclass(slots=True)
class AuthResult:
    auth: ModelAuth
    env: ProviderEnv | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class AuthCheck:
    type: Literal["api_key"]
    source: str | None = None


AuthType = Literal["api_key"]


@dataclass(frozen=True, slots=True)
class AuthPrompt:
    type: Literal["text", "secret", "select"]
    message: str
    placeholder: str | None = None
    options: tuple[tuple[str, str], ...] | None = None


class AuthInteraction(Protocol):
    signal: AbortSignal | None

    def prompt(self, prompt: AuthPrompt) -> Awaitable[str]: ...


@dataclass(slots=True)
class ApiKeyAuth:
    name: str
    resolve: Callable[..., Awaitable[AuthResult | None]]
    login: Callable[[AuthInteraction], Awaitable[ApiKeyCredential]] | None = None
    check: Callable[..., Awaitable[AuthCheck | None]] | None = None


@dataclass(slots=True)
class ProviderAuth:
    api_key: ApiKeyAuth | None = None
