from __future__ import annotations

from collections.abc import Sequence

from omh.llm.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    ModelAuth,
)
from omh.llm.types import AbortSignal


def env_api_key_auth(name: str, env_vars: Sequence[str]) -> ApiKeyAuth:
    async def login(interaction: AuthInteraction) -> ApiKeyCredential:
        if interaction.signal is not None:
            interaction.signal.throw_if_aborted()
        key = await interaction.prompt(AuthPrompt(type="secret", message=f"Enter {name}"))
        if interaction.signal is not None:
            interaction.signal.throw_if_aborted()
        return ApiKeyCredential(key=key)

    async def resolve(*, ctx: AuthContext, credential: ApiKeyCredential | None, signal: AbortSignal) -> AuthResult | None:
        signal.throw_if_aborted()
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
        for env_var in env_vars:
            value = await ctx.env(env_var)
            signal.throw_if_aborted()
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name=name, login=login, resolve=resolve)
