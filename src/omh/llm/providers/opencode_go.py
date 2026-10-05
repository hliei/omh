from __future__ import annotations

from omh.llm.api.openai_completions import openai_completions_api
from omh.llm.auth.helpers import env_api_key_auth
from omh.llm.auth.types import ProviderAuth
from omh.llm.models import CreatedProvider, CreateProviderOptions, create_provider
from omh.llm.providers.opencode_go_models import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
)


def opencode_go_provider() -> CreatedProvider:
    """OpenCode Go's fixed Chat Completions route with API-key auth."""

    return create_provider(
        CreateProviderOptions(
            id="opencode-go",
            name="OpenCode Go",
            base_url=OPENCODE_GO_BASE_URL,
            auth=ProviderAuth(api_key=env_api_key_auth("OpenCode API key", ("OPENCODE_API_KEY",))),
            models=OPENCODE_GO_MODELS,
            api=openai_completions_api(),
        )
    )
