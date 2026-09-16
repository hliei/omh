from __future__ import annotations

from omh.llm.api.openai_completions import openai_completions_api
from omh.llm.auth.helpers import env_api_key_auth
from omh.llm.auth.types import ProviderAuth
from omh.llm.models import CreatedProvider, CreateProviderOptions, create_provider
from omh.llm.providers.deepseek_models import DEEPSEEK_MODELS


def deepseek_provider() -> CreatedProvider:
    return create_provider(
        CreateProviderOptions(
            id="deepseek",
            name="DeepSeek",
            base_url="https://api.deepseek.com",
            auth=ProviderAuth(api_key=env_api_key_auth("DeepSeek API key", ("DEEPSEEK_API_KEY",))),
            models=DEEPSEEK_MODELS,
            api=openai_completions_api(),
        )
    )
