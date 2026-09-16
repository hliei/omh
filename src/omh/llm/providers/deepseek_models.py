from __future__ import annotations

from omh.llm.types import Model, ModelCost, OpenAICompletionsCompat, ThinkingLevelMap

DEEPSEEK_V4_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}

DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    **DEEPSEEK_V4_THINKING_LEVEL_MAP,
    "low": "low",
}

_DEEPSEEK_COMPAT = OpenAICompletionsCompat(
    requires_reasoning_content_on_assistant_messages=True,
    thinking_format="deepseek",
)

DEEPSEEK_MODELS: tuple[Model, ...] = (
    Model(
        id="deepseek-flash",
        name="DeepSeek V4.1 Flash",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        reasoning=True,
        thinking_level_map=DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP,
        input=("text", "image"),
        cost=ModelCost(input=0.3, output=1.2, cache_read=0.006, cache_write=0),
        context_window=1_000_000,
        max_tokens=384_000,
        compat=_DEEPSEEK_COMPAT,
    ),
    Model(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        api="openai-completions",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        reasoning=True,
        thinking_level_map=DEEPSEEK_V4_THINKING_LEVEL_MAP,
        input=("text",),
        cost=ModelCost(input=1.32, output=3.96, cache_read=0.044, cache_write=0),
        context_window=1_000_000,
        max_tokens=384_000,
        compat=_DEEPSEEK_COMPAT,
    ),
)
