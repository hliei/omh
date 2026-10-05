"""Built-in model directory for read-only listing and selection validation.

The directory is metadata only. It registers the providers whose runtime
support ships with this product so ``--list-models`` and explicit
``--provider``/``--model`` validation stay offline and never resolve
credentials. Provider routing and capability verification belong to the
provider deliveries; a provider is listed only after it is registered here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omh.llm import (
    create_models,
    deepseek_provider,
    get_supported_thinking_levels,
    opencode_go_provider,
)
from omh.llm.models import Models, Provider
from omh.llm.types import Model

#: Marks model metadata that ships with the product rather than user config.
SOURCE_BUILTIN = "builtin"

ProviderFactory = Callable[[], Provider]

#: Providers whose models this product can actually run. Registering one here
#: makes its models appear in ``--list-models`` and valid for exact selection.
_BUILTIN_PROVIDERS: tuple[ProviderFactory, ...] = (deepseek_provider, opencode_go_provider)


def builtin_models() -> Models:
    """Assemble the registered providers without reading credentials."""
    models = create_models()
    for factory in _BUILTIN_PROVIDERS:
        models.set_provider(factory())
    return models


@dataclass(frozen=True, slots=True)
class ModelListing:
    """One provider/model entry with the effective options this product exposes."""

    provider: str
    model: Model
    thinking_levels: tuple[str, ...]
    source: str = SOURCE_BUILTIN


class ModelDirectory:
    """Read-only view over the registered providers and their models."""

    def __init__(self, models: Models | None = None) -> None:
        self._models = models if models is not None else builtin_models()

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(provider.id for provider in self._models.get_providers())

    def listings(self, search: str | None = None) -> tuple[ModelListing, ...]:
        """Return every model, ordered by provider and ID, optionally searched."""
        entries = [
            ModelListing(
                provider=model.provider,
                model=model,
                thinking_levels=self.thinking_levels(model),
            )
            for model in self._models.get_models()
        ]
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
