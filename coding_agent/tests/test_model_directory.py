"""Model directory metadata, source dates and explicit ``models.json`` overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from coding_agent.model_directory import (
    SOURCE_BUILTIN,
    SOURCE_USER,
    ModelDirectory,
)


def write_models(agent_dir: Path, document: object) -> Path:
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "models.json"
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    return path


def test_builtin_catalog_lists_both_supported_providers_with_metadata() -> None:
    directory = ModelDirectory()
    listings = {entry.model.id: entry for entry in directory.listings()}
    assert "deepseek-flash" in listings
    assert "deepseek-v4-pro" in listings
    assert "deepseek-v4.1-flash" in listings
    assert directory.provider_ids == ("deepseek", "opencode-go")

    flash = listings["deepseek-v4.1-flash"]
    assert flash.provider == "opencode-go"
    assert flash.model.api == "openai-completions"
    assert flash.model.input == ("text", "image")
    assert flash.thinking_levels == ("low", "high", "max")
    assert flash.model.context_window > 0
    assert flash.model.max_tokens > 0
    assert flash.model.cost.input > 0
    assert flash.source == SOURCE_BUILTIN
    assert flash.source_date


def test_models_json_override_marks_user_source_and_preserves_identity(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "opencode-go": {
                "modelOverrides": {
                    "deepseek-v4.1-flash": {"cost": {"input": 9.5, "output": 20.0}},
                },
            },
        },
    })
    os.utime(agent_dir / "models.json", (1_577_923_200, 1_577_923_200))  # 2020-01-02 UTC
    directory = ModelDirectory(agent_dir=agent_dir)
    listing = next(entry for entry in directory.listings() if entry.model.id == "deepseek-v4.1-flash")
    assert listing.model.cost.input == 9.5
    assert listing.model.cost.output == 20.0
    # The override merges; fields the override omits keep the built-in value.
    assert listing.model.cost.cache_read == 0
    assert listing.source == SOURCE_USER
    # User metadata carries its own configuration date, not the built-in catalog date.
    assert listing.source_date == "2020-01-02"
    assert directory.diagnostics == ()


def test_models_json_adds_a_model_through_the_supported_protocol(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "deepseek": {
                "models": [{
                    "id": "deepseek-experiment",
                    "name": "DeepSeek Experiment",
                    "api": "openai-completions",
                    "reasoning": True,
                    "input": ["text"],
                    "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 128000,
                    "maxTokens": 8192,
                }],
            },
        },
    })
    directory = ModelDirectory(agent_dir=agent_dir)
    assert directory.diagnostics == ()
    matches = directory.find_models("deepseek", "deepseek-experiment")
    assert len(matches) == 1
    assert matches[0].provider == "deepseek"
    listing = next(entry for entry in directory.listings() if entry.model.id == "deepseek-experiment")
    assert listing.source == SOURCE_USER


def test_models_json_duplicate_definition_is_diagnosed(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "deepseek": {
                "models": [{
                    "id": "deepseek-flash",
                    "api": "openai-completions",
                    "reasoning": True,
                    "input": ["text"],
                    "cost": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                }],
            },
        },
    })
    directory = ModelDirectory(agent_dir=agent_dir)
    assert any("duplicates" in diagnostic.message for diagnostic in directory.diagnostics)
    matches = directory.find_models("deepseek", "deepseek-flash")
    assert len(matches) == 1 and matches[0].cost.input == 0.3


def test_models_json_override_field_types_are_diagnosed(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "deepseek": {
                "bogusProviderKey": True,
                "modelOverrides": {
                    "deepseek-flash": {"headers": "not-an-object", "thinkingLevelMap": {"high": 5}},
                },
            },
        },
    })
    directory = ModelDirectory(agent_dir=agent_dir)
    reasons = [diagnostic.reason for diagnostic in directory.diagnostics]
    assert "unknown-key" in reasons
    assert "invalid-type" in reasons
    assert "invalid-value" in reasons
    listing = next(entry for entry in directory.listings() if entry.model.id == "deepseek-flash")
    assert listing.model.headers is None


def test_models_json_missing_fields_are_diagnosed_not_inferred(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "deepseek": {"models": [{"id": "deepseek-experiment", "api": "openai-completions"}]},
        },
    })
    directory = ModelDirectory(agent_dir=agent_dir)
    assert any(diagnostic.reason == "invalid-schema" for diagnostic in directory.diagnostics)
    assert directory.find_models("deepseek", "deepseek-experiment") == ()


def test_models_json_rejects_unsupported_provider_and_protocol(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "openai": {"models": [{"id": "gpt", "api": "openai-completions"}]},
            "deepseek": {
                "models": [{
                    "id": "deepseek-responses",
                    "api": "openai-responses",
                    "reasoning": True,
                    "input": ["text"],
                    "cost": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                }],
            },
        },
    })
    directory = ModelDirectory(agent_dir=agent_dir)
    reasons = [diagnostic.reason for diagnostic in directory.diagnostics]
    assert reasons.count("invalid-value") >= 2
    assert directory.find_models("deepseek", "deepseek-responses") == ()
    assert "openai" not in directory.provider_ids


def test_models_json_invalid_json_is_diagnosed(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, "{not json")
    directory = ModelDirectory(agent_dir=agent_dir)
    assert [diagnostic.reason for diagnostic in directory.diagnostics] == ["invalid-json"]


def test_models_json_unknown_keys_are_hinted(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {
        "providers": {
            "deepseek": {"modelOverrides": {"deepseek-flash": {"mystery": 1}}},
        },
    })
    directory = ModelDirectory(agent_dir=agent_dir)
    assert any(diagnostic.reason == "unknown-key" for diagnostic in directory.diagnostics)


def test_directory_never_contacts_a_provider(tmp_path: Path) -> None:
    # A directory lookup only reads local files; there is no HTTP boundary to call.
    agent_dir = tmp_path / "agent"
    directory = ModelDirectory(agent_dir=agent_dir)
    assert directory.listings()
    assert not agent_dir.exists()


@pytest.mark.parametrize("value", [17, None, {"high": 5}, {"bogus": "max"}])
def test_bad_thinking_override_preserves_builtin_levels(tmp_path: Path, value: object) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {"providers": {"opencode-go": {"modelOverrides": {
        "deepseek-v4.1-flash": {"thinkingLevelMap": value},
    }}}})
    directory = ModelDirectory(agent_dir=agent_dir)
    listing = next(entry for entry in directory.listings() if entry.model.id == "deepseek-v4.1-flash")
    assert listing.thinking_levels == ("low", "high", "max")
    assert any(diagnostic.blocking for diagnostic in directory.diagnostics)


def test_empty_thinking_override_is_fixed_without_adjustable_levels(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {"providers": {"opencode-go": {"modelOverrides": {
        "deepseek-v4.1-flash": {"thinkingLevelMap": {}},
    }}}})
    directory = ModelDirectory(agent_dir=agent_dir)
    listing = next(entry for entry in directory.listings() if entry.model.id == "deepseek-v4.1-flash")
    assert listing.thinking_levels == ()
    assert directory.diagnostics == ()


def test_partial_thinking_map_only_exposes_declared_levels(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_models(agent_dir, {"providers": {"opencode-go": {"modelOverrides": {
        "deepseek-v4.1-flash": {"thinkingLevelMap": {"max": "max"}},
    }}}})
    directory = ModelDirectory(agent_dir=agent_dir)
    listing = next(entry for entry in directory.listings() if entry.model.id == "deepseek-v4.1-flash")
    assert listing.thinking_levels == ("max",)
    assert directory.diagnostics == ()


@pytest.mark.parametrize("compat", [{"thinkingFormat": "other"}, {"maxTokensField": []}, {"supportsReasoningEffort": 17}])
def test_bad_compat_override_keeps_original_contract(tmp_path: Path, compat: object) -> None:
    agent_dir = tmp_path / "agent"
    original = ModelDirectory().find_models("opencode-go", "deepseek-v4.1-flash")[0]
    write_models(agent_dir, {"providers": {"opencode-go": {"modelOverrides": {
        "deepseek-v4.1-flash": {"compat": compat},
    }}}})
    directory = ModelDirectory(agent_dir=agent_dir)
    actual = directory.find_models("opencode-go", "deepseek-v4.1-flash")[0]
    assert actual.compat == original.compat
    assert any(diagnostic.blocking for diagnostic in directory.diagnostics)
