"""Configuration paths, merge, diagnostics, persistence and credentials."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from omh.llm.auth.types import ApiKeyCredential

from coding_agent import (
    ConfigError,
    FileCredentialStore,
    load_settings,
    merge_settings,
    project_settings_path,
    resolve_agent_dir,
    update_settings,
)


def test_agent_dir_default_and_environment_override(tmp_path: Path) -> None:
    assert resolve_agent_dir(env={}, home=tmp_path) == (tmp_path / ".omh" / "agent").resolve()
    override = tmp_path / "custom-agent"
    assert resolve_agent_dir(env={"OMH_CODING_AGENT_DIR": str(override)}) == override.resolve()


def test_project_settings_path_uses_effective_cwd(tmp_path: Path) -> None:
    assert project_settings_path(tmp_path) == tmp_path / ".omh" / "settings.json"


def test_merge_objects_recursively_and_replace_arrays() -> None:
    base = {"retry": {"enabled": True, "maxRetries": 3}, "defaultTools": ["read", "bash"]}
    override = {"retry": {"maxRetries": 1}, "defaultTools": ["write"]}
    merged = merge_settings(base, override)
    assert merged["retry"] == {"enabled": True, "maxRetries": 1}
    assert merged["defaultTools"] == ["write"]
    # The replaced array source is not revived from the lower layer.
    assert merged["defaultTools"] != ["read", "bash", "write"]


def test_settings_merge_order_and_project_trust(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultModel": "deepseek/deepseek-flash",
        "theme": "dark",
        "retry": {"enabled": True, "maxRetries": 3},
    }))
    cwd = tmp_path / "project"
    (cwd / ".omh").mkdir(parents=True)
    (cwd / ".omh" / "settings.json").write_text(json.dumps({
        "defaultModel": "deepseek/deepseek-v4-pro",
        "theme": "light",
        "retry": {"maxRetries": 1},
    }))

    trusted = load_settings(agent_dir=agent_dir, cwd=cwd, project_trusted=True)
    assert trusted.values["defaultModel"] == "deepseek/deepseek-v4-pro"
    assert trusted.values["theme"] == "light"
    assert trusted.values["retry"] == {"enabled": True, "maxRetries": 1}
    assert trusted.source_of("defaultModel") == "project"

    untrusted = load_settings(agent_dir=agent_dir, cwd=cwd, project_trusted=False)
    assert untrusted.values["defaultModel"] == "deepseek/deepseek-flash"
    assert untrusted.values["theme"] == "dark"
    assert untrusted.source_of("defaultModel") == "global"


def test_explicit_settings_override_trusted_project(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"defaultModel": "deepseek/deepseek-flash"}))
    cwd = tmp_path / "project"
    (cwd / ".omh").mkdir(parents=True)
    (cwd / ".omh" / "settings.json").write_text(json.dumps({"defaultModel": "deepseek/deepseek-v4-pro"}))
    snapshot = load_settings(
        agent_dir=agent_dir, cwd=cwd, project_trusted=True,
        explicit={"defaultModel": "opencode-go/deepseek-v4.1-flash"},
    )
    assert snapshot.values["defaultModel"] == "opencode-go/deepseek-v4.1-flash"


def test_settings_diagnostics_hint_unknown_and_drop_invalid(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultModel": 5,
        "defaultThinkingLevel": "extreme",
        "defaultTools": ["read", "grep"],
        "unknownSetting": True,
        "compaction": {"bogus": 1},
    }))
    snapshot = load_settings(agent_dir=agent_dir, cwd=tmp_path)
    reasons = [diagnostic.reason for diagnostic in snapshot.diagnostics]
    assert "invalid-type" in reasons
    assert "invalid-value" in reasons
    assert "unknown-key" in reasons
    assert "defaultThinkingLevel" not in snapshot.values
    assert "defaultTools" not in snapshot.values
    assert snapshot.values["unknownSetting"] is True


def test_settings_credential_keys_are_never_accepted(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    secrets = {"apiKey": "super-secret", "DEEPSEEK_API_KEY": "super-secret"}
    (agent_dir / "settings.json").write_text(json.dumps(secrets))
    snapshot = load_settings(agent_dir=agent_dir, cwd=tmp_path)
    assert all(diagnostic.reason == "unknown-key" for diagnostic in snapshot.diagnostics)
    assert all("super-secret" not in diagnostic.message for diagnostic in snapshot.diagnostics)


def test_nested_setting_types_are_validated(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({
        "compaction": {"enabled": "yes", "reserveTokens": True},
        "retry": {"maxRetries": "many"},
    }))
    snapshot = load_settings(agent_dir=agent_dir, cwd=tmp_path)
    assert [diagnostic.reason for diagnostic in snapshot.diagnostics].count("invalid-type") == 3

    from coding_agent.config import settings_compaction, settings_retry

    assert settings_compaction(snapshot) is not None  # defaults survive; invalid fields were dropped
    assert settings_compaction(snapshot).enabled is True
    assert settings_retry(snapshot) is not None
    assert settings_retry(snapshot).max_retries == 3


def test_invalid_json_is_diagnosed_without_raising(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text("{not json")
    snapshot = load_settings(agent_dir=agent_dir, cwd=tmp_path)
    assert [diagnostic.reason for diagnostic in snapshot.diagnostics] == ["invalid-json"]


def test_update_settings_writes_only_selected_fields(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "defaultModel": "deepseek/deepseek-flash",
        "retry": {"enabled": True, "maxRetries": 3},
        "unknownSetting": {"kept": True},
    }))
    update_settings(path, {"retry": {"maxRetries": 1}, "theme": "dark"})
    stored = json.loads(path.read_text())
    assert stored["defaultModel"] == "deepseek/deepseek-flash"
    assert stored["retry"] == {"enabled": True, "maxRetries": 1}
    assert stored["theme"] == "dark"
    assert stored["unknownSetting"] == {"kept": True}


def test_update_settings_rejects_corrupt_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError):
        update_settings(path, {"theme": "dark"})


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes are required")
async def test_credential_store_writes_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "agent" / "auth.json"
    store = FileCredentialStore(path)
    credential = await store.modify("deepseek", lambda current: ApiKeyCredential(key="secret"))
    assert credential is not None and credential.key == "secret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    stored = await store.read("deepseek")
    assert stored is not None and stored.key == "secret"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes are required")
async def test_credential_store_tightens_permissive_file(tmp_path: Path) -> None:
    path = tmp_path / "agent" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"deepseek": {"type": "api_key", "key": "secret"}}))
    os.chmod(path, 0o644)
    store = FileCredentialStore(path)
    assert await store.read("deepseek") is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_credential_store_delete_preserves_other_providers(tmp_path: Path) -> None:
    path = tmp_path / "agent" / "auth.json"
    store = FileCredentialStore(path)
    await store.modify("deepseek", lambda current: ApiKeyCredential(key="one"))
    await store.modify("opencode-go", lambda current: ApiKeyCredential(key="two"))
    await store.delete("deepseek")
    assert await store.read("deepseek") is None
    remaining = await store.read("opencode-go")
    assert remaining is not None and remaining.key == "two"


async def test_credential_store_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "agent" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    store = FileCredentialStore(path)
    with pytest.raises(ConfigError):
        await store.read("deepseek")


@pytest.mark.parametrize("env", [{"KEY": 17}, {"KEY": []}])
async def test_credential_store_rejects_non_string_environment_values(tmp_path: Path, env: object) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"deepseek": {"type": "api_key", "key": "temporary", "env": env}}))
    with pytest.raises(ConfigError):
        await FileCredentialStore(path).read("deepseek")
