"""Common host selection, credentials and actual-request behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from omh.agent import (
    AgentHistory,
    MessageHistoryEntry,
    ThinkingLevelChangeHistoryEntry,
)
from omh.llm.auth.types import ApiKeyCredential
from omh.llm.types import AssistantMessage, TextContent, empty_usage
from support import RecordingFetch

from coding_agent import (
    AgentSessionRuntime,
    CodingAgentHost,
    FileCredentialStore,
    decode_history,
    encode_history,
)

STAMP = datetime(2026, 10, 5, tzinfo=UTC)


def write_history(
    path: Path, *, cwd: Path, provider: str = "deepseek", model_id: str = "deepseek-flash",
    thinking: str = "high",
) -> None:
    history = AgentHistory(conversation_id="conversation", created_at=STAMP, leaf_id="thinking", entries=(
        MessageHistoryEntry(
            id="assistant", parent_id=None, timestamp=STAMP,
            message=AssistantMessage(
                api="openai-completions", provider=provider, model=model_id, timestamp=1,
                content=[TextContent(text="earlier response")], usage=empty_usage(), stop_reason="stop",
            ),
        ),
        ThinkingLevelChangeHistoryEntry(id="thinking", parent_id="assistant", timestamp=STAMP, thinking_level=thinking),
    ))
    path.write_text(encode_history(history, cwd=str(cwd), display_name="saved"))


def write_settings(agent_dir: Path, values: object) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text(json.dumps(values))


def test_new_session_uses_product_defaults(tmp_path: Path) -> None:
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    selection = host.select_new()
    assert selection.ready
    assert selection.model is not None
    assert (selection.model.provider, selection.model.id) == ("opencode-go", "deepseek-v4.1-flash")
    assert selection.thinking_level == "high"
    assert selection.tools == ("read", "bash", "edit", "write")
    assert selection.cwd == tmp_path.resolve()
    assert selection.diagnostics == ()


def test_explicit_selection_wins_and_invalid_input_is_rejected(tmp_path: Path) -> None:
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    explicit = host.select_new(model="deepseek/deepseek-flash", thinking="off", tools=("read",))
    assert explicit.ready
    assert explicit.model is not None and explicit.model.id == "deepseek-flash"
    assert explicit.thinking_level == "off"
    assert explicit.tools == ("read",)

    unknown = host.select_new(model="deepseek/does-not-exist")
    assert not unknown.ready
    assert any("Unknown model" in diagnostic.message for diagnostic in unknown.diagnostics)

    bad_thinking = host.select_new(model="deepseek/deepseek-v4-pro", thinking="xhigh")
    assert not bad_thinking.ready
    assert any("not supported" in diagnostic.message for diagnostic in bad_thinking.diagnostics)


def test_trusted_project_settings_replace_global_arrays(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"defaultModel": "deepseek/deepseek-flash", "defaultTools": ["read"]})
    cwd = tmp_path / "project"
    (cwd / ".omh").mkdir(parents=True)
    (cwd / ".omh" / "settings.json").write_text(json.dumps({
        "defaultModel": "deepseek/deepseek-v4-pro",
        "defaultTools": ["bash", "edit"],
    }))

    trusted = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, project_trusted=True)
    trusted_selection = trusted.select_new()
    assert trusted_selection.model is not None and trusted_selection.model.id == "deepseek-v4-pro"
    assert trusted_selection.tools == ("bash", "edit")

    untrusted = CodingAgentHost(startup_dir=cwd, agent_dir=agent_dir, project_trusted=False)
    untrusted_selection = untrusted.select_new()
    assert untrusted_selection.model is not None and untrusted_selection.model.id == "deepseek-flash"
    assert untrusted_selection.tools == ("read",)


def test_explicit_cli_settings_override_trusted_project(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"defaultModel": "deepseek/deepseek-flash"})
    cwd = tmp_path / "project"
    (cwd / ".omh").mkdir(parents=True)
    (cwd / ".omh" / "settings.json").write_text(json.dumps({"defaultModel": "deepseek/deepseek-v4-pro"}))
    host = CodingAgentHost(
        startup_dir=cwd, agent_dir=agent_dir, project_trusted=True,
        explicit_settings={"defaultModel": "opencode-go/deepseek-v4.1-flash"},
    )
    selection = host.select_new()
    assert selection.model is not None and selection.model.id == "deepseek-v4.1-flash"


def test_configured_tools_can_disable_every_tool(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"defaultTools": []})
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    assert host.select_new().tools == ()


def test_invalid_configured_default_model_is_not_ready(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"defaultModel": "deepseek/does-not-exist"})
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    selection = host.select_new()
    assert not selection.ready
    assert any("not available" in diagnostic.message for diagnostic in selection.diagnostics)


def test_cwd_resolution_prefers_explicit_then_startup(tmp_path: Path) -> None:
    startup = tmp_path / "start"
    startup.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    host = CodingAgentHost(startup_dir=startup, agent_dir=tmp_path / "agent")
    assert host.select_new().cwd == startup.resolve()
    assert host.select_new(cwd="../other").cwd == other.resolve()

    missing = host.select_new(cwd="../missing")
    assert not missing.ready
    assert any("does not exist" in diagnostic.message for diagnostic in missing.diagnostics)


def test_project_settings_follow_the_effective_cwd(tmp_path: Path) -> None:
    startup = tmp_path / "start"
    other = tmp_path / "other"
    for directory, model_id in ((startup, "deepseek-flash"), (other, "deepseek-v4-pro")):
        (directory / ".omh").mkdir(parents=True)
        (directory / ".omh" / "settings.json").write_text(json.dumps({"defaultModel": f"deepseek/{model_id}"}))
    host = CodingAgentHost(startup_dir=startup, agent_dir=tmp_path / "agent", project_trusted=True)
    assert host.select_new().model is not None
    assert host.select_new().model.id == "deepseek-flash"
    explicit = host.select_new(cwd="../other")
    assert explicit.cwd == other.resolve()
    assert explicit.model is not None and explicit.model.id == "deepseek-v4-pro"


def test_selection_never_writes_settings(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"defaultModel": "deepseek/deepseek-flash"})
    before = (agent_dir / "settings.json").read_text()
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    host.select_new(model="opencode-go/deepseek-v4.1-flash", thinking="max")
    host.select_new()
    assert (agent_dir / "settings.json").read_text() == before


def test_open_restores_saved_cwd_model_and_thinking(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "history.jsonl"
    write_history(path, cwd=project, provider="deepseek", model_id="deepseek-v4-pro", thinking="max")
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    selection = host.select_open(path)
    assert selection.ready
    assert selection.cwd == project.resolve()
    assert selection.model is not None and selection.model.id == "deepseek-v4-pro"
    assert selection.thinking_level == "max"

    overridden = host.select_open(path, model="deepseek/deepseek-flash", cwd=".")
    assert overridden.model is not None and overridden.model.id == "deepseek-flash"
    assert overridden.cwd == tmp_path.resolve()


def test_open_missing_history_cwd_is_not_ready(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    write_history(path, cwd=tmp_path / "gone")
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    selection = host.select_open(path)
    assert not selection.ready
    assert any("replacement" in diagnostic.message for diagnostic in selection.diagnostics)


def test_open_unavailable_history_model_does_not_fall_back(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    write_history(path, cwd=tmp_path, provider="test", model_id="saved-model")
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    selection = host.select_open(path)
    assert not selection.ready
    assert selection.model is None
    assert any("not available" in diagnostic.message for diagnostic in selection.diagnostics)

    explicit = host.select_open(path, model="deepseek/deepseek-flash")
    assert explicit.ready
    assert explicit.model is not None and explicit.model.id == "deepseek-flash"


def test_open_reports_visible_thinking_clamp(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    write_history(path, cwd=tmp_path, provider="deepseek", model_id="deepseek-v4-pro", thinking="low")
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    selection = host.select_open(path)
    assert selection.ready
    assert selection.thinking_level == "high"
    clamp = next(d for d in selection.diagnostics if "using" in d.message)
    assert "low" in clamp.message and "high" in clamp.message
    assert clamp.source == "history"


def test_configured_thinking_clamp_reports_settings_source(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {
        "defaultModel": "deepseek/deepseek-v4-pro",
        "defaultThinkingLevel": "low",
    })
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    selection = host.select_new()
    assert selection.ready
    assert selection.thinking_level == "high"
    clamp = next(d for d in selection.diagnostics if "using" in d.message)
    assert clamp.source == "global"


async def test_key_source_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent_dir = tmp_path / "agent"
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    model = host.select_new().model
    assert model is not None
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    assert await host.key_source(model) is None

    monkeypatch.setenv("OPENCODE_API_KEY", "from-env")
    assert await host.key_source(model) == "OPENCODE_API_KEY"

    store = FileCredentialStore(agent_dir / "auth.json")
    await store.modify("opencode-go", lambda current: ApiKeyCredential(key="stored"))
    assert await host.key_source(model) == "stored credential"

    override = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir, api_key="temporary")
    assert await override.key_source(model) == "cli"


async def test_readiness_reports_missing_key_with_repair_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent_dir = tmp_path / "agent"
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    selection = host.select_new()
    assert selection.ready
    diagnostics = await host.readiness(selection)
    assert any("OPENCODE_API_KEY" in diagnostic.message for diagnostic in diagnostics)

    store = FileCredentialStore(agent_dir / "auth.json")
    await store.modify("opencode-go", lambda current: ApiKeyCredential(key="stored"))
    assert await host.readiness(selection) == ()


async def test_readiness_keeps_unresolved_selection_diagnostics(tmp_path: Path) -> None:
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent")
    selection = host.select_new(model="deepseek/does-not-exist")
    diagnostics = await host.readiness(selection)
    assert any("Unknown model" in diagnostic.message for diagnostic in diagnostics)


async def test_actual_request_uses_stored_credential_and_selected_model(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    store = FileCredentialStore(agent_dir / "auth.json")
    await store.modify("opencode-go", lambda current: ApiKeyCredential(key="stored-secret"))
    fetch = RecordingFetch()
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir, fetch=fetch)
    selection = host.select_new()
    assert selection.ready
    runtime = AgentSessionRuntime(host.build_options(selection))
    session = await runtime.new_session()

    # Configuring and selecting never sends a verification request.
    assert fetch.requests == []
    await session.prompt("hello")
    assert len(fetch.requests) == 1
    request = fetch.requests[0]
    assert request.url == "https://opencode.ai/zen/go/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer stored-secret"
    assert request.json_body["model"] == "deepseek-v4.1-flash"


async def test_credentials_never_reach_saved_history_or_export(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    store = FileCredentialStore(agent_dir / "auth.json")
    await store.modify("opencode-go", lambda current: ApiKeyCredential(key="stored-secret"))
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir, fetch=RecordingFetch())
    selection = host.select_new()
    runtime = AgentSessionRuntime(host.build_options(selection))
    session = await runtime.new_session()
    await session.prompt("hello")

    saved = tmp_path / "saved.jsonl"
    await session.save(saved)
    exported = await session.export()
    assert "stored-secret" not in saved.read_text()
    assert "stored-secret" not in exported


async def test_build_options_maps_compaction_and_retry_settings(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {
        "compaction": {"enabled": False, "reserveTokens": 2048},
        "retry": {"maxRetries": 1},
    })
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    options = host.build_options(host.select_new())
    assert options.agent_options.compaction is not None
    assert options.agent_options.compaction.enabled is False
    assert options.agent_options.compaction.reserve_tokens == 2048
    assert options.agent_options.retry is not None
    assert options.agent_options.retry.max_retries == 1


def test_build_options_carries_configured_resource_paths(tmp_path: Path) -> None:
    skill = tmp_path / "skills"
    template = tmp_path / "templates"
    skill.mkdir()
    template.mkdir()
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"skills": [str(skill)], "prompts": [str(template)]})
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=agent_dir)
    options = host.build_options(host.select_new())
    assert [str(source.path) for source in options.skill_sources] == [str(skill)]
    assert [str(source.path) for source in options.template_sources] == [str(template)]


async def test_open_through_runtime_keeps_history_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / "history.jsonl"
    write_history(path, cwd=project)
    original = decode_history(path.read_text()).history
    host = CodingAgentHost(startup_dir=tmp_path, agent_dir=tmp_path / "agent", fetch=RecordingFetch())
    selection = host.select_open(path)
    runtime = AgentSessionRuntime(host.build_options(selection))
    session = await runtime.open_session(path)
    assert session.agent.history == original
    assert session.cwd == project.resolve()
