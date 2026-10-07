"""Global and project configuration, credentials and agent-directory paths.

This module owns the installed product's user-visible file interface:

- the global agent directory (``~/.omh/agent``, overridable with
  ``OMH_CODING_AGENT_DIR``) holds ``settings.json``, ``auth.json``,
  ``models.json`` and the session root;
- a project's ``.omh/settings.json`` supplies a trusted project layer.

Settings merge recursively: explicit CLI values override trusted project
settings, which override global settings; nested objects merge and arrays are
replaced. Credentials and the model directory are global-only. Nothing here
resolves a model or sends a request; the host decides selection separately.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import stat
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from omh.agent import CompactionSettings, RetryPolicy
from omh.llm.auth.types import (
    ApiKeyCredential,
    AuthOperationOptions,
    Credential,
    CredentialInfo,
)
from omh.llm.models import EXTENDED_THINKING_LEVELS
from omh.llm.types import ModelThinkingLevel

#: The built-in model tool identifiers shared by settings and the host.
ToolName = Literal["read", "bash", "edit", "write"]
#: One explicit system-prompt addendum: literal text or a file path.
AppendKind = Literal["text", "file"]

#: Product application name; drives the environment variable name and defaults.
APP_NAME = "omh"
#: Directory name used for the global agent directory and project configuration.
CONFIG_DIR_NAME = ".omh"
#: Environment variable replacing the global agent directory.
ENV_AGENT_DIR = "OMH_CODING_AGENT_DIR"

SETTINGS_FILE = "settings.json"
AUTH_FILE = "auth.json"
MODELS_FILE = "models.json"
#: Remembered project resource-trust decisions; global-only.
TRUST_FILE = "trust.json"

#: Project and global files replacing or extending the base system prompt.
SYSTEM_FILE = "SYSTEM.md"
APPEND_SYSTEM_FILE = "APPEND_SYSTEM.md"
#: Directories holding automatically discovered skills and prompt templates.
SKILLS_DIR = "skills"
PROMPTS_DIR = "prompts"
#: User-authored resource directory name, relative to a project or home directory.
AGENTS_RESOURCES_DIR = ".agents"

#: Private creation mode for the credential file.
AUTH_FILE_MODE = 0o600

#: Provider environment variables, used for repair diagnostics and documentation.
PROVIDER_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
}

SettingsScope = Literal["global", "project"]
DiagnosticSource = Literal[
    "global", "project", "cli", "settings", "history", "models", "credentials", "trust",
]
DiagnosticReason = Literal[
    "invalid-json", "invalid-schema", "unknown-key", "invalid-type", "invalid-value", "unavailable",
    "adjusted", "untrusted", "recoverable",
]

#: The built-in tool selection shared by settings validation and the host.
DEFAULT_TOOLS: tuple[ToolName, ...] = ("read", "bash", "edit", "write")


def validate_tools(names: Sequence[str]) -> tuple[ToolName, ...] | None:
    """Accept a distinct subset of the built-in tools; an empty subset disables them."""
    if len(set(names)) != len(names) or any(name not in DEFAULT_TOOLS for name in names):
        return None
    return tuple(cast(ToolName, name) for name in names)


@dataclass(frozen=True, slots=True)
class ConfigDiagnostic:
    """One explainable configuration or directory problem with its source."""

    path: str
    source: DiagnosticSource
    reason: DiagnosticReason
    message: str

    @property
    def blocking(self) -> bool:
        # ``untrusted`` records a skipped controlled layer, ``recoverable``
        # records a fallback resource read, and any ``trust`` diagnostic only
        # reports a lost remembered default; all still leave a runnable session.
        if self.source == "trust":
            return False
        return self.reason not in {"unknown-key", "adjusted", "untrusted", "recoverable"}


class ConfigError(Exception):
    """A configuration file could not be read or written as required."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def resolve_agent_dir(
    *, env: Mapping[str, str] | None = None, home: str | Path | None = None,
) -> Path:
    """Resolve the global agent directory from the environment or its default."""
    environment = os.environ if env is None else env
    override = environment.get(ENV_AGENT_DIR)
    if override:
        return Path(override).expanduser().resolve()
    base = Path.home() if home is None else Path(home)
    return (base / CONFIG_DIR_NAME / "agent").expanduser().resolve()


def project_settings_path(cwd: str | Path) -> Path:
    """Return the project configuration path for an effective working directory."""
    return Path(cwd).expanduser() / CONFIG_DIR_NAME / SETTINGS_FILE


def provider_api_key_env(provider_id: str) -> str | None:
    """Return the common environment variable for a supported provider, if any."""
    return PROVIDER_API_KEY_ENV.get(provider_id)


# --------------------------------------------------------------------------- #
# JSON objects and recursive merge
# --------------------------------------------------------------------------- #


def _read_json_object(path: Path, source: DiagnosticSource) -> tuple[dict[str, object], ConfigDiagnostic | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, None
    except OSError as error:
        return {}, ConfigDiagnostic(str(path), source, "invalid-json", f"Cannot read {path.name}: {error}")
    if not text.strip():
        return {}, None
    try:
        parsed = json.loads(text)
    except ValueError as error:
        return {}, ConfigDiagnostic(str(path), source, "invalid-json", f"Invalid JSON in {path.name}: {error}")
    if not isinstance(parsed, dict):
        return {}, ConfigDiagnostic(
            str(path), source, "invalid-schema", f"{path.name} must contain a JSON object",
        )
    return parsed, None


def merge_settings(
    base: Mapping[str, object], override: Mapping[str, object],
) -> dict[str, object]:
    """Merge ``override`` onto ``base``; nested objects merge, arrays replace."""
    result: dict[str, object] = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = merge_settings(cast(Mapping[str, object], current), cast(Mapping[str, object], value))
        else:
            result[key] = value
    return result


# --------------------------------------------------------------------------- #
# Settings schema
# --------------------------------------------------------------------------- #


def _is_object(value: object) -> bool:
    return isinstance(value, dict)


_SETTINGS_TYPES: dict[str, type] = {
    "defaultProvider": str,
    "defaultModel": str,
    "defaultThinkingLevel": str,
    "enabledModels": list,
    "defaultTools": list,
    "skills": list,
    "prompts": list,
    "theme": str,
    "hideThinking": bool,
    "collapseTools": bool,
    "compaction": dict,
    "retry": dict,
    "sessionDir": str,
}


def _validate_object(
    values: Mapping[str, object], *, path: str, source: DiagnosticSource,
    schema: Mapping[str, type], label: str,
) -> tuple[dict[str, object], list[ConfigDiagnostic]]:
    """Validate one settings object, dropping invalid values and hinting unknown keys."""
    accepted: dict[str, object] = {}
    diagnostics: list[ConfigDiagnostic] = []
    for key, value in values.items():
        expected = schema.get(key)
        if expected is None:
            diagnostics.append(ConfigDiagnostic(
                path, source, "unknown-key", f"Unknown {label} key {key!r}",
            ))
            accepted[key] = value
            continue
        if expected is dict:
            if not _is_object(value):
                diagnostics.append(ConfigDiagnostic(
                    path, source, "invalid-type", f"{label} {key} must be an object",
                ))
                continue
            accepted[key] = value
            continue
        if expected is bool:
            if not isinstance(value, bool):
                diagnostics.append(ConfigDiagnostic(
                    path, source, "invalid-type", f"{label} {key} must be a boolean",
                ))
                continue
            accepted[key] = value
            continue
        if expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                diagnostics.append(ConfigDiagnostic(
                    path, source, "invalid-type", f"{label} {key} must be an integer",
                ))
                continue
            accepted[key] = value
            continue
        if expected is float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                diagnostics.append(ConfigDiagnostic(
                    path, source, "invalid-type", f"{label} {key} must be a number",
                ))
                continue
            accepted[key] = value
            continue
        if expected is list:
            if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
                diagnostics.append(ConfigDiagnostic(
                    path, source, "invalid-type", f"{label} {key} must be an array of strings",
                ))
                continue
            accepted[key] = value
            continue
        if expected is str:
            if not isinstance(value, str):
                diagnostics.append(ConfigDiagnostic(
                    path, source, "invalid-type", f"{label} {key} must be a string",
                ))
                continue
            accepted[key] = value
            continue
        accepted[key] = value
    return accepted, diagnostics


_COMPACTION_TYPES: dict[str, type] = {
    "enabled": bool,
    "reserveTokens": int,
    "keepRecentTokens": int,
}

_RETRY_TYPES: dict[str, type] = {
    "enabled": bool,
    "maxRetries": int,
    "baseDelayMs": float,
    "maxAgentDelayMs": float,
}


def _validate_settings(
    values: Mapping[str, object], *, path: str, source: DiagnosticSource,
) -> tuple[dict[str, object], list[ConfigDiagnostic]]:
    accepted, diagnostics = _validate_object(
        values, path=path, source=source, schema=_SETTINGS_TYPES, label="settings",
    )
    thinking = accepted.get("defaultThinkingLevel")
    if isinstance(thinking, str) and thinking not in EXTENDED_THINKING_LEVELS:
        diagnostics.append(ConfigDiagnostic(
            path, source, "invalid-value",
            f"defaultThinkingLevel must be one of {', '.join(EXTENDED_THINKING_LEVELS)}",
        ))
        accepted.pop("defaultThinkingLevel", None)
    tools = accepted.get("defaultTools")
    if isinstance(tools, list) and validate_tools(tools) is None:
        diagnostics.append(ConfigDiagnostic(
            path, source, "invalid-value",
            f"defaultTools must be a distinct subset of {', '.join(DEFAULT_TOOLS)}",
        ))
        accepted.pop("defaultTools", None)
    if isinstance(accepted.get("compaction"), dict):
        nested, nested_diagnostics = _validate_object(
            cast(Mapping[str, object], accepted["compaction"]),
            path=path, source=source, schema=_COMPACTION_TYPES, label="compaction",
        )
        accepted["compaction"] = nested
        diagnostics.extend(nested_diagnostics)
        try:
            settings_compaction(SettingsSnapshot(values={"compaction": nested}))
        except ValueError as error:
            diagnostics.append(ConfigDiagnostic(path, source, "invalid-value", str(error)))
            accepted.pop("compaction")
    if isinstance(accepted.get("retry"), dict):
        nested, nested_diagnostics = _validate_object(
            cast(Mapping[str, object], accepted["retry"]),
            path=path, source=source, schema=_RETRY_TYPES, label="retry",
        )
        accepted["retry"] = nested
        diagnostics.extend(nested_diagnostics)
        try:
            settings_retry(SettingsSnapshot(values={"retry": nested}))
        except ValueError as error:
            diagnostics.append(ConfigDiagnostic(path, source, "invalid-value", str(error)))
            accepted.pop("retry")
    return accepted, diagnostics


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """The merged effective settings with their global/project provenance."""

    values: Mapping[str, object] = field(default_factory=dict)
    global_values: Mapping[str, object] = field(default_factory=dict)
    project_values: Mapping[str, object] = field(default_factory=dict)
    explicit_values: Mapping[str, object] = field(default_factory=dict)
    diagnostics: tuple[ConfigDiagnostic, ...] = ()

    def source_of(self, key: str) -> DiagnosticSource:
        """Report which scope supplied the effective value for ``key``."""
        if key in self.explicit_values:
            return "cli"
        return "project" if key in self.project_values else "global"


def load_settings(
    *, agent_dir: Path, cwd: Path, project_trusted: bool = False,
    explicit: Mapping[str, object] | None = None,
) -> SettingsSnapshot:
    """Load and merge global, trusted project and explicit settings.

    The project layer is skipped entirely when the project is not trusted, so a
    replaced source never revives values from a lower priority layer.
    """
    diagnostics: list[ConfigDiagnostic] = []
    global_path = agent_dir / SETTINGS_FILE
    global_values, error = _read_json_object(global_path, "global")
    if error is not None:
        diagnostics.append(error)
    global_values, global_diagnostics = _validate_settings(
        global_values, path=str(global_path), source="global",
    )
    diagnostics.extend(global_diagnostics)

    project_values: dict[str, object] = {}
    if project_trusted:
        project_path = project_settings_path(cwd)
        project_values, error = _read_json_object(project_path, "project")
        if error is not None:
            diagnostics.append(error)
        project_values, project_diagnostics = _validate_settings(
            project_values, path=str(project_path), source="project",
        )
        diagnostics.extend(project_diagnostics)

    merged = merge_settings(global_values, project_values)
    explicit_clean: dict[str, object] = {}
    if explicit:
        explicit_clean, explicit_diagnostics = _validate_settings(
            explicit, path="<cli>", source="cli",
        )
        diagnostics.extend(explicit_diagnostics)
        merged = merge_settings(merged, explicit_clean)
    return SettingsSnapshot(
        values=merged, global_values=global_values, project_values=project_values,
        explicit_values=explicit_clean,
        diagnostics=tuple(diagnostics),
    )


def _as_str(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    return value if isinstance(value, str) else None


def settings_model(snapshot: SettingsSnapshot) -> tuple[str | None, str] | None:
    """Return the configured provider and exact model reference when set."""
    provider = _as_str(snapshot.values, "defaultProvider")
    model = _as_str(snapshot.values, "defaultModel")
    if model is None:
        return None
    return (provider, model)


def settings_thinking(snapshot: SettingsSnapshot) -> ModelThinkingLevel | None:
    value = _as_str(snapshot.values, "defaultThinkingLevel")
    if value in EXTENDED_THINKING_LEVELS:
        return value
    return None


def settings_tools(snapshot: SettingsSnapshot) -> tuple[ToolName, ...] | None:
    value = snapshot.values.get("defaultTools")
    if not isinstance(value, list):
        return None
    return validate_tools(value)


def settings_strings(snapshot: SettingsSnapshot, key: str) -> tuple[str, ...]:
    value = snapshot.values.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str))


def settings_theme(snapshot: SettingsSnapshot) -> str | None:
    return _as_str(snapshot.values, "theme")


def settings_compaction(snapshot: SettingsSnapshot) -> CompactionSettings | None:
    """Map the settings object onto the SDK's existing public compaction settings."""
    value = snapshot.values.get("compaction")
    if not isinstance(value, dict):
        return None
    options: dict[str, object] = {}
    if isinstance(value.get("enabled"), bool):
        options["enabled"] = value["enabled"]
    if isinstance(value.get("reserveTokens"), int) and not isinstance(value.get("reserveTokens"), bool):
        options["reserve_tokens"] = value["reserveTokens"]
    if isinstance(value.get("keepRecentTokens"), int) and not isinstance(value.get("keepRecentTokens"), bool):
        options["keep_recent_tokens"] = value["keepRecentTokens"]
    return CompactionSettings(**options)  # type: ignore[arg-type]


def settings_retry(snapshot: SettingsSnapshot) -> RetryPolicy | None:
    """Map the settings object onto the SDK's existing public retry policy."""
    value = snapshot.values.get("retry")
    if not isinstance(value, dict):
        return None
    options: dict[str, object] = {}
    if isinstance(value.get("enabled"), bool):
        options["enabled"] = value["enabled"]
    for key, field_name in (("maxRetries", "max_retries"), ("baseDelayMs", "base_delay_ms"),
                            ("maxAgentDelayMs", "max_agent_delay_ms")):
        entry = value.get(key)
        if not isinstance(entry, bool) and isinstance(entry, int | float):
            options[field_name] = entry
    return RetryPolicy(**options)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _atomic_write(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        if mode is not None and os.name == "posix":
            os.chmod(handle.name, mode)
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def update_settings(
    path: str | Path, changes: Mapping[str, object],
) -> dict[str, object]:
    """Persist only the selected fields, preserving every other key.

    Nested objects merge with the stored object; arrays and scalars replace.
    The stored file must already be a JSON object when it exists.
    """
    destination = Path(path)
    current, error = _read_json_object(destination, "global")
    if error is not None:
        raise ConfigError(error.message)
    merged = merge_settings(current, changes)
    _atomic_write(destination, json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return merged


def validate_settings_changes(changes: Mapping[str, object]) -> None:
    """Reject invalid or unknown fields before an explicit settings write."""
    _, diagnostics = _validate_settings(changes, path="settings", source="settings")
    if diagnostics:
        raise ConfigError("; ".join(item.message for item in diagnostics))
    if "theme" in changes and changes["theme"] not in {"dark", "light"}:
        raise ConfigError("theme must be dark or light")


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


def _parse_auth(text: str, path: Path) -> dict[str, ApiKeyCredential]:
    try:
        parsed = json.loads(text) if text.strip() else {}
    except ValueError as error:
        raise ConfigError(f"Invalid JSON in {path.name}: {error}") from error
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path.name} must contain a JSON object")
    credentials: dict[str, ApiKeyCredential] = {}
    for provider_id, entry in parsed.items():
        if not isinstance(entry, dict) or entry.get("type") != "api_key":
            raise ConfigError(f"Invalid {path.name} credential for provider {provider_id!r}")
        key = entry.get("key")
        env = entry.get("env")
        if key is not None and not isinstance(key, str):
            raise ConfigError(f"Invalid {path.name} credential for provider {provider_id!r}")
        if env is not None and (
            not isinstance(env, dict)
            or not all(isinstance(name, str) and isinstance(value, str) for name, value in env.items())
        ):
            raise ConfigError(f"Invalid {path.name} credential for provider {provider_id!r}")
        credentials[provider_id] = ApiKeyCredential(
            key=key, env=cast(dict[str, str] | None, env),
        )
    return credentials


class FileCredentialStore:
    """Credential store backed by the global ``auth.json`` with mode 0600."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    def _enforce_mode(self) -> None:
        if os.name != "posix" or not self.path.exists():
            return
        try:
            current = stat.S_IMODE(self.path.stat().st_mode)
        except OSError:
            return
        if current & 0o077:
            try:
                os.chmod(self.path, AUTH_FILE_MODE)
            except OSError:
                pass

    def _load(self) -> dict[str, ApiKeyCredential]:
        if not self.path.exists():
            return {}
        self._enforce_mode()
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as error:
            raise ConfigError(f"Cannot read {self.path.name}: {error}") from error
        return _parse_auth(text, self.path)

    @staticmethod
    def _check_signal(options: AuthOperationOptions | None) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        self._check_signal(options)
        async with self._lock:
            return self._load().get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> Sequence[CredentialInfo]:
        self._check_signal(options)
        async with self._lock:
            return tuple(
                CredentialInfo(provider_id=provider_id, type="api_key")
                for provider_id in self._load()
            )

    async def modify(
        self, provider_id: str,
        fn: Callable[[Credential | None], Credential | None | Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        self._check_signal(options)
        async with self._lock:
            credentials = self._load()
            current = credentials.get(provider_id)
            result = fn(current)
            if inspect.isawaitable(result):
                result = await result
            self._check_signal(options)
            if result is None:
                return current
            if not isinstance(result, ApiKeyCredential):
                raise ConfigError("Credential modification must return an api_key credential")
            credentials[provider_id] = result
            self._write(credentials)
            return result

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        self._check_signal(options)
        async with self._lock:
            credentials = self._load()
            credentials.pop(provider_id, None)
            self._write(credentials)

    def _write(self, credentials: Mapping[str, ApiKeyCredential]) -> None:
        payload = {
            provider_id: {"type": "api_key", **({"key": credential.key} if credential.key is not None else {}),
                          **({"env": credential.env} if credential.env is not None else {})}
            for provider_id, credential in credentials.items()
        }
        _atomic_write(self.path, json.dumps(payload, indent=2) + "\n", mode=AUTH_FILE_MODE)
        self._enforce_mode()
