"""Embeddable coding application consuming the omh SDK."""

from coding_agent.agent_session import AgentSession, CodingAgentOptions, ToolName
from coding_agent.agent_session_runtime import AgentSessionRuntime
from coding_agent.config import (
    ConfigDiagnostic,
    ConfigError,
    FileCredentialStore,
    SettingsSnapshot,
    load_settings,
    merge_settings,
    project_settings_path,
    resolve_agent_dir,
    update_settings,
)
from coding_agent.history import DecodedHistory, decode_history, encode_history
from coding_agent.host import CodingAgentHost, SessionSelection
from coding_agent.model_directory import ModelDirectory, ModelListing
from coding_agent.resources import ApplicationDiagnostic, ApplicationResources
from coding_agent.session_manager import SaveMode, SaveState, SessionManager

__all__ = [
    "AgentSession", "AgentSessionRuntime", "ApplicationDiagnostic", "ApplicationResources",
    "CodingAgentHost", "CodingAgentOptions", "ConfigDiagnostic", "ConfigError",
    "DecodedHistory", "FileCredentialStore", "ModelDirectory", "ModelListing", "SaveMode",
    "SaveState", "SessionManager", "SessionSelection", "SettingsSnapshot", "ToolName",
    "decode_history", "encode_history", "load_settings", "merge_settings",
    "project_settings_path", "resolve_agent_dir", "update_settings",
]
