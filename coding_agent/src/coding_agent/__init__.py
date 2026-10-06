"""Embeddable coding application consuming the omh SDK."""

from coding_agent.agent_session import (
    AgentSession,
    CodingAgentOptions,
    ToolName,
    UnsupportedImageModelError,
)
from coding_agent.agent_session_runtime import AgentSessionRuntime
from coding_agent.attachments import (
    AttachmentError,
    FileAttachment,
    attachment_images,
    compose_first_task,
    read_file_attachment,
    render_attachment,
)
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
from coding_agent.images import (
    IMAGE_MAX_BASE64_BYTES,
    IMAGE_MAX_DIMENSION,
    ImageInputError,
    ImageLimits,
    ProcessedImage,
    count_history_images,
    create_read_image_processor,
    degradation_notice,
    process_image,
    read_image,
)
from coding_agent.model_directory import ModelDirectory, ModelListing
from coding_agent.resources import ApplicationDiagnostic, ApplicationResources
from coding_agent.session_manager import SaveState, SessionManager
from coding_agent.trust import TrustDecision, TrustStore

__all__ = [
    "AgentSession", "AgentSessionRuntime", "ApplicationDiagnostic", "ApplicationResources",
    "AttachmentError", "CodingAgentHost", "CodingAgentOptions", "ConfigDiagnostic", "ConfigError",
    "DecodedHistory", "FileAttachment", "FileCredentialStore", "IMAGE_MAX_BASE64_BYTES",
    "IMAGE_MAX_DIMENSION", "ImageInputError", "ImageLimits", "ModelDirectory", "ModelListing",
    "ProcessedImage", "SaveState", "SessionManager", "SessionSelection", "SettingsSnapshot",
    "ToolName", "TrustDecision", "TrustStore", "UnsupportedImageModelError", "attachment_images", "compose_first_task",
    "count_history_images", "create_read_image_processor", "decode_history", "degradation_notice",
    "encode_history", "load_settings", "merge_settings", "process_image",
    "project_settings_path", "read_file_attachment", "read_image", "render_attachment",
    "resolve_agent_dir", "update_settings",
]
