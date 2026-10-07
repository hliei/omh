"""Parse explicit settings edits without changing a conversation selection."""

from __future__ import annotations

import json

from omh.llm.types import Model

from coding_agent.config import (
    ConfigError,
    ToolName,
    validate_settings_changes,
    validate_tools,
)
from coding_agent.model_directory import ModelDirectory

# Short UI names and their persisted settings fields.
FIELDS = {
    'thinking': 'defaultThinkingLevel', 'tools': 'defaultTools',
    'models': 'enabledModels', 'theme': 'theme',
    'hideThinking': 'hideThinking', 'collapseTools': 'collapseTools',
    'compaction': 'compaction', 'retry': 'retry', 'skills': 'skills', 'prompts': 'prompts',
}


def resolve_model(directory: ModelDirectory, reference: str) -> Model:
    """Resolve an exact reference; never choose another provider on failure."""
    provider, slash, model_id = reference.partition('/')
    matches = directory.find_models(provider, model_id) if slash else directory.find_models(None, reference)
    if len(matches) != 1:
        raise ConfigError(f'Unknown or ambiguous model {reference!r}; use /model to select an exact provider/model')
    return matches[0]


def parse_tools(value: str) -> tuple[ToolName, ...]:
    names = () if value in {'none', 'off'} else tuple(value.split(','))
    validated = validate_tools(names)
    if validated is None:
        raise ConfigError('tools must be a distinct subset of read,bash,edit,write, or none')
    return validated


def default_change(directory: ModelDirectory, field: str, raw: str) -> dict[str, object]:
    """Validate one selected field, retaining every unselected stored field."""
    if field == 'model':
        model = resolve_model(directory, raw)
        return {'defaultProvider': model.provider, 'defaultModel': model.id}
    key = FIELDS.get(field)
    if key is None:
        raise ConfigError(f'Unknown settings field {field!r}; fields: model, ' + ', '.join(FIELDS))
    if field == 'tools':
        value: object = list(parse_tools(raw))
    elif field in {'models', 'skills', 'prompts', 'compaction', 'retry', 'hideThinking', 'collapseTools'}:
        try:
            value = json.loads(raw)
        except ValueError as error:
            raise ConfigError(f'{field} requires a JSON value: {error}') from error
    else:
        value = raw
    changes = {key: value}
    validate_settings_changes(changes)
    if field == 'models':
        assert isinstance(value, list)
        for reference in value:
            resolve_model(directory, reference)
    return changes


def effect_of(field: str) -> str:
    if field == 'models':
        return 'cycle set updated now; current model unchanged'
    if field in {'compaction', 'retry'}:
        return 'takes effect on new/reopened sessions'
    if field in {'skills', 'prompts'}:
        return 'takes effect after /reload, on the next new prompt; accepted inputs stay expanded'
    return 'takes effect on new/reopened sessions; current selection unchanged'
