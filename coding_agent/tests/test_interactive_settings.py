"""Current selections and scoped defaults through the installed terminal UI."""

import json
from pathlib import Path

import pytest
from test_interactive import InteractiveSession, provider_env, sends
from test_interactive_sessions import start


def test_current_selection_and_defaults_are_independent(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    settings = home / '.omh' / 'agent' / 'settings.json'
    try:
        ui.wait_for('phase input')
        ui.send(b'/model opencode-go/kimi-k3\r')
        ui.wait_for('current model opencode-go/kimi-k3')
        ui.wait_for('thinking max')
        assert not settings.exists()
        ui.send(b'/settings global theme light\r')
        ui.wait_for(f'saved global {settings}')
        assert json.loads(settings.read_text()) == {'theme': 'light'}
        ui.send(b'/settings current\r')
        ui.wait_for('current theme dark')
        ui.send(b'/tools none\r')
        ui.wait_for('current tools none')
        ui.send(b'/thinking low\r')
        ui.wait_for('valid thinking: max')
        ui.send(b'hello\r')
        ui.wait_for('reply:hello')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == ['dialogue']
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


@pytest.mark.parametrize("scenario", ["settings-batch", "settings-request"])
def test_busy_choices_apply_to_tool_continuation_and_keep_captured_tools(tmp_path: Path, scenario: str) -> None:
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    payloads = home / 'selections.jsonl'
    ui = InteractiveSession(
        '--no-approve', '--no-context-files', '--api-key', 'offline', '--no-session',
        home=home, cwd=project,
        env=provider_env(home, scenario, INTERACTIVE_SELECTIONS=str(payloads)),
    )
    try:
        ui.wait_for('phase input')
        ui.send(b'go\r')
        ui.wait_for('settings-ready' if scenario == 'settings-batch' else 'request opencode-go/deepseek-v4.1-flash')
        ui.send(b'/model opencode-go/glm-5.3\r')
        ui.wait_for('current model opencode-go/glm-5.3')
        ui.send(b'/thinking low\r')
        ui.wait_for('current thinking low')
        ui.send(b'/tools none\r')
        ui.wait_for('current tools none')
        ui.wait_for('next opencode-go/glm-5.3 thinking low tools none')
        captured = json.loads(payloads.read_text().splitlines()[0])
        assert captured == {'model': 'opencode-go/deepseek-v4.1-flash', 'thinking': 'high', 'tools': ['read', 'bash', 'edit', 'write']}
        (project / 'release-settings').touch()
        ui.wait_for('settings batch done' if scenario == 'settings-batch' else 'reply:go')
        assert (project / 'first.txt').read_text() == ('a' if scenario == 'settings-batch' else 'old snapshot')
        if scenario == 'settings-batch':
            assert (project / 'second.txt').read_text() == 'b'
        selections = [json.loads(line) for line in payloads.read_text().splitlines()]
        assert selections[1] == {'model': 'opencode-go/glm-5.3', 'thinking': 'low', 'tools': []}
        assert len(selections) == 2
        ui.send(b'!printf user-shell\r')
        ui.wait_for('user-shell')
        ui.wait_for('shell completed')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_model_and_thinking_keys_use_effective_levels_and_preserve_the_draft(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    settings = home / '.omh' / 'agent' / 'settings.json'
    try:
        ui.wait_for('phase input')
        ui.send(b'/settings global models ["opencode-go/kimi-k3", "opencode-go/kimi-k2.7-code"]\r')
        ui.wait_for(f'saved global {settings}')
        ui.send(b'draft\x0c')
        ui.wait_for('Select model')
        ui.send(b'\x1b')
        ui.wait_for('model selection cancelled; draft preserved')
        ui.wait_for('> draft', after=ui.visible().index('model selection cancelled'))
        ui.send(b'\x03\x10')
        ui.wait_for('current model opencode-go/kimi-k3')
        ui.wait_for('adjusted from high to max')
        ui.send(b'\x1b[Z')
        ui.wait_for('thinking max; no other effective level')
        ui.send(b'\x10')
        ui.wait_for('current model opencode-go/kimi-k2.7-code thinking fixed-on')
        ui.send(b'\x1b[Z')
        ui.wait_for('thinking fixed-on; no other effective level')
        ui.send(b'\x1b[112;6u')
        ui.wait_for('current model opencode-go/kimi-k3', after=ui.visible().index('thinking fixed-on; no other'))
        ui.send(b'\x13')
        ui.wait_for('current thinking max unchanged')
        assert json.loads(settings.read_text()) == {
            'enabledModels': ['opencode-go/kimi-k3', 'opencode-go/kimi-k2.7-code'],
            'defaultThinkingLevel': 'max',
        }
        ui.send(b'\x0c\x1b[A\x1b[A\r')
        ui.wait_for('current model opencode-go/glm-5.3-flash')
        ui.send(b'\x1b[Z')
        ui.wait_for('current thinking low')
        ui.send(b'/quit\r')
        ui.wait_for('Enter discards it and exits')
        ui.send(b'\r')
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_scoped_writes_preserve_fields_reopen_history_and_report_failed_target(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    agent = home / '.omh' / 'agent'
    agent.mkdir(parents=True)
    settings = agent / 'settings.json'
    settings.write_text(json.dumps({'retry': {'enabled': False, 'maxRetries': 7}, 'future': {'keep': 1}}))
    ui, _ = start(tmp_path)
    project = tmp_path / 'project'
    try:
        ui.wait_for('phase input')
        ui.send(b'/settings global retry {"enabled":true}\r')
        ui.wait_for('takes effect on new/reopened sessions')
        assert json.loads(settings.read_text()) == {'retry': {'enabled': True, 'maxRetries': 7}, 'future': {'keep': 1}}
        ui.send(b'/settings project model deepseek/deepseek-flash\r')
        project_settings = project / '.omh' / 'settings.json'
        ui.wait_for(f'saved project {project_settings}')
        ui.wait_for('project defaults require trust before loading')
        assert json.loads(project_settings.read_text()) == {'defaultProvider': 'deepseek', 'defaultModel': 'deepseek-flash'}
        ui.send(b'/model opencode-go/kimi-k3\r')
        ui.wait_for('current model opencode-go/kimi-k3')
        ui.send(b'/name preserved selection\r')
        ui.wait_for('name preserved selection')
        saved = next((home / 'sessions').glob('*.jsonl'))
        ui.send(b'/settings global model deepseek/deepseek-flash\r')
        ui.wait_for(f'saved global {settings}: model')
        ui.send(b'/settings global thinking low\r')
        ui.wait_for(f'saved global {settings}: thinking')
        ui.send(b'/settings current\r')
        ui.wait_for('current model opencode-go/kimi-k3 thinking max')
        # A failed replace leaves the existing file/directory visible and does
        # not claim a successful write or affect the selected conversation.
        settings.unlink()
        settings.mkdir()
        offset = len(ui.visible())
        ui.send(b'/settings global theme light\r')
        ui.wait_for(f'global write failed at {settings}', after=offset)
        assert f'saved global {settings}: theme' not in ui.visible()[offset:]
        settings.rmdir()
        settings.write_text('{"defaultProvider":"deepseek","defaultModel":"deepseek-flash","defaultThinkingLevel":"low"}')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        ui.close()
        ui, _ = start(tmp_path, '--session', str(saved))
        ui.wait_for('model opencode-go/kimi-k3 | thinking max')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_settings_reject_invalid_fields_and_keep_low_color_theme_fallback(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    settings = home / '.omh' / 'agent' / 'settings.json'
    try:
        ui.wait_for('phase input')
        for command, message in [
            ('/settings theme light', 'needs current, global or project scope'),
            ('/settings global theme neon', 'theme must be dark or light'),
            ('/settings global tools read,unknown', 'tools must be a distinct subset'),
            ('/settings global models ["opencode-go/missing"]', 'Unknown or ambiguous model'),
            ('/settings global mystery true', 'Unknown settings field'),
            ('/tools read,read', 'tools must be a distinct subset'),
        ]:
            ui.send((command + '\r').encode())
            ui.wait_for(message)
        assert not settings.exists()
        ui.send(b'/settings current theme light\r')
        ui.wait_for('current theme light')
        ui.send(b'/settings current hideThinking false\r')
        ui.wait_for('current hideThinking false')
        ui.send(b'/settings current collapseTools false\r')
        ui.wait_for('current collapseTools false')
        assert not settings.exists()
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


@pytest.mark.parametrize('fallback', [{'NO_COLOR': '1'}, {'TERM': 'dumb'}])
def test_current_theme_respects_terminal_fallback(tmp_path: Path, fallback: dict[str, str]) -> None:
    home = tmp_path / 'home'
    home.mkdir()
    env = provider_env(home, 'echo', **fallback)
    ui = InteractiveSession('--no-approve', '--api-key', 'offline', home=home, cwd=tmp_path, env=env)
    try:
        ui.wait_for('theme plain')
        ui.send(b'/settings current theme light\r')
        ui.wait_for('current theme plain')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert b'\x1b[38;5;' not in ui.output
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()



def test_partial_write_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    ui, home = start(tmp_path, scenario='settings-write-failure')
    target = home / '.omh' / 'agent' / 'settings.json'
    try:
        ui.wait_for('phase input')
        ui.send(b'/settings global theme light\r')
        ui.wait_for('controlled failure after replacement')
        ui.wait_for('no success reported, inspect target before retrying')
        assert json.loads(target.read_text()) == {'theme': 'light'}
        assert 'saved global' not in ui.visible()
        ui.send(b'/settings current\r')
        ui.wait_for('current theme dark')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_resource_defaults_wait_for_reload_and_do_not_reexpand_inputs(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    agent = home / '.omh' / 'agent'
    agent.mkdir(parents=True)
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    (old / 'greet.md').write_text('old accepted input')
    (new / 'greet.md').write_text('new resource input $@')
    target = agent / 'settings.json'
    target.write_text(json.dumps({'prompts': [str(old)]}))
    ui, _ = start(tmp_path, '--no-prompt-templates')
    try:
        ui.wait_for('phase input')
        ui.send(('/settings global prompts ' + json.dumps([str(new)]) + '\r').encode())
        ui.wait_for('takes effect after /reload')
        ui.send(b'/greet\r')
        ui.wait_for('reply:old accepted input')
        ui.wait_for('phase input', after=ui.visible().index('reply:old accepted input'))
        ui.send(b'/reload\r')
        ui.wait_for('resources reloaded; next new prompt; accepted inputs unchanged')
        ui.send(b'/greet\r')
        ui.wait_for('reply:new resource input')
        ui.wait_for('phase input', after=ui.visible().index('reply:new resource input'))
        target.write_text('broken JSON')
        ui.send(b'/reload\r')
        ui.wait_for('Invalid JSON')
        ui.send(b'/greet after-failure\r')
        ui.wait_for('reply:new resource input after-failure')
        saved = next((home / 'sessions').glob('*.jsonl')).read_text()
        assert 'old accepted input' in saved
        assert 'new resource input' in saved
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == ['dialogue', 'dialogue', 'dialogue']
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_model_selection_diagnoses_missing_credentials_without_fallback(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    home.mkdir()
    ui = InteractiveSession('--no-approve', '--no-session', home=home, cwd=tmp_path,
                            env=provider_env(home, 'echo'))
    try:
        ui.wait_for('No API key')
        ui.send(b'/model deepseek/deepseek-flash\r')
        ui.wait_for('current model deepseek/deepseek-flash')
        ui.wait_for('DEEPSEEK_API_KEY')
        ui.send(b'/model opencode-go/missing\r')
        ui.wait_for('Unknown or ambiguous model')
        ui.send(b'/settings current\r')
        ui.wait_for('current model deepseek/deepseek-flash thinking high')
        ui.send(b'hello\r')
        ui.wait_for('No request was sent.', after=ui.visible().index('current model deepseek/deepseek-flash thinking high'))
        assert sends(home) == []
        ui.send(b'/quit\r')
        assert ui.finish() == 0
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_replacing_an_unavailable_history_model_preserves_conversation_identity(tmp_path: Path) -> None:
    ui, home = start(tmp_path)
    try:
        ui.wait_for('phase input')
        ui.send(b'/model opencode-go/kimi-k3\r')
        ui.wait_for('current model opencode-go/kimi-k3')
        ui.send(b'/name repair model\r')
        ui.wait_for('name repair model')
        path = next((home / 'sessions').glob('*.jsonl'))
        from coding_agent import decode_history
        identity = decode_history(path.read_bytes()).history.conversation_id
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        ui.close()
        path.write_text(path.read_text().replace('kimi-k3', 'missing-model'))
        ui, _ = start(tmp_path, '--session', str(path))
        ui.wait_for('Saved model opencode-go/missing-model is not available')
        ui.send(b'/model opencode-go/glm-5.3\r')
        ui.wait_for('current model opencode-go/glm-5.3')
        ui.wait_for(f'session {identity}')
        ui.send(b'/session\r')
        ui.wait_for('repair model')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert decode_history(path.read_bytes()).history.conversation_id == identity
        assert len(list((home / 'sessions').glob('*.jsonl'))) == 1
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()



def test_model_repair_retains_explicit_thinking_and_disabled_tools(tmp_path: Path) -> None:
    agent = tmp_path / 'home' / '.omh' / 'agent'
    agent.mkdir(parents=True)
    (agent / 'settings.json').write_text('{"defaultProvider":"opencode-go","defaultModel":"missing"}')
    ui, home = start(tmp_path, '--no-tools', '--thinking', 'low')
    try:
        ui.wait_for('No request was sent.')
        ui.send(b'/model opencode-go/glm-5.3\r')
        ui.wait_for('current model opencode-go/glm-5.3 thinking low')
        ui.send(b'/settings current\r')
        ui.wait_for('thinking low tools none')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


@pytest.mark.parametrize('policy', ['retry', 'compaction'])
def test_policy_defaults_wait_for_a_new_session(tmp_path: Path, policy: str) -> None:
    agent = tmp_path / 'home' / '.omh' / 'agent'
    agent.mkdir(parents=True)
    configuration = ({'enabled': False, 'baseDelayMs': 0, 'maxRetries': 1} if policy == 'retry'
                     else {'enabled': False, 'reserveTokens': 999999, 'keepRecentTokens': 0})
    target = agent / 'settings.json'
    target.write_text(json.dumps({policy: configuration}))
    ui, home = start(tmp_path, scenario=f'settings-{policy}')
    try:
        ui.wait_for('phase input')
        ui.send(f'/settings global {policy} {{"enabled":true}}\r'.encode())
        ui.wait_for('takes effect on new/reopened sessions')
        ui.send(b'first\r')
        first_result = 'model error 503' if policy == 'retry' else 'reply:first'
        ui.wait_for(first_result)
        ui.wait_for('phase input', after=ui.visible().index(first_result))
        assert sends(home) == ['dialogue']
        assert json.loads(target.read_text())[policy]['enabled'] is True
        ui.send(b'/new\r')
        ui.wait_for('new current')
        ui.send(b'second\r')
        if policy == 'retry':
            ui.wait_for('phase retry dialogue 1/')
            ui.wait_for('reply:second')
        else:
            ui.wait_for('phase compact')
            ui.wait_for('reply:second')
        ui.wait_for('phase input', after=ui.visible().index('reply:second'))
        if policy == 'retry':
            assert sends(home) == ['dialogue', 'dialogue', 'dialogue']
        else:
            assert 'summary' in sends(home)
        ui.send(b'/quit\r')
        assert ui.finish() == 0
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()


def test_project_defaults_follow_trusted_effective_cwd_on_new_sessions(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    agent = home / '.omh' / 'agent'
    agent.mkdir(parents=True)
    (agent / 'settings.json').write_text('{"defaultProvider":"deepseek","defaultModel":"deepseek-flash"}')
    startup, project = tmp_path / 'startup', tmp_path / 'effective'
    startup.mkdir()
    (project / '.omh').mkdir(parents=True)
    target = project / '.omh' / 'settings.json'
    target.write_text('{"defaultProvider":"opencode-go","defaultModel":"kimi-k3","theme":"light","hideThinking":false}')
    ui = InteractiveSession('--approve', '--no-context-files', '--api-key', 'offline', '--cwd', str(project),
                            home=home, cwd=startup, env=provider_env(home, 'echo'))
    try:
        ui.wait_for('model opencode-go/kimi-k3 | thinking max | theme light')
        ui.send(b'/settings project model opencode-go/kimi-k2.7-code\r')
        ui.wait_for(f'saved project {target}')
        assert not (startup / '.omh' / 'settings.json').exists()
        assert json.loads(target.read_text()) == {'defaultProvider': 'opencode-go', 'defaultModel': 'kimi-k2.7-code', 'theme': 'light', 'hideThinking': False}
        ui.send(b'/settings current\r')
        ui.wait_for('current model opencode-go/kimi-k3 thinking max')
        ui.send(b'/new\r')
        ui.wait_for('new current')
        ui.wait_for('model opencode-go/kimi-k2.7-code | thinking fixed-on')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == []
    finally:
        if ui.process.poll() is None:
            ui.process.kill()
            ui.process.wait()
        ui.close()
