"""First configuration and credential commands through the installed PTY."""

import json
import stat
from dataclasses import asdict
from pathlib import Path

import pytest
from omh.llm.types import SystemMessage
from omh.llm.utils.transcript import get_current_system_message
from test_cli import run_cli
from test_interactive import InteractiveSession, provider_env, sends, settle


def test_missing_key_can_be_saved_without_echo_or_verification(tmp_path: Path) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    ui = InteractiveSession('--no-context-files', home=home, cwd=project,
                            env=provider_env(home, 'echo'))
    secret = 'private-test-api-key'
    auth = home / '.omh' / 'agent' / 'auth.json'
    try:
        ui.wait_for('No API key available')
        ui.send(b'/login opencode-go\r')
        ui.wait_for('API key for opencode-go (hidden)')
        ui.send(secret.encode() + b'\r')
        ui.wait_for('opencode-go key source: stored credential')
        assert sends(home) == []
        assert stat.S_IMODE(auth.stat().st_mode) == 0o600
        assert json.loads(auth.read_text())['opencode-go']['key'] == secret
        ui.send(b'hello\r')
        ui.wait_for('reply:hello')
        ui.send(b'/export jsonl backup.jsonl\r')
        ui.wait_for('exported')
        ui.send(b'/logout opencode-go\r')
        ui.wait_for('opencode-go key source: none')
        ui.send(b'blocked\r')
        ui.wait_for('No API key available', after=ui.visible().rfind('key source: none'))
        ui.send(b'\x03/quit\r')
        assert ui.finish() == 0
        assert sends(home) == ['dialogue']
        assert secret not in ui.output.decode()
        for saved in [project / 'backup.jsonl', *home.rglob('sessions/**/*.jsonl')]:
            assert secret not in saved.read_text()
        assert not (project / '.omh' / 'auth.json').exists()
    finally:
        ui.close()


def test_first_trust_decision_is_remembered_and_denial_keeps_other_resources(tmp_path: Path) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    agent = home / '.omh' / 'agent'
    agent.mkdir(parents=True)
    (agent / 'SYSTEM.md').write_text('GLOBAL BASE')
    (project / 'AGENTS.md').write_text('ANCESTOR INSTRUCTIONS')
    config = project / '.omh'
    config.mkdir()
    (config / 'SYSTEM.md').write_text('PROJECT BASE')
    (config / 'settings.json').write_text('{"defaultModel":"glm-5.3"}')
    templates = config / 'prompts'
    templates.mkdir()
    (templates / 'local.md').write_text('PROJECT TEMPLATE')
    explicit = project / 'explicit.md'
    explicit.write_text('EXPLICIT TEMPLATE')
    options = ('--api-key', 'offline', '--prompt-template', str(explicit))
    ui = InteractiveSession(*options, home=home, cwd=project, env=provider_env(home, 'echo'))
    try:
        ui.wait_for('Trust project configuration and resources?')
        assert sends(home) == []
        ui.send(b'n')
        ui.wait_for('trust denied')
        ui.wait_for('model opencode-go/deepseek-v4.1-flash')
        trust = json.loads((agent / 'trust.json').read_text())
        assert trust['projects'][str(project.resolve())] == 'denied'
        ui.send(b'/explicit\r')
        ui.wait_for('reply:EXPLICIT TEMPLATE')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
    finally:
        ui.close()
    reopened = InteractiveSession(*options, home=home, cwd=project, env=provider_env(home, 'echo'))
    try:
        reopened.wait_for('phase input')
        assert 'Trust project configuration and resources?' not in reopened.visible()
        reopened.send(b'/trust approve\r')
        reopened.wait_for('trust approved')
        reopened.wait_for('run /reload when idle')
        reopened.send(b'/reload\r')
        reopened.wait_for('resources reloaded')
        reopened.send(b'/local\r')
        reopened.wait_for('reply:PROJECT TEMPLATE')
        assert 'current model opencode-go/glm-5.3' not in reopened.visible()
        reopened.send(b'/quit\r')
        assert reopened.finish() == 0
    finally:
        reopened.close()


@pytest.mark.parametrize('provider,env_name', [('deepseek', 'DEEPSEEK_API_KEY'), ('opencode-go', 'OPENCODE_API_KEY')])
@pytest.mark.parametrize('override', [False, True])
def test_logout_reports_the_remaining_source(tmp_path: Path, provider: str, env_name: str, override: bool) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    extra = ('--api-key', 'temporary-secret') if override else ()
    model = 'deepseek-flash' if provider == 'deepseek' else 'glm-5.3'
    ui = InteractiveSession('--no-session', '--no-context-files', '--model', f'{provider}/{model}', *extra,
                            home=home, cwd=project,
                            env=provider_env(home, 'echo', **{env_name: 'environment-secret'}))
    try:
        ui.wait_for('phase input')
        ui.send(f'/login {provider}\r'.encode())
        ui.wait_for(f'API key for {provider} (hidden)')
        ui.send(b'saved-secret\r')
        ui.wait_for('saved global API key')
        source = 'cli' if override else 'stored credential'
        ui.wait_for(f'{provider} key source: {source}')
        before = len(ui.visible())
        ui.send(b'/logout\r')
        ui.wait_for('deleted saved global key')
        ui.wait_for(f'{provider} key source: {"cli" if override else env_name}', after=before)
        assert json.loads((home / '.omh' / 'agent' / 'auth.json').read_text()) == {}
        assert sends(home) == []
        ui.send(b'hello\r')
        ui.wait_for('reply:hello')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert all(secret not in ui.output.decode() for secret in ('temporary-secret', 'environment-secret', 'saved-secret'))
    finally:
        ui.close()


def test_hidden_input_cancel_and_invalid_auth_commands_never_send(tmp_path: Path) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    ui = InteractiveSession('--no-session', '--no-context-files', home=home, cwd=project,
                            env=provider_env(home, 'echo'))
    try:
        ui.wait_for('No API key available')
        for command in (b'/login other\r', b'/login deepseek inline-secret\r', b'/logout other\r', b'/trust invalid\r'):
            before = len(ui.visible())
            ui.send(command)
            ui.wait_for('nothing was sent', after=before)
        ui.send(b'/login deepseek\r')
        ui.wait_for('API key for deepseek (hidden)')
        ui.send(b'cancelled-secret\x1b')
        ui.wait_for('login cancelled')
        ui.send(b'/login\r')
        ui.wait_for('deepseek key source: none')
        ui.wait_for('opencode-go key source: none')
        ui.send(b'/help login\r')
        ui.wait_for('/login [deepseek | opencode-go]')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == []
        assert 'inline-secret' not in ui.output.decode()
        assert 'cancelled-secret' not in ui.output.decode()
        assert not (home / '.omh' / 'agent' / 'auth.json').exists()
    finally:
        ui.close()


def test_authentication_error_can_be_repaired_without_verification(tmp_path: Path) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    env = provider_env(home, 'auth-error', OPENCODE_API_KEY='expired-secret')
    ui = InteractiveSession('--no-session', '--no-context-files', home=home, cwd=project, env=env)
    try:
        ui.wait_for('phase input')
        ui.send(b'hello\r')
        ui.wait_for('Authentication failed for opencode-go')
        settle(ui)
        ui.send(b'/login opencode-go\r')
        ui.wait_for('API key for opencode-go (hidden)')
        ui.send(b'replacement-secret\r')
        ui.wait_for('opencode-go key source: stored credential')
        assert sends(home) == ['dialogue']
        ui.send(b'retry\r')
        ui.wait_for('reply:retry')
        ui.send(b'/quit\r')
        assert ui.finish() == 0
        assert sends(home) == ['dialogue', 'dialogue']
        assert 'replacement-secret' not in ui.output.decode()
    finally:
        ui.close()


def test_busy_trust_waits_for_reload_and_keeps_accepted_inputs(tmp_path: Path) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    agent = home / '.omh' / 'agent'
    agent.mkdir(parents=True)
    (agent / 'SYSTEM.md').write_text('GLOBAL BASE')
    (project / 'AGENTS.md').write_text('ANCESTOR INSTRUCTIONS')
    config = project / '.omh'
    config.mkdir()
    (config / 'SYSTEM.md').write_text('PROJECT BASE')
    templates = config / 'prompts'
    templates.mkdir()
    (templates / 'local.md').write_text('ORIGINAL TEMPLATE')
    contexts = home / 'contexts.jsonl'
    ui = InteractiveSession('--no-session', '--api-key', 'offline', home=home, cwd=project,
                            env=provider_env(home, 'settings-request', INTERACTIVE_PAYLOADS=str(contexts)))
    try:
        ui.wait_for('Trust project configuration and resources?')
        ui.send(b'y')
        ui.wait_for('trust approved')
        ui.send(b'go\r')
        ui.wait_for('request opencode-go/deepseek-v4.1-flash')
        ui.send(b'/local\r')
        ui.wait_for('queued steering (1 waiting): /local')
        ui.send(b'/trust deny\r')
        ui.wait_for('trust denied remembered')
        ui.send(b'/reload\r')
        ui.wait_for('preparation is busy')
        (templates / 'local.md').write_text('CHANGED TEMPLATE')
        (project / 'release-settings').touch()
        ui.wait_for('reply:ORIGINAL TEMPLATE')
        settle(ui)
        assert 'PROJECT BASE' in contexts.read_text()
        assert 'CHANGED TEMPLATE' not in contexts.read_text()
        assert (project / 'first.txt').read_text() == 'old snapshot'
        ui.send(b'/reload\r')
        ui.wait_for('resources reloaded')
        ui.send(b'after reload\r')
        ui.wait_for('reply:after reload')
        latest = json.loads(contexts.read_text().splitlines()[-1])
        system = json.dumps(asdict(get_current_system_message([
            SystemMessage(content=item['content'], timestamp=item['timestamp'], sections=item['sections'])
            for item in latest if item.get('role') == 'system'
        ])))
        assert 'GLOBAL BASE' in system
        assert 'PROJECT BASE' not in system
        assert 'ANCESTOR INSTRUCTIONS' in system
        ui.send(b'/quit\r')
        assert ui.finish() == 0
    finally:
        ui.close()


@pytest.mark.parametrize('flags,remembered,expected', [
    ((), None, 'GLOBAL BASE'),
    (('--approve',), None, 'PROJECT BASE'),
    (('--no-approve',), 'approved', 'GLOBAL BASE'),
    ((), 'approved', 'PROJECT BASE'),
    ((), 'denied', 'GLOBAL BASE'),
])
def test_print_never_asks_trust_or_dispatches_login(tmp_path: Path, flags: tuple[str, ...], remembered: str | None, expected: str) -> None:
    home, project = tmp_path / 'home', tmp_path / 'project'
    home.mkdir()
    project.mkdir()
    agent = home / '.omh' / 'agent'
    agent.mkdir(parents=True)
    (agent / 'SYSTEM.md').write_text('GLOBAL BASE')
    config = project / '.omh'
    config.mkdir()
    (config / 'SYSTEM.md').write_text('PROJECT BASE')
    if remembered:
        (agent / 'trust.json').write_text(json.dumps({'projects': {str(project.resolve()): remembered}}))
    payloads = home / 'payloads.jsonl'
    result = run_cli('--print', '--no-session', '--cwd', str(project), '--api-key', 'offline', *flags,
                     '/login deepseek', home=home, env=provider_env(home, 'echo', INTERACTIVE_PAYLOADS=str(payloads)))
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'reply:/login deepseek'
    assert 'Trust project configuration and resources?' not in result.stdout + result.stderr
    assert not (agent / 'auth.json').exists()
    first = json.loads(payloads.read_text().splitlines()[0])
    assert expected in json.dumps(first)
    if not flags and remembered is None:
        assert 'not trusted and were skipped' in result.stderr
        assert not (agent / 'trust.json').exists()
