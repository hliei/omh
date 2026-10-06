
import pytest
from omh.agent import AgentInitialState, AgentOptions, PromptTemplateSource, SkillSource
from omh.llm.utils.transcript import get_current_system_prompt, get_current_tools
from support import OfflineStream, model

from coding_agent import AgentSessionRuntime, CodingAgentOptions


def user_texts(context):
    return [message.content if isinstance(message.content, str) else "".join(
        block.text for block in message.content if block.type == "text"
    ) for message in context.messages if message.role == "user"]


def skill(path, *, name="review", description="Review code", body="Review carefully", hidden=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\ndisable-model-invocation: {str(hidden).lower()}\n---\n{body}")
    return path


@pytest.mark.parametrize("reopen", [False, True])
async def test_new_and_reopened_sessions_assemble_ordered_resources_and_actual_tools(tmp_path, reopen):
    global_dir = tmp_path / "global"
    project = tmp_path / "project"
    explicit = tmp_path / "explicit"
    for directory, text in ((global_dir, "global instructions"), (project, "project instructions"),
                            (explicit, "explicit instructions")):
        directory.mkdir()
        (directory / "AGENTS.md").write_text(text)
        skill(directory / "skills" / "review" / "SKILL.md", description=f"{directory.name} review")
        (directory / "prompts").mkdir()
        (directory / "prompts" / "review.md").write_text(f"{directory.name} template $1")
    stream = OfflineStream()
    options = CodingAgentOptions(
        cwd=project, agent_dir="../global", context_dirs=("../explicit",),
        model=model(), stream_fn=stream, tools=("read", "write"),
        skill_sources=tuple(SkillSource(directory / "skills", source) for directory, source in (
            (explicit, "explicit"), (project, "project"), (global_dir, "global"))),
        template_sources=tuple(PromptTemplateSource(directory / "prompts", source) for directory, source in (
            (explicit, "explicit"), (project, "project"), (global_dir, "global"))),
        agent_options=AgentOptions(initial_state=AgentInitialState(system_prompt="raw host instructions")),
    )
    session = await AgentSessionRuntime(options).new_session()
    if reopen:
        path = await session.save(project / "history.jsonl")
        await session.close()
        session = await AgentSessionRuntime(options).open_session(path)
    assert not stream.requests
    assert session.resources.skills[0].source == "global"
    assert [template.source for template in session.resources.templates] == ["global", "project", "explicit"]
    assert len([diagnostic for diagnostic in session.resources.diagnostics if diagnostic.reason == "collision"]) == 4
    await session.prompt("inspect")
    context = stream.requests[-1][1]
    prompt = get_current_system_prompt(context.messages)
    assert prompt.index("global instructions") < prompt.index("project instructions") < prompt.index("explicit instructions")
    assert "raw host instructions" in prompt
    assert "global review" in prompt
    assert "project review" not in prompt
    assert "- read:" in prompt and "- write:" in prompt
    assert "- bash:" not in prompt and "- edit:" not in prompt
    assert [tool.name for tool in get_current_tools(context.messages)] == ["read", "write"]


@pytest.mark.parametrize("tools,reader", [((), None), (("write",), None), (("bash",), "bash"), (("read",), "read")])
async def test_custom_base_retains_context_and_catalog_requires_actual_reader(tmp_path, tools, reader):
    (tmp_path / "AGENTS.md").write_text("Keep project context")
    visible = skill(tmp_path / "visible" / "SKILL.md", name="visible", description="Visible guidance")
    hidden = skill(tmp_path / "hidden" / "SKILL.md", name="hidden", description="Hidden guidance", hidden=True)
    stream = OfflineStream()
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=tools, custom_prompt="Custom base",
        skill_sources=(SkillSource(visible), SkillSource(hidden)),
    )).new_session()
    await session.prompt("work")
    prompt = get_current_system_prompt(stream.requests[-1][1].messages)
    assert "Custom base" in prompt and "Keep project context" in prompt
    assert "You are a coding assistant" not in prompt
    assert "Hidden guidance" not in prompt
    assert ("Visible guidance" in prompt) == (reader is not None)
    if reader:
        assert f"Use {'the read tool' if reader == 'read' else 'bash'} to load" in prompt


@pytest.mark.parametrize("entry", ["prompt", "steer", "follow_up"])
async def test_string_input_expands_skill_before_template_at_acceptance(tmp_path, entry):
    path = skill(tmp_path / "review" / "SKILL.md", hidden=True)
    template = tmp_path / "skill:review.md"
    template.write_text("template must not override skill")
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        skill_sources=(SkillSource(path),), template_sources=(PromptTemplateSource(template),),
    ))
    await runtime.new_session()
    if entry == "prompt":
        await runtime.prompt("/skill:review original args")
    else:
        getattr(runtime, entry)("/skill:review original args")
        path.write_text("body changed after input was queued")
        await runtime.prompt("start")
    users = user_texts(stream.requests[-1][1])
    expanded = next(text for text in users if "<skill " in text)
    assert "Review carefully" in expanded
    assert "References are relative to" in expanded
    assert "</skill>\n\noriginal args" in expanded
    assert "body changed" not in expanded
    assert "template must not override" not in expanded


async def test_templates_cache_until_reload_and_explicit_skills_reread_body(tmp_path):
    path = skill(tmp_path / "review" / "SKILL.md")
    template = tmp_path / "greet.md"
    template.write_text('Hello $1 / $2 / ${3:-default}')
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        skill_sources=(SkillSource(path),), template_sources=(PromptTemplateSource(template),),
    ))
    session = await runtime.new_session()
    template.write_text('Updated $1')
    skill(path, body="New skill body", description="New catalog description")
    await runtime.prompt('/greet "two words" second')
    assert user_texts(stream.requests[-1][1])[-1] == 'Hello two words / second / default'
    await session.prompt('/skill:review')
    assert "New skill body" in user_texts(stream.requests[-1][1])[-1]
    assert session.resources.skills[0].description == "Review code"
    await runtime.reload_resources()
    await runtime.prompt('/greet final')
    assert user_texts(stream.requests[-1][1])[-1] == 'Updated final'
    assert session.resources.skills[0].description == "New catalog description"


@pytest.mark.parametrize("entry", ["prompt", "steer", "follow_up"])
async def test_unknown_and_unreadable_skill_keeps_helper_passthrough_and_diagnostics(tmp_path, entry):
    path = skill(tmp_path / "review" / "SKILL.md")
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        skill_sources=(SkillSource(path), SkillSource("missing-skills")),
    ))
    session = await runtime.new_session()
    assert any(item.reason == "read_error" and item.path.endswith("missing-skills")
               for item in session.resources.diagnostics)
    path.unlink()
    for text in ("/skill:missing args", "/unknown args", "/skill:review args"):
        if entry == "prompt":
            await session.prompt(text)
        else:
            getattr(session, entry)(text)
            await session.prompt("start")
        assert text in user_texts(stream.requests[-1][1])
    assert len(session.input_diagnostics) == 1
    assert session.input_diagnostics[0].reason == "read_error"
    assert session.input_diagnostics[0].path == str(path)


async def test_busy_reload_preserves_sections_and_tool_batch_then_syncs_next_prompt(tmp_path):
    import asyncio

    from omh.llm.types import SystemMessage, ToolCall, UserMessage

    (tmp_path / "AGENTS.md").write_text("old project instructions")
    (tmp_path / "input.txt").write_text("read completed")
    path = skill(tmp_path / "review" / "SKILL.md", description="old catalog")
    entered, release = asyncio.Event(), asyncio.Event()

    async def before_tool(context, signal):
        entered.set()
        await release.wait()

    stream = OfflineStream([[ToolCall(id="read-call", name="read", arguments={"path": "input.txt"})]])
    options = CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("read",),
        skill_sources=(SkillSource(path),),
        agent_options=AgentOptions(initial_state=AgentInitialState(system_prompt="raw original"),
                                   before_tool_call=before_tool),
    )
    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session()
    waiter = asyncio.create_task(session.prompt([
        SystemMessage(content="raw appended", timestamp=1), UserMessage(content="read file", timestamp=1),
    ]))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        snapshot = session.agent.history
        (tmp_path / "AGENTS.md").write_text("new project instructions")
        options.tools = ("write",)
        options.skill_sources = ()
        await runtime.reload_resources()
        assert session.agent.history == snapshot
        assert [tool.name for tool in session.agent.state.tools] == ["write"]
        session.follow_up("finish current prompt")
    finally:
        release.set()
        await waiter
    assert len(stream.requests) == 3
    for _, context, _ in stream.requests:
        prompt = get_current_system_prompt(context.messages)
        assert "old project instructions" in prompt and "new project instructions" not in prompt
        assert "old catalog" in prompt
        assert "raw original" in prompt and "raw appended" in prompt
    assert [tool.name for tool in get_current_tools(stream.requests[0][1].messages)] == ["read"]
    assert [tool.name for tool in get_current_tools(stream.requests[1][1].messages)] == ["write"]
    results = [message for message in session.agent.state.messages if message.role == "toolResult"]
    assert len(results) == 1 and not results[0].is_error
    assert "read completed" in results[0].content[0].text
    session.steer("continue with old sections")
    await runtime.continue_()
    continued = get_current_system_prompt(stream.requests[-1][1].messages)
    assert "old project instructions" in continued and "new project instructions" not in continued
    await runtime.prompt("next new prompt")
    prompt = get_current_system_prompt(stream.requests[-1][1].messages)
    assert "new project instructions" in prompt and "old project instructions" not in prompt
    assert "old catalog" not in prompt
    assert "raw original" in prompt and "raw appended" in prompt
    assert "- write:" in prompt and "- read:" not in prompt


@pytest.mark.parametrize("command", ["/skill:review args", "/greet args"])
async def test_retry_uses_expanded_input_and_original_sections_after_reload(tmp_path, command):
    from omh.agent import RetryPolicy, RetryStartEvent
    from omh.llm.types import AssistantMessage, ErrorEvent, empty_usage
    from omh.llm.utils.event_stream import create_assistant_message_event_stream

    (tmp_path / "AGENTS.md").write_text("original resources")
    path = skill(tmp_path / "review" / "SKILL.md", body="original skill")
    template = tmp_path / "greet.md"
    template.write_text("original template $1")
    captured = OfflineStream()

    def stream(selected, context, options):
        if not captured.requests:
            captured.requests.append((selected, context, options))
            result = create_assistant_message_event_stream()
            result.push(ErrorEvent(reason="error", error=AssistantMessage(
                api=selected.api, provider=selected.provider, model=selected.id,
                content=[], usage=empty_usage(), stop_reason="error", timestamp=1,
                error_message="503 overloaded",
            )))
            return result
        return captured(selected, context, options)

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        skill_sources=(SkillSource(path),), template_sources=(PromptTemplateSource(template),),
        agent_options=AgentOptions(retry=RetryPolicy(base_delay_ms=0)),
    ))
    session = await runtime.new_session()

    async def reloader(event, signal):
        if isinstance(event, RetryStartEvent):
            path.unlink()
            template.write_text("changed template $1")
            (tmp_path / "AGENTS.md").write_text("changed resources")
            await session.reload_resources()

    session.agent.subscribe(reloader)
    await runtime.prompt(command)
    assert len(captured.requests) == 2
    first = user_texts(captured.requests[0][1])
    assert first == user_texts(captured.requests[1][1])
    if command.startswith("/skill:"):
        assert "original skill" in first[0]
    else:
        assert first == ["original template args"]
    assert not session.input_diagnostics
    for _, context, _ in captured.requests:
        prompt = get_current_system_prompt(context.messages)
        assert "original resources" in prompt and "changed resources" not in prompt
    await runtime.prompt("new prompt")
    prompt = get_current_system_prompt(captured.requests[-1][1].messages)
    assert "changed resources" in prompt and "original resources" not in prompt


async def test_failed_resource_preparation_leaves_session_and_destination_untouched(tmp_path):
    stream = OfflineStream()
    options = CodingAgentOptions(cwd=tmp_path, model=model(), stream_fn=stream, tools=("read",))
    runtime = AgentSessionRuntime(options)
    session = await runtime.new_session()
    destination = await session.save("backup.jsonl")
    raw = destination.read_bytes().rstrip(b"\n")
    destination.write_bytes(raw)
    snapshot = session.agent.history
    resources = session.resources
    sections = session.agent.state.system_sections
    options.tools = ("write",)
    options.context_dirs = ("invalid\0directory",)
    with pytest.raises(ValueError):
        await session.reload_resources()
    assert session.agent.history == snapshot
    assert session.resources == resources
    assert session.agent.state.system_sections == sections
    assert [tool.name for tool in session.agent.state.tools] == ["read"]
    with pytest.raises(ValueError):
        await runtime.open_session(destination)
    assert runtime.current_session is session
    assert destination.read_bytes() == raw
    assert not stream.requests
    await session.prompt("old session still usable")
    assert [tool.name for tool in get_current_tools(stream.requests[-1][1].messages)] == ["read"]


@pytest.mark.parametrize("entry", ["steer", "follow_up"])
async def test_queued_template_is_expanded_before_reload_and_keeps_images(tmp_path, entry):
    from omh.llm.types import ImageContent

    template = tmp_path / "greet.md"
    template.write_text('old $1')
    stream = OfflineStream()
    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        template_sources=(PromptTemplateSource("greet.md"),),
    ))
    session = await runtime.new_session()
    image = ImageContent(data="aW1hZ2U=", mime_type="image/png")
    getattr(runtime, entry)('/greet "two words"', [image])
    template.write_text('new $1')
    await runtime.reload_resources()
    await runtime.prompt("start")
    assert "old two words" in user_texts(stream.requests[-1][1])
    queued = next(message for message in stream.requests[-1][1].messages
                  if message.role == "user" and not isinstance(message.content, str)
                  and any(block.type == "image" for block in message.content))
    assert queued.content[-1] == image
    await session.prompt("/greet final")
    assert user_texts(stream.requests[-1][1])[-1] == "new final"


async def test_typed_messages_and_direct_sdk_input_keep_literal_slash_syntax(tmp_path):
    from omh.llm.types import UserMessage

    (tmp_path / "greet.md").write_text("expanded $1")
    stream = OfflineStream()
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        template_sources=(PromptTemplateSource("greet.md"),),
    )).new_session()
    literal = UserMessage(content="/greet literal", timestamp=1)
    await session.prompt(literal)
    assert user_texts(stream.requests[-1][1])[-1] == "/greet literal"
    session.steer(literal)
    await session.continue_()
    assert user_texts(stream.requests[-1][1])[-1] == "/greet literal"
    await session.agent.prompt("/greet SDK")
    assert user_texts(stream.requests[-1][1])[-1] == "/greet SDK"


async def test_reopen_syncs_current_resources_without_rewriting_history_or_raw_content(tmp_path):
    (tmp_path / "AGENTS.md").write_text("old context")
    path = skill(tmp_path / "review" / "SKILL.md", description="old skill catalog")
    original = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=("read",),
        skill_sources=(SkillSource(path),), session_file="reopen.jsonl",
        agent_options=AgentOptions(initial_state=AgentInitialState(system_prompt="raw preserved")),
    )).new_session()
    await original.prompt("first")
    history = original.agent.history
    await original.close()
    (tmp_path / "AGENTS.md").write_text("new context")
    stream = OfflineStream()
    reopened = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=("write",), custom_prompt="new custom base",
    )).open_session(tmp_path / "reopen.jsonl")
    assert reopened.agent.history == history and not stream.requests
    await reopened.prompt("new prompt")
    prompt = get_current_system_prompt(stream.requests[-1][1].messages)
    assert "new context" in prompt and "old context" not in prompt
    assert "new custom base" in prompt and "raw preserved" in prompt
    assert "old skill catalog" not in prompt
    assert "You are a coding assistant" not in prompt and "<rules>" not in prompt
    assert [tool.name for tool in get_current_tools(stream.requests[-1][1].messages)] == ["write"]


async def test_failed_skill_expansion_still_runs_template_helper_and_parse_diagnostics_survive_reload(tmp_path):
    path = skill(tmp_path / "review" / "SKILL.md")
    template = tmp_path / "skill:review.md"
    template.write_text("fallback template $1")
    broken = tmp_path / "broken.md"
    broken.write_text("---\ninvalid frontmatter line\n---\nbroken body")
    stream = OfflineStream()
    session = await AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=stream, tools=(),
        skill_sources=(SkillSource(path),),
        template_sources=(PromptTemplateSource(template), PromptTemplateSource(broken)),
    )).new_session()
    assert any(item.reason == "parse_error" and item.path == str(broken) for item in session.resources.diagnostics)
    path.unlink()
    await session.prompt("/skill:review args")
    assert user_texts(stream.requests[-1][1])[-1] == "fallback template args"
    assert session.input_diagnostics[0].reason == "read_error"
    await session.reload_resources()
    assert not session.resources.skills
    assert any(item.reason == "read_error" and item.path == str(path) for item in session.resources.diagnostics)
