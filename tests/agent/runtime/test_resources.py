from __future__ import annotations

from pathlib import Path

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    AgentHarnessResources,
    BeforeRunHook,
    BranchScan,
    ConfigUpdateEvent,
    Context,
    DriveOptions,
    EntryAddedEvent,
    Err,
    InvalidMessage,
    LocalExecutionEnv,
    MemorySessionRepo,
    PromptTemplate,
    PromptTemplateRequest,
    RunEndEvent,
    RunStartEvent,
    SessionCreateOptions,
    Skill,
    SkillRequest,
    UnknownSkill,
    UnknownTemplate,
    format_skill_invocation,
    load_prompt_templates,
    load_skills,
)
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    Model,
    StartEvent,
    TextContent,
    Usage,
    UsageCost,
)
from omh.llm import Context as LlmContext

NOW = 1_700_000_000_000
MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="test",
    base_url="https://example.invalid",
    reasoning=False,
    input=("text",),
    cost=UsageCost(),
    context_window=8_192,
    max_tokens=1_024,
)
USAGE = Usage(
    input=1,
    output=1,
    cache_read=0,
    cache_write=0,
    total_tokens=2,
    cost=UsageCost(),
)


class RecordingModels:
    def __init__(self) -> None:
        self.contexts: list[LlmContext] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (MODEL.provider, MODEL.id):
            return MODEL
        return None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is MODEL
        self.contexts.append(context)
        message = AssistantMessage(
            api=MODEL.api,
            provider=MODEL.provider,
            model=MODEL.id,
            usage=USAGE,
            stop_reason="stop",
            timestamp=NOW,
            content=[TextContent(text=f"answer {len(self.contexts)}")],
        )
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=message))
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end()
        return stream


def _message_text(message: object) -> str:
    content = getattr(message, "content")
    if isinstance(content, str):
        return content
    return "".join(item.text for item in content if isinstance(item, TextContent))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def _create(
    models: RecordingModels, resources: AgentHarnessResources
) -> tuple[MemorySessionRepo, AgentHarness, RecordingModels]:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session, models=models, model=MODEL, resources=resources
        ),
        BACKGROUND_CONTEXT,
    )
    return repo, created.harness, models


async def test_default_resources_are_empty_and_unknown_names_return_err() -> None:
    repo, harness, models = await _create(RecordingModels(), AgentHarnessResources())
    lane = await harness.lane("main", BACKGROUND_CONTEXT)

    assert await harness.get_resources(BACKGROUND_CONTEXT) == AgentHarnessResources()

    skill_result = await lane.skill("missing", None, BACKGROUND_CONTEXT)
    assert isinstance(skill_result, Err)
    assert skill_result.error == UnknownSkill(name="missing")

    template_result = await lane.prompt_from_template(
        "missing", None, BACKGROUND_CONTEXT
    )
    assert isinstance(template_result, Err)
    assert template_result.error == UnknownTemplate(name="missing")

    admission = await lane.accept(SkillRequest(name="missing"), BACKGROUND_CONTEXT)
    assert isinstance(admission, Err)
    assert admission.error == UnknownSkill(name="missing")
    template_admission = await lane.accept(
        PromptTemplateRequest(name="missing"), BACKGROUND_CONTEXT
    )
    assert isinstance(template_admission, Err)
    assert template_admission.error == UnknownTemplate(name="missing")

    assert models.contexts == []

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_registered_skill_reports_hook_events_and_result() -> None:
    skill = Skill(
        name="greet",
        description="Say hello",
        content="Greet the user.",
        file_path="/skills/greet/SKILL.md",
    )
    resources = AgentHarnessResources(skills=(skill,))
    repo, harness, models = await _create(RecordingModels(), resources)
    lane = await harness.lane("main", BACKGROUND_CONTEXT)

    hook_events: list[BeforeRunHook] = []
    events: list[object] = []

    async def capture_hook(event: object, _context: Context) -> None:
        assert isinstance(event, BeforeRunHook)
        hook_events.append(event)

    async def capture_event(event: object, _context: Context) -> None:
        events.append(event)

    harness.hooks.on("before_run", capture_hook)
    for name in ("run_start", "entry_added", "run_end"):
        harness.events.on(name, capture_event)

    result = await lane.skill("greet", "Be brief.", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.kind == "run"
    assert result.value.status == "completed"
    assert len(models.contexts) == 1
    user_message = models.contexts[0].messages[0]
    assert _message_text(user_message) == format_skill_invocation(skill, "Be brief.")
    assert hook_events[0].resources == resources
    assert hook_events[0].prompt == (user_message,)
    assert any(isinstance(event, RunStartEvent) for event in events)
    assert any(isinstance(event, RunEndEvent) for event in events)
    assert any(isinstance(event, EntryAddedEvent) for event in events)

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_prompt_from_template_substitutes_args_and_rejects_empty_content() -> None:
    template = PromptTemplate(
        name="review", content="Review $1 against $2", description="Review"
    )
    empty = PromptTemplate(name="empty", content="", description=None)
    resources = AgentHarnessResources(prompt_templates=(template, empty))
    repo, harness, models = await _create(RecordingModels(), resources)
    lane = await harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt_from_template("review", ("a", "b"), BACKGROUND_CONTEXT)

    assert result.ok is True
    assert _message_text(models.contexts[0].messages[0]) == "Review a against b"

    empty_result = await lane.prompt_from_template("empty", None, BACKGROUND_CONTEXT)
    assert isinstance(empty_result, Err)
    assert empty_result.error == InvalidMessage(reason="empty")
    assert len(models.contexts) == 1

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_primitive_accept_and_drive_match_the_skill_convenience() -> None:
    skill = Skill(
        name="greet",
        description="Say hello",
        content="Greet the user.",
        file_path="/skills/greet/SKILL.md",
    )
    resources = AgentHarnessResources(skills=(skill,))
    repo, harness, models = await _create(RecordingModels(), resources)
    lane = await harness.lane("main", BACKGROUND_CONTEXT)

    admission = await lane.accept(SkillRequest(name="greet"), BACKGROUND_CONTEXT)
    assert admission.ok is True
    driven = await lane.drive(
        DriveOptions(operation_id=admission.value.operation_id, wait_for_retry=True),
        BACKGROUND_CONTEXT,
    )

    assert driven.ok is True
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "completed"
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [
        _message_text(entry.message) for entry in history if entry.type == "message"
    ] == [format_skill_invocation(skill), "answer 1"]
    assert len(models.contexts) == 1

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_set_resources_updates_lookup_and_emits_config_update() -> None:
    repo, harness, _models = await _create(RecordingModels(), AgentHarnessResources())
    lane = await harness.lane("main", BACKGROUND_CONTEXT)
    config_events: list[ConfigUpdateEvent] = []

    async def capture(event: object, _context: Context) -> None:
        assert isinstance(event, ConfigUpdateEvent)
        config_events.append(event)

    harness.events.on("config_update", capture)

    skill = Skill(
        name="greet",
        description="Say hello",
        content="Greet the user.",
        file_path="/skills/greet/SKILL.md",
    )
    updated = AgentHarnessResources(skills=(skill,))
    await harness.set_resources(updated, BACKGROUND_CONTEXT)

    assert await harness.get_resources(BACKGROUND_CONTEXT) == updated
    assert config_events == [ConfigUpdateEvent(property="resources")]
    result = await lane.skill("greet", None, BACKGROUND_CONTEXT)
    assert result.ok is True

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_temporary_skill_files_persist_as_ordinary_messages(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "skills" / "greet" / "SKILL.md",
        "---\ndescription: Say hello\n---\nGreet the user.",
    )
    env = LocalExecutionEnv(str(tmp_path))
    skills = await load_skills(env, str(tmp_path / "skills"), BACKGROUND_CONTEXT)
    assert skills.diagnostics == ()
    skill = skills.skills[0]

    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=RecordingModels(),
            model=MODEL,
            resources=AgentHarnessResources(skills=(skill,)),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.skill("greet", None, BACKGROUND_CONTEXT)

    assert result.ok is True
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    texts = [
        _message_text(entry.message) for entry in history if entry.type == "message"
    ]
    assert texts[0] == format_skill_invocation(skill)
    assert texts[1] == "answer 1"

    await created.harness.close(BACKGROUND_CONTEXT)
    reopened_session = await repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session, models=RecordingModels(), model=MODEL
        ),
        BACKGROUND_CONTEXT,
    )
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)
    reopened_history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )

    assert [
        _message_text(entry.message)
        for entry in reopened_history
        if entry.type == "message"
    ] == texts

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_temporary_prompt_template_files_expand_arguments(tmp_path: Path) -> None:
    _write(
        tmp_path / "prompts" / "review.md",
        "---\ndescription: Review a change\n---\nReview $1 against $2",
    )
    env = LocalExecutionEnv(str(tmp_path))
    prompts = await load_prompt_templates(
        env, str(tmp_path / "prompts"), BACKGROUND_CONTEXT
    )
    assert prompts.diagnostics == ()

    repo, harness, models = await _create(
        RecordingModels(),
        AgentHarnessResources(prompt_templates=prompts.prompt_templates),
    )
    lane = await harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt_from_template(
        "review", ("main", "beta"), BACKGROUND_CONTEXT
    )

    assert result.ok is True
    assert _message_text(models.contexts[0].messages[0]) == "Review main against beta"

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
