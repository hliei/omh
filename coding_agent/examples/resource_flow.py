"""Run offline: python coding_agent/examples/resource_flow.py."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.agent import PromptTemplateSource, SkillSource
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream
from omh.llm.utils.transcript import get_current_system_prompt

from coding_agent import AgentSessionRuntime, CodingAgentOptions


async def main() -> None:
    model = Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",), context_window=100000, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    requests = []

    def stream_fn(selected, context, options):
        requests.append(context)
        response = AssistantMessage(
            api=selected.api, provider=selected.provider, model=selected.id,
            content=[TextContent(text="Offline response")], usage=empty_usage(),
            stop_reason="stop", timestamp=len(requests),
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason="stop", message=response))
        return stream

    with TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        (cwd / "AGENTS.md").write_text("Keep changes focused.")
        skill_dir = cwd / "skills" / "review"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\nname: review\ndescription: Review changes\n---\nInspect the code carefully.")
        prompts = cwd / "prompts"
        prompts.mkdir()
        template = prompts / "review.md"
        template.write_text("Review $1 and explain the result.")
        runtime = AgentSessionRuntime(CodingAgentOptions(
            cwd=cwd, model=model, stream_fn=stream_fn, tools=("read",),
            session_file="conversation.jsonl", custom_prompt="You review code.",
            skill_sources=(SkillSource("skills", "project"),),
            template_sources=(PromptTemplateSource("prompts", "project"),),
        ))
        session = await runtime.new_session()
        await runtime.prompt('/review "changed files"')
        runtime.follow_up("/skill:review focus on public behavior")
        skill_file.write_text("---\nname: review\ndescription: Updated catalog\n---\nNew instructions.")
        template.write_text("Updated review of $1.")
        await runtime.reload_resources()
        await runtime.prompt("Consume the accepted follow-up.")
        queued = [message for message in requests[-1].messages if message.role == "user"][-1]
        assert "Inspect the code carefully" in queued.content[0].text
        await runtime.prompt("/skill:review")
        explicit = [message for message in requests[-1].messages if message.role == "user"][-1]
        assert "New instructions" in explicit.content[0].text
        await runtime.prompt("/review final")
        last = [message for message in requests[-1].messages if message.role == "user"][-1]
        assert last.content[0].text == "Updated review of final."
        history = session.agent.history
        await session.close()
        reopened = await runtime.open_session("conversation.jsonl")
        assert reopened.agent.history == history
        await reopened.prompt("Reopened with current resources.")
        prompt = get_current_system_prompt(requests[-1].messages)
        assert "Keep changes focused" in prompt and "Updated catalog" in prompt
        print(f"Resource flow: {len(requests)} requests; {len(reopened.resources.skills)} skill; "
              f"{len(reopened.resources.templates)} template; {len(reopened.resources.diagnostics)} diagnostics")
        await reopened.close()


asyncio.run(main())
