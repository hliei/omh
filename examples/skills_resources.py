"""Host-selected skills and explicit input: python examples/skills_resources.py.

Fully offline; the host injects a catalog and submits freshly expanded text.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    SkillSource,
    build_system_sections,
    create_read_tool,
    expand_skill_command,
    format_skills_for_prompt,
    load_skills,
)
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    SystemMessage,
    TextContent,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream
from omh.llm.utils.transcript import get_current_system_prompt


async def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, hidden in [("inspect", False), ("release", True)]:
            directory = root / "skills" / name
            directory.mkdir(parents=True)
            (directory / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Instructions for {name}\n"
                f"disable-model-invocation: {str(hidden).lower()}\n---\n"
                f"Follow the {name} checklist.\n", encoding="utf-8",
            )
        resources = load_skills([SkillSource("skills", "project")], cwd=root)
        tools = [create_read_tool(root)]
        sections = build_system_sections(cwd=root, selected_tools=[tool.name for tool in tools])
        sections["skills"] = format_skills_for_prompt(resources.skills, tools=tools)
        expanded = expand_skill_command('/skill:release "version 1.0"', resources.skills)
        for diagnostic in [*resources.diagnostics, *expanded.diagnostics]:
            print(f"{diagnostic.path}: {diagnostic.reason}: {diagnostic.message}")

        def stream_fn(model, context, options):
            catalog = get_current_system_prompt(context.messages)
            assert "<name>inspect</name>" in catalog
            assert "<name>release</name>" not in catalog
            user = next(message for message in context.messages if isinstance(message, UserMessage))
            assert user.content == [TextContent(text=expanded.text)]
            assert "Follow the release checklist." in expanded.text
            print("Model received expanded input:\n", expanded.text)
            message = AssistantMessage(
                api=model.api, provider=model.provider, model=model.id,
                timestamp=1000, usage=empty_usage(), stop_reason="stop",
                content=[TextContent(text="Skill instructions received.")],
            )
            stream = create_assistant_message_event_stream()
            stream.push(DoneEvent(reason="stop", message=message))
            return stream

        model = Model(
            id="offline", name="Offline", provider="example", api="openai-completions",
            base_url="", reasoning=False, input=("text",), context_window=128000,
            max_tokens=1024, cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        )
        agent = Agent(AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(
                model=model, tools=tools,
                messages=[SystemMessage(content="", sections=sections, timestamp=1000)],
            ),
        ))
        await agent.prompt(expanded.text)
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
