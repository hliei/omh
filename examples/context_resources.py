"""Explicit context loading and system sections: python examples/context_resources.py.

Fully offline; temporary project resources are injected through SystemMessage.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    build_system_sections,
    create_read_tool,
    load_project_context_files,
)
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    SystemMessage,
    TextContent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream
from omh.llm.utils.transcript import get_current_system_prompt


async def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        global_dir, project = root / "global", root / "project"
        global_dir.mkdir()
        project.mkdir()
        (global_dir / "AGENTS.md").write_text("Explain changes clearly.", encoding="utf-8")
        (project / "AGENTS.md").write_text("Use project conventions.", encoding="utf-8")
        resources = load_project_context_files(cwd=project, agent_dir=global_dir)
        for diagnostic in resources.diagnostics:
            print(f"{diagnostic.path}: {diagnostic.reason}: {diagnostic.message}")
        sections = build_system_sections(
            cwd=project,
            context_files=resources.files,
            custom_prompt="Help maintain this project.",
            selected_tools=["read"],
        )

        def stream_fn(model, context, options):
            prompt = get_current_system_prompt(context.messages)
            assert "Help maintain this project." in prompt
            assert "Explain changes clearly." in prompt
            assert "Use project conventions." in prompt
            print("Model received:\n", prompt)
            message = AssistantMessage(
                api=model.api, provider=model.provider, model=model.id,
                timestamp=1000, usage=empty_usage(), stop_reason="stop",
                content=[TextContent(text="Project context received.")],
            )
            stream = create_assistant_message_event_stream()
            stream.push(DoneEvent(reason="stop", message=message))
            return stream

        model = Model(
            id="offline", name="Offline", provider="example", api="openai-completions",
            base_url="", reasoning=False, input=("text",),
            context_window=128000, max_tokens=1024,
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        )
        agent = Agent(AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(
                model=model, tools=[create_read_tool(project)],
                messages=[SystemMessage(content="", sections=dict(sections), timestamp=1000)],
            ),
        ))
        await agent.prompt("Inspect the project instructions.")
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
