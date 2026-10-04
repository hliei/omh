"""Host-selected prompt templates: python examples/prompt_templates.py.

Fully offline; the host loads templates, expands input and submits plain text.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    PromptTemplateSource,
    expand_prompt_template,
    load_prompt_templates,
)
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


async def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        prompts = root / "prompts"
        prompts.mkdir()
        (prompts / "review.md").write_text(
            "---\ndescription: Review selected files\n---\n"
            "Review $1 with ${2:-care}. Remaining: ${@:3}. All: $ARGUMENTS",
            encoding="utf-8",
        )
        loaded = load_prompt_templates([PromptTemplateSource("prompts", "project")], cwd=root)
        for diagnostic in loaded.diagnostics:
            print(f"{diagnostic.path}: {diagnostic.reason}: {diagnostic.message}")
        text = expand_prompt_template('/review "two files $2"', loaded.templates)
        assert text == "Review two files $2 with care. Remaining: . All: two files $2"

        def stream_fn(model, context, options):
            user = next(message for message in context.messages if isinstance(message, UserMessage))
            assert user.content == [TextContent(text=text)]
            print("Model received expanded input:\n", text)
            message = AssistantMessage(
                api=model.api, provider=model.provider, model=model.id,
                timestamp=1000, usage=empty_usage(), stop_reason="stop",
                content=[TextContent(text="Template received.")],
            )
            stream = create_assistant_message_event_stream()
            stream.push(DoneEvent(reason="stop", message=message))
            return stream

        model = Model(
            id="offline", name="Offline", provider="example", api="openai-completions",
            base_url="", reasoning=False, input=("text",), context_window=128000,
            max_tokens=1024, cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        )
        agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=model)))
        await agent.prompt(text)
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
