from __future__ import annotations

from omh.llm.types import Context, SystemMessage, Tool, ToolReference, UserMessage
from omh.llm.utils.transcript import (
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_initial_system_message,
    normalize_context,
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def test_create_initial_system_message_empty() -> None:
    assert create_initial_system_message(None, None) is None
    assert create_initial_system_message("", []) is None


def test_create_initial_system_message_from_prompt_and_tools() -> None:
    tool = Tool(name="run", description="Run it", parameters={"type": "object"})
    message = create_initial_system_message("Base prompt", [tool])
    assert message is not None
    assert message.content == "Base prompt"
    assert message.tools_added == [tool]


def test_normalize_context_prepends_leading_system_message() -> None:
    context = normalize_context(
        Context(
            messages=[_user("hi")],
            system_prompt="system",
            tools=[Tool(name="t", description="d", parameters={})],
        )
    )
    assert isinstance(context.messages[0], SystemMessage)
    assert context.messages[0].content == "system"
    assert context.messages[1].role == "user"


def test_normalize_context_leaves_messages_without_prompt() -> None:
    context = normalize_context(Context(messages=[_user("hi")]))
    assert [message.role for message in context.messages] == ["user"]


def test_get_initial_system_message_only_reads_a_leading_message() -> None:
    leading = SystemMessage(content="lead", timestamp=0)
    assert get_initial_system_message([leading, _user("hi")]) is leading
    assert get_initial_system_message([_user("hi"), leading]) is None


def test_get_current_system_message_replays_content_sections_and_tools() -> None:
    messages = [
        SystemMessage(
            content="First",
            timestamp=0,
            sections={"style": "concise"},
            tools_added=[Tool(name="a", description="A", parameters={})],
        ),
        SystemMessage(content="Second", timestamp=1, sections={"style": None, "tone": "warm"}),
        SystemMessage(
            timestamp=2,
            content="",
            tools_removed=[ToolReference(name="a")],
            tools_added=[Tool(name="b", description="B", parameters={})],
        ),
    ]

    replayed = get_current_system_message(messages)
    assert replayed is not None
    assert "First" in replayed.content and "Second" in replayed.content
    assert replayed.sections == {"tone": "warm"}
    assert [tool.name for tool in replayed.tools_added or []] == ["b"]
    assert get_current_system_prompt(messages) == "First\n\nSecond\n\nwarm"
    assert [tool.name for tool in get_current_tools(messages)] == ["b"]


def test_get_current_system_message_returns_none_without_system_state() -> None:
    assert get_current_system_message([_user("hi")]) is None


def test_tool_redefinition_is_a_removal_followed_by_an_addition() -> None:
    from omh.llm.utils.transcript import get_tool_state_changes

    previous = [Tool(name="run", description="Old", parameters={"type": "object"})]
    current = [Tool(name="run", description="New", parameters={"type": "object"})]

    changes = get_tool_state_changes(previous, current)
    assert [tool.name for tool in changes.tools_added] == ["run"]
    assert [tool.description for tool in changes.tools_added] == ["New"]
    assert [tool.name for tool in changes.tools_removed] == ["run"]


def test_collapse_system_messages_replays_prompt_sections_and_tools() -> None:
    from omh.llm.types import TranscriptContext
    from omh.llm.utils.transcript import collapse_system_messages

    transcript = TranscriptContext(
        messages=[
            SystemMessage(
                content="Base",
                timestamp=0,
                sections={"style": "concise", "tone": "warm"},
                tools_added=[Tool(name="a", description="A", parameters={})],
            ),
            _user("hi"),
            SystemMessage(
                content="Later",
                timestamp=2,
                sections={"style": None},
                tools_removed=[ToolReference(name="a")],
                tools_added=[Tool(name="b", description="B", parameters={})],
            ),
        ]
    )

    collapsed = collapse_system_messages(transcript)
    assert [message.role for message in collapsed.messages] == ["system", "user"]
    head = collapsed.messages[0]
    assert isinstance(head, SystemMessage)
    assert get_current_system_prompt(collapsed.messages) == "Base\n\nLater\n\nwarm"
    assert [tool.name for tool in get_current_tools(collapsed.messages)] == ["b"]
