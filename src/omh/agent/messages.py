"""Application messages and their conversion to model input."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from omh.llm.types import AbortSignal, Message


#: Application message union. The traditional Agent starts with the standard LLM
#: message roles; applications add their own roles by extending the conversion
#: boundary rather than by widening the LLM message union.
@runtime_checkable
class CustomAgentMessage(Protocol):
    """A message recognised by its ``role`` but outside the standard LLM roles.

    Custom messages live in the application history. The default model conversion
    filters them out; applications that want them in a request provide
    ``AgentOptions.convert_to_llm``.
    """

    role: str


AgentMessage = Message | CustomAgentMessage

#: Converts application messages to LLM messages before transcript normalization.
ConvertToLlm = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]

#: Transforms application history before conversion; runs before ``convert_to_llm``.
TransformContext = Callable[
    [list[AgentMessage], AbortSignal | None],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]
