import json
from collections.abc import Mapping
from typing import Any

from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    FetchRequest,
    FetchResponse,
    Model,
    ModelCost,
    TextContent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


def model(id="offline", provider="test"):
    return Model(id=id, name=id, api="openai-completions", provider=provider,
                 base_url="", reasoning=True, input=("text", "image"),
                 context_window=100000, max_tokens=1024,
                 cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0))


class OfflineStream:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, selected_model, context, options):
        self.requests.append((selected_model, context, options))
        response = self.responses.pop(0) if self.responses else [TextContent(text="done")]
        message = AssistantMessage(
            api=selected_model.api, provider=selected_model.provider, model=selected_model.id,
            usage=empty_usage(), stop_reason="toolUse" if any(
                item.type == "toolCall" for item in response
            ) else "stop", timestamp=1000, content=response,
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
        return stream


def sse_response(*chunks: Mapping[str, Any]) -> FetchResponse:
    """Build a controlled OpenAI-compatible SSE response."""
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return FetchResponse(
        status=200, headers={"content-type": "text/event-stream"}, text="\n\n".join(lines) + "\n",
    )


def text_stream(text: str = "done") -> FetchResponse:
    """One SSE response whose final assistant message has the given text."""
    return sse_response({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})


def tool_call_stream(tool_call_id: str, name: str, arguments: Mapping[str, Any]) -> FetchResponse:
    """One SSE response that requests a tool call."""
    return sse_response(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": json.dumps(arguments)}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )


class SequencedFetch:
    """Controlled HTTP boundary returning queued responses in order."""

    def __init__(self, *responses: FetchResponse) -> None:
        self.responses = list(responses)
        self.requests: list[FetchRequest] = []

    async def __call__(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        assert self.responses, "SequencedFetch received more requests than queued responses"
        return self.responses.pop(0)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [request.json_body for request in self.requests]


class RecordingFetch:
    """Controlled HTTP boundary that records every request and returns a queued response."""

    def __init__(self, response: FetchResponse | None = None) -> None:
        self.response = response if response is not None else text_stream()
        self.requests: list[FetchRequest] = []

    async def __call__(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        return self.response
