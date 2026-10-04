from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
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
