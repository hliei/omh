from dataclasses import replace

import pytest

from omh.llm.models import create_models
from omh.llm.providers.deepseek import deepseek_provider
from omh.llm.types import (
    AssistantMessage,
    Context,
    ImageContent,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    empty_usage,
)

from .http_samples import RecordingFetch, sse_response


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_images", [True, False])
async def test_tool_images_follow_all_consecutive_tool_results(supports_images: bool) -> None:
    models = create_models()
    models.set_provider(deepseek_provider())
    catalog_model = models.get_model("deepseek", "deepseek-flash")
    assert catalog_model is not None
    model = replace(catalog_model, input=("text", "image") if supports_images else ("text",))
    fetch = RecordingFetch(
        sse_response({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    context = Context(
        messages=[
            AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=empty_usage(),
                stop_reason="toolUse",
                timestamp=1,
                content=[
                    ToolCall(id="first", name="read", arguments={}),
                    ToolCall(id="second", name="read", arguments={}),
                ],
            ),
            ToolResultMessage(
                tool_call_id="first",
                tool_name="read",
                content=[ImageContent(data="QQ==", mime_type="image/png")],
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="second",
                tool_name="read",
                content=[TextContent(text="second image"), ImageContent(data="Qg==", mime_type="image/jpeg")],
                timestamp=3,
            ),
        ]
    )

    result = await models.complete_simple(model, context, SimpleStreamOptions(api_key="test", fetch=fetch))

    assert result.stop_reason == "stop"
    messages = fetch.body["messages"]
    assert [message["role"] for message in messages] == (
        ["assistant", "tool", "tool", "user"] if supports_images else ["assistant", "tool", "tool"]
    )
    assert [message["tool_call_id"] for message in messages[1:3]] == ["first", "second"]
    if supports_images:
        assert messages[1]["content"] == "(see attached image)"
        assert messages[2]["content"] == "second image"
        assert [part["image_url"]["url"] for part in messages[3]["content"] if part["type"] == "image_url"] == [
            "data:image/png;base64,QQ==",
            "data:image/jpeg;base64,Qg==",
        ]
    else:
        assert messages[1]["content"] == "(tool image omitted: model does not support images)"
        assert messages[2]["content"] == "second image\n(tool image omitted: model does not support images)"
