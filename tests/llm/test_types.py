from __future__ import annotations

from omh.llm.types import ImageContent, Usage, UserMessage, empty_usage


def test_public_message_and_image_types() -> None:
    image = ImageContent(data="abc", mime_type="image/png")
    message = UserMessage(
        content=[image],
        timestamp=1,
    )
    assert message.role == "user"
    assert message.content[0].type == "image"
    assert message.content[0].mime_type == "image/png"


def test_usage_records_token_and_cost_fields() -> None:
    usage = empty_usage()
    assert usage.input == 0
    assert usage.output == 0
    assert usage.cache_read == 0
    assert usage.cache_write == 0
    assert usage.total_tokens == 0
    assert usage.cost.total == 0
    Usage(
        input=10,
        output=4,
        cache_read=2,
        cache_write=1,
        total_tokens=17,
        reasoning=3,
        cost=usage.cost,
    )
