from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from omh.llm.types import FetchRequest, FetchResponse


def sse_response(*chunks: Mapping[str, Any], status: int = 200) -> FetchResponse:
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return FetchResponse(
        status=status,
        headers={"content-type": "text/event-stream"},
        text="\n\n".join(lines) + "\n",
    )


def json_error_response(status: int, body: Mapping[str, Any]) -> FetchResponse:
    return FetchResponse(
        status=status,
        headers={"content-type": "application/json"},
        text=json.dumps(body),
    )


class RecordingFetch:
    def __init__(self, response: FetchResponse) -> None:
        self.response = response
        self.requests: list[FetchRequest] = []

    async def __call__(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        return self.response

    @property
    def body(self) -> dict[str, Any]:
        assert self.requests, "fetch was not called"
        return self.requests[0].json_body
