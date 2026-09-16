from __future__ import annotations

import asyncio
import json
import platform
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from omh.llm.api.simple_options import build_base_options
from omh.llm.api.transform_messages import transform_messages
from omh.llm.models import calculate_cost, clamp_thinking_level
from omh.llm.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    FetchFunction,
    FetchRequest,
    FetchResponse,
    MaxTokensField,
    Message,
    Model,
    OpenAICompletionsOptions,
    ProviderHeaders,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingFormat,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    empty_usage,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream
from omh.llm.utils.json_parse import parse_streaming_json

_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


@dataclass(frozen=True, slots=True)
class ResolvedOpenAICompletionsCompat:
    supports_store: bool
    supports_developer_role: bool
    supports_reasoning_effort: bool
    supports_usage_in_streaming: bool
    supports_finish_reason: bool
    max_tokens_field: MaxTokensField
    requires_tool_result_name: bool
    requires_assistant_after_tool_result: bool
    requires_thinking_as_text: bool
    requires_reasoning_content_on_assistant_messages: bool
    thinking_format: ThinkingFormat
    supports_strict_mode: bool


class OpenAICompletionsApi:
    def stream(
        self,
        model: Model,
        context: Context,
        options: OpenAICompletionsOptions | StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        completions_options = options if isinstance(options, OpenAICompletionsOptions) or options is None else OpenAICompletionsOptions(
            signal=options.signal,
            api_key=options.api_key,
            fetch=options.fetch,
            env=options.env,
            headers=options.headers,
            temperature=options.temperature,
            sampling_params=options.sampling_params,
            max_tokens=options.max_tokens,
            timeout_ms=options.timeout_ms,
            on_payload=options.on_payload,
        )
        return stream(model, context, completions_options)

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)


def openai_completions_api() -> OpenAICompletionsApi:
    return OpenAICompletionsApi()


def _user_agent() -> str:
    return f"omh ({platform.system()} {platform.release()}; {platform.machine()})"


def _has_header(headers: ProviderHeaders | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def _client_api_key(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> str:
    if api_key:
        return api_key
    if _has_header(headers, "authorization"):
        return "unused"
    raise ValueError(f"No API key for provider: {provider}")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def detect_compat(model: Model) -> ResolvedOpenAICompletionsCompat:
    provider = model.provider
    base_url = model.base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()
    is_non_standard = is_deepseek
    return ResolvedOpenAICompletionsCompat(
        supports_store=not is_non_standard,
        supports_developer_role=not is_non_standard,
        supports_reasoning_effort=True,
        supports_usage_in_streaming=True,
        supports_finish_reason=True,
        max_tokens_field="max_tokens" if is_deepseek else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        requires_reasoning_content_on_assistant_messages=is_deepseek,
        thinking_format="deepseek" if is_deepseek else "openai",
        supports_strict_mode=True,
    )


def get_compat(model: Model) -> ResolvedOpenAICompletionsCompat:
    detected = detect_compat(model)
    compat = model.compat
    if compat is None:
        return detected
    return ResolvedOpenAICompletionsCompat(
        supports_store=detected.supports_store if compat.supports_store is None else compat.supports_store,
        supports_developer_role=(
            detected.supports_developer_role
            if compat.supports_developer_role is None
            else compat.supports_developer_role
        ),
        supports_reasoning_effort=(
            detected.supports_reasoning_effort
            if compat.supports_reasoning_effort is None
            else compat.supports_reasoning_effort
        ),
        supports_usage_in_streaming=(
            detected.supports_usage_in_streaming
            if compat.supports_usage_in_streaming is None
            else compat.supports_usage_in_streaming
        ),
        supports_finish_reason=(
            detected.supports_finish_reason if compat.supports_finish_reason is None else compat.supports_finish_reason
        ),
        max_tokens_field=detected.max_tokens_field if compat.max_tokens_field is None else compat.max_tokens_field,
        requires_tool_result_name=(
            detected.requires_tool_result_name
            if compat.requires_tool_result_name is None
            else compat.requires_tool_result_name
        ),
        requires_assistant_after_tool_result=(
            detected.requires_assistant_after_tool_result
            if compat.requires_assistant_after_tool_result is None
            else compat.requires_assistant_after_tool_result
        ),
        requires_thinking_as_text=(
            detected.requires_thinking_as_text
            if compat.requires_thinking_as_text is None
            else compat.requires_thinking_as_text
        ),
        requires_reasoning_content_on_assistant_messages=(
            detected.requires_reasoning_content_on_assistant_messages
            if compat.requires_reasoning_content_on_assistant_messages is None
            else compat.requires_reasoning_content_on_assistant_messages
        ),
        thinking_format=detected.thinking_format if compat.thinking_format is None else compat.thinking_format,
        supports_strict_mode=(
            detected.supports_strict_mode if compat.supports_strict_mode is None else compat.supports_strict_mode
        ),
    )


def _convert_tools(tools: list[Tool], compat: ResolvedOpenAICompletionsCompat) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for tool in tools:
        function: dict[str, object] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            function["strict"] = False
        converted.append({"type": "function", "function": function})
    return converted


def _has_tool_history(messages: list[Message]) -> bool:
    for message in messages:
        if message.role == "toolResult":
            return True
        if message.role == "assistant" and any(block.type == "toolCall" for block in message.content):
            return True
    return False


def convert_messages(
    model: Model,
    context: Context,
    compat: ResolvedOpenAICompletionsCompat,
) -> list[dict[str, object]]:
    params: list[dict[str, object]] = []
    transformed = transform_messages(context.messages, model)
    if context.system_prompt:
        role = "developer" if model.reasoning and compat.supports_developer_role else "system"
        params.append({"role": role, "content": context.system_prompt})

    last_role: str | None = None
    tool_images: list[dict[str, object]] = []
    for message_index, message in enumerate(transformed):
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and message.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})
        if message.role == "user":
            if isinstance(message.content, str):
                params.append({"role": "user", "content": message.content})
            else:
                content: list[dict[str, object]] = []
                for item in message.content:
                    if item.type == "text":
                        content.append({"type": "text", "text": item.text})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                            }
                        )
                if content:
                    params.append({"role": "user", "content": content})
        elif message.role == "assistant":
            assistant: dict[str, object] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }
            text_parts = [block.text for block in message.content if block.type == "text" and block.text.strip()]
            assistant_text = "".join(text_parts)
            thinking_blocks = [
                block for block in message.content if block.type == "thinking" and block.thinking.strip()
            ]
            tool_calls = [block for block in message.content if block.type == "toolCall"]
            if thinking_blocks:
                if compat.requires_thinking_as_text:
                    thinking_text = "\n\n".join(block.thinking for block in thinking_blocks)
                    assistant["content"] = [{"type": "text", "text": thinking_text}, *[{"type": "text", "text": text} for text in text_parts]]
                else:
                    if assistant_text:
                        assistant["content"] = assistant_text
                    signature = thinking_blocks[0].thinking_signature
                    if signature in _REASONING_FIELDS:
                        assistant[signature] = "\n".join(block.thinking for block in thinking_blocks)
            elif assistant_text:
                assistant["content"] = assistant_text
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in tool_calls
                ]
            if (
                compat.requires_reasoning_content_on_assistant_messages
                and model.reasoning
                and "reasoning_content" not in assistant
            ):
                assistant["reasoning_content"] = ""
            content_value = assistant.get("content")
            has_content = (isinstance(content_value, str) and len(content_value) > 0) or (
                isinstance(content_value, list) and len(content_value) > 0
            )
            if has_content or assistant.get("tool_calls"):
                params.append(assistant)
        elif isinstance(message, ToolResultMessage):
            text_result = "\n".join(block.text for block in message.content if block.type == "text")
            has_images = any(block.type == "image" for block in message.content)
            tool_result_text = text_result or ("(see attached image)" if has_images else "(no tool output)")
            tool_result: dict[str, object] = {
                "role": "tool",
                "content": tool_result_text,
                "tool_call_id": message.tool_call_id,
            }
            if compat.requires_tool_result_name and message.tool_name:
                tool_result["name"] = message.tool_name
            params.append(tool_result)
            if "image" in model.input:
                tool_images.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                    }
                    for block in message.content
                    if block.type == "image"
                )
            next_message = transformed[message_index + 1] if message_index + 1 < len(transformed) else None
            if tool_images and (next_message is None or next_message.role != "toolResult"):
                if compat.requires_assistant_after_tool_result:
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Attached image(s) from tool result:"},
                            *tool_images,
                        ],
                    }
                )
                tool_images = []
                last_role = "user"
                continue
        last_role = message.role
    return params


def build_params(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None,
    compat: ResolvedOpenAICompletionsCompat,
) -> dict[str, object]:
    params: dict[str, object] = {
        "model": model.id,
        "messages": convert_messages(model, context, compat),
        "stream": True,
    }
    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}
    if compat.supports_store:
        params["store"] = False
    if options and options.max_tokens:
        params[compat.max_tokens_field] = options.max_tokens
    if options and options.temperature is not None:
        params["temperature"] = options.temperature
    if context.tools:
        params["tools"] = _convert_tools(context.tools, compat)
    elif _has_tool_history(context.messages):
        params["tools"] = []
    if options and options.tool_choice:
        params["tool_choice"] = options.tool_choice
    if compat.thinking_format == "deepseek" and model.reasoning:
        if options and options.reasoning_effort:
            params["thinking"] = {"type": "enabled"}
        elif model.thinking_level_map is None or "off" not in model.thinking_level_map or model.thinking_level_map["off"] is not None:
            params["thinking"] = {"type": "disabled"}
        if options and options.reasoning_effort and compat.supports_reasoning_effort:
            mapped = None if model.thinking_level_map is None else model.thinking_level_map.get(options.reasoning_effort)
            params["reasoning_effort"] = mapped if isinstance(mapped, str) else options.reasoning_effort
    elif options and options.reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        mapped = None if model.thinking_level_map is None else model.thinking_level_map.get(options.reasoning_effort)
        params["reasoning_effort"] = mapped if isinstance(mapped, str) else options.reasoning_effort
    if options and options.sampling_params:
        params.update(options.sampling_params)
    return params


def _request_headers(model: Model, api_key: str, options: OpenAICompletionsOptions | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "accept": "text/event-stream",
        "user-agent": _user_agent(),
    }
    if model.headers:
        headers.update(model.headers)
    if options and options.headers:
        for name, value in options.headers.items():
            if value is None:
                headers.pop(name, None)
                headers.pop(name.lower(), None)
            else:
                headers[name] = value
    return headers


@asynccontextmanager
async def _fetch_response(request: FetchRequest, fetch: FetchFunction | None) -> AsyncIterator[FetchResponse]:
    if fetch is not None:
        yield await fetch(request)
        return

    import httpx

    timeout = None if request.timeout_ms is None else request.timeout_ms / 1000
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(request.method, request.url, headers=request.headers, json=request.json_body) as response:
            if response.status_code >= 400:
                await response.aread()
            yield FetchResponse(
                status=response.status_code,
                headers=dict(response.headers),
                text=response.text if response.status_code >= 400 else "",
                lines=response.aiter_lines(),
            )


def _map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason is None:
        return "stop", None
    if reason in {"stop", "end"}:
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in {"function_call", "tool_calls"}:
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    return "error", f"Provider finish_reason: {reason}"


def parse_chunk_usage(raw_usage: dict[str, Any], model: Model) -> Usage:
    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    details = raw_usage.get("prompt_tokens_details") or {}
    completion_details = raw_usage.get("completion_tokens_details") or {}
    cached = 0
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        cached = int(details["cached_tokens"] or 0)
    elif raw_usage.get("prompt_cache_hit_tokens") is not None:
        cached = int(raw_usage["prompt_cache_hit_tokens"] or 0)
    elif raw_usage.get("cached_tokens") is not None:
        cached = int(raw_usage["cached_tokens"] or 0)
    cache_read = cached
    cache_write = int(details.get("cache_write_tokens") or 0) if isinstance(details, dict) else 0
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    output_tokens = int(raw_usage.get("completion_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0) if isinstance(completion_details, dict) else 0
    usage = empty_usage()
    usage.input = input_tokens
    usage.output = output_tokens
    usage.cache_read = cache_read
    usage.cache_write = cache_write
    usage.reasoning = reasoning
    usage.total_tokens = input_tokens + output_tokens + cache_read + cache_write
    calculate_cost(model, usage)
    return usage


def _format_error(error: object) -> str:
    return str(error)


def stream(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> AssistantMessageEventStream:
    events = AssistantMessageEventStream()

    async def run() -> None:
        output = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=empty_usage(),
            stop_reason="pending",
            timestamp=_now_ms(),
        )
        task = asyncio.current_task()
        assert task is not None
        signal = options.signal if options else None

        def cancel_request() -> None:
            task.cancel()

        try:
            if signal is not None:
                signal.throw_if_aborted()
                signal.add_callback(cancel_request)
            api_key = _client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)
            compat = get_compat(model)
            params = build_params(model, context, options, compat)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if isawaitable(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = next_params
            request = FetchRequest(
                method="POST",
                url=f"{model.base_url.rstrip('/')}/chat/completions",
                headers=_request_headers(model, api_key, options),
                json_body=params,
                timeout_ms=options.timeout_ms if options else None,
            )
            async with _fetch_response(request, options.fetch if options else None) as response:
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {response.text}")
                events.push(StartEvent(partial=output))

                text_block: TextContent | None = None
                thinking_block: ThinkingContent | None = None
                has_finish_reason = False
                tool_blocks_by_index: dict[int, ToolCall] = {}
                tool_blocks_by_id: dict[str, ToolCall] = {}
                partial_args: dict[int, str] = {}

                def content_index(block: TextContent | ThinkingContent | ToolCall) -> int:
                    return output.content.index(block)

                def finish_block(block: TextContent | ThinkingContent | ToolCall) -> None:
                    index = content_index(block)
                    if isinstance(block, TextContent):
                        events.push(TextEndEvent(content_index=index, content=block.text, partial=output))
                    elif isinstance(block, ThinkingContent):
                        events.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=output))
                    else:
                        raw = partial_args.get(index)
                        if raw is not None:
                            block.arguments = parse_streaming_json(raw)
                        events.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))

                def ensure_text() -> TextContent:
                    nonlocal text_block
                    if text_block is None:
                        text_block = TextContent(text="")
                        output.content.append(text_block)
                        events.push(TextStartEvent(content_index=content_index(text_block), partial=output))
                    return text_block

                def ensure_thinking(signature: str) -> ThinkingContent:
                    nonlocal thinking_block
                    if thinking_block is None:
                        thinking_block = ThinkingContent(thinking="", thinking_signature=signature)
                        output.content.append(thinking_block)
                        events.push(ThinkingStartEvent(content_index=content_index(thinking_block), partial=output))
                    return thinking_block

                def ensure_tool(delta: dict[str, Any]) -> ToolCall:
                    stream_index = delta.get("index")
                    function = delta.get("function") or {}
                    name = function.get("name") or ""
                    block = tool_blocks_by_index.get(stream_index) if isinstance(stream_index, int) else None
                    tool_id = delta.get("id")
                    if block is None and isinstance(tool_id, str):
                        block = tool_blocks_by_id.get(tool_id)
                    if block is None:
                        block = ToolCall(id=tool_id if isinstance(tool_id, str) else "", name=name, arguments={})
                        output.content.append(block)
                        index = content_index(block)
                        partial_args[index] = ""
                        if isinstance(stream_index, int):
                            tool_blocks_by_index[stream_index] = block
                        if isinstance(tool_id, str) and tool_id:
                            tool_blocks_by_id[tool_id] = block
                        events.push(ToolCallStartEvent(content_index=index, partial=output))
                    if isinstance(tool_id, str) and tool_id:
                        block.id = tool_id
                        tool_blocks_by_id[tool_id] = block
                    if name:
                        block.name = name
                    return block

                async for line in response.aiter_lines():
                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    stripped = line.strip()
                    if not stripped or not stripped.startswith("data:"):
                        continue
                    payload = stripped[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    chunk_id = chunk.get("id")
                    if isinstance(chunk_id, str) and chunk_id and not output.response_id:
                        output.response_id = chunk_id
                    chunk_model = chunk.get("model")
                    if isinstance(chunk_model, str) and chunk_model and chunk_model != model.id and not output.response_model:
                        output.response_model = chunk_model
                    if isinstance(chunk.get("usage"), dict):
                        output.usage = parse_chunk_usage(chunk["usage"], model)
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        output.raw_stop_reason = str(finish_reason)
                        stop_reason, error_message = _map_stop_reason(str(finish_reason))
                        output.stop_reason = stop_reason
                        if error_message:
                            output.error_message = error_message
                        has_finish_reason = True
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        block = ensure_text()
                        block.text += content
                        events.push(
                            TextDeltaEvent(content_index=content_index(block), delta=content, partial=output)
                        )
                    found_reasoning: str | None = None
                    for field in _REASONING_FIELDS:
                        value = delta.get(field)
                        if isinstance(value, str) and value:
                            found_reasoning = field
                            break
                    if found_reasoning:
                        value = delta[found_reasoning]
                        if isinstance(value, str) and value:
                            thinking = ensure_thinking(found_reasoning)
                            thinking.thinking += value
                            events.push(
                                ThinkingDeltaEvent(content_index=content_index(thinking), delta=value, partial=output)
                            )
                    tool_calls = delta.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tool_delta in tool_calls:
                            if not isinstance(tool_delta, dict):
                                continue
                            tool_block = ensure_tool(tool_delta)
                            index = content_index(tool_block)
                            function = tool_delta.get("function") or {}
                            arguments = function.get("arguments") if isinstance(function, dict) else None
                            delta_text = arguments if isinstance(arguments, str) else ""
                            partial_args[index] = partial_args.get(index, "") + delta_text
                            tool_block.arguments = parse_streaming_json(partial_args[index])
                            events.push(
                                ToolCallDeltaEvent(content_index=index, delta=delta_text, partial=output)
                            )

            for item in list(output.content):
                finish_block(item)
            if options and options.signal and options.signal.aborted:
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "aborted":
                raise RuntimeError("Request was aborted")
            if not has_finish_reason and not compat.supports_finish_reason:
                output.stop_reason = "toolUse" if any(block.type == "toolCall" for block in output.content) else "stop"
            if output.stop_reason == "error":
                raise RuntimeError(output.error_message or "Provider returned an error stop reason")
            if (compat.supports_finish_reason and not has_finish_reason) or output.stop_reason == "pending":
                raise RuntimeError("Stream ended without finish_reason")
            if output.stop_reason not in {"stop", "length", "toolUse"}:
                raise RuntimeError(output.error_message or "Stream ended without a successful stop reason")
            events.push(DoneEvent(reason=output.stop_reason, message=output))
            events.end()
        except (Exception, asyncio.CancelledError) as error:
            output.stop_reason = "aborted" if options and options.signal and options.signal.aborted else "error"
            output.error_message = "Request was aborted" if output.stop_reason == "aborted" else _format_error(error)
            events.push(ErrorEvent(reason=output.stop_reason, error=output))
            events.end()

        finally:
            if signal is not None:
                signal.remove_callback(cancel_request)

    asyncio.get_running_loop().create_task(run())
    return events


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    _client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)
    base = build_base_options(model, context, options, options.api_key if options else None)
    clamped = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped == "off" else clamped
    return stream(
        model,
        context,
        OpenAICompletionsOptions(
            temperature=base.temperature,
            sampling_params=base.sampling_params,
            max_tokens=base.max_tokens,
            signal=base.signal,
            api_key=base.api_key,
            fetch=base.fetch,
            headers=base.headers,
            on_payload=base.on_payload,
            timeout_ms=base.timeout_ms,
            env=base.env,
            tool_choice=options.tool_choice if options else None,
            reasoning_effort=reasoning_effort,
        ),
    )
