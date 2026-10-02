"""Agent-only data boundaries around the standalone loop's open contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import cast

from omh.agent._async import maybe_await
from omh.agent.context import AgentContext
from omh.agent.data import snapshot_messages
from omh.agent.hooks import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentLoopTurnUpdate,
    AgentRequestUpdate,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareRequestContext,
)
from omh.agent.loop_config import AgentLoopConfig
from omh.agent.messages import ConvertToLlm, LoopMessage, TransformContext
from omh.agent.model_response import _default_convert_to_llm
from omh.agent.state import snapshot_tools
from omh.agent.tools import AgentTool, AgentToolPrepareArguments, AgentToolResult
from omh.llm.types import AbortSignal, AssistantMessage, Message, ToolResultMessage


def snapshot_context(context: AgentContext) -> AgentContext:
    return AgentContext(messages=list(snapshot_messages(context.messages)), tools=snapshot_tools(context.tools))


def _isolate_preparation(prepare: AgentToolPrepareArguments) -> AgentToolPrepareArguments:
    def isolated(args: dict[str, object]) -> dict[str, object]:
        return prepare(copy.deepcopy(args))
    return isolated


def isolate_tools(tools: list[AgentTool]) -> list[AgentTool]:
    return [replace(tool, prepare_arguments=_isolate_preparation(tool.prepare_arguments))
            if tool.prepare_arguments is not None else tool for tool in tools]


def snapshot_turn(turn: AgentTurnContext) -> AgentTurnContext:
    # Copy the message group together to preserve relationships within one view.
    message, results, messages, context_messages = copy.deepcopy((
        turn.message, turn.tool_results, turn.new_messages, turn.context.messages,
    ))
    return AgentTurnContext(
        message=message, tool_results=results, new_messages=messages,
        context=AgentContext(messages=context_messages, tools=snapshot_tools(turn.context.tools)),
    )


def _snapshot_result(result: AgentToolResult) -> AgentToolResult:
    message = ToolResultMessage(
        tool_call_id="", tool_name="", timestamp=0,
        content=result.content, details=result.details, usage=result.usage,
    )
    snapshot = cast(ToolResultMessage, snapshot_messages([message])[0])
    return AgentToolResult(
        content=snapshot.content, details=snapshot.details,
        usage=snapshot.usage, terminate=result.terminate,
    )


def isolate_loop_config(
    config: AgentLoopConfig,
    convert: ConvertToLlm | None,
    transform: TransformContext | None,
) -> AgentLoopConfig:
    """Keep request projections and callbacks separate from the loop's messages."""
    async def convert_messages(messages: list[LoopMessage]) -> list[Message]:
        snapshots = snapshot_messages(messages)
        if convert is None:
            return _default_convert_to_llm(list(snapshots))
        return copy.deepcopy(await maybe_await(convert(snapshots)))

    async def transform_messages(messages: list[LoopMessage], signal: AbortSignal | None) -> list[LoopMessage]:
        assert transform is not None
        projected = await maybe_await(transform(snapshot_messages(messages), signal))
        return list(snapshot_messages(projected))

    async def prepare(request: PrepareRequestContext, signal: AbortSignal | None) -> AgentRequestUpdate | None:
        assert config.prepare_request is not None
        update = await maybe_await(config.prepare_request(
            replace(request, context=snapshot_context(request.context), model=copy.deepcopy(request.model)), signal,
        ))
        if update is None:
            return None
        context = snapshot_context(update.context) if update.context is not None else None
        if context is not None:
            context.tools = isolate_tools(context.tools)
        return replace(update, context=context)

    async def finish(turn: AgentTurnContext, signal: AbortSignal | None) -> AgentTurnDecision | None:
        assert config.finish_turn is not None
        return await maybe_await(config.finish_turn(snapshot_turn(turn), signal))

    async def next_turn(turn: AgentTurnContext) -> AgentLoopTurnUpdate | None:
        assert config.prepare_next_turn is not None
        update = await maybe_await(config.prepare_next_turn(snapshot_turn(turn)))
        if update is None:
            return None
        context = snapshot_context(update.context) if update.context is not None else None
        if context is not None:
            context.tools = isolate_tools(context.tools)
        return replace(
            update, context=context,
            messages=list(snapshot_messages(update.messages)) if update.messages is not None else None,
        )

    async def before(context: BeforeToolCallContext, signal: AbortSignal | None) -> BeforeToolCallResult | None:
        assert config.before_tool_call is not None
        return await maybe_await(config.before_tool_call(replace(
            context, assistant_message=copy.deepcopy(context.assistant_message),
            tool_call=copy.deepcopy(context.tool_call), context=snapshot_context(context.context),
        ), signal))

    async def after(context: AfterToolCallContext, signal: AbortSignal | None) -> AfterToolCallResult:
        result = _snapshot_result(context.result)
        is_error = context.is_error
        if config.after_tool_call is not None:
            override = await maybe_await(config.after_tool_call(replace(
                context, assistant_message=copy.deepcopy(context.assistant_message),
                tool_call=copy.deepcopy(context.tool_call), result=_snapshot_result(result),
                context=snapshot_context(context.context),
            ), signal))
            if override is not None:
                result = _snapshot_result(AgentToolResult(
                    content=override.content if override.content is not None else result.content,
                    details=override.details if override.details is not None else result.details,
                    usage=override.usage if override.usage is not None else result.usage,
                    terminate=override.terminate if override.terminate is not None else result.terminate,
                ))
                if override.is_error is not None:
                    is_error = override.is_error
        return AfterToolCallResult(
            content=result.content, details=result.details, usage=result.usage,
            terminate=result.terminate, is_error=is_error,
        )

    def snapshot_response(message: AssistantMessage) -> AssistantMessage:
        return cast(AssistantMessage, snapshot_messages([message])[0])

    return replace(
        config, convert_to_llm=convert_messages,
        transform_context=transform_messages if transform is not None else None,
        prepare_request=prepare if config.prepare_request is not None else None,
        finish_turn=finish if config.finish_turn is not None else None,
        prepare_next_turn=next_turn if config.prepare_next_turn is not None else None,
        before_tool_call=before if config.before_tool_call is not None else None,
        after_tool_call=after, _snapshot_response=snapshot_response,
    )
