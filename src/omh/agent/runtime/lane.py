from __future__ import annotations

import time
from asyncio import Lock, sleep
from dataclasses import replace
from typing import Literal, cast

from omh.agent.agent_harness import (
    AgentHarnessOptions,
    CurrentOperationInfo,
    DriveOptions,
    DriveResult,
    InvalidMessage,
    LaneBusy,
    LaneExecutionInfo,
    NothingToResume,
    OperationAdmission,
    OperationAdmissionResult,
    OperationError,
    OperationMismatch,
    OperationResultRecord,
    PromptRequest,
    ResumeResult,
    RunResult,
    SettledDriveOutcome,
    WaitingDriveOutcome,
)
from omh.agent.context import Context
from omh.agent.result import err, ok
from omh.agent.runtime.codec import decode_assistant_frame, encode_assistant_frame
from omh.agent.runtime.retry import is_retryable_assistant_error, retry_delay_ms
from omh.agent.session.commit import insert_entry, insert_usage
from omh.agent.session.types import (
    BranchScan,
    Entry,
    NewMessageEntry,
    SessionMutator,
    UsageRow,
    Write,
)
from omh.agent.session.values import (
    ListCursor,
    ListReadOptions,
    append_list,
    branch_tip,
    delete_list,
    delete_value,
    lane_config,
    lane_state,
    operation_meta,
    operation_result,
    operation_state,
    pending_assistant_frames,
    set_value,
)
from omh.agent.types import AgentMessage
from omh.agent.utils.usage import empty_usage
from omh.llm.models import Models
from omh.llm.types import (
    AssistantMessage,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    UserMessage,
)
from omh.llm.types import (
    Context as LlmContext,
)
from omh.llm.types import (
    ThinkingLevel as LlmThinkingLevel,
)
from omh.llm.utils.assistant_message_frame import (
    AssistantMessageFrame,
    AssistantMessageFrameEncoder,
    reduce_assistant_message_frames,
)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _normalize_prompt(prompt: str | AgentMessage | list[AgentMessage]) -> list[AgentMessage]:
    if isinstance(prompt, str):
        if not prompt:
            return []
        return [UserMessage(content=[TextContent(text=prompt)], timestamp=_now_ms())]
    if isinstance(prompt, list):
        return list(prompt)
    return [prompt]


class AgentLane:
    def __init__(self, name: str, options: AgentHarnessOptions) -> None:
        self.name = name
        self._options = options
        self._drive_lock = Lock()

    async def accept(self, request: PromptRequest, context: Context) -> OperationAdmissionResult:
        messages = _normalize_prompt(request.prompt)
        if not messages:
            return err(InvalidMessage(reason="empty"))
        operation_id = request.operation_id or self._options.session.id_generator.next()
        started_at = _now_ms()
        entry_ids = [self._options.session.id_generator.next(started_at) for _ in messages]

        async def accept(mutator: SessionMutator, mutation_context: Context) -> OperationAdmissionResult:
            stored_lane = await mutator.get_value(lane_state(self.name), mutation_context)
            stored_tip = await mutator.get_value(branch_tip(self.name), mutation_context)
            if stored_lane is None or stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing durable state")
            durable_lane = cast(dict[str, object], stored_lane.value)
            current = durable_lane.get("currentOperationId")
            if isinstance(current, str):
                return err(LaneBusy(operation_id=current))

            parent_id = stored_tip.value
            entries: list[NewMessageEntry] = []
            for entry_id, message in zip(entry_ids, messages, strict=True):
                entries.append(NewMessageEntry(id=entry_id, parent_id=parent_id, message=message))
                parent_id = entry_id
            meta: object = {
                "operationId": operation_id,
                "lane": self.name,
                "sourceTipId": stored_tip.value,
                "startedAt": started_at,
                "intent": {"kind": "run", "promptEntryIds": entry_ids},
            }
            state: object = {
                "at": "starting",
                "latestAssistantEntryId": None,
                "control": {"status": "running"},
                "settings": {
                    "compaction": {
                        "enabled": True,
                        "reserveTokens": 16_384,
                        "keepRecentTokens": 20_000,
                    },
                    "steeringMode": "all",
                    "followUpMode": "all",
                    "toolExecution": "parallel",
                },
            }
            next_lane: object = {
                "currentOperationId": operation_id,
                "lastOperationId": durable_lane.get("lastOperationId"),
                "inbox": durable_lane.get("inbox", []),
            }
            await mutator.commit(
                [
                    *(insert_entry(entry) for entry in entries),
                    set_value(branch_tip(self.name), parent_id),
                    set_value(operation_meta(operation_id), meta),
                    set_value(operation_state(operation_id), state),
                    set_value(lane_state(self.name), next_lane),
                ],
                mutation_context,
            )
            return ok(OperationAdmission(operation_id=operation_id, kind="run", started_at=started_at))

        return await self._options.session.mutate(accept, context)

    async def drive(self, options: DriveOptions, context: Context) -> DriveResult:
        async with self._drive_lock:
            existing = await self.get_result(options.operation_id, context)
            if existing is not None:
                return ok(SettledDriveOutcome(outcome=existing))
            execution = await self.inspect_execution(context)
            if execution.current is None or execution.current.operation_id != options.operation_id:
                return err(
                    OperationMismatch(
                        expected_operation_id=None if execution.current is None else execution.current.operation_id,
                        operation_id=options.operation_id,
                    )
                )
            while True:
                current = await self.inspect_execution(context)
                if current.current is None:
                    result = await self.get_result(options.operation_id, context)
                    if result is None:
                        raise RuntimeError(f"Operation {options.operation_id!r} ended without a result")
                    return ok(SettledDriveOutcome(outcome=result))
                match current.current.at:
                    case "starting":
                        await self._start_operation(options.operation_id, context)
                    case "checkpoint":
                        outcome = await self._advance_checkpoint(options.operation_id, context)
                        if outcome is not None:
                            return ok(SettledDriveOutcome(outcome=outcome))
                    case "assistant.ready":
                        await self._run_generation(options.operation_id, context)
                    case "assistant.retry_wait":
                        waiting = await self._advance_retry_wait(options, context)
                        if waiting is not None:
                            return ok(waiting)
                    case "assistant.effect_pending":
                        await self._recover_generation(options.operation_id, context)
                    case other:
                        raise RuntimeError(f"Unsupported operation state: {other}")

    async def prompt(
        self,
        prompt: str | AgentMessage | list[AgentMessage],
        context: Context,
    ) -> RunResult:
        admission = await self.accept(PromptRequest(prompt=prompt), context)
        if not admission.ok:
            return err(admission.error)
        driven = await self.drive(
            DriveOptions(operation_id=admission.value.operation_id, wait_for_retry=True),
            context,
        )
        if not driven.ok:
            return err(driven.error)
        if driven.value.kind != "settled":
            raise RuntimeError("Prompt returned a retry wait despite wait_for_retry=True")
        return ok(driven.value.outcome)

    async def _start_operation(self, operation_id: str, context: Context) -> None:
        async def transition(mutator: SessionMutator, mutation_context: Context) -> None:
            durable_lane, meta, state = await self._read_operation(mutator, operation_id, mutation_context)
            if state.get("at") != "starting":
                return
            intent = cast(dict[str, object], meta["intent"])
            prompt_entry_ids = cast(list[str], intent["promptEntryIds"])
            trigger_entry_id = prompt_entry_ids[-1] if prompt_entry_ids else cast(str | None, meta["sourceTipId"])
            if trigger_entry_id is None:
                raise RuntimeError("Run start has no trigger entry")
            checkpoint = {
                **state,
                "at": "checkpoint",
                "continuation": {"kind": "need_assistant", "overflowRecoveryUsed": False},
                "triggerEntryId": trigger_entry_id,
            }
            await mutator.commit([set_value(operation_state(operation_id), checkpoint)], mutation_context)
            del durable_lane

        await self._options.session.mutate(transition, context)

    async def _advance_checkpoint(
        self,
        operation_id: str,
        context: Context,
    ) -> OperationResultRecord | None:
        async def transition(
            mutator: SessionMutator,
            mutation_context: Context,
        ) -> OperationResultRecord | None:
            durable_lane, meta, state = await self._read_operation(mutator, operation_id, mutation_context)
            if state.get("at") != "checkpoint":
                return None
            continuation = cast(dict[str, object], state["continuation"])
            if continuation.get("kind") == "need_assistant":
                stored_configuration = await mutator.get_value(lane_config(self.name), mutation_context)
                if stored_configuration is None:
                    raise RuntimeError(f"Lane {self.name!r} is missing configuration")
                retry = self._options.retry
                generation_context = {
                    "stepId": self._options.session.id_generator.next(),
                    "triggerEntryId": state["triggerEntryId"],
                    "configuration": stored_configuration.value,
                    "streamOptions": {},
                    "retryPolicy": {
                        "maxAttempts": retry.max_retries + 1 if retry.enabled else 1,
                        "baseDelayMs": retry.base_delay_ms,
                        "maxAgentDelayMs": retry.max_agent_delay_ms,
                    },
                    "overflowRecoveryUsed": continuation.get("overflowRecoveryUsed", False),
                }
                ready = {
                    "at": "assistant.ready",
                    "control": state["control"],
                    "settings": state["settings"],
                    "latestAssistantEntryId": state["latestAssistantEntryId"],
                    "generationContext": generation_context,
                    "nextAttempt": 1,
                }
                await mutator.commit([set_value(operation_state(operation_id), ready)], mutation_context)
                return None

            stored_tip = await mutator.get_value(branch_tip(self.name), mutation_context)
            if stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing branch state")
            record = self._result_record(meta, "completed", stored_tip.value)
            result_payload = self._encode_result(record)
            next_lane = {
                "currentOperationId": None,
                "lastOperationId": operation_id,
                "inbox": durable_lane.get("inbox", []),
            }
            await mutator.commit(
                [
                    delete_value(operation_meta(operation_id)),
                    delete_value(operation_state(operation_id)),
                    set_value(operation_result(operation_id), result_payload),
                    set_value(lane_state(self.name), next_lane),
                ],
                mutation_context,
            )
            return record

        return await self._options.session.mutate(transition, context)

    async def _run_generation(self, operation_id: str, context: Context) -> None:
        model = await self._resolve_ready_model(operation_id, context)
        if model is None:
            await self._finish_ready_failure(
                operation_id,
                OperationError(
                    code="model_unavailable",
                    message="The configured model is unavailable in this process",
                ),
                context,
            )
            return
        intent = await self._publish_generation_intent(operation_id, model, context)
        messages = await self.find_entries(BranchScan(order="oldest_first"), context)
        provider_messages = [
            entry.message
            for entry in messages
            if entry.type == "message"
            and not (isinstance(entry.message, AssistantMessage) and entry.message.stop_reason in {"error", "aborted"})
        ]
        generation_context = cast(dict[str, object], intent["generationContext"])
        configuration = cast(dict[str, object], generation_context["configuration"])
        thinking_level = cast(str, configuration["thinkingLevel"])
        reasoning = None if thinking_level == "off" else cast(LlmThinkingLevel, thinking_level)
        stream = self._models().stream_simple(
            model,
            LlmContext(messages=provider_messages),
            SimpleStreamOptions(reasoning=reasoning),
        )
        encoder = AssistantMessageFrameEncoder()
        async for event in stream:
            frame = encoder.encode(event)
            if frame is not None:
                await self._append_frame(operation_id, cast(str, intent["responseEntryId"]), frame, context)
        response = await stream.result()
        await self._settle_response(operation_id, intent, response, context)

    async def resume(self, context: Context) -> ResumeResult:
        execution = await self.inspect_execution(context)
        if execution.current is None:
            return err(NothingToResume())
        return await self.drive(
            DriveOptions(operation_id=execution.current.operation_id, wait_for_retry=True),
            context,
        )

    async def _recover_generation(self, operation_id: str, context: Context) -> None:
        stored = await self._options.session.get_value(operation_state(operation_id), context)
        if stored is None:
            raise RuntimeError(f"Operation {operation_id!r} is missing state")
        intent = cast(dict[str, object], stored.value)
        if intent.get("at") != "assistant.effect_pending":
            return
        response_entry_id = cast(str, intent["responseEntryId"])
        address = pending_assistant_frames(operation_id, response_entry_id)
        frames: list[AssistantMessageFrame] = []
        cursor: ListCursor | None = None
        while True:
            page = await self._options.session.read_list(
                address,
                ListReadOptions(cursor=cursor, order="asc", limit=1_000),
                context,
            )
            frames.extend(decode_assistant_frame(item.value) for item in page)
            if len(page) < 1_000:
                break
            cursor = ListCursor(seq=page[-1].seq)
        partial = reduce_assistant_message_frames(frames)
        warning = (
            "Assistant request was interrupted. The preceding content is the latest committed partial; "
            "newer live output may be missing and the external outcome is unknown."
        )
        generation_context = cast(dict[str, object], intent["generationContext"])
        configuration = cast(dict[str, object], generation_context["configuration"])
        identity = cast(dict[str, object], configuration["model"])
        if partial is None:
            recovered = AssistantMessage(
                api="unknown",
                provider=cast(str, identity["provider"]),
                model=cast(str, identity["modelId"]),
                usage=empty_usage(),
                stop_reason="error",
                timestamp=_now_ms(),
                error_message=warning,
            )
        else:
            recovered = replace(
                partial,
                usage=empty_usage(),
                stop_reason="error",
                error_message=warning,
            )
        await self._settle_response(operation_id, intent, recovered, context, recovery=True)

    async def _advance_retry_wait(
        self,
        options: DriveOptions,
        context: Context,
    ) -> WaitingDriveOutcome | None:
        stored = await self._options.session.get_value(operation_state(options.operation_id), context)
        if stored is None:
            raise RuntimeError(f"Operation {options.operation_id!r} is missing state")
        state = cast(dict[str, object], stored.value)
        if state.get("at") != "assistant.retry_wait":
            return None
        not_before = cast(int, state["notBefore"])
        remaining_ms = not_before - _now_ms()
        if remaining_ms > 0 and not options.wait_for_retry:
            return WaitingDriveOutcome(
                operation_id=options.operation_id,
                reason="retry",
                not_before=not_before,
            )
        if remaining_ms > 0:
            await sleep(remaining_ms / 1_000)

        async def transition(mutator: SessionMutator, mutation_context: Context) -> None:
            _lane, _meta, current = await self._read_operation(mutator, options.operation_id, mutation_context)
            if current.get("at") != "assistant.retry_wait":
                return
            ready = {
                "at": "assistant.ready",
                "control": current["control"],
                "settings": current["settings"],
                "latestAssistantEntryId": current["latestAssistantEntryId"],
                "generationContext": current["generationContext"],
                "nextAttempt": current["nextAttempt"],
            }
            await mutator.commit([set_value(operation_state(options.operation_id), ready)], mutation_context)

        await self._options.session.mutate(transition, context)
        return None

    async def _resolve_ready_model(self, operation_id: str, context: Context) -> Model | None:
        stored = await self._options.session.get_value(operation_state(operation_id), context)
        if stored is None or not isinstance(stored.value, dict):
            raise RuntimeError(f"Operation {operation_id!r} is missing state")
        state = cast(dict[str, object], stored.value)
        if state.get("at") != "assistant.ready":
            raise RuntimeError(f"Operation {operation_id!r} is not ready for generation")
        generation_context = cast(dict[str, object], state["generationContext"])
        configuration = cast(dict[str, object], generation_context["configuration"])
        identity = cast(dict[str, object], configuration["model"])
        return self._models().get_model(cast(str, identity["provider"]), cast(str, identity["modelId"]))

    async def _finish_ready_failure(
        self,
        operation_id: str,
        error: OperationError,
        context: Context,
    ) -> None:
        async def finish(mutator: SessionMutator, mutation_context: Context) -> None:
            durable_lane, meta, state = await self._read_operation(mutator, operation_id, mutation_context)
            if state.get("at") != "assistant.ready":
                return
            stored_tip = await mutator.get_value(branch_tip(self.name), mutation_context)
            if stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing branch state")
            record = self._result_record(meta, "failed", stored_tip.value, error)
            await mutator.commit(
                [
                    delete_value(operation_meta(operation_id)),
                    delete_value(operation_state(operation_id)),
                    set_value(operation_result(operation_id), self._encode_result(record)),
                    set_value(
                        lane_state(self.name),
                        {
                            "currentOperationId": None,
                            "lastOperationId": operation_id,
                            "inbox": durable_lane.get("inbox", []),
                        },
                    ),
                ],
                mutation_context,
            )

        await self._options.session.mutate(finish, context)

    async def _publish_generation_intent(
        self,
        operation_id: str,
        model: Model,
        context: Context,
    ) -> dict[str, object]:
        async def transition(
            mutator: SessionMutator,
            mutation_context: Context,
        ) -> dict[str, object]:
            _lane, _meta, state = await self._read_operation(mutator, operation_id, mutation_context)
            if state.get("at") != "assistant.ready":
                raise RuntimeError(f"Operation {operation_id!r} is not ready for generation")
            generation_context = cast(dict[str, object], state["generationContext"])
            pending = {
                "at": "assistant.effect_pending",
                "control": state["control"],
                "settings": state["settings"],
                "latestAssistantEntryId": state["latestAssistantEntryId"],
                "generationContext": generation_context,
                "attempt": state["nextAttempt"],
                "responseEntryId": self._options.session.id_generator.next(),
                "usageId": self._options.session.id_generator.next(),
                "intendedOutputLimit": model.max_tokens,
                "contextWindow": model.context_window,
            }
            await mutator.commit([set_value(operation_state(operation_id), pending)], mutation_context)
            return pending

        return await self._options.session.mutate(transition, context)

    async def _append_frame(
        self,
        operation_id: str,
        response_entry_id: str,
        frame: AssistantMessageFrame,
        context: Context,
    ) -> None:
        async def append(mutator: SessionMutator, mutation_context: Context) -> None:
            _lane, _meta, state = await self._read_operation(mutator, operation_id, mutation_context)
            if state.get("at") != "assistant.effect_pending" or state.get("responseEntryId") != response_entry_id:
                return
            await mutator.commit(
                [
                    append_list(
                        pending_assistant_frames(operation_id, response_entry_id),
                        encode_assistant_frame(frame),
                    )
                ],
                mutation_context,
            )

        await self._options.session.mutate(append, context)

    async def _settle_response(
        self,
        operation_id: str,
        intent: dict[str, object],
        response: AssistantMessage,
        context: Context,
        *,
        recovery: bool = False,
    ) -> None:
        async def settle(mutator: SessionMutator, mutation_context: Context) -> None:
            durable_lane, _meta, state = await self._read_operation(mutator, operation_id, mutation_context)
            if state.get("at") != "assistant.effect_pending" or state.get("responseEntryId") != intent["responseEntryId"]:
                raise RuntimeError("Assistant settlement no longer owns its intent")
            stored_tip = await mutator.get_value(branch_tip(self.name), mutation_context)
            if stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing branch state")
            response_entry_id = cast(str, intent["responseEntryId"])
            next_state: dict[str, object] | None = None
            failure: OperationError | None = None
            if response.stop_reason == "error":
                generation_context = cast(dict[str, object], state["generationContext"])
                policy = cast(dict[str, object], generation_context["retryPolicy"])
                attempt = cast(int, state["attempt"])
                if (recovery or is_retryable_assistant_error(response)) and attempt < cast(
                    int, policy["maxAttempts"]
                ):
                    next_state = {
                        "at": "assistant.retry_wait",
                        "control": state["control"],
                        "settings": state["settings"],
                        "latestAssistantEntryId": response_entry_id,
                        "generationContext": generation_context,
                        "nextAttempt": attempt + 1,
                        "notBefore": _now_ms()
                        + retry_delay_ms(
                            cast(int, policy["baseDelayMs"]),
                            cast(int, policy["maxAgentDelayMs"]),
                            attempt,
                        ),
                        "errorMessage": response.error_message or "Assistant request failed",
                    }
                else:
                    failure = OperationError(
                        code="assistant_error",
                        message=response.error_message or "Assistant request failed",
                    )
            elif response.stop_reason in {"stop", "length"} and not any(
                isinstance(content, ToolCall) for content in response.content
            ):
                next_state = {
                    "at": "checkpoint",
                    "control": state["control"],
                    "settings": state["settings"],
                    "latestAssistantEntryId": response_entry_id,
                    "continuation": {"kind": "may_finish", "includeFinalAssistant": True},
                    "triggerEntryId": response_entry_id,
                }
            else:
                failure = OperationError(
                    code="unsupported_assistant_response",
                    message=f"Unsupported assistant response: {response.stop_reason}",
                )

            writes: list[Write] = [
                insert_entry(
                    NewMessageEntry(
                        id=response_entry_id,
                        parent_id=stored_tip.value,
                        message=response,
                    )
                ),
                insert_usage(
                    UsageRow(
                        id=cast(str, intent["usageId"]),
                        usage=response.usage,
                        adjustment=False,
                        entry_id=response_entry_id,
                    )
                ),
                set_value(branch_tip(self.name), response_entry_id),
                delete_list(pending_assistant_frames(operation_id, response_entry_id)),
            ]
            if failure is None:
                if next_state is None:
                    raise RuntimeError("Assistant settlement has no successor")
                writes.extend(
                    [
                        set_value(operation_state(operation_id), next_state),
                        set_value(
                            lane_state(self.name),
                            {**durable_lane, "currentOperationId": operation_id},
                        ),
                    ]
                )
            else:
                record = self._result_record(_meta, "failed", response_entry_id, failure)
                writes.extend(
                    [
                        delete_value(operation_meta(operation_id)),
                        delete_value(operation_state(operation_id)),
                        set_value(operation_result(operation_id), self._encode_result(record)),
                        set_value(
                            lane_state(self.name),
                            {
                                "currentOperationId": None,
                                "lastOperationId": operation_id,
                                "inbox": durable_lane.get("inbox", []),
                            },
                        ),
                    ]
                )
            await mutator.commit(writes, mutation_context)

        await self._options.session.mutate(settle, context)

    async def get_result(self, operation_id: str, context: Context) -> OperationResultRecord | None:
        stored = await self._options.session.get_value(operation_result(operation_id), context)
        if stored is None:
            return None
        return self._decode_result(cast(dict[str, object], stored.value))

    def _models(self) -> Models:
        return self._options.models

    async def _read_operation(
        self,
        mutator: SessionMutator,
        operation_id: str,
        context: Context,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        durable_lane = await mutator.get_value(lane_state(self.name), context)
        meta = await mutator.get_value(operation_meta(operation_id), context)
        state = await mutator.get_value(operation_state(operation_id), context)
        if durable_lane is None or meta is None or state is None:
            raise RuntimeError(f"Operation {operation_id!r} has incomplete durable state")
        lane_value = cast(dict[str, object], durable_lane.value)
        if lane_value.get("currentOperationId") != operation_id:
            raise RuntimeError(f"Operation {operation_id!r} is not current")
        return lane_value, cast(dict[str, object], meta.value), cast(dict[str, object], state.value)

    @staticmethod
    def _result_record(
        meta: dict[str, object],
        status: Literal["completed", "aborted", "failed"],
        tip_id: str | None,
        error: OperationError | None = None,
    ) -> OperationResultRecord:
        return OperationResultRecord(
            operation_id=cast(str, meta["operationId"]),
            kind="run",
            status=status,
            error=error,
            from_tip_id=cast(str | None, meta["sourceTipId"]),
            tip_id=tip_id,
            started_at=cast(int, meta["startedAt"]),
            ended_at=_now_ms(),
        )

    @staticmethod
    def _encode_result(record: OperationResultRecord) -> dict[str, object]:
        return {
            "operationId": record.operation_id,
            "kind": record.kind,
            "status": record.status,
            "fromTipId": record.from_tip_id,
            "tipId": record.tip_id,
            "startedAt": record.started_at,
            "endedAt": record.ended_at,
            **(
                {}
                if record.error is None
                else {"error": {"code": record.error.code, "message": record.error.message}}
            ),
        }

    @staticmethod
    def _decode_result(value: dict[str, object]) -> OperationResultRecord:
        error_value = value.get("error")
        error = None
        if isinstance(error_value, dict):
            error = OperationError(code=cast(str, error_value["code"]), message=cast(str, error_value["message"]))
        return OperationResultRecord(
            operation_id=cast(str, value["operationId"]),
            kind="run",
            status=cast(Literal["completed", "aborted", "failed"], value["status"]),
            error=error,
            from_tip_id=cast(str | None, value["fromTipId"]),
            tip_id=cast(str | None, value["tipId"]),
            started_at=cast(int, value["startedAt"]),
            ended_at=cast(int, value["endedAt"]),
        )

    async def inspect_execution(self, context: Context) -> LaneExecutionInfo:
        stored_lane = await self._options.session.get_value(lane_state(self.name), context)
        if stored_lane is None:
            raise RuntimeError(f"Lane {self.name!r} is missing durable state")
        durable_lane = cast(dict[str, object], stored_lane.value)
        operation_id = durable_lane.get("currentOperationId")
        current: CurrentOperationInfo | None = None
        if isinstance(operation_id, str):
            meta = await self._options.session.get_value(operation_meta(operation_id), context)
            state = await self._options.session.get_value(operation_state(operation_id), context)
            if meta is None or state is None:
                raise RuntimeError(f"Operation {operation_id!r} has incomplete durable state")
            durable_meta = cast(dict[str, object], meta.value)
            durable_state = cast(dict[str, object], state.value)
            current = CurrentOperationInfo(
                operation_id=operation_id,
                kind="run",
                started_at=cast(int, durable_meta["startedAt"]),
                at=cast(str, durable_state["at"]),
            )
        return LaneExecutionInfo(
            current=current,
            last_operation_id=cast(str | None, durable_lane.get("lastOperationId")),
        )

    async def get_tip_id(self, context: Context) -> str | None:
        stored = await self._options.session.get_value(branch_tip(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return stored.value

    async def find_entries(self, query: BranchScan | None, context: Context) -> list[Entry]:
        branch = await self._options.session.branch(self.name, context)
        if branch is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return await branch.find_entries(query, context)
