from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import DriveOptions, DriveOutcome, SettledDriveOutcome
from omh.agent.context import Context
from omh.agent.hooks import BeforeDriveHook
from omh.agent.runtime.drive.checkpoint import run_checkpoint, start_run
from omh.agent.runtime.drive.generation import run_generation
from omh.agent.runtime.drive.reconcile import reconcile_abort
from omh.agent.runtime.drive.recovery import recover_assistant_generation
from omh.agent.runtime.drive.retry import run_retry_wait
from omh.agent.runtime.drive.structural import (
    prepare_compaction_threshold,
    recover_structural_generation,
    run_structural_decision,
    run_structural_generation,
    run_structural_retry_wait,
)
from omh.agent.runtime.drive.tools import run_tools

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def drive_operation(
    lane: AgentLane, options: DriveOptions, context: Context
) -> DriveOutcome:
    execution = await lane.inspect_execution(context)
    operation = (
        "run"
        if execution.current is None
        else execution.current.kind
    )
    await lane.hooks.run(
        "before_drive",
        BeforeDriveHook(
            lane=lane.name,
            run_id=options.operation_id,
            operation=operation,
        ),
        context,
    )
    while True:
        if await lane.is_abort_requested(options.operation_id, context):
            return SettledDriveOutcome(
                outcome=await reconcile_abort(lane, options.operation_id, context)
            )
        current = await lane.inspect_execution(context)
        if current.current is None:
            result = await lane.get_result(options.operation_id, context)
            if result is None:
                raise RuntimeError(
                    f"Operation {options.operation_id!r} ended without a result"
                )
            return SettledDriveOutcome(outcome=result)
        match current.current.at:
            case "starting":
                await start_run(lane, options.operation_id, context)
            case "checkpoint":
                from omh.agent.runtime.codec import decode_operation_state
                from omh.agent.runtime.types import CheckpointOperation
                from omh.agent.session.values import operation_state

                stored = await lane._options.session.get_value(
                    operation_state(options.operation_id), context
                )
                if stored is None:
                    raise RuntimeError(
                        f"Operation {options.operation_id!r} is missing state"
                    )
                checkpoint = decode_operation_state(stored.value)
                if not isinstance(checkpoint, CheckpointOperation):
                    continue
                if await prepare_compaction_threshold(
                    lane, options.operation_id, checkpoint, context
                ):
                    continue
                outcome = await run_checkpoint(lane, options.operation_id, context)
                if outcome is not None:
                    return SettledDriveOutcome(outcome=outcome)
            case "assistant.ready":
                await run_generation(lane, options.operation_id, context)
            case "assistant.retry_wait":
                waiting = await run_retry_wait(lane, options, context)
                if waiting is not None:
                    return waiting
            case "assistant.effect_pending":
                await recover_assistant_generation(lane, options.operation_id, context)
            case "tools":
                await run_tools(lane, options.operation_id, context)
            case "summary.deciding":
                await run_structural_decision(lane, options.operation_id, context)
            case "summary.ready":
                await run_structural_generation(lane, options.operation_id, context)
            case "summary.retry_wait":
                waiting = await run_structural_retry_wait(lane, options, context)
                if waiting is not None:
                    return waiting
            case "summary.effect_pending":
                await recover_structural_generation(
                    lane, options.operation_id, context
                )
            case other:
                raise RuntimeError(f"Unsupported operation state: {other}")
