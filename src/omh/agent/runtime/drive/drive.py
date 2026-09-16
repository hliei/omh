from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import DriveOptions, DriveOutcome, SettledDriveOutcome
from omh.agent.context import Context
from omh.agent.runtime.drive.checkpoint import run_checkpoint, start_run
from omh.agent.runtime.drive.generation import run_generation
from omh.agent.runtime.drive.recovery import recover_assistant_generation
from omh.agent.runtime.drive.retry import run_retry_wait

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def drive_operation(
    lane: AgentLane, options: DriveOptions, context: Context
) -> DriveOutcome:
    while True:
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
            case other:
                raise RuntimeError(f"Unsupported operation state: {other}")
