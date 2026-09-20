from __future__ import annotations

import time
from typing import Literal

from omh.agent.agent_harness import OperationError, OperationResultRecord
from omh.agent.runtime.codec import encode_lane_state, encode_operation_result
from omh.agent.runtime.types import LaneState, OperationMeta, OperationSnapshot
from omh.agent.session.types import Write
from omh.agent.session.values import (
    delete_value,
    lane_state,
    operation_meta,
    operation_result,
    operation_state,
    set_value,
)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def result_record(
    meta: OperationMeta,
    status: Literal["completed", "declined", "aborted", "failed"],
    tip_id: str | None,
    error: OperationError | None = None,
) -> OperationResultRecord:
    return OperationResultRecord(
        operation_id=meta.operation_id,
        kind=meta.intent.kind,
        status=status,
        error=error,
        from_tip_id=meta.source_tip_id,
        tip_id=tip_id,
        started_at=meta.started_at,
        ended_at=now_ms(),
    )


def terminal_writes(
    lane_name: str,
    snapshot: OperationSnapshot,
    record: OperationResultRecord,
) -> list[Write]:
    return [
        delete_value(operation_meta(record.operation_id)),
        delete_value(operation_state(record.operation_id)),
        set_value(
            operation_result(record.operation_id), encode_operation_result(record)
        ),
        set_value(
            lane_state(lane_name),
            encode_lane_state(
                LaneState(
                    current_operation_id=None,
                    last_operation_id=record.operation_id,
                    inbox=snapshot.lane.inbox,
                )
            ),
        ),
    ]
