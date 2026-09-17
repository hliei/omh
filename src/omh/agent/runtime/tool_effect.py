from __future__ import annotations

from dataclasses import dataclass

from omh.agent.runtime.types import (
    EffectPendingToolCall,
    OperationState,
    ToolsOperation,
)


@dataclass(frozen=True, slots=True)
class ToolEffect:
    operation_id: str
    turn_id: str
    source_index: int
    invocation_id: str

    def owns(self, state: OperationState) -> bool:
        return (
            isinstance(state, ToolsOperation)
            and state.batch.turn_id == self.turn_id
            and any(
                isinstance(call, EffectPendingToolCall)
                and call.source_index == self.source_index
                and call.result_entry_id == self.invocation_id
                for call in state.batch.calls
            )
        )
