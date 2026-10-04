# Agent 的显式 end 结束整个对话活动

状态：已接受并实施。有效 finish_turn="end" 阻止本活动的自然续行、队列消费及后续自动压缩／重试／容量恢复。沿用 [ADR-0010](0010-agent-conversation-history-and-lifetime.md) 的 Agent 活动与内层循环区分；当前 hook 契约见 [Agent 契约](../agent.md)。

有效的 finish_turn="end" 决定结束整个 Agent 对话活动，保留未消费队列，该活动不再启动后续自动压缩、恢复或续行。相比仅终止内层循环、再由外层策略和队列重新启动，这保留现有 SDK 显式 end 的停止含义，使增加运行时策略后原调用者仍能控制活动停止。代价是该显式停止优先于原本可以进行的自动恢复或压缩；之后的新调用仍按其正常前置条件和策略执行。

error/aborted 响应保留既有忽略 finish 决定的规则，按选定重试或取消规则收尾。最终 agent_settled 回调可以显式提交新 prompt，作为新的活动按 [ADR-0011](0011-agent-awaited-event-listeners.md) 接纳；这不把旧活动恢复为运行中。工具批次 terminate 保留抑制自然工具续行的范围，不由本决定升级为整个活动取消。独立 loop 的 end 继续终止它自己的循环。
