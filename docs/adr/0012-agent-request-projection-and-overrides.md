# Agent 每次请求重建有效上下文

状态：设计已接受，尚未实施。沿用 [ADR-0009](0009-agent-runtime-policy-ownership.md) 的运行策略归属及 [ADR-0010](0010-agent-conversation-history-and-lifetime.md) 的权威历史与有效上下文区分；当前已交付 hook 契约仍见 [Agent 契约](../agent.md)。

Agent 每次模型请求先从权威历史重建有效上下文，并读取当前模型与 thinking 配置，再调用 `prepare_request`。hook 返回的 context、model、thinking 覆盖当前请求，下一请求重新准备并再次调用 hook；替代请求上下文不改写权威历史，请求级模型覆盖不改写公开模型选择。

相比沿用同一次运行内持续保留前次 hook 替代值，这让压缩后的上下文和运行中配置变更进入后续请求，避免先前替代上下文遮住新的历史投影。代价是依赖“一次返回、整个运行持续生效”的 Agent 调用者需要迁移；需要持续覆盖时，hook 应在每次请求明确返回覆盖值。

Agent 的 prepare_next_turn 保留原执行时点、两种调用形式的优先级及追加消息能力，请求 context、model、thinking 的返回覆盖统一放到 prepare_request。追加消息纳入权威历史，随后参与当前有效上下文。相比同时支持两处请求覆盖，这减少覆盖与 live 配置及压缩投影的优先级组合；现有依赖 next-turn 返回覆盖的 Agent 调用者需迁移到 prepare_request，不默默忽略不支持的返回字段。

独立 loop 保留原有局部持续替代契约及完整 next-turn 返回能力，由宿主编排其执行状态。Agent 的逐请求刷新属于完整运行时协调，不要求独立 loop 持有权威历史或运行策略。读取隔离见 [ADR-0010](0010-agent-conversation-history-and-lifetime.md)，显式终止的活动范围见 [ADR-0013](0013-agent-explicit-activity-end.md)；具体返回类型与签名在规格阶段展开。
