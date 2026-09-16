# 对齐 pi 的恢复语义

本项目以 pi 的 durable harness 为 Python SDK 的参考，恢复行为严格对齐 pi，而不另行设计默认重试或人工介入策略。对已支持的能力，已结算结果不得重复执行，结果未知的工具调用按 pi 的持久化声明及当前声明决定安全重放或生成中断结果；这不承诺外部副作用恰好发生一次。

这一选择优先保持可验证的行为一致性，而非为 Python 版提供不同的恢复策略；后续功能范围及 Python 接口形式另行讨论。核对依据为 [pi harness §4.5，提交 f9bcd351dc3cedf989bc5fc0f8aa012db5737df2](https://github.com/earendil-works/pi/blob/f9bcd351dc3cedf989bc5fc0f8aa012db5737df2/packages/agent/docs/harness.md#45-driving-and-crash-recovery)，参考基线及兼容边界见 [ADR-0002](0002-pi-reference-and-compatibility.md)。
