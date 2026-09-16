# 保留 pi 的 SQLite 分支索引结构及已知性能限制

为保持文件、主要函数和存储行为便于与 pi 核对，首版沿用固定基线的 SQLite 分支索引结构，不在移植过程中另行设计索引。接受上游已明确的限制：对未压缩长历史产生分叉时，可能复制 O(history) 索引行，因此不承诺长历史分叉具有固定开销。

这一取舍以结构可核对性和首版工作范围为先；查询无遗漏、无重复及分支历史正确性仍须验证。未来优化应作为独立架构变更，避免将已知性能限制误当成实现遗漏。依据：[pi 分支索引 §2.6](https://github.com/earendil-works/pi/blob/f9bcd351dc3cedf989bc5fc0f8aa012db5737df2/packages/agent/docs/harness.md#26-the-branch-index)。
