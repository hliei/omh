# SQLite 会话容器布局与宿主所有权边界

SQLite 后端是本项目第一个跨进程持久的存储实现，本 ADR 记录该票确定、后续不应在未察觉的情况下改变的三个选择：容器命名、枚举的容错程度，以及谁被允许写一个 Session。

## Context

固定 pi 基线的 `sqlite-node` 后端默认每个 Session 一个 `{id}.sqlite` 文件，并对不安全 id 做编码；它的 `list` 遍历目录中的 `*.sqlite` 并逐文件尽力读取；它明确说明所有权由宿主负责，存储层不提供租约、fence 或自动接管。本项目的 T03 规范化了这些行为，同时把元数据中的物理路径去掉（`open`/`delete` 由 id 与仓库配置推导路径，因为跨 Session fork 与外部源不在首版范围）。

可选做法：

1. 让 `metadata` 携带物理路径，并在 `open`/`delete` 时校验路径身份（上游做法）。需要 fork/外部源才有意义，首版没有这些调用者。
2. 用 id 直接拼接文件名。无法阻止 `../`、分隔符或 Unicode 逃出配置目录。
3. 在存储层实现跨进程租约或文件锁。上游明确不做，且需要实现接管与 fencing 语义。

## Decision

- **命名**：安全 id（`[A-Za-z0-9_-]+`）用 `{id}.sqlite`；其余用 `~` + base64url(UTF-16LE) 编码后加扩展名，任何 id 都不能让文件落到 `directory` 之外。
- **枚举**：`list` 是 best-effort 发现，逐文件读取会话行，遇到损坏文件、版本不符或非会话库时跳过；显式 `open` 才报错。因此 `list` 不保证“目录里有什么就能列出什么”，也不负责修复或清理。
- **所有权**：仓库只在自身实例内保留已在手的 id 并拒绝重复可写句柄；不提供租约、fence、心跳、跨进程/跨仓库检测或自动接管。跨仓库共用同一文件属于宿主信任边界之外的行为，不作承诺。Session 的读写以 `BEGIN IMMEDIATE` 和 WAL 保证事务原子性与一致性快照，不保证互斥所有权。

## Consequences

- 会话文件可以在目录间整体搬迁与备份，`open` 只依赖 id 与仓库配置；代价是元数据不能证明它属于哪个目录，外部源路径身份校验要等到 fork 票实现。
- 目录中的无关 `.sqlite` 文件、损坏文件不会让 `list` 失败，但也可能在清单中静默缺席；应用若需要严格清单，应在文档化路径下自行管理文件。
- 第二个进程或第二个仓库实例同时打开同一个 Session 不会被检测；正确性依赖宿主保证单写者。若将来需要多进程调度，必须新增租约/接管能力，而不是放宽这里的约束。
- 已确认的 O(history) 分叉复制限制与分支索引结构无关本决定，见 [ADR-0005](0005-preserve-sqlite-branch-index.md)。

依据：pi 固定基线的 `packages/session-backends/sqlite-node/src/sqlite/repo.ts`（`sessionFileName`、`pendingIds`、best-effort `list`）与 `packages/agent/docs/harness.md` §1.7、§2.7、§2.8（宿主所有权、无租约）。本项目的对应与差异记录在 [sqlite-session-upstream.md](../sqlite-session-upstream.md)。
