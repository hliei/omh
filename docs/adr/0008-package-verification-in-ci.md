# CI 中的打包验证

状态：已接受，已实施。

## Context

[ADR-0003](0003-single-sdk-distribution.md) 把首版验收收敛为离线 pytest，明确不将仓库外 wheel 安装与独立发布检查作为门槛，留到需要正式交付安装产物时再验证构建内容。

此后 SDK 中出现两处只在发行产物里才有意义的内容：`omh/py.typed` 声明类型信息，SQLite 迁移以包数据形式分发（`[tool.setuptools.package-data]`）。这类回归在源码树中不可见。测试、CI 与日常开发都通过 editable 安装工作，而 editable 安装写入的是指向 `src/` 的 `.pth` 指针，其可编辑 wheel 不包含包数据，因此 `py.typed` 与迁移文件是否进入 wheel 从未被验证。漏打包的后果落在下游：类型信息静默失效，使下游类型检查退化为无类型库；或到会话迁移阶段才失败。

## Decision

CI 增加独立的 `package` job，与既有 SDK 质量矩阵并列，在 Ubuntu 24.04 上验证发行产物：

- 用 `python -m build` 走完整两阶段构建（先 sdist，再从 sdist 构建 wheel），使不完整的 sdist 在 CI 中失败。
- 校验 wheel 包含 `omh/py.typed` 与 SQLite 迁移文件，不包含 `tests/`、`docs/`、`examples/` 目录；校验 sdist 包含 `py.typed`，且不包含应用、测试与本地工作材料。
- 安装构建出的 wheel 与其 dev extra，从仓库外运行完整测试，并断言 `omh` 解析到 `site-packages`。src 布局使仓库根目录不在导入路径上，该断言因此能证明测试执行的是发行产物而非源码树。
- 不上传构建产物。该 job 只做验证，不产生留存文件，也不构成发布流程。

该 job 独占 Ubuntu：发行包是 `py3-none-any` 纯 Python wheel，产物内容与平台无关。

## Consequences

发行产物的内容与"装在仓库外仍可用"进入 CI 门槛，取代 ADR-0003 中把该项排除在外的范围约定。代价是每次触发多一个约 30 秒的 job，且打包问题会阻塞合并；收益是漏打包在合并前暴露，而不是在下游安装或运行时暴露。

本决定不覆盖发布。构建产物不上传、不签名，未区分版本发布渠道，也未涉及 sdist 与平台相关 wheel 的分发；需要正式交付安装产物时仍需单独确定发布流程。
