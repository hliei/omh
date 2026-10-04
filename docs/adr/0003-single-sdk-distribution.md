# 单个 SDK 发行包与内部职责分层

2026-09-29 更新：发行产物的构建内容与仓库外安装验证已进入 CI，见 [ADR-0008](0008-package-verification-in-ci.md)；下文首版验收范围不再排除该项。

Python SDK 以根 pyproject.toml 定义一个发行包，内部保留 llm、agent、session_backends/sqlite 的职责边界，统一版本与发布。llm 可独立调用且不依赖 agent；当前不提供独立安装与发布，以减少跨层开发时的版本协调成本。

项目名称为 oh-my-harness，Python 发行名称与导入名统一为 `omh`。SDK 采用 src/omh/ 布局，将可导入源码与仓库其他文件隔离；根 pyproject.toml 只构建 SDK。消费者独立管理构建与依赖，SDK 不依赖具体应用。相比 flat layout 或共享 src 目录，这一结构优先明确安装与项目边界。构建时应明确限定 SDK 包含范围。

首版实现验收收敛为离线 pytest，不将仓库外 wheel 安装与独立发布检查作为门槛；需要正式交付安装产物时再验证构建内容。
