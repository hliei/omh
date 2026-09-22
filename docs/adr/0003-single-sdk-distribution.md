# 单个 SDK 发行包与内部职责分层

Python SDK 以根 pyproject.toml 定义一个发行包，内部保留 llm、agent、session_backends/sqlite 的职责边界，统一版本与发布。llm 可独立调用且不依赖 agent；当前不提供独立安装与发布，以减少跨层开发时的版本协调成本。

项目名称为 oh-my-harness，Python 发行名称与导入名统一为 `omh`。SDK 采用 src/omh/ 布局，将可导入源码与仓库其他文件隔离；根 pyproject.toml 只构建 SDK。未来 Console、coding agent 等应用分别位于根目录的 console/、coding_agent/ 项目下，各自管理构建与依赖，消费 SDK；SDK 不依赖应用。相比 flat layout 或共享 src 目录，这一结构优先明确安装与项目边界。构建时应明确限定 SDK 包含范围；应用项目在实际开发时创建。

首版实现验收收敛为离线 pytest，不将仓库外 wheel 安装与独立发布检查作为门槛；需要正式交付安装产物时再验证构建内容。
