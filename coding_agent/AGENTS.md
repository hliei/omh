# Coding-agent project guide

## Product contracts

Before changing this product, read [the application contracts](README.md#application-contracts)
and the chapter owning the behavior: [AgentSession](docs/agent-session.md),
[SessionManager](docs/session-manager.md), or [AgentSessionRuntime](docs/agent-session-runtime.md).
[CONTEXT.md](CONTEXT.md) defines product vocabulary;
[the product architecture decision](docs/adr/0001-sdk-composition-boundary.md)
records composition and ownership choices.

## SDK boundary

The product consumes the public `omh.agent` and `omh.llm` APIs. Read the relevant
[Agent](../docs/agent.md) or [LLM](../docs/llm.md) contract when changing that integration.
Keep product documentation, examples and terms in this project. SDK documentation
owns generic SDK behavior and does not reference this product.

## Development

Follow the root repository's Git and task conventions. This project has its own
build configuration and checks; use the [README development commands](README.md)
from this directory. Saving changes need history reopen and repair coverage;
switching changes need cancellation, subscription and retained-session coverage.
