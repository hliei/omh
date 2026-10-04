# Documentation

omh's main SDK is the in-process Agent in `omh.agent`. Start with [Getting started](getting-started.md) to connect a model, continue a conversation, stream output, and execute a tool. The [project README](../README.md) introduces the SDK and development workflow.

## Agent

The [Agent contract](agent.md) describes the stateful Agent and its standalone loop:

- [State and ownership](agent.md#state-and-ownership), [running and continuing](agent.md#running-and-continuing).
- [Input queues](agent.md#input-queues), [tools](agent.md#tools), [request and turn hooks](agent.md#request-and-turn-hooks).
- [Built-in read, bash, edit, and write](agent.md#built-in-read-bash-edit-and-write), with an offline [file-tools example](../examples/file_tools.py).
- [Project context and system sections](agent.md#project-context-and-system-sections), with an offline [context resource example](../examples/context_resources.py).
- [Skills resources](agent.md#skills-resources), with an offline [skills example](../examples/skills_resources.py).
- [Prompt templates](agent.md#prompt-templates), with an offline [template example](../examples/prompt_templates.py).
- [Events](agent.md#events-and-subscribers), [cancellation](agent.md#cancellation), [standalone loop](agent.md#standalone-loop).

## LLM layer

[LLM layer](llm.md) covers model/provider configuration, transcript inputs, request options, credentials, and streaming. `omh.llm` is shared by both Agent SDKs and can be used independently.

## Durable Agent SDK (experimental)

`omh.durable` provides persistent Sessions and recoverable execution through `AgentHarness`. Read the [durable overview](durable/README.md) for the execution model or [Durable getting started](durable/getting-started.md) for runnable examples, including SQLite persistence.

| Contract | Questions it answers |
| --- | --- |
| [Storage](durable/storage.md) | What is durable, what commits atomically, and who owns a Session? |
| [Conversation tree](durable/conversation-tree.md) | How do branches, history, compaction, and context relate? |
| [Operations](durable/operations.md) | What is accepted, queued, persisted, and settled? |
| [Execution and recovery](durable/execution.md) | What happens after interruption, cancellation, close, or failure? |
| [Public surface](durable/public-api.md) | How do applications run, observe, and extend a durable harness? |
| [Invariants and verification](durable/invariants.md) | Which guarantees must changes preserve, and where are they tested? |

The contracts describe implemented behavior and identify limitations. Experimental status does not relax the documented persistence and recovery guarantees. Offline tests do not validate a live provider or establish performance.

## Project knowledge and contribution

- [Domain vocabulary](../CONTEXT.md) defines terms.
- [Architecture decisions](adr/README.md) explain significant trade-offs and their scope.
- [Project guide](../AGENTS.md) routes development tasks to the relevant conventions.
- [Domain documentation](agents/domain.md), [task tracking](agents/issue-tracker.md), [triage](agents/triage-labels.md), and [Git workflow](agents/git-workflow.md) describe repository work.
