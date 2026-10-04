# Domain documentation

Before exploring domain concepts or changing a design, read the root [CONTEXT.md](../../CONTEXT.md), the relevant [Agent](../agent.md), [LLM](../llm.md), or [durable](../durable/README.md) contract, and applicable [architecture decisions](../adr/README.md).

## Document responsibilities

- `CONTEXT.md` defines canonical terms and distinctions. Use that vocabulary in proposals, tests, and implementation discussions.
- `docs/agent.md` describes the main in-process Agent and standalone loop. `docs/llm.md` describes the shared, independently usable model layer. `docs/durable/` describes the experimental Durable Agent SDK's state transitions and guarantees.
- SDK documentation describes SDK contracts and generic host responsibilities. Product-specific terms, decisions and guides belong to their owning project; SDK documents have no product references.
- `docs/adr/` records consequential SDK choices with context, alternatives, decisions, and consequences. Keep numbers stable and update incoming links when files move.
- Local specifications and tickets track proposed work according to [task tracking](issue-tracker.md). They do not replace lasting contracts.

When a term is unclear, check code and the glossary before introducing a synonym. Add a glossary entry when a new domain concept is agreed. Record an ADR when a decision involves a real trade-off, is costly to reverse, and would otherwise surprise a maintainer. Create documentation when its content is concrete.

If a proposal conflicts with an ADR, name the decision and explain the conflict before changing the design. Verify claims against implementation and tests; distinguish current behavior, accepted limits, and proposed changes. Preserve one authoritative home for each rule and link from other documents.
