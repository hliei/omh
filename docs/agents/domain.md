# Domain documentation

Before exploring domain concepts or changing a design, read the root [CONTEXT.md](../../CONTEXT.md), the relevant [harness chapter](../harness.md), and applicable [architecture decisions](../adr/README.md).

## Document responsibilities

- `CONTEXT.md` defines canonical terms and distinctions. Use that vocabulary in proposals, tests, and implementation discussions.
- `docs/harness.md` and `docs/harness/` describe runtime behavior, state transitions, and guarantees. `docs/llm.md` describes the independent model layer.
- `docs/adr/` records consequential choices with context, alternatives, decisions, and consequences. Keep numbers stable and update incoming links when files move.
- Local specifications and tickets track proposed work according to [task tracking](issue-tracker.md). They do not replace lasting contracts.

When a term is unclear, check code and the glossary before introducing a synonym. Add a glossary entry when a new domain concept is agreed. Record an ADR when a decision involves a real trade-off, is costly to reverse, and would otherwise surprise a maintainer. Create documentation when its content is concrete.

If a proposal conflicts with an ADR, name the decision and explain the conflict before changing the design. Verify claims against implementation and tests; distinguish current behavior, accepted limits, and proposed changes. Preserve one authoritative home for each rule and link from other documents.
