# Task tracker: local files

Specifications, decision questions, execution tickets, and their current status live in `.scratch/<feature-slug>/`. GitHub Issues are optional for external reports or explicitly requested collaboration; routine planning uses local files. PRs record delivered changes and validation.

## Local layout

- `spec.md`: authoritative requirements, scope, and acceptance criteria.
- `tickets/<NN>-<slug>.md`: execution ticket with a stable ID, objective, acceptance criteria, direct blockers, status, and branch/PR links when assigned.
- `ticket-plan.md`: links to tickets, dependency order, and proposed delivery groups. Ticket files own their status.
- `issue/<NN>-<slug>.md`: decision question with type, state (`open`, `claimed`, `resolved`), and eventual answer.
- `map.md`: feature entry point and links to the current planning material.
- `archive/`: historical drafts and imported records; consult for provenance, not current scope or status.

`.scratch/` is ignored by Git. Copy or back up local planning separately when changing machines or checkouts. A fresh clone contains the project conventions, not the active backlog.

## Skill integration

When a skill says “publish to the issue tracker”, save or update the authoritative local file and return its path. Map issue references and blocking edges to local ticket IDs and relative Markdown links. Keep requirements and acceptance criteria readable without the original conversation. Split tickets remain ready for execution; incoming requests use [triage roles](triage-labels.md).

When fetching work, read the ticket, its parent specification, direct blockers, and relevant project documentation. Claim decision questions before resolving them and record answers locally. Create GitHub Issues only when explicitly requested; an existing external issue may remain the discussion source for that report.

## Execution and status

Use ticket status `todo`, `in-progress`, `in-review`, `done`, or `cancelled`. Readiness and blockers are separate from execution status. Record branch and PR links in the ticket once available.

1. Select a ticket or cohesive group whose external blockers are `done`. Work internal dependencies in order when several tickets share a branch.
2. Follow [the Git workflow](git-workflow.md) to define a reviewable delivery and create its branch. Mark selected tickets `in-progress`.
3. Verify acceptance criteria and record results. Mark tickets `in-review` when their PR is ready.
4. Mark tickets `done` after their acceptance criteria are met and the changes merge. Complete a parent specification only when its full scope is delivered.

A closed historical GitHub issue does not imply its migrated local ticket is done. Migration closes records administratively; implementation status stays local.

## Public delivery and durable knowledge

PR descriptions explain the problem, delivered scope, and validation without depending on `.scratch/` paths. Local ticket IDs may supplement that explanation. Issue links and closing keywords are needed only for actual GitHub-tracked work.

Keep agreed domain terms in `CONTEXT.md`. Record important architectural trade-offs in `docs/adr/`, including context, alternatives, decision, and consequences. Maintain lasting user and developer guidance with the code. Temporary plans remain local.
