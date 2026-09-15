# Issue tracker: GitHub

GitHub Issues are the shared source for specifications, decision questions, and execution tickets. Pull requests record the implementation and validation and link to the relevant issues.

## GitHub operations

Use the `gh` CLI from this repository; resolve the target from `git remote -v` (currently `hliei/omh`).

- Read a ticket with `gh issue view <number> --comments` and inspect its labels and dependencies.
- List work with `gh issue list --state open`, filtering by the triage labels when needed.
- Create an issue with `gh issue create --title "..." --body-file <draft-path>`; use a body file for multiline descriptions and comments.
- Update labels with `gh issue edit <number> --add-label <label>` or `--remove-label <label>`.
- Close completed work with `gh issue close <number>` after recording its outcome.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Local preparation

Use `.scratch/<feature-slug>/` for drafts, exploration, and unpublished work:

- `spec.md`: draft specification.
- `issue/<NN>-<slug>.md`: one decision question per file.
- `tickets/<NN>-<slug>.md`: one execution ticket per file, ordered by dependencies.
- `map.md`: local wayfinding map and links to decision questions.

The entire `.scratch/` directory is ignored by Git, including `issue/` and `tickets/`. It is available to local sessions but is not backed up by a Git push. Shared work must include its requirements and acceptance criteria on GitHub rather than relying on a local path.

## Publishing and fetching

When a skill says "publish to the issue tracker":

- Publish a specification as an issue with the problem, scope, and acceptance criteria.
- Publish each execution ticket as a separate issue linked to its parent specification. Publish blockers first; use native dependency relationships when available, otherwise a `Blocked by` section with issue links.
- Publish shared decision questions as separate issues, linked from a parent planning issue. Record the question type and state (`open`, `claimed`, or `resolved`) in the body. Claim before working; resolve with the answer and update the parent map.
- Apply [triage labels](triage-labels.md) to execution issues. Approved tickets produced by the ticket-splitting flow are `ready-for-agent`.
- Record the resulting issue URLs in any local drafts that remain in use. Once published, update shared scope and status on GitHub.

When fetching a ticket, read the GitHub issue body, comments, and dependencies. A local file path explicitly supplied by the user can be used for unpublished work.

Choose work whose blocking issues are complete. Link the implementation PR to its issue; use a closing reference only when the PR fulfills that issue's acceptance criteria. Keep parent planning issues open until their full scope is complete.

## Before a GitHub repository is connected

Prepare drafts in `.scratch/` and report them as unpublished. Resolve the target repository from its configured remote or a user-provided repository URL before publishing. A local draft does not count as a published issue.

## Durable decisions

When discussion settles an important architecture trade-off, capture its context, alternatives, decision, and consequences in `docs/adr/`, with links to related issues or PRs when available. Ordinary implementation tasks do not require ADRs. Record agreed domain terms in `CONTEXT.md` according to [the domain documentation](domain.md).
