# Git workflow

## Repository contents

Commit source code, tests, build configuration, examples, user documentation, `AGENTS.md`, project-specific `docs/agents/` conventions, domain vocabulary, and architecture decisions. Keep documents alongside the code changes that affect them.

The root `.gitignore` excludes local exploration and personal tool files. Personal skills are installed outside the repository; local links may support an individual developer's tools. Cloning the project does not install those skills or restore `.scratch/` drafts. Keep a separate backup of local material that needs to survive machine loss.

## First commit

Review the intended files and ignore rules before staging. Inspect the staged diff for local material and credentials. Configure the intended GitHub repository before pushing; do not infer a remote from an example project.

## Feature work

1. Define one cohesive delivery from a local ticket, several closely related tickets, or a small direct request. One branch and PR should be independently reviewable, mergeable, and reversible; ticket count does not determine branch count.
2. Start a `<type>/<delivery-name>` branch (see naming below) from up-to-date `main`. Create branches when work starts. For dependent deliveries, merge the prerequisite first, then branch from updated `main`; unrelated work can proceed independently.
3. Implement the selected scope and update relevant tests and documentation. Keep local execution status according to [the task tracker](issue-tracker.md).
4. Run checks appropriate to the change, select files to stage, and inspect the staged diff before committing.
5. Push the feature branch and open a PR describing the problem, resulting behavior, scope, and validation. Include necessary context directly; reviewers must not need local planning files. A GitHub Issue is optional.
6. Merge after review and required checks pass. Keep `main` usable and verifiable. Update local ticket outcomes and PR links, sync local `main`, and retain the remote feature branch at its final head. Local feature branches may be removed after delivery.

Keep GitHub automatic head-branch deletion disabled and merge PRs without `--delete-branch`. Delete remote branches only when explicitly requested. Start later deliveries on new branches from updated `main` rather than reusing retained branches.

Split a large feature into independently useful deliveries. Group small tickets when they contribute to the same result. Code, tests, and necessary documentation for one behavior normally belong together. Routine feature development reaches `main` through PRs.

Pushing a branch shares its reachable commits, including their history. Ignore rules affect untracked files; they do not remove previously committed files or history.

## Branch naming

Use `<type>/<short-delivery-name>` with lowercase English words separated by hyphens. Name the delivered behavior or change; ticket IDs belong in local planning and may supplement a name when useful. Use the same convention for human and agent work.

| Type | Purpose | Example |
| --- | --- | --- |
| `feat` | New capability | `feat/deepseek-streaming` |
| `fix` | Correct broken behavior | `fix/session-recovery` |
| `docs` | Documentation or project conventions | `docs/branch-naming` |
| `refactor` | Restructure without changing intended behavior | `refactor/provider-adapters` |
| `test` | Tests without a production behavior change | `test/sqlite-transactions` |
| `chore` | Tooling, dependencies, or maintenance | `chore/update-dev-dependencies` |

Choose the type for the primary purpose: tests and documentation accompanying a feature stay on its `feat/` branch. Use `feat` consistently rather than `feature`, and `fix` rather than `bugfix`. Keep `main` as the integration branch. A prototype workflow may retain its separate `prototype/<name>` evidence branch as required by that workflow.

## Releases

Mark a release with a version tag such as `v0.1.0` on the reviewed commit and document the changes. Treat package publication or deployment as a separate release action.
