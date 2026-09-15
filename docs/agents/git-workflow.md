# Git workflow

## Repository contents

Commit source code, tests, build configuration, examples, user documentation, `AGENTS.md`, project-specific `docs/agents/` conventions, domain vocabulary, and architecture decisions. Keep documents alongside the code changes that affect them.

The root `.gitignore` excludes local exploration and personal tool files. Personal skills are installed outside the repository; local links may support an individual developer's tools. Cloning the project does not install those skills or restore `.scratch/` drafts. Keep a separate backup of local material that needs to survive machine loss.

## First commit

Review the intended files and ignore rules before staging. Inspect the staged diff for local material and credentials. Configure the intended GitHub repository before pushing; do not infer a remote from an example project.

## Feature work

1. Start a short-lived feature or fix branch from `main`.
2. Implement the change and update relevant tests and documentation.
3. Run the checks appropriate to the change and inspect the staged diff before committing.
4. Push the branch and open a PR describing the problem, resulting behavior, validation, and related issues.
5. Merge after review and required checks pass. Keep `main` usable and verifiable.

Pushing a branch shares its reachable commits, including their history. Ignore rules affect untracked files; they do not remove previously committed files or history.

## Releases

Mark a release with a version tag such as `v0.1.0` on the reviewed commit and document the changes. Treat package publication or deployment as a separate release action.
