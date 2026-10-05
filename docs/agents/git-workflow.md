# Git workflow

## Repository contents

Commit source code, tests, build configuration, examples, user documentation, `AGENTS.md`, project-specific `docs/agents/` conventions, domain vocabulary, and architecture decisions. Keep documents alongside the code changes that affect them.

The root `.gitignore` excludes local exploration and personal tool files. Personal skills are installed outside the repository; local links may support an individual developer's tools. Cloning the project does not install those skills or restore `.scratch/` drafts. Keep a separate backup of local material that needs to survive machine loss.

## First commit

Review the intended files and ignore rules before staging. Inspect the staged diff for local material and credentials. Configure the intended GitHub repository before pushing; do not infer a remote from an example project.

## Feature work

1. Define one cohesive delivery from a local ticket, several closely related tickets, or a small direct request. One branch and PR should be independently reviewable, mergeable, and reversible; ticket count does not determine branch count.
2. Run `scripts/workflow.sh start <type> <delivery-name>` before the first implementation edit to create a new worktree and `<type>/<delivery-name>` branch from updated `origin/main`. Work in the directory it prints and verify the branch with `git status --short --branch`. `docs` and `chore` use `scripts/workflow.sh start <type>` and go directly to `main` (see Direct commits to `main`). For dependent deliveries, merge the prerequisite first, then start the dependent delivery; unrelated work can proceed independently.
3. Implement the selected scope and update relevant tests and documentation. Keep local execution status according to [the task tracker](issue-tracker.md).
4. Before pushing Python code, dependency, or check-configuration changes, run `ruff check src tests`, `mypy`, and `pytest` in the development environment and resolve failures. For documentation-only changes, verify the affected content and links. Select files to stage and inspect the staged diff before committing.
5. Push the feature branch and open a PR describing the problem, resulting behavior, scope, and validation. Include necessary context directly; reviewers must not need local planning files. A GitHub Issue is optional.
6. Merge after review and required checks pass. Use GitHub's **Create a merge commit** for every PR so `main` preserves the delivery's branch boundary and original commit history; do not use squash merges or rebase merges. Keep `main` usable and verifiable. Update local ticket outcomes and PR links, sync local `main`, and retain the remote feature branch at its final head. Local feature branches may be removed after delivery.

Keep GitHub automatic head-branch deletion disabled and merge PRs without `--delete-branch`. Delete remote branches only when explicitly requested. Start later deliveries on new branches from updated `main` rather than reusing retained branches.

Split a large feature into independently useful deliveries. Group small tickets when they contribute to the same result. Code, tests, and necessary documentation for one behavior normally belong together. Feature development reaches `main` through PRs.

Pushing a branch shares its reachable commits, including their history. Ignore rules affect untracked files; they do not remove previously committed files or history.

## Workflow commands and local guards

Run `bash scripts/workflow.sh install` once per clone. It installs this repository's existing commit-message check and branch/push guards through an absolute `core.hooksPath`, shared by its worktrees. An unrelated custom hook path must be integrated before installing.

```bash
# New independent delivery: creates a sibling directory such as omh-feat-streaming.
bash scripts/workflow.sh start feat streaming
# Independent documentation or maintenance: prepares the primary main checkout.
bash scripts/workflow.sh start docs
bash scripts/workflow.sh start chore
# After the delivery's PR merges (123 is an example PR number).
bash scripts/workflow.sh finish feat/streaming 123
```

Feature starts leave existing edits in the primary checkout untouched. The new branch has no upstream until `git push -u origin <type>/<delivery-name>`, so it cannot accidentally track `origin/main`. Local `.scratch/` and `AGENTS.override.md`, when present, are shared through symlinks; local planning remains authoritative in the primary checkout. Install adds both link paths to the shared Git exclude file so they stay local.

The commit-message hook allows only `docs`/`chore` subjects for direct commits on `main`. The push hook checks every new commit in a `main` update, allows only direct `docs`/`chore` commits, and rejects other source branches, deletions, and non-fast-forward updates. Feature branches push normally. These guards check declared commit types, not semantic scope; feature documentation and tests still belong to their delivery branch. Local hooks can be deliberately bypassed or disabled; they do not restrict file editing or replace review and CI.

Finish requires `gh` access to the repository and a merged PR into `main` whose head matches the worktree and retained remote branch. It confirms that head is in `origin/main` and refuses tracked changes, untracked files, extra commits, and ignored local material other than the shared planning/override symlinks. Preserve needed ignored files and remove disposable environments/caches before retrying. It removes the script-created worktree without force, preserves the shared material and both feature branches, and leaves the primary checkout untouched. Use Codex's archive operation for app-managed worktrees instead.

Verify these commands and hooks without touching this repository's remote with `bash tests/workflow.sh`.

## Parallel deliveries

Each concurrent delivery uses a separate checkout or Git worktree and its own branch. One writer owns each working directory; separate branch names do not isolate uncommitted files, the index, or branch switches in a shared checkout. Keep the primary `main` checkout available for integration and serial direct commits.

1. Read each ticket and its parent specification, confirm external blockers are `done`, and check for missing interface or behavior dependencies. Record selected tickets as `in-progress` with the responsible session and branch/worktree path before implementation so another session can avoid claiming the same scope. Related tickets may share a delivery; work their internal dependencies in order.
2. Use the start command to create an isolated checkout for each delivery, with its branch based on updated `main`. Give that session the absolute working directory and canonical local ticket/specification paths. The script shares `.scratch/` and the local override from the primary checkout; when using another worktree mechanism, restore access to that material before implementation. Update ticket status in the canonical files.
3. Implement, verify, and open each PR independently. Coordinate changes to shared contracts; if a delivery requires another delivery's unfinished behavior, record the blocker and defer the dependent scope until the prerequisite merges.
4. Integrate PRs one at a time. Before merging a PR, fetch the latest `origin/main`; if it has advanced beyond the branch's base, merge it into the delivery branch. Resolve text conflicts and check that both deliveries' acceptance criteria still hold together. Run the applicable local checks on the resulting branch, push any integration changes, and wait for review and required checks to pass before merging the PR.
5. Mark tickets `done` after merge, run the finish command to clean up its worktree, and synchronize the primary `main` checkout when it is clean and idle. Retain the remote delivery branch and preserve needed local material before cleanup.

If implementation edits already exist on `main`, stop concurrent writes to that checkout and inspect its diff and status. When all edits belong to one delivery, move them onto its feature branch before continuing, then isolate other sessions. If edits from several deliveries are mixed, separate them deliberately before committing. Creating a branch at PR time does not recover who owns mixed changes.

## Direct commits to `main`

`docs` and `chore` commits go straight to `main`: documentation, repository conventions and process (including this workflow), tooling, dependencies, and maintenance. They need no branch, PR, or pre-merge review; push them to `main` directly.

`feat`, `fix`, `refactor`, and `test` deliveries start on a `<type>/<delivery-name>` branch and reach `main` through a PR. Documentation and tests that belong to a delivery's scope stay on that branch; do not split them out to take this shortcut.

Direct commits still follow the commit-message convention and, when they touch Python code, dependency, or check configuration, the local checks below. `main` therefore mixes delivery merge commits with direct commits; that is accepted for these changes.

Serialize direct commits in the primary `main` checkout. Before editing, confirm that it is clean, idle, and synchronized with `origin/main`; feature work continues in its isolated checkouts. Coordinate with the current writer before synchronizing or committing in the primary checkout.

## Commit messages

For commits you write, use a single-line Conventional Commit subject: `<type>(<scope>): <short imperative summary>` (omit the scope when it adds no value). Keep the subject within 72 characters and use no body by default. For example: `feat(agent): add Session-free conversation path and durable namespace`. Put delivery details and validation in the PR description; add a commit body only when explicitly requested.

## Local checks and CI failures

Run the local checks before each push of code changes. `ruff check --fix src tests` can fix supported lint findings; review its changes and rerun the checks. Local checks run on the developer platform; CI repeats Ruff, mypy, and pytest on macOS and Ubuntu 24.04 to cover both supported platforms. CI also builds the sdist and wheel on Ubuntu 24.04, verifies the wheel carries `py.typed` and the SQLite migrations, and runs the test suite against the installed wheel rather than the source tree.

If CI fails, inspect the failing job and reproduce its command locally where possible. Fix the cause on the same feature branch, verify the fix, commit, and push again. The existing PR updates and CI reruns; merge only after the required checks pass. Platform-specific failures need verification on the affected platform. CI reports failures and does not automatically commit fixes.

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

Choose the type for the primary purpose: tests and documentation accompanying a feature stay on its `feat/` branch. The `docs` and `chore` types commit directly to `main`; the other types start on a branch. Use `feat` consistently rather than `feature`, and `fix` rather than `bugfix`. Keep `main` as the integration branch. A prototype workflow may retain its separate `prototype/<name>` evidence branch as required by that workflow.

## Releases

Mark a release with a version tag such as `v0.1.0` on the reviewed commit and document the changes. Treat package publication or deployment as a separate release action.
