#!/usr/bin/env bash
set -euo pipefail

fail() { echo "workflow: $*" >&2; exit 1; }

common=$(git rev-parse --path-format=absolute --git-common-dir)
primary=$(cd "$common/.." && pwd -P)
cd "$primary"

usage() {
  cat <<'EOF'
Usage:
  scripts/workflow.sh install
  scripts/workflow.sh start <feat|fix|refactor|test> <delivery-name>
  scripts/workflow.sh start <docs|chore>
  scripts/workflow.sh finish <type/delivery-name> <merged-PR-number>

Feature work gets a new sibling worktree and branch from origin/main.
Docs/chore use the clean, synchronized primary main checkout.
Finish removes only a clean worktree at the merged PR's final head.
Remote and local feature branches are retained.
EOF
}

delivery_branch() {
  [[ "$1" =~ ^(feat|fix|refactor|test)/[a-z0-9]+(-[a-z0-9]+)*$ ]] ||
    fail "expected <feat|fix|refactor|test>/<lowercase-delivery-name>"
}

clean() {
  [[ -z "$(git -C "$1" status --porcelain --untracked-files=all)" ]] ||
    fail "working directory has changes: $1"
}

case "${1:-help}" in
  install)
    [[ $# == 1 ]] || fail "install takes no arguments"
    current=$(git config --get core.hooksPath || true)
    case "$current" in
      ""|.githooks|"$primary/.githooks") ;;
      *) fail "existing hooksPath is $current; integrate those hooks before installing" ;;
    esac
    git config --local core.hooksPath "$primary/.githooks"
    # Directory-only ignore patterns do not cover the symlink used in worktrees.
    for pattern in /.scratch /AGENTS.override.md; do
      if ! grep -Fxq -- "$pattern" "$common/info/exclude"; then
        printf '\n%s\n' "$pattern" >> "$common/info/exclude"
      fi
    done
    echo "Hooks installed for this repository and its worktrees: $primary/.githooks"
    ;;
  start)
    [[ $# -ge 2 ]] || fail "start needs a task type"
    hooks=$(git config --get core.hooksPath || true)
    [[ "$hooks" == "$primary/.githooks" ]] || fail "run scripts/workflow.sh install first"
    case "$2" in
      docs|chore)
        [[ $# == 2 ]] || fail "start docs/chore takes no delivery name"
        [[ "$(git symbolic-ref --short HEAD)" == main ]] || fail "primary checkout must be on main"
        clean "$primary"
        git fetch origin
        git merge --ff-only origin/main
        [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] ||
          fail "main has local commits; reconcile them before starting another task"
        echo "Ready for $2 in $primary (main); keep one writer in this checkout."
        ;;
      feat|fix|refactor|test)
        [[ $# == 3 ]] || fail "start needs a type and delivery name"
        branch="$2/$3"
        delivery_branch "$branch"
        worktree="$primary-${branch//\//-}"
        [[ ! -e "$worktree" && ! -L "$worktree" ]] || fail "worktree path already exists: $worktree"
        if git show-ref --verify --quiet "refs/heads/$branch"; then
          fail "branch already exists: $branch; choose a new delivery name"
        fi
        git fetch origin
        if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
          fail "remote branch already exists: $branch; choose a new delivery name"
        fi
        git worktree add -b "$branch" "$worktree" origin/main
        # Planning and local alignment references remain authoritative in primary.
        if [[ -d "$primary/.scratch" ]]; then
          ln -s "$primary/.scratch" "$worktree/.scratch"
        fi
        if [[ -f "$primary/AGENTS.override.md" ]]; then
          ln -s "$primary/AGENTS.override.md" "$worktree/AGENTS.override.md"
        fi
        # A new delivery must push to its own branch, never the starting main ref.
        git config --unset-all "branch.$branch.remote" || true
        git config --unset-all "branch.$branch.merge" || true
        echo "Work only in: $worktree"
        echo "Branch: $branch"
        echo "First push: git push -u origin $branch"
        ;;
      *) fail "unknown task type: $2" ;;
    esac
    ;;
  finish)
    [[ $# == 3 ]] || fail "finish needs a delivery branch and PR number"
    branch=$2
    delivery_branch "$branch"
    [[ "$3" =~ ^[1-9][0-9]*$ ]] || fail "PR number must be a positive integer"
    worktree="$primary-${branch//\//-}"
    [[ -d "$worktree" ]] || fail "worktree not found: $worktree"
    [[ "$(git -C "$worktree" rev-parse --show-toplevel)" == "$worktree" ]] || fail "unexpected worktree root"
    [[ "$(git -C "$worktree" symbolic-ref --short HEAD)" == "$branch" ]] || fail "worktree is on another branch"
    clean "$worktree"
    # Never delete ignored local data. Only the shared planning links are exempt.
    while IFS= read -r -d '' item; do
      case "$item" in
        .scratch|AGENTS.override.md)
          [[ -L "$worktree/$item" && "$(readlink "$worktree/$item")" == "$primary/$item" ]] ||
            fail "preserve ignored material before cleanup: $worktree/$item"
          ;;
        *) fail "preserve or remove ignored material before cleanup: $worktree/$item" ;;
      esac
    done < <(git -C "$worktree" ls-files --others --ignored --exclude-standard --directory -z)
    info=$(gh api "repos/{owner}/{repo}/pulls/$3" --jq '[.merged, .head.ref, .head.sha, .base.ref] | @tsv')
    IFS=$'\t' read -r merged pr_branch pr_head base <<< "$info"
    [[ "$merged" == true && "$pr_branch" == "$branch" && "$base" == main ]] ||
      fail "PR is not a merged delivery of $branch into main"
    [[ "$(git -C "$worktree" rev-parse HEAD)" == "$pr_head" ]] || fail "worktree has commits outside the merged PR"
    git fetch origin
    [[ "$(git rev-parse "refs/remotes/origin/$branch")" == "$pr_head" ]] || fail "remote delivery branch must retain the PR's final head"
    git merge-base --is-ancestor "$pr_head" origin/main || fail "PR head is not in origin/main's merge history"
    git worktree remove "$worktree"
    echo "Removed $worktree; retained local and remote $branch."
    ;;
  help|--help|-h) usage ;;
  *) usage >&2; exit 1 ;;
esac
