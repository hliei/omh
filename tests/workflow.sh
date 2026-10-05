#!/usr/bin/env bash
# Exercise the workflow against a disposable repository, never the real remote.
set -euo pipefail
source_root=$(cd "$(dirname "$0")/.." && pwd -P)
test_root=$(mktemp -d "${TMPDIR:-/tmp}/omh-workflow.XXXXXX")
trap 'rm -rf "$test_root"' EXIT

reject() {
  local expected=$1
  shift
  if "$@" >"$test_root/rejection.log" 2>&1; then
    echo "Expected rejection: $*" >&2
    exit 1
  fi
  if ! grep -q "$expected" "$test_root/rejection.log"; then
    cat "$test_root/rejection.log" >&2
    exit 1
  fi
}

git init --bare --initial-branch=main "$test_root/remote.git" >/dev/null
git clone "$test_root/remote.git" "$test_root/main repo" >/dev/null 2>&1
cd "$test_root/main repo"
git config user.name 'Workflow test'
git config user.email 'workflow@example.invalid'
mkdir scripts .githooks
cp "$source_root/scripts/workflow.sh" scripts/
cp "$source_root/.githooks/commit-msg" "$source_root/.githooks/pre-push" .githooks/
printf '/.scratch/\n.env\n' > .gitignore
printf '/AGENTS.override.md\n' >> .git/info/exclude
git add .
git commit -m 'chore: initialize fixture' >/dev/null
git push -u origin main >/dev/null 2>&1
mkdir -p .scratch/reference
echo 'local reference' > .scratch/reference/README.md
echo 'local override' > AGENTS.override.md

git config core.hooksPath custom-hooks
reject 'existing hooksPath' bash scripts/workflow.sh install
[[ "$(git config --get core.hooksPath)" == custom-hooks ]]
git config --unset core.hooksPath
bash scripts/workflow.sh install >/dev/null
bash scripts/workflow.sh install >/dev/null
[[ "$(git config --get core.hooksPath)" == "$PWD/.githooks" ]]
bash scripts/workflow.sh start docs >/dev/null 2>&1
bash scripts/workflow.sh start chore >/dev/null 2>&1
reject 'lowercase-delivery-name' bash scripts/workflow.sh start feat '../invalid'

echo docs > docs.txt
git add docs.txt
reject 'has changes' bash scripts/workflow.sh start docs
reject 'main accepts direct docs/chore' git commit -m 'feat: wrong branch'
reject 'integrate deliveries through PRs' git commit -m 'Revert delivery'
reject 'not a Conventional Commit' git commit -m 'invalid message'
git commit -m 'docs: add documentation' >/dev/null
git push origin main >/dev/null 2>&1
echo maintenance > maintenance.txt
git add maintenance.txt
git commit -m 'chore: maintain fixture' >/dev/null
git push origin main >/dev/null 2>&1

# Even a commit made while bypassing commit-msg is stopped at normal push time.
git -c core.hooksPath=/dev/null commit --allow-empty -m 'feat: bypass commit guard' >/dev/null
reject 'main direct pushes allow only docs/chore' git push origin main
git reset --hard origin/main >/dev/null

primary=$PWD
bash scripts/workflow.sh start feat alpha >/dev/null 2>&1
bash scripts/workflow.sh start fix beta >/dev/null 2>&1
alpha="$primary-feat-alpha"
beta="$primary-fix-beta"
[[ "$(git -C "$alpha" symbolic-ref --short HEAD)" == feat/alpha ]]
[[ "$(git -C "$beta" symbolic-ref --short HEAD)" == fix/beta ]]
[[ -L "$alpha/.scratch" && -L "$alpha/AGENTS.override.md" ]]
[[ "$(cat "$alpha/.scratch/reference/README.md")" == 'local reference' ]]
if git -C "$alpha" rev-parse --verify '@{upstream}' >/dev/null 2>&1; then
  echo 'Feature branch unexpectedly tracks origin/main' >&2
  exit 1
fi
reject 'already exists' bash scripts/workflow.sh start feat alpha
echo alpha > "$alpha/alpha.txt"
[[ ! -e "$primary/alpha.txt" && ! -e "$beta/alpha.txt" ]]
[[ -z "$(git -C "$beta" status --porcelain)" ]]
git -C "$alpha" add alpha.txt
git -C "$alpha" commit -m 'feat: implement alpha' >/dev/null
reject 'push main only from local main' git -C "$alpha" push origin HEAD:main
git -C "$alpha" push -u origin feat/alpha >/dev/null 2>&1

# Stub GitHub metadata only; all worktree, history, hook and push checks use Git.
mkdir "$test_root/bin"
cat > "$test_root/bin/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\t%s\t%s\t%s\n' "$TEST_PR_MERGED" "$TEST_PR_BRANCH" "$TEST_PR_HEAD" "$TEST_PR_BASE"
EOF
chmod +x "$test_root/bin/gh"
export PATH="$test_root/bin:$PATH"
export TEST_PR_MERGED=false TEST_PR_BRANCH=feat/alpha TEST_PR_BASE=main
TEST_PR_HEAD=$(git -C "$alpha" rev-parse HEAD)
export TEST_PR_HEAD
reject 'not a merged delivery' bash scripts/workflow.sh finish feat/alpha 1
[[ -d "$alpha" ]]
git -c core.hooksPath=/dev/null merge --no-ff feat/alpha -m 'Merge delivery' >/dev/null
# The server-side PR merge is simulated without the direct-push guard.
git -c core.hooksPath=/dev/null push origin main >/dev/null 2>&1
export TEST_PR_MERGED=true
echo private > "$alpha/.env"
reject 'ignored material' bash scripts/workflow.sh finish feat/alpha 1
[[ -f "$alpha/.env" ]]
rm "$alpha/.env"
echo pending > "$alpha/pending.txt"
reject 'has changes' bash scripts/workflow.sh finish feat/alpha 1
rm "$alpha/pending.txt"
git -C "$alpha" commit --allow-empty -m 'fix: extra unmerged work' >/dev/null
reject 'commits outside the merged PR' bash scripts/workflow.sh finish feat/alpha 1
git -C "$alpha" reset --hard "$TEST_PR_HEAD" >/dev/null
bash scripts/workflow.sh finish feat/alpha 1 >/dev/null 2>&1
[[ ! -d "$alpha" && -f "$primary/.scratch/reference/README.md" ]]
git show-ref --verify --quiet refs/heads/feat/alpha
git show-ref --verify --quiet refs/remotes/origin/feat/alpha
[[ -d "$beta" ]]
echo 'Workflow integration checks passed.'
