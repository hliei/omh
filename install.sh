#!/bin/sh
set -eu

fail() { printf 'omh installer: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) fail "supported platforms are macOS and Linux" ;;
esac

for command in curl git; do
    command -v "$command" >/dev/null 2>&1 || fail "install $command and retry"
done

repository=https://github.com/hliei/omh.git
commit=$(git ls-remote "$repository" refs/heads/main | cut -f1)
[ -n "$commit" ] || fail "could not resolve the repository's main branch"

if command -v uv >/dev/null 2>&1; then
    uv_command=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
    uv_command="$HOME/.local/bin/uv"
else
    installer_dir=$(mktemp -d)
    trap 'rm -rf "$installer_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf 'Installing uv...\n'
    curl -fsSL https://astral.sh/uv/install.sh -o "$installer_dir/uv-install.sh"
    UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer_dir/uv-install.sh"
    uv_command="$HOME/.local/bin/uv"
fi

printf 'Installing omh from commit %s...\n' "$commit"
"$uv_command" tool install --python 3.14 --force \
    --with "omh @ git+$repository@$commit" \
    "git+$repository@$commit#subdirectory=coding_agent"

bin_dir=$("$uv_command" tool dir --bin)
"$bin_dir/omh" --version
case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) printf '\nAdd the command directory to your PATH:\n\n  export PATH="%s:$PATH"\n' "$bin_dir" ;;
esac
printf '\nRun omh in your project directory to get started.\n'
