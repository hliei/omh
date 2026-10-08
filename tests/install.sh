#!/bin/sh
# Offline installer checks; no user tools, Python installations or shell files change.
set -eu

root=$(CDPATH='' cd "$(dirname "$0")/.." && pwd)
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT
mkdir -p "$test_dir/commands" "$test_dir/bin"

cat > "$test_dir/commands/stub" <<'EOF'
#!/bin/sh
set -eu
case "${0##*/}" in
    uname) printf '%s\n' "${INSTALL_TEST_PLATFORM:-Linux}" ;;
    git)
        [ "${INSTALL_TEST_FAIL_GIT:-0}" = 0 ] || exit 1
        printf '0123456789abcdef0123456789abcdef01234567\trefs/heads/main\n'
        ;;
    curl) exit 1 ;;
    uv)
        printf '%s\n' "$*" >> "$INSTALL_TEST_LOG"
        case "$1 $2" in
            'tool install')
                [ "${INSTALL_TEST_FAIL_UV:-0}" = 0 ] || exit 41
                printf '#!/bin/sh\nprintf "omh 0.1.0\\n"\n' > "$INSTALL_TEST_BIN/omh"
                chmod +x "$INSTALL_TEST_BIN/omh"
                ;;
            'tool dir') printf '%s\n' "$INSTALL_TEST_BIN" ;;
        esac
        ;;
esac
EOF
chmod +x "$test_dir/commands/stub"
for command in uname git curl uv; do
    ln -s stub "$test_dir/commands/$command"
done

export PATH="$test_dir/commands:/usr/bin:/bin"
export INSTALL_TEST_LOG="$test_dir/uv.log"
export INSTALL_TEST_BIN="$test_dir/bin"

# Use uv on PATH, pin both packages together, and print PATH guidance.
sh "$root/install.sh" > "$test_dir/output"
grep -F "export PATH=\"$test_dir/bin:\$PATH\"" "$test_dir/output" >/dev/null
grep -F 'omh 0.1.0' "$test_dir/output" >/dev/null
grep -F 'tool install --python 3.14 --force --with omh @ git+https://github.com/hliei/omh.git@0123456789abcdef0123456789abcdef01234567 git+https://github.com/hliei/omh.git@0123456789abcdef0123456789abcdef01234567#subdirectory=coding_agent' "$INSTALL_TEST_LOG" >/dev/null

# Reinstall with the existing uv command.
sh "$root/install.sh" > "$test_dir/output"
if grep -F 'Installing uv...' "$test_dir/output" >/dev/null; then
    echo 'Unexpected uv bootstrap on reinstall' >&2
    exit 1
fi

# Omit PATH guidance when the CLI directory is present.
export PATH="$PATH:$test_dir/bin"
sh "$root/install.sh" > "$test_dir/output"
if grep -F 'export PATH=' "$test_dir/output" >/dev/null; then
    echo 'Unexpected PATH guidance' >&2
    exit 1
fi

# Stop on an unsupported platform, a failed install, or a failed source lookup.
if INSTALL_TEST_PLATFORM=Windows_NT sh "$root/install.sh" > "$test_dir/output" 2>&1; then
    exit 1
fi
grep -F 'supported platforms are macOS and Linux' "$test_dir/output" >/dev/null
if INSTALL_TEST_FAIL_UV=1 sh "$root/install.sh" > "$test_dir/output" 2>&1; then
    exit 1
fi
if grep -F 'Run omh' "$test_dir/output" >/dev/null; then
    exit 1
fi
if INSTALL_TEST_FAIL_GIT=1 sh "$root/install.sh" > "$test_dir/output" 2>&1; then
    exit 1
fi
grep -F 'could not resolve' "$test_dir/output" >/dev/null

printf 'Installer checks passed.\n'
