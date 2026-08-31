#!/usr/bin/env bash
# Build (or reuse) the ABC binary pinned in abc.lock.
#
# ABC is shared across projects, not vendored per-repo, so the source and
# build output live outside this repo by default:
#   src:     $ABC_SRC_ROOT      (default: ~/opt/eda/src/abc)
#   builds:  $ABC_INSTALL_ROOT  (default: ~/opt/eda/abc)
# Each build is installed under a commit-named directory, e.g.
#   ~/opt/eda/abc/<commit12>/<platform>/bin/abc
# and ~/opt/eda/abc/current is symlinked to the version this repo pins.
# Re-running this script is a no-op once that path already has the binary;
# pass --force to rebuild anyway.
#
# Usage: scripts/setup_abc.sh [--force]

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$REPO_ROOT/abc.lock"

if [[ ! -f "$LOCK_FILE" ]]; then
  echo "error: $LOCK_FILE not found" >&2
  exit 1
fi

REPO_URL="$(grep -E '^repo=' "$LOCK_FILE" | cut -d= -f2-)"
COMMIT="$(grep -E '^commit=' "$LOCK_FILE" | cut -d= -f2-)"
if [[ -z "$REPO_URL" || -z "$COMMIT" ]]; then
  echo "error: could not read repo/commit from $LOCK_FILE" >&2
  exit 1
fi
COMMIT_SHORT="${COMMIT:0:12}"

ABC_SRC_ROOT="${ABC_SRC_ROOT:-$HOME/opt/eda/src/abc}"
ABC_INSTALL_ROOT="${ABC_INSTALL_ROOT:-$HOME/opt/eda/abc}"

OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH_NAME="$(uname -m)"
case "$ARCH_NAME" in
  aarch64) ARCH_NAME="arm64" ;;
esac
PLATFORM="${OS_NAME}-${ARCH_NAME}"

VERSION_DIR="$ABC_INSTALL_ROOT/$COMMIT_SHORT"
BIN_DIR="$VERSION_DIR/$PLATFORM/bin"
BIN_PATH="$BIN_DIR/abc"

if [[ -x "$BIN_PATH" && "$FORCE" -eq 0 ]]; then
  echo "ABC already built for $PLATFORM at commit $COMMIT_SHORT:" >&2
  echo "$BIN_PATH"
  ln -sfn "$VERSION_DIR" "$ABC_INSTALL_ROOT/current"
  exit 0
fi

echo "== Setting up ABC ($COMMIT_SHORT) for $PLATFORM ==" >&2

# ABC has no tags, so there's no lightweight ref to shallow-clone by name.
# Instead we fetch exactly the pinned commit at depth 1 (GitHub allows
# fetching an arbitrary reachable SHA for public repos). This keeps the
# clone at a few dozen MB permanently instead of the ~80 MB full history,
# and re-running this after abc.lock's commit changes re-fetches at depth 1
# again rather than accumulating history.
mkdir -p "$(dirname "$ABC_SRC_ROOT")"
if [[ ! -d "$ABC_SRC_ROOT/.git" ]]; then
  echo "-- initializing shallow clone at $ABC_SRC_ROOT" >&2
  mkdir -p "$ABC_SRC_ROOT"
  git -C "$ABC_SRC_ROOT" init -q
  git -C "$ABC_SRC_ROOT" remote add origin "$REPO_URL"
fi

echo "-- fetching pinned commit $COMMIT_SHORT (depth 1)" >&2
git -C "$ABC_SRC_ROOT" fetch --depth 1 origin "$COMMIT"
git -C "$ABC_SRC_ROOT" checkout --detach FETCH_HEAD

# Windows-only files (MSVC project files, pthreads-win32 headers/prebuilt
# libs, upstream's own Windows CI config) are never touched by the Unix
# Makefile build on macOS/Linux; drop them so a mac-only checkout doesn't
# carry Windows-only weight.
rm -rf "$ABC_SRC_ROOT/lib"
rm -f "$ABC_SRC_ROOT"/abcexe.dsp "$ABC_SRC_ROOT"/abclib.dsp "$ABC_SRC_ROOT"/abcspace.dsw
rm -rf "$ABC_SRC_ROOT/.github"

# ABC links libreadline for line editing at its own interactive `abc>` prompt,
# and its Makefile assumes the header is simply there: no configure step, no
# fallback. On a machine without the development headers (a stock GitHub
# ubuntu runner, for one) the build dies at
# `src/base/main/mainUtils.c:32: fatal error: readline/readline.h`.
#
# Nothing in this project needs it. The engine drives ABC non-interactively
# (`abc -c "..."`), so readline only matters when a human types at the prompt
# by hand. Probe for the header and, if it's missing, build without it --
# ABC guards every use behind `#ifdef ABC_USE_READLINE`, so this costs the
# interactive line editor and nothing else.
#
# The probe compiles rather than looking in a fixed list of directories:
# readline lives in different places on macOS (SDK/libedit, or Homebrew's
# prefix) than on Linux, and the compiler already knows its own search path.
#
# A plain string, not an array: this script runs under `set -u`, and macOS
# ships bash 3.2, where expanding an EMPTY array as "${arr[@]}" is an
# "unbound variable" error rather than zero arguments -- which would fire on
# exactly the common path here (readline present, so nothing to add).
# The value below has no spaces or glob characters, so the deliberately
# unquoted expansion at the `make` call is safe and yields no argument when
# it's empty.
MAKE_READLINE_VAR=""
_probe_dir="$(mktemp -d)"
if printf '#include <readline/readline.h>\nint main(void){return 0;}\n' \
     > "$_probe_dir/probe.c" \
   && "${CC:-cc}" -c "$_probe_dir/probe.c" -o "$_probe_dir/probe.o" 2>/dev/null; then
  echo "-- libreadline headers found: building WITH readline" >&2
else
  echo "-- libreadline headers not found: building with ABC_USE_NO_READLINE=1" >&2
  echo "   (interactive 'abc>' line editing only; nothing this project uses)" >&2
  MAKE_READLINE_VAR="ABC_USE_NO_READLINE=1"
fi
rm -rf "$_probe_dir"

# Clear intermediates left behind by an earlier FAILED build before starting.
# The success path below deletes them, so anything still here came from a
# build that died partway -- and if it died at the readline header, its .o
# files were compiled with the opposite -DABC_USE_READLINE setting from the
# one we just chose. Linking those together is the kind of mismatch that
# produces a binary rather than an error.
find "$ABC_SRC_ROOT" -name "*.o" -delete
find "$ABC_SRC_ROOT" -name "*.d" -delete

echo "-- building (make)" >&2
NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
make -C "$ABC_SRC_ROOT" -j"$NPROC" $MAKE_READLINE_VAR

if [[ ! -x "$ABC_SRC_ROOT/abc" ]]; then
  echo "error: build finished but $ABC_SRC_ROOT/abc was not produced" >&2
  exit 1
fi

# .o/.d intermediates are disposable once the binary is linked; `make` will
# regenerate whatever it needs on the next build.
find "$ABC_SRC_ROOT" -name "*.o" -delete
find "$ABC_SRC_ROOT" -name "*.d" -delete
rm -f "$ABC_SRC_ROOT/abc.history"

echo "-- installing to $BIN_PATH" >&2
mkdir -p "$BIN_DIR"
cp "$ABC_SRC_ROOT/abc" "$BIN_PATH"
ln -sfn "$VERSION_DIR" "$ABC_INSTALL_ROOT/current"

echo "-- verifying" >&2
"$BIN_PATH" -c "version; quit" >&2

echo "$BIN_PATH"
