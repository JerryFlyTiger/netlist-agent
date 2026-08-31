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

echo "-- building (make)" >&2
NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
make -C "$ABC_SRC_ROOT" -j"$NPROC"

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
