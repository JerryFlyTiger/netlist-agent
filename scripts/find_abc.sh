#!/usr/bin/env bash
# Resolve which ABC binary this project should use and print its path.
# Any language can shell out to this instead of hard-coding a path.
#
# Resolution order:
#   1. $ABC_BIN                        - explicit override, for debugging
#   2. <repo>/vendor/<platform>/abc     - bundled binary, if this project
#                                         ever ships one alongside itself
#   3. $ABC_INSTALL_ROOT/current/<platform>/bin/abc
#                                       - the shared install set up by
#                                         setup_abc.sh (default install
#                                         root: ~/opt/eda/abc)
#   4. `abc` on PATH                   - last resort
#
# Exits non-zero with a clear message on stderr if nothing is found, rather
# than silently falling back to some other tool.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH_NAME="$(uname -m)"
case "$ARCH_NAME" in
  aarch64) ARCH_NAME="arm64" ;;
esac
PLATFORM="${OS_NAME}-${ARCH_NAME}"

ABC_INSTALL_ROOT="${ABC_INSTALL_ROOT:-$HOME/opt/eda/abc}"

# 1. explicit override
if [[ -n "${ABC_BIN:-}" ]]; then
  if [[ -x "$ABC_BIN" ]]; then
    echo "$ABC_BIN"
    exit 0
  fi
  echo "error: \$ABC_BIN is set to '$ABC_BIN' but it is not an executable file" >&2
  exit 1
fi

# 2. vendored copy shipped with this repo
VENDORED="$REPO_ROOT/vendor/$PLATFORM/abc"
if [[ -x "$VENDORED" ]]; then
  echo "$VENDORED"
  exit 0
fi

# 3. shared install managed by setup_abc.sh
SHARED="$ABC_INSTALL_ROOT/current/$PLATFORM/bin/abc"
if [[ -x "$SHARED" ]]; then
  echo "$SHARED"
  exit 0
fi

# 4. PATH
if command -v abc >/dev/null 2>&1; then
  command -v abc
  exit 0
fi

cat >&2 <<EOF
error: could not find an ABC binary for platform '$PLATFORM'.

Checked, in order:
  \$ABC_BIN                (unset)
  $VENDORED
  $SHARED
  abc on \$PATH

Run scripts/setup_abc.sh to build ABC into the shared location, or set
\$ABC_BIN to point at a binary directly.
EOF
exit 1
