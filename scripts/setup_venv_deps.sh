#!/usr/bin/env bash
# Copy the comma-deps shim packages the build system imports out of the checkout
# venv.
#
# SConstruct and the SConscripts import a handful of C libraries as Python
# modules purely to learn their include/lib directories:
#
#   SConstruct:58  pkg_names = ['acados','capnproto','eigen','ffmpeg',
#                                'json11','ncurses','zeromq','zstd']
#   selfdrive/pandad/SConscript:4  import libusb
#   ... ui assets: import imgui, import bootstrap_icons
#
# Those modules ship with comma's device venv (/usr/local/venv on AGNOS). On this
# bench the venv lives in the checkout and four of them are missing, which makes
# scons die while *reading* the SConscripts - so nothing rebuilds at all:
#
#   ModuleNotFoundError: No module named 'ncurses'   (SConstruct:59)
#   ModuleNotFoundError: No module named 'libusb'    (pandad/SConscript:4)
#
# acados/capnproto/eigen/ffmpeg/json11/zeromq/zstd are already present here.
#
# Usage:  scripts/setup_venv_deps.sh [source-venv]
# Default source is the cp-egpucp checkout, which has all of them. The packages
# are plain Python shims; imgui and bootstrap_icons additionally carry static
# libs and headers, so they are copied rather than re-created.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_VENV="${1:-/home/nvidia/公共/openpilot_cp_egpucp/openpilot/.venv}"
NEEDED=(ncurses libusb imgui bootstrap_icons)

SITE=$(echo "$SRC_VENV"/lib/python3*/site-packages)
DST=$(echo "$REPO_ROOT"/.venv/lib/python3*/site-packages)

if [ ! -d "$SITE" ]; then
  echo "ERROR: source site-packages not found: $SITE" >&2
  echo "       pass the venv to copy from: $0 /path/to/.venv" >&2
  exit 1
fi
if [ ! -d "$DST" ]; then
  echo "ERROR: target venv not found: $DST" >&2
  exit 1
fi

for m in "${NEEDED[@]}"; do
  if [ ! -d "$SITE/$m" ]; then
    echo "SKIP $m (not in $SITE)"
    continue
  fi
  rm -rf "${DST:?}/$m/__pycache__"
  cp -r "$SITE/$m" "$DST/"
  echo "installed $m ($(du -sh "$SITE/$m" | cut -f1))"
done

echo
echo "verifying every import the build system needs:"
status=0
while read -r mod; do
  if "$REPO_ROOT/.venv/bin/python" -c "import $mod" 2>/dev/null; then
    echo "  ok      $mod"
  else
    echo "  MISSING $mod"
    status=1
  fi
done < <(grep -rhoE "^[[:space:]]*import [a-zA-Z_][a-zA-Z0-9_]*" \
           "$REPO_ROOT/SConstruct" \
           $(find "$REPO_ROOT/openpilot" "$REPO_ROOT/msgq_repo" "$REPO_ROOT/opendbc_repo" \
                  -name SConscript 2>/dev/null) 2>/dev/null \
         | sed 's/.*import //' | sort -u)

exit $status