#!/usr/bin/env bash
# Run the repo's own validation, and if it fails, give opencode exactly one
# bounded attempt to fix it. Never changes the exit status into a pass on its own.
set -euo pipefail

VALIDATE_CMD="${1:?usage: upstream_sync_test_with_fixup.sh <validate-command> [prompt-file]}"
PROMPT_FILE="${2:-}"
LOG="${RUNNER_TEMP:-/tmp}/upstream_test.log"

: > "$LOG"

set +e
bash -c "$VALIDATE_CMD" >> "$LOG" 2>&1
rc=$?
set -e

if [ $rc -eq 0 ]; then
  echo "validation passed"
  exit 0
fi

echo "::warning::validation failed (rc=${rc}); tail of ${LOG}:"
tail -60 "$LOG" || true

if [ -z "$PROMPT_FILE" ] || [ ! -f "$PROMPT_FILE" ]; then
  echo "::error::no fix prompt available; leaving the failure for a human"
  exit 1
fi

if ! bash "$(dirname "$0")/upstream_sync_opencode_test_fix.sh" "$PROMPT_FILE" "$LOG"; then
  echo "::error::opencode could not fix the failure"
  exit 1
fi

echo "re-running validation after the fix attempt"
set +e
bash -c "$VALIDATE_CMD" >> "$LOG" 2>&1
rc=$?
set -e

if [ $rc -ne 0 ]; then
  echo "::error::validation still failing after the fix attempt; this needs a human"
  tail -60 "$LOG" || true
  exit 1
fi

echo "validation passed after the fix attempt"
