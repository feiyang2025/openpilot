#!/usr/bin/env bash
# One bounded attempt at fixing a validation failure. Evidence only: the failure
# log goes in, the AI is told it may not declare success without running the
# check itself.
set -euo pipefail

PROMPT_FILE="${1:?usage: upstream_sync_opencode_test_fix.sh <prompt-file> <log-file>}"
LOG_FILE="${2:?}"
MODEL="${OPENCODE_MODEL:-opencode/deepseek-v4-flash}"

export OPENCODE_API_KEY="${OPENCODE_API_KEY:?OPENCODE_API_KEY is not set}"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "::error::prompt file not found: $PROMPT_FILE"
  exit 1
fi

tail -400 "$LOG_FILE" > "${RUNNER_TEMP:-/tmp}/opencode_test_fix_context.txt"

set +e
npx --yes opencode-ai run \
  --model "$MODEL" \
  "$(cat "$PROMPT_FILE")" \
  >> "$RUNNER_TEMP/opencode_test_fix.log" 2>&1
rc=$?
set -e

git add -A
exit $rc
