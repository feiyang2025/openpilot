#!/usr/bin/env bash
# Hand merge conflicts to opencode, bounded by MAX_FIX_ATTEMPTS.
#
# The AI is allowed to edit code. It is NOT allowed to decide the change is done:
# anything it cannot resolve with evidence stops the job for a human.
set -euo pipefail

PROMPT_FILE="${1:?usage: upstream_sync_opencode.sh <prompt-file>}"
ATTEMPT_FILE="${RUNNER_TEMP:-/tmp}/opencode_fix_attempts"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "::error::prompt file not found: $PROMPT_FILE"
  exit 1
fi

echo "::group::opencode conflict resolution"
echo "prompt: ${PROMPT_FILE}"

export OPENCODE_API_KEY="${OPENCODE_API_KEY:?OPENCODE_API_KEY is not set}"
MODEL="${OPENCODE_MODEL:-opencode/deepseek-v4-flash}"

npx --yes @anthropic-ai/claude-code --version >/dev/null 2>&1 || true

if ! command -v opencode >/dev/null 2>&1 && ! npx --yes opencode-ai --version >/dev/null 2>&1; then
  echo "::error::opencode CLI not available in this runner"
  exit 1
fi

attempt=0
max_attempts="${MAX_FIX_ATTEMPTS:-2}"

while [ "$attempt" -lt "$max_attempts" ]; do
  attempt=$((attempt + 1))
  remaining=$(awk "/^remaining:/{print \$2}" .git/rebase-merge/git-rebase-todo 2>/dev/null | head -1 || true)
  conflicted="$(git diff --name-only --diff-filter=U | tr '\n' ' ')"
  if [ -z "$conflicted" ]; then
    echo "no conflicted files left"
    break
  fi
  echo "attempt ${attempt}/${max_attempts}; conflicted: ${conflicted}"

  set +e
  npx --yes opencode-ai run \
    --model "$MODEL" \
    "$(cat "$PROMPT_FILE")" \
    >> "$RUNNER_TEMP/opencode_fix.log" 2>&1
  rc=$?
  set -e

  if [ -n "$remaining" ] && [ "$remaining" != "0" ]; then
    git add -A
    GIT_EDITOR=true git rebase --continue || true
  fi

  still="$(git diff --name-only --diff-filter=U | tr '\n' ' ')"
  if [ -z "$still" ]; then
    echo "::notice::conflicts resolved on attempt ${attempt}"
    git add -A
    echo "$attempt" > "$ATTEMPT_FILE"
    echo "::endgroup::"
    exit 0
  fi

  if [ $rc -ne 0 ] && [ $attempt -ge "$max_attempts" ]; then
    echo "::error::opencode exited $rc with conflicts still unresolved"
    break
  fi
done

still="$(git diff --name-only --diff-filter=U | tr '\n' ' ')"
if [ -n "$still" ]; then
  echo "::error::giving up after ${max_attempts} attempts; a human must resolve: ${still}"
  echo "::endgroup::"
  exit 1
fi

echo "::endgroup::"
