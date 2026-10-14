#!/usr/bin/env bash
# Cheap structural checks after opencode touched the tree.
#
# These run before the expensive CI suite so an obviously broken resolution fails
# in seconds instead of minutes. They do NOT prove correctness - a car-control
# change that compiles and passes here can still be wrong, which is why the PR
# stays a draft.
set -euo pipefail

fails=0
note() { echo "::error::$1"; fails=$((fails + 1)); }

conflicted="$(git diff --name-only --diff-filter=U | tr '\n' ' ')"
if [ -n "$conflicted" ]; then
  note "unresolved conflict markers remain in: ${conflicted}"
fi

if git diff --cached --check 2>/dev/null | grep -q .; then
  note "whitespace errors introduced by the merge"
fi

# Conflict markers can survive in files git considers resolved.
if git grep -nE '^(<<<<<<<|>>>>>>>)( |$)' -- . ':!*.md' ':!*.txt' >/dev/null 2>&1; then
  note "conflict markers still present:"
  git grep -nE '^(<<<<<<<|>>>>>>>)( |$)' -- . ':!*.md' ':!*.txt' | head -20
fi

if grep -qE '^\s*(<<<<<<<|>>>>>>>)\s*$' .gitmodules 2>/dev/null; then
  note ".gitmodules looks corrupted"
fi

# The submodule pointer must have moved with the merge, or CI passes and the
# device build breaks.
for sub in opendbc_repo panda; do
  if [ -f ".gitmodules" ] && grep -q "path = ${sub}$" .gitmodules; then
    if ! git submodule status -- "${sub}" | grep -q '^-'; then
      echo "::notice::submodule ${sub} is populated"
    else
      note "submodule ${sub} is not initialised"
    fi
  fi
done

if [ "$fails" -ne 0 ]; then
  echo "::error::${fails} structural check(s) failed"
  exit 1
fi

echo "structural checks passed (NOTE: this does not prove the resolution is correct)"
