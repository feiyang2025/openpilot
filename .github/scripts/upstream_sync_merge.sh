#!/usr/bin/env bash
# Merge upstream into OUTPUT_BRANCH, reporting whether conflicts appeared.
#
# Never force-resolves. A conflict is reported via the `merge_conflicts` output so
# the workflow can hand it to opencode; if opencode cannot resolve it, the job
# stops and a human looks at it.
set -euo pipefail

FORK_BRANCH="${FORK_BRANCH:?}"
OUTPUT_BRANCH="${OUTPUT_BRANCH:?}"
UPSTREAM_URL="${UPSTREAM_URL:?}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:?}"

echo "::group::upstream sync"
echo "fork      : origin/${FORK_BRANCH}"
echo "output    : ${OUTPUT_BRANCH}"
echo "upstream  : ${UPSTREAM_URL} ${UPSTREAM_BRANCH}"

git config user.name  "opencode-sync-bot"
git config user.email "opencode-sync-bot@users.noreply.github.com"

git fetch origin "$FORK_BRANCH" --depth=1 || git fetch origin "$FORK_BRANCH"
git checkout -B "$OUTPUT_BRANCH" "origin/${FORK_BRANCH}"

git remote get-url upstream >/dev/null 2>&1 \
  && git remote set-url upstream "$UPSTREAM_URL" \
  || git remote add upstream "$UPSTREAM_URL"

git fetch upstream "$UPSTREAM_BRANCH" --depth=200

base_commit="$(git merge-base HEAD "upstream/${UPSTREAM_BRANCH}")"
upstream_commit="$(git rev-parse "upstream/${UPSTREAM_BRANCH}")"

if [ "$base_commit" = "$upstream_commit" ]; then
  echo "merge_conflicts=false" >> "$GITHUB_OUTPUT"
  echo "::endgroup::"
  echo "Already up to date with upstream (${upstream_commit})"
  exit 0
fi

echo "upstream moved: ${base_commit} -> ${upstream_commit}"

set +e
git merge --no-edit --no-stat "upstream/${UPSTREAM_BRANCH}"
merge_rc=$?
set -e

if [ $merge_rc -ne 0 ]; then
  echo "merge_conflicts=true" >> "$GITHUB_OUTPUT"
  echo "::warning::Merge produced conflicts. opencode will attempt to resolve them; if it cannot, this job stops and a human must look."
  git --no-pager diff --name-only --diff-filter=U > "$RUNNER_TEMP/conflicted_files.txt" 2>/dev/null || true
  echo "conflicted files:"
  cat "$RUNNER_TEMP/conflicted_files.txt" 2>/dev/null || echo "(none listed)"
else
  echo "merge_conflicts=false" >> "$GITHUB_OUTPUT"
fi

echo "::endgroup::"
