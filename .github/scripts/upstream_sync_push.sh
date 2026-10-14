#!/usr/bin/env bash
# Push OUTPUT_BRANCH only. Never touches FORK_BRANCH: the fork's C3 branch must
# stay a clean mirror of the upstream so the next sync starts from a known state.
set -euo pipefail

FORK_BRANCH="${FORK_BRANCH:?}"
OUTPUT_BRANCH="${OUTPUT_BRANCH:?}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "::error::working tree is dirty; refusing to push"
  git status --short
  exit 1
fi

if git diff --name-only --diff-filter=U | grep -q .; then
  echo "::error::unresolved conflicts remain; refusing to push"
  git diff --name-only --diff-filter=U
  exit 1
fi

if [ -z "$(git rev-parse HEAD)" ] || [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/${FORK_BRANCH}")" ]; then
  echo "::notice::nothing to push: HEAD matches origin/${FORK_BRANCH}"
  exit 0
fi

git push --force-with-lease origin "HEAD:refs/heads/${OUTPUT_BRANCH}"

echo "::notice::pushed ${OUTPUT_BRANCH}; ${FORK_BRANCH} is untouched"
echo "pushed_sha=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"
