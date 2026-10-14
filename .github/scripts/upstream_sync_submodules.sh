#!/usr/bin/env bash
# Point the opendbc / panda submodules at the matching branch on the upstream
# owner's fork, so the child repos do not lag behind the parent.
#
# A submodule pointer that does not move with the merge is the single most common
# cause of a build that passes CI and then fails on the device.
set -euo pipefail

OUTPUT_BRANCH="${OUTPUT_BRANCH:?}"
SUBMODULE_FORK="${SUBMODULE_FORK:-https://github.com/mouxangithub}"

declare -A BRANCHES=(
  [opendbc_repo]="master-c3"
  [panda]="master-c3"
)

changed=0
for path in "${!BRANCHES[@]}"; do
  branch="${BRANCHES[$path]}"
  if [ ! -f ".gitmodules" ]; then
    echo "::warning::no .gitmodules, skipping"
    exit 0
  fi
  if ! grep -q "path = ${path}$" .gitmodules; then
    echo "::notice::${path} is not a submodule in .gitmodules, skipping"
    continue
  fi

  url="${SUBMODULE_FORK}/${path/opendbc_repo/opendbc}.git"
  echo "::group::${path} -> ${url} ${branch}"

  git submodule sync -- "${path}" >/dev/null 2>&1 || true
  git config "submodule.${path}.url" "${url}"

  if git ls-remote --exit-code --heads "${url}" "${branch}" >/dev/null 2>&1; then
    if git submodule update --init --remote --depth=1 -- "${path}" >/dev/null 2>&1; then
      git add "${path}"
      changed=1
      echo "updated ${path} -> $(git rev-parse --short HEAD "${path}" 2>/dev/null || echo unknown)"
    else
      echo "::warning::could not update ${path}; leaving the existing pointer in place"
    fi
  else
    echo "::warning::${url} has no branch '${branch}'; leaving ${path} untouched"
  fi
  echo "::endgroup::"
done

if [ "$changed" = "1" ]; then
  git diff --cached --name-only -- . | sed 's/^/submodule pointer changed: /'
else
  echo "no submodule pointers changed"
fi
