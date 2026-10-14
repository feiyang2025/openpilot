#!/usr/bin/env bash
# Point the opendbc / panda submodules at the matching branch on the upstream
# owner's fork, so the child repos do not lag behind the parent.
#
# A submodule pointer that does not move with the merge is the single most common
# cause of a build that passes CI and then fails on the device.
set -euo pipefail

OUTPUT_BRANCH="${OUTPUT_BRANCH:?}"
# Mirror the branches .gitmodules declares. Note opendbc tracks tn-c3, not
# master-c3 - reading it from .gitmodules rather than hardcoding it keeps this
# correct when upstream renames a line.
SUBMODULE_FORK="${SUBMODULE_FORK:-https://github.com/mouxangithub}"

declare -A REPO_NAMES=(
  [opendbc_repo]="opendbc"
  [panda]="panda"
  [jetlink_repo]="jetlink"
)

declared_branch() {
  # .gitmodules stanza for a submodule path -> its declared branch, or empty.
  awk -v want="path = $1" '
    $0 ~ want { inblock = 1; next }
    inblock && /^\[/ { inblock = 0 }
    inblock && /branch *= */ {
      gsub(/^[ \t]*branch *= */, "")
      gsub(/[\r\n]/, "")
      print
      exit
    }
  ' .gitmodules
}

changed=0
for path in "${!REPO_NAMES[@]}"; do
  [ -f ".gitmodules" ] || { echo "::warning::no .gitmodules, skipping"; break; }
  if ! grep -q "path = ${path}$" .gitmodules; then
    echo "::notice::${path} is not a submodule in .gitmodules, skipping"
    continue
  fi

  repo="${REPO_NAMES[$path]}"
  branch="$(declared_branch "$path")"
  if [ -z "$branch" ]; then
    echo "::warning::${path} declares no branch in .gitmodules; leaving it untouched"
    continue
  fi

  url="${SUBMODULE_FORK}/${repo}.git"
  echo "::group::${path} -> ${url} ${branch}"

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
