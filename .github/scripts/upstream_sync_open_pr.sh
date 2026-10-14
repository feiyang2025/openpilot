#!/usr/bin/env bash
# Open a DRAFT PR OUTPUT_BRANCH -> FORK_BRANCH. Never auto-merges.
#
# A draft is the whole point: control code cannot be validated by CI. The owner
# has to drive the car before this lands.
set -euo pipefail

FORK_BRANCH="${FORK_BRANCH:?}"
OUTPUT_BRANCH="${OUTPUT_BRANCH:?}"
TITLE="${1:-sync: upstream update}"
EXTRA_BODY="${OPEN_PR_EXTRA_BODY:-}"
SYNC_TEST_PASSED="${SYNC_TEST_PASSED:-false}"
SYNC_TEST_LOG="${SYNC_TEST_LOG:-}"

repo="${GITHUB_REPOSITORY:?}"
sha="$(git rev-parse HEAD)"

status_line="**validation:** $([ "$SYNC_TEST_PASSED" = "true" ] && echo 'passed' || echo 'FAILED — see the workflow summary')"

if [ -n "$SYNC_TEST_LOG" ] && [ -f "$SYNC_TEST_LOG" ]; then
  {
    echo "## validation log"
    echo
    echo '```'
    tail -200 "$SYNC_TEST_LOG"
    echo '```'
  } > "$RUNNER_TEMP/pr_log_section.md"
  log_section="$(cat "$RUNNER_TEMP/pr_log_section.md")"
else
  log_section=""
fi

body="$(cat <<EOF
Automated upstream sync. **Draft on purpose — do not merge without a road test.**

${status_line}

### What this PR is

\`${OUTPUT_BRANCH}\` = \`${FORK_BRANCH}\` merged with upstream. Conflicts were resolved by
opencode following \`.github/scripts/prompts/upstream_sync_openpilot.txt\`; verify the
resolution yourself, the AI's confidence is not evidence.

### Check before merging

1. **C3 / tici / DOS compatibility kept?** \`common/hardware/comma/hardware.py\`,
   \`is_tici_dos()\`, \`launch_chffrplus.sh:117-169\` (MCU detection exits 1 on
   failure — confirm it still does).
2. **Submodule pointers** for \`opendbc_repo\` and \`panda\` moved with the merge.
3. **Nothing from upstream was silently dropped.** AI-reported "removed duplicate"
   is the failure mode to look for.
4. **opendbc / safety / selfdrive high-risk areas** reviewed line by line.
5. **Road test.** CI cannot tell you whether longitudinal or lateral behaviour
   regressed.

${extra_bODY}

${log_section}

commit: ${sha}
EOF
)"

gh pr create \
  --repo "$repo" \
  --base "$FORK_BRANCH" \
  --head "$OUTPUT_BRANCH" \
  --title "$TITLE" \
  --body "$body" \
  --draft

echo "::notice::draft PR opened: ${OUTPUT_BRANCH} -> ${FORK_BRANCH}"
