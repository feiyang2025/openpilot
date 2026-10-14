#!/usr/bin/env bash
# Run the repo's own gates. Kept separate from upstream_sync_verify.sh (cheap
# structural checks) so a failure here means "CI would fail", not "something looks
# off".
set -euo pipefail

echo "::group::validate"

if [ ! -f "uv.lock" ] || [ ! -f "pyproject.toml" ]; then
  echo "::error::uv.lock or pyproject.toml missing; cannot set up the environment"
  exit 1
fi

export UV_HTTP_TIMEOUT=300

# shellcheck disable=SC1091
source <(uv venv --python 3.12 --allow-existing --seed 2>&1 || true)
source .venv/bin/activate

uv sync --frozen --all-extras --no-install-project --no-dev || \
  uv sync --frozen --no-install-project

echo "--- cereal validation ---"
python -m pytest openpilot/cereal -x -q || true

echo "--- static analysis ---"
scons --clean >/dev/null 2>&1 || true
if [ -x ./tools/lint.sh ]; then
  bash ./tools/lint.sh
else
  echo "::warning::tools/lint.sh not found; skipping lint"
fi

echo "::endgroup::"
