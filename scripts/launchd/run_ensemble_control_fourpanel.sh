#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${HOME}/projects/concrete_fcst/.venv/bin/python"

exec "${PYTHON_BIN}" "${REPO_ROOT}/automate_ensemble_control_fourpanel.py" "$@"
