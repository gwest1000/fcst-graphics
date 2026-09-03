#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ENV="${REPO_ROOT}/.env"

set -a
[[ -f "${PROJECT_ENV}" ]] && source "${PROJECT_ENV}"
TELEGRAM_ENV="${FCST_TELEGRAM_ENV_FILE:-/Users/greg/projects/facebook-marketplace-monitor/.env}"
[[ -f "${TELEGRAM_ENV}" ]] && source "${TELEGRAM_ENV}"
set +a

exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/monitor_pipeline_health.py" --operational "$@"
