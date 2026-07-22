#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  print -u2 "Fire-activity overlay publisher is waiting for ${ENV_FILE}."
  exit 0
fi

set -a
source "${ENV_FILE}"
set +a

export MPLCONFIGDIR="${REPO_ROOT}/logs/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/automate_fire_activity_overlay.py" "$@"
