#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  print -u2 "R2 publisher is waiting for ${ENV_FILE}."
  exit 0
fi

set -a
source "${ENV_FILE}"
set +a

exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/automate_r2_publish.py" "$@"
