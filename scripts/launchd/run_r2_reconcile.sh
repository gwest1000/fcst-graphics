#!/bin/zsh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${FCST_R2_ENV_FILE:-${REPO_ROOT}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  print -u2 "R2 reconciliation is waiting for ${ENV_FILE}."
  exit 0
fi

set -a
source "${ENV_FILE}"
set +a

cd "${REPO_ROOT}"
exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/reconcile_r2_objects.py" --apply "$@"
