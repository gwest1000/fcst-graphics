#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FCST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RADAR_ROOT="${RADARSAT_PROJECT_ROOT:-/Users/greg/projects/radar-sat}"

if [[ ! -f "${FCST_ROOT}/.env" ]]; then
  print -u2 "Missing ${FCST_ROOT}/.env."
  exit 2
fi
if [[ ! -x "${FCST_ROOT}/.venv/bin/python" ]]; then
  print -u2 "Missing forecast-graphics Python environment."
  exit 2
fi
if [[ ! -x "${RADAR_ROOT}/.venv/bin/python" ]]; then
  print -u2 "Missing radar-sat Python environment at ${RADAR_ROOT}."
  exit 2
fi

read "admin_access_key?Temporary R2 Admin Access Key ID: "
read -s "admin_secret_key?Temporary R2 Admin Secret Access Key: "
print

if [[ -z "${admin_access_key}" || -z "${admin_secret_key}" ]]; then
  print -u2 "Both fields are required; no bucket configuration was changed."
  exit 2
fi

cleanup() {
  unset admin_access_key admin_secret_key
}
trap cleanup EXIT INT TERM

set -a
source "${FCST_ROOT}/.env"
set +a

print "Applying forecast-graphics CORS and lifecycle rules..."
(
  cd "${FCST_ROOT}"
  FCST_R2_ACCESS_KEY_ID="${admin_access_key}" \
  FCST_R2_SECRET_ACCESS_KEY="${admin_secret_key}" \
    .venv/bin/python configure_r2_bucket.py
)

print "Applying radar-sat CORS and lifecycle rules..."
(
  cd "${RADAR_ROOT}"
  RADARSAT_R2_ACCESS_KEY_ID="${admin_access_key}" \
  RADARSAT_R2_SECRET_ACCESS_KEY="${admin_secret_key}" \
  PYTHONPATH="${RADAR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    .venv/bin/python scripts/configure_r2.py
)

print "Both R2 bucket configurations were applied. Revoke the temporary admin token."
