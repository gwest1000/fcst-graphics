#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENT_DIR="${HOME}/Library/LaunchAgents"
LOG_DIR="${REPO_ROOT}/logs"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
AUTOMATION="${REPO_ROOT}/automate_hrdps_west.py"
ENSEMBLE_RUNNER="${REPO_ROOT}/scripts/launchd/run_ensemble_control_fourpanel.sh"
LPI_VERIFICATION_AUTOMATION="${REPO_ROOT}/automate_lpi_verification.py"
FIRE_DANGER_VERIFICATION_AUTOMATION="${REPO_ROOT}/automate_fire_danger_verification.py"
R2_PUBLISHER="${REPO_ROOT}/scripts/launchd/run_r2_publisher.sh"

mkdir -p "${AGENT_DIR}" "${LOG_DIR}"

install_agent() {
  local label="$1"
  local model="$2"
  local cycle="$3"
  local hour="$4"
  local minute="$5"
  local wait_minutes="$6"
  local max_runtime_minutes="$7"
  local log_prefix="$8"
  local target="${AGENT_DIR}/${label}.plist"

  cat > "${target}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${label}</string>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${HOME}</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>${AUTOMATION}</string>
    <string>--model</string>
    <string>${model}</string>
    <string>--cycle</string>
    <string>${cycle}</string>
    <string>--wait-minutes</string>
    <string>${wait_minutes}</string>
    <string>--poll-minutes</string>
    <string>5</string>
    <string>--max-runtime-minutes</string>
    <string>${max_runtime_minutes}</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${hour}</integer>
    <key>Minute</key>
    <integer>${minute}</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/${log_prefix}.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/${log_prefix}.err.log</string>
</dict>
</plist>
PLIST

  plutil -lint "${target}" >/dev/null
  launchctl bootout "gui/${UID}/${label}" 2>/dev/null || true
  sleep 1
  if ! launchctl bootstrap "gui/${UID}" "${target}"; then
    sleep 2
    launchctl bootstrap "gui/${UID}" "${target}"
  fi
  launchctl enable "gui/${UID}/${label}"
}

install_agent "com.greg.hrdps-west-convective-00" "west" "00" "17" "30" "690" "900" "hrdps_west_00"
install_agent "com.greg.hrdps-west-convective-12" "west" "12" "5" "30" "690" "900" "hrdps_west_12"
install_agent "com.greg.hrdps-continental-00" "continental" "00" "17" "30" "360" "540" "hrdps_continental_00"
install_agent "com.greg.hrdps-continental-06" "continental" "06" "23" "30" "360" "540" "hrdps_continental_06"
install_agent "com.greg.hrdps-continental-12" "continental" "12" "5" "30" "360" "540" "hrdps_continental_12"
install_agent "com.greg.hrdps-continental-18" "continental" "18" "11" "30" "360" "540" "hrdps_continental_18"

install_ensemble_agent() {
  local label="$1"
  local model="$2"
  local cycle="$3"
  local hour="$4"
  local minute="$5"
  local log_prefix="$6"
  local target="${AGENT_DIR}/${label}.plist"

  cat > "${target}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${label}</string>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${HOME}</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>${ENSEMBLE_RUNNER}</string>
    <string>--model</string>
    <string>${model}</string>
    <string>--cycle</string>
    <string>${cycle}</string>
    <string>--max-runtime-minutes</string>
    <string>360</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${hour}</integer>
    <key>Minute</key>
    <integer>${minute}</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/${log_prefix}.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/${log_prefix}.err.log</string>
</dict>
</plist>
PLIST

  plutil -lint "${target}" >/dev/null
  launchctl bootout "gui/${UID}/${label}" 2>/dev/null || true
  sleep 1
  if ! launchctl bootstrap "gui/${UID}" "${target}"; then
    sleep 2
    launchctl bootstrap "gui/${UID}" "${target}"
  fi
  launchctl enable "gui/${UID}/${label}"
}

install_ensemble_agent "com.greg.gefs-control-fourpanel-00" "gefs_control" "0" "5" "0" "gefs_control_00"
install_ensemble_agent "com.greg.ecmwf-control-fourpanel-00" "ecmwf_control" "0" "5" "30" "ecmwf_control_00"

install_interval_agent() {
  local label="$1"
  local interval_seconds="$2"
  local automation="$3"
  local log_prefix="$4"
  local target="${AGENT_DIR}/${label}.plist"

  cat > "${target}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${label}</string>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${HOME}</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>${automation}</string>
  </array>
  <key>StartInterval</key>
  <integer>${interval_seconds}</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/${log_prefix}.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/${log_prefix}.err.log</string>
</dict>
</plist>
PLIST

  plutil -lint "${target}" >/dev/null
  launchctl bootout "gui/${UID}/${label}" 2>/dev/null || true
  sleep 1
  if ! launchctl bootstrap "gui/${UID}" "${target}"; then
    sleep 2
    launchctl bootstrap "gui/${UID}" "${target}"
  fi
  launchctl enable "gui/${UID}/${label}"
}

install_interval_agent "com.greg.lpi-verification" "600" "${LPI_VERIFICATION_AUTOMATION}" "lpi_verification"
install_interval_agent "com.greg.fire-danger-verification" "1800" "${FIRE_DANGER_VERIFICATION_AUTOMATION}" "fire_danger_verification"

install_r2_agent() {
  local model="$1"
  local target="${AGENT_DIR}/com.greg.fcst-r2-${model}.plist"
  cat > "${target}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.greg.fcst-r2-${model}</string>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>${R2_PUBLISHER}</string>
    <string>--model</string>
    <string>${model}</string>
    <string>--once</string>
    <string>--sync-retained</string>
  </array>
  <key>StartInterval</key>
  <integer>180</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/r2_${model}.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/r2_${model}.err.log</string>
</dict>
</plist>
PLIST
  plutil -lint "${target}" >/dev/null
  launchctl bootout "gui/${UID}/com.greg.fcst-r2-${model}" 2>/dev/null || true
  launchctl bootstrap "gui/${UID}" "${target}"
  launchctl enable "gui/${UID}/com.greg.fcst-r2-${model}"
}

install_r2_agent "continental"
install_r2_agent "west"
install_r2_agent "gefs_control"
install_r2_agent "ecmwf_control"

launchctl print "gui/${UID}" | grep 'com.greg' || true
