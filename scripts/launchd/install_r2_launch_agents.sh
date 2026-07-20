#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENT_DIR="${HOME}/Library/LaunchAgents"
LOG_DIR="${REPO_ROOT}/logs"
R2_PUBLISHER="${REPO_ROOT}/scripts/launchd/run_r2_publisher.sh"

mkdir -p "${AGENT_DIR}" "${LOG_DIR}"

install_r2_agent() {
  local model="$1"
  local label="com.greg.fcst-r2-${model}"
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
  <key>ProcessType</key>
  <string>Background</string>
  <key>LowPriorityIO</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/r2_${model}.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/r2_${model}.err.log</string>
</dict>
</plist>
PLIST

  plutil -lint "${target}" >/dev/null
  launchctl bootout "gui/${UID}/${label}" 2>/dev/null || true
  launchctl bootstrap "gui/${UID}" "${target}"
  launchctl enable "gui/${UID}/${label}"
}

for model in continental west gefs_control ecmwf_control; do
  install_r2_agent "${model}"
done

launchctl print "gui/${UID}" | grep 'com.greg.fcst-r2-' || true
