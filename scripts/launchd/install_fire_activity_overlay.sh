#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENT_DIR="${HOME}/Library/LaunchAgents"
LOG_DIR="${REPO_ROOT}/logs"
LABEL="com.greg.fcst-fire-activity-overlay"
TARGET="${AGENT_DIR}/${LABEL}.plist"

mkdir -p "${AGENT_DIR}" "${LOG_DIR}"

cat > "${TARGET}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
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
    <string>${REPO_ROOT}/scripts/launchd/run_fire_activity_overlay.sh</string>
  </array>
  <key>StartInterval</key>
  <integer>3600</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>LowPriorityIO</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/fire_activity_overlay.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/fire_activity_overlay.err.log</string>
</dict>
</plist>
PLIST

plutil -lint "${TARGET}" >/dev/null
launchctl bootout "gui/${UID}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID}" "${TARGET}"
launchctl enable "gui/${UID}/${LABEL}"
launchctl print "gui/${UID}/${LABEL}" | head -40
