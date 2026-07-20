#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENT_DIR="${HOME}/Library/LaunchAgents"
LOG_DIR="${REPO_ROOT}/logs"
LABEL="com.greg.fcst-r2-usage-monitor"
TARGET="${AGENT_DIR}/${LABEL}.plist"
WEEKLY_LABEL="com.greg.fcst-r2-usage-weekly-report"
WEEKLY_TARGET="${AGENT_DIR}/${WEEKLY_LABEL}.plist"

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
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>${REPO_ROOT}/scripts/launchd/run_r2_usage_monitor.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>2</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>14</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/r2_usage_monitor.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/r2_usage_monitor.err.log</string>
</dict>
</plist>
PLIST

plutil -lint "${TARGET}" >/dev/null
launchctl bootout "gui/${UID}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID}" "${TARGET}"
launchctl enable "gui/${UID}/${LABEL}"
launchctl print "gui/${UID}/${LABEL}" | head -40

cat > "${WEEKLY_TARGET}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${WEEKLY_LABEL}</string>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>${REPO_ROOT}/scripts/launchd/run_r2_usage_monitor.sh</string>
    <string>--always-notify</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key>
    <integer>1</integer>
    <key>Hour</key>
    <integer>10</integer>
    <key>Minute</key>
    <integer>5</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/r2_usage_weekly_report.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/r2_usage_weekly_report.err.log</string>
</dict>
</plist>
PLIST

plutil -lint "${WEEKLY_TARGET}" >/dev/null
launchctl bootout "gui/${UID}/${WEEKLY_LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID}" "${WEEKLY_TARGET}"
launchctl enable "gui/${UID}/${WEEKLY_LABEL}"
launchctl print "gui/${UID}/${WEEKLY_LABEL}" | head -40
