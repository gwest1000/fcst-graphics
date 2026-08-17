#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENT_DIR="${HOME}/Library/LaunchAgents"
LOG_DIR="${REPO_ROOT}/logs"
RUNNER="${REPO_ROOT}/scripts/launchd/run_pipeline_health_monitor.sh"

mkdir -p "${AGENT_DIR}" "${LOG_DIR}"

install_agent() {
  local label="$1"
  local schedule="$2"
  local argument="$3"
  local log_name="$4"
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
    <string>${RUNNER}</string>
    ${argument}
  </array>
  ${schedule}
  <key>ProcessType</key>
  <string>Background</string>
  <key>LowPriorityIO</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/${log_name}.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/${log_name}.err.log</string>
</dict>
</plist>
PLIST

  plutil -lint "${target}" >/dev/null
  launchctl bootout "gui/${UID}/${label}" 2>/dev/null || true
  launchctl bootstrap "gui/${UID}" "${target}"
  launchctl enable "gui/${UID}/${label}"
}

DAILY_SCHEDULE='<key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>7</integer>
    <key>Minute</key>
    <integer>5</integer>
  </dict>'
HOURLY_SCHEDULE='<key>StartInterval</key>
  <integer>3600</integer>
  <key>RunAtLoad</key>
  <true/>'

# Install the daily service first so the hourly RunAtLoad check sees both labels.
install_agent \
  "com.greg.fcst-pipeline-health-daily" \
  "${DAILY_SCHEDULE}" \
  '<string>--always-notify</string>' \
  "pipeline_health_daily"
install_agent \
  "com.greg.fcst-pipeline-health" \
  "${HOURLY_SCHEDULE}" \
  '' \
  "pipeline_health"

launchctl print "gui/${UID}/com.greg.fcst-pipeline-health" | head -40
