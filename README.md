# Forecast Graphics

Operational forecast-map generation and publication for:

- HRDPS 2.5 km
- GEFS control
- ECMWF IFS control
- ECMWF ENS 500 hPa mean and spread

The HRDPS surface products include three-hourly 2 m temperature for the
BC-wide HRDPS 2.5 km domain. HRDPS-West 1 km products were retired in
September 2026 ahead of the upstream model's announced decommissioning.

The model jobs are managed independently. They share plotting code, but each has
its own run locks, status, R2 upload state, and publication worker. GitHub Pages
serves the static viewer in `site/`; forecast imagery and run manifests are
stored in Cloudflare R2.

## Local setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Do not commit `.env` or R2 credentials.

## Runtime storage

Machine-level storage is configured in `~/.config/project-data.env`:

```text
PROJECT_DATA_ROOT=/Volumes/Greg1_2tb/project-data
```

Forecast Graphics then uses `${PROJECT_DATA_ROOT}/fcstGraphics/data` for
downloaded model fields and durable caches, and
`${PROJECT_DATA_ROOT}/fcstGraphics/plots` for generated frames. Static tracked
map assets remain in the repository's `data/` directory. The project-specific
`FCSTGRAPHICS_DATA_ROOT` and `FCSTGRAPHICS_PLOTS_ROOT` variables take
precedence; without any configured root, development commands fall back to the
repository's `data/` and `plots/` directories. A configured but unavailable
root is an error so a missing SSD cannot silently fill internal storage.

ECMWF ensemble-mean and spread GRIBs are durable forecast inputs owned by the
`concrete_fcst` archive rather than graphic-specific caches. Forecast Graphics
reads them from `${CONCRETE_FCST_DATA_ROOT}/raw/ecmwf/realtime` (or the data
root in `concrete_fcst/configs/project.toml`) and invokes the concrete archive
module when a 00Z or 12Z cycle is incomplete. `CONCRETE_FCST_REPO_ROOT`,
`CONCRETE_FCST_DATA_ROOT`, and `CONCRETE_FCST_PYTHON` can override the local
repository, data, and interpreter paths.

## R2 publication

Synchronize retained frames for one model:

```bash
.venv/bin/python automate_r2_publish.py --model continental --once --sync-retained
```

Supported model publication groups are `continental`, `gefs_control`,
`ecmwf_control`, and `ecmwf_ensemble`. Verification products are included in
their associated HRDPS manifests.

Install or refresh the independent R2 publication workers with:

```bash
scripts/launchd/install_r2_launch_agents.sh
```

The workers run every three minutes. They publish complete PNG files only,
compress and upload two files in parallel by default, and update a model's
manifest only after its images are available. Set `FCST_R2_UPLOAD_WORKERS` in
`.env` to tune that bounded concurrency.

## Free-tier monitoring

`monitor_r2_usage.py` checks Cloudflare's account-wide analytics for current R2
storage plus billing-period Class A and Class B operations, projects operation
usage through the end of the billing period, and sends a macOS notification at
70% projected usage (critical at 90%). After three consecutive check failures it
also alerts that monitoring is unavailable. The analytics-only check does not
consume an R2 operation.
Install its four-times-daily launch agent with:

```bash
scripts/launchd/install_r2_usage_monitor.sh
```

The default checks run at 02:00, 10:00, 14:00, and 22:00 local time. Current
results are written to `logs/r2_usage_latest.json`. A separate weekly heartbeat
runs every Monday at 10:05 local time and always sends a usage notification,
including when usage is healthy. R2 credentials and the Cloudflare analytics
token are kept in macOS Keychain.
The publisher token is intentionally limited to bucket object read/write. Bucket
CORS and lifecycle configuration therefore requires a separate administrative
credential if `configure_r2_bucket.py` must be rerun.

To apply the forecast and radar bucket lifecycle backstops with a temporary R2
Admin Read & Write credential, run:

```bash
scripts/configure_r2_lifecycle_with_temporary_admin.zsh
```

The helper prompts locally for the Access Key ID and hidden Secret Access Key,
applies both bucket configurations, and does not store the administrative
credential. Revoke the temporary token after the command succeeds.

## Pipeline health monitoring

`monitor_pipeline_health.py` checks the complete forecast pipeline every 10 minutes:
required launch agents, the external data volume, public R2 model manifests,
BCWS fire activity, the ECCC lightning archive, and daily CWFIS FFMC/DMC/DC
anchors. Missing launch agents are reloaded automatically. Disk loss and failed
scheduler repair are eligible for alerts immediately, subject to the hourly limit
below. Missing or incomplete model runs only alert once they are at least one hour
past their website-ready target. This also applies to failed, stopped, or stalled
jobs; a fresh heartbeat does not suppress an overdue-run alert.
Targets reflect local processing schedules, not guarantees from the model providers:

- HRDPS: 5.5 hours after each 00/06/12/18Z initialization.
- GEFS 00Z: 06:00 Pacific.
- ECMWF control: 06:30 Pacific for 00Z; 14:00 Pacific for 12Z.
- ECMWF ensemble: 07:00 Pacific for 00Z; 14:30 Pacific for 12Z.

Alerts name the model, dated run, hours past the target, plain-English cause,
and the affected products. When only part of a run is missing, alerts name the
missing graphics and any missing-frame count, and list what is already available.
A missing fire-danger product is not described as a missing weather forecast;
recorded CWFIS failures are distinguished from other fire-danger calculation errors.
Publication requires every expected forecast hour and product; a partial upload
does not count as recovery. A newer complete run
supersedes historical failed cycles. Each dated run has its own incident identity.
All health notifications are limited to one per rolling hour, including new or
escalating incidents, recovery messages, and daily summaries. Problems detected
during that hour remain pending and are reported on the next eligible check if
still present. The daily summary is skipped if another notification was sent
within the preceding hour.

CWFIS monitoring validates the actual BC-domain FFMC/DMC/DC GeoTIFF files, including
georeferencing and plausible values, for one common observation date. Empty date
directories, temporary downloads, corrupt files, and future-dated inputs do not
count as usable data. An early warning begins at 36 hours after the observation's
20Z reference time, leaving 12 hours before the shared 48-hour initialization limit;
the warning gives the expiry time and states what will be affected. This allows
for the usual evening download while warning before fire-danger guidance is lost.
Inputs older than 48 hours are critical, rather than merely showing old date labels.
The monitor also requests a tiny 16-by-16 GeoTIFF sample for each fuel-moisture field
every ten minutes, using a known usable date. HTTP 200 responses containing XML
errors do not count as success. Export-service failures persisting for 55 minutes
trigger an early warning even when cached inputs remain fresh. These probes are
read-only and never replace forecast input caches. All warnings retain the hourly
notification limit.

Model-run reminders repeat every 4 hours; other critical incidents repeat every
6 hours and warnings every 24 hours. Successful automatic repairs and transient
remote failures must persist for 55 minutes before alerting, and recovery must
remain stable for 55 minutes. These are elapsed-time checks, not poll counts. Related
public-manifest or scheduler failures are grouped into one actionable incident,
with a bounded message size and a link to the graphics page. A shared lock
prevents the ten-minute and daily processes from racing. Each run is retained in a
rotating JSONL history so an alert can be investigated after recovery. Notification
history includes the exact message text and Telegram's message ID/acceptance time.
Acceptance confirms Telegram received the message, not that a phone displayed it.
Only the launch-agent wrapper passes `--operational`; direct/manual invocations
are isolated from durable incident state, cannot repair services, and cannot
send Telegram alerts.

Telegram alerts reuse `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from the
existing Monitor Search environment, so credentials are not copied into this
repository. A concise daily report is scheduled at 07:05 Pacific even when every
check is healthy, subject to the same hourly limit. Install or refresh both launch
agents with:

```bash
scripts/launchd/install_pipeline_health_monitor.sh
```

## Live fire activity

Fire-weather forecast fields are immutable base PNGs. Current BCWS and NIFC
incidents are rendered separately into a transparent, map-aligned overlay and
reused across every forecast hour. Install the hourly retrieval and R2 publisher
with:

```bash
scripts/launchd/install_fire_activity_overlay.sh
```

The viewer checks the small live manifest every hour and displays its
overlay only for a product selected in `latest` mode. Unchanged PNGs are not
uploaded again; the manifest observation time is refreshed every hour.

## Web viewer

`site/config.json` lists the independently published model manifests and the
live fire-activity manifest. During
migration it can point to the legacy combined manifest. In production it points
to R2 URLs under `manifests/`.
