# Forecast Graphics

Operational forecast-map generation and publication for:

- HRDPS 2.5 km
- HRDPS-West 1 km
- GEFS control
- ECMWF IFS control

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

## R2 publication

Synchronize retained frames for one model:

```bash
.venv/bin/python automate_r2_publish.py --model continental --once --sync-retained
```

Supported model publication groups are `continental`, `west`, `gefs_control`,
and `ecmwf_control`. Verification products are included in their associated
HRDPS manifests.

Install or refresh only the four independent R2 publication workers with:

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

## Web viewer

`site/config.json` lists the independently published model manifests. During
migration it can point to the legacy combined manifest. In production it points
to R2 URLs under `manifests/`.
