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

## Web viewer

`site/config.json` lists the independently published model manifests. During
migration it can point to the legacy combined manifest. In production it points
to R2 URLs under `manifests/`.
