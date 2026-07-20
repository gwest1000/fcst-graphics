# Lightning Development Archive

This archive preserves the forecast and observation data needed to evaluate and improve the BC lightning potential index.

## Forecast Target

The proposed predictand is occurrence of at least one observed lightning flash within 30 km during the three-hour period ending at the forecast-panel valid time. The archive retains undilated three-hour flash density, so other radii and thresholds can be tested without rebuilding the observation history.

F003 through F048 map cleanly to trailing three-hour blocks. F000 is archived as an environmental analysis but should not be treated as a three-hour forecast sample.

## Archive Root

Default location:

```text
/Volumes/Greg1_2tb/concrete_fcst_data/derived/lightning_ml
```

Set `LIGHTNING_ML_ARCHIVE_ROOT` or use the command-line archive-root option to override it.

The root contains:

```text
observations/eccc_lightning_3h/schema_v1/
model/hrdps_continental_5km/schema_v1/
baseline/hrdps_continental_lpi_5km/schema_v1/
status.json
```

## Observations

Every complete three-hour block is aggregated from the 18 ECCC ten-minute lightning-density GeoTIFFs in the interval `(start, end]`. Source units of flash/km2/minute are multiplied by ten and summed, producing total flash/km2 for the block.

The three-hour GeoTIFF and JSON manifest are written atomically and verified before the source files are removed. Incomplete blocks remain in the local mirror. The mirror skips ten-minute files whose block is already archived, preventing repeated downloads.

The LPI verification job reads three-hour aggregates first and falls back to ten-minute files, preserving existing verification graphics.

## HRDPS Predictors

Only continental 00Z and 12Z runs are archived. Fields are cropped to the existing BC plotting domain, sampled at 5 km, packed to fixed-precision int16 values, and compressed into one NPZ per forecast hour.

The schema includes:

- temperature and specific-humidity profiles at 20 pressure levels from 1000 to 250 hPa;
- U/V wind at 850, 700, 500 and 250 hPa;
- pressure vertical velocity at 1000, 850, 700, 500 and 250 hPa;
- CAPE, most-unstable lifted index and lifted index;
- surface pressure, MSLP, 2 m temperature/dewpoint and 10 m wind;
- PBL height, storm-relative helicity, precipitation rate and accumulation;
- column cloud water, total cloud cover, 500 hPa height and absolute vorticity.

Static latitude, longitude and terrain are stored once. A grid hash prevents incompatible grids from being mixed. Every forecast-hour sidecar records source filenames, clipping counts, a checksum and valid time. A run is complete only when all 17 hours and its manifest are present.

The handmade LPI is archived separately at 0.5-point precision with its formula version. This supports direct evaluation of later formula changes against the forecast that was actually issued.

## Operational Safety

The archiver runs before plotting cleanup. If the external archive is unavailable or a model hour fails validation, plotting continues but the raw eligible run is retained locally for retry. Observation source files are retained whenever aggregate archiving fails.

Each scheduled job first performs an atomic create/write/fsync/delete probe in the archive root. On macOS, a launchd `System Policy` denial for the external volume must be resolved by enabling Removable Volumes or Full Disk Access for the exact Python executable used by the agent. The current virtual environment resolves to:

```text
/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9
```

An interactive Terminal write is not a sufficient test; verify through the launchd job and inspect its status or log.

## Inspection

```bash
.venv/bin/python lightning_ml_archive.py status
```

The status reports observation blocks, model runs and hours, completed runs, baseline hours, archive bytes and filesystem free space.

Retained handmade-LPI cache files can be backfilled with:

```bash
.venv/bin/python lightning_ml_archive.py archive-baseline --run YYYYMMDDTHHZ
```

## Initial Review

After at least three weeks of convective-season observations, evaluate the handmade LPI before fitting ML:

1. Match each F003-F048 forecast to its trailing three-hour observed-density block.
2. Create the binary 30 km occurrence target from the undilated density grids.
3. Measure event frequency and reliability by LPI bin, forecast lead and local time.
4. Compare current LPI components for hits, misses and false alarms.
5. Adjust the handmade formulation only when a repeatable conditional bias appears across multiple storm days.

Do not introduce terrain, interior or marine priors during this first review. Use terrain and region only to stratify verification and identify whether the model fields already explain any differences.

The scheduled verification job declares the initial dataset ready at a 21-day archive span, at least 95% three-hour coverage, and at least 12 lightning-active blocks. It records readiness in `logs/state/lpi_verification.status.json` and sends a one-time macOS notification. This threshold starts an initial tuning review; it does not make three weeks sufficient for final out-of-sample validation.
