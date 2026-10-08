# Experimental ECMWF LPI and HRDPS transfer assessment

Implemented 7 October 2026. The fitted transfer correction **failed the held-out
assessment** and is not the default. Published ECMWF panels retain their current
display. The reconstructed ingredient recipe is available as an experimental
HRDPS-scale diagnostic, not an independently verified ECMWF lightning probability.

## Ingredients

`ecmwf_lpi.py` reconstructs the same ten HRDPS ingredients and twenty temporal
features. ECMWF IFS Open Data `oper/fc`, 0.25 degree:

- Temperature and relative humidity at 1000, 925, 850, 700, 600, 500, 400, 300,
  250 hPa. Five levels missing from the existing local download are retrieved by
  validated byte ranges from the source inventory. Log-pressure interpolation
  supplies the HRDPS charging-layer quadrature; it adds no new vertical information.
- Charging RH and pressure depth use the HRDPS temperature gates and weighting.
  Surface pressure clips underground layers. The subcloud humidity definition
  includes the surface and above-ground 850/800/750/700 hPa values.
- Approximate most-unstable LI selects the most buoyant lifted parcel among the
  surface and downloaded low-level parcels within 300 hPa of the surface. It is
  not an exact reconstruction of HRDPS MU virtual-temperature LI.
- MUCAPE, pressure velocity at 500/700 hPa converted to upward velocity, accumulated
  precipitation differences, instantaneous precipitation rate, 2-m temperature
  and dewpoint provide the remaining ingredients.
- Features are smoothed with Gaussian sigma 20 km using latitude-specific
  east-west distances. Missing ingredients remain missing.
- Three-hour windows use native endpoint snapshots; daily windows use native
  snapshots, with interval-weighted means and persistence. Hourly peaks are not
  invented. Source intervals become six hours after F144.

## Archive and execution

Persistent regional arrays live at:

`/Volumes/Greg1_2tb/concrete_fcst_data/derived/ecmwf_lpi/schema_v1/<stamp>/ingredients.npz`

They are outside the seven-day raw GRIB cleanup directory. Source inventory URLs,
hashes and selected message offsets persist in `source_profiles/*.source.json`.
Temporary global supplementary GRIB messages are removed after regional extraction.
The ECMWF control automation now attempts to retain ingredients for each processed
run before cleanup, without requiring an image regeneration. Archive failure is
logged separately and does not invalidate completed public plots.

Examples, from the repository root:

```sh
.venv/bin/python scripts/run_ecmwf_lpi.py --stamp 20261007T12Z --hours 360 --archive-only
.venv/bin/python scripts/run_ecmwf_lpi.py --stamp 20261007T12Z --hours 24 --duration 24h
.venv/bin/python scripts/run_ecmwf_lpi.py --stamp 20261007T12Z --hours 24 --duration 24h --apply-experimental-transfer
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/calibrate_ecmwf_lpi.py
```

The final example rebuilds the fixed Oct 1–7 study. The experimental transfer flag
explicitly applies the rejected correction for research comparisons. Default
outputs use the unadjusted HRDPS recipe. Complete daily windows are available
through F360; applying the teacher recipe beyond day one is an unverified
extrapolation. The CLI rejects three-hour products after F144 and incomplete daily
windows before F024. F000 is an instantaneous diagnostic in the runtime, excluded
from fitting; no three-hour occurrence claim is made for that snapshot.

## Assessment

`scripts/calibrate_ecmwf_lpi.py` rebuilds HRDPS teacher probabilities from the frozen
hourly ingredient archive using `bc_lpi_v4_random32`. Source file hashes are saved.
All seven October 1–7 12Z ECMWF runs have complete day-one ingredients. October 3
has insufficient finite HRDPS probability support to yield matched footprints,
leaving **six usable weather days**.

Each native ECMWF cell is compared with its HRDPS footprint mean, requiring at
least three archived points and 80% finite teacher coverage. BC land cells only;
scores weight cell area and give each usable issued run equal total weight.
Coverage averages approximately 22.5% for daily products and 7.7% for three-hour
products across all seven archived runs. Missing support is not no-lightning.

Two correction families are compared: log-odds intercept/slope, and a regularized
log-odds residual using the twenty ingredient features. Each has three shrinkage
strengths. Nested whole-run withholding chooses the correction on the other five
usable runs, then evaluates the withheld sixth. No pixels or windows from the
withheld run enter fitting or method selection.

Held-out daily RMSE worsens from **3.28 to 4.18 percentage points**, and correlation
falls from **0.318 to 0.050**. Daily mean bias improves from −0.83 to +0.11 points.
Three-hour RMSE worsens from **2.24 to 2.36 points**. The correction mainly changes
the average scale; it does not reliably reproduce HRDPS's weather pattern.

Recommendation: keep the correction disabled, retain ingredients for subsequent
independent lightning verification, and investigate the sparse HRDPS support
before interpreting this as an all-BC assessment. More spatial samples do not
replace independent weather days or seasonal coverage.

Readable report:
https://gregs-mac-studio.tailb1004b.ts.net:8454/ecmwf_lpi_transfer.html

Machine results: `output/ecmwf_lpi_transfer_20261007/results.json`.
Fitted research coefficients: `models/lpi/ecmwf_lpi_transfer_v1.json`.
