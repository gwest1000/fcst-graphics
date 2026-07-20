# Regional New-Fire-Start Forecast Plan

Status: design plan, revised 2026-07-17

## Objective

Forecast the number of newly reported wildfires during each Pacific-time calendar day for:

- all of British Columbia;
- seven non-overlapping operational weather regions; and
- separate lightning-caused and human-caused components, plus their total.

The forecaster-facing result should support statements such as: **"10-20 new starts are likely in the Southwest and South-Central Interior today."** It should show an expected count and a calibrated likely range, not a false-precision integer.

## Decisions

- Use the HRDPS 2.5 km forecast as the operational weather input.
- Predict on an equal-area cell grid, then sum cells into regions and all of BC.
- Start by testing **20 km x 20 km cells (400 km2)**. This is not 20 km2. A 20 km2 cell would only be about 4.5 km on a side and is too fine relative to fire-report sparsity, lightning-location uncertainty, and practical forecast skill.
- Compare 10 km, 20 km, and 30 km square cells using blocked-year validation. Retain 20 km only if it calibrates as well as or better than the alternatives.
- Define forecast days as 00:00-24:00 America/Vancouver. Noon local standard time remains important for daily FWI-code anchoring, but it is not a useful boundary for the public meaning of "new starts today."
- Model lightning and human starts separately because their predictors and timing differ, even though the operational total is the primary product.
- Use three weeks, not two, as the minimum prospective lightning archive for the first LPI calibration attempt. This is enough for an initial diagnostic calibration during an active season, not enough to validate a fire-start model.

## Revised Regions

The regions must form a topologically valid partition of BC: no gaps, no overlaps, and every point assigned exactly once.

1. South Coast
2. Southwest and South-Central Interior
3. Columbia-Kootenay
4. Central Interior
5. North and Central Coast
6. Northern Interior
7. Northeast BC

The former Southern Interior is split between regions 2 and 3. Kinbasket Reservoir should be entirely within Columbia-Kootenay unless the final operational boundary map indicates otherwise.

The boundary workflow will use BC Albers (EPSG:3005), clip every region to the official BC boundary, snap shared boundaries, and test that the union equals BC and pairwise intersections have zero area. A user-provided operational map should be used to finalize the Coast Mountain and Rocky Mountain crest boundaries.

## Critique of the Earlier Plan

1. **The time requirement was too optimistic.** Two or three weeks can support an initial LPI-to-observed-lightning calibration, but not a credible fire-occurrence model. Fire-start fitting and validation need multiple historical fire seasons, including quiet and severe years.

2. **The proposed cell size was ambiguous.** The Canadian literature commonly uses 20 km x 20 km cells, not 20 km2 cells. Cell size should still be selected by validation rather than copied blindly.

3. **Regional averaging would hide important gradients.** Lytton and Vancouver cannot share one set of averaged predictors. Cell-level prediction followed by regional summation avoids that failure.

4. **Ignition and reporting were not separated strongly enough.** The operational target is newly reported starts. Lightning may ignite a fire that is not detected for several days, so a holdover/reporting-delay model is required.

5. **The uncertainty plan was incomplete.** Simply summing independent Poisson cells produces intervals that are too narrow during province-wide outbreak days. The regional simulation must include overdispersion, shared day effects, and spatially correlated error.

6. **The role of LPI needed tighter limits.** LPI estimates lightning occurrence, not fire ignition. It must feed an ignition/survival model alongside fuel moisture and storm rainfall, not act as a direct fire-count formula.

7. **The human-fire component was missing.** Human starts need a separate baseline based on season, access/exposure, day of week, and fire weather. Combining causes too early would make the model harder to diagnose and less portable.

## Modelling Unit

Create a fixed equal-area grid in BC Albers. Derive HRDPS predictors at native 2.5 km resolution, then summarize them into each candidate cell using physically appropriate statistics:

- area mean for broad moisture and temperature fields;
- maximum and upper percentiles for LPI and convective signals;
- storm-total precipitation and lightning exposure;
- forest/fuel area as an exposure term rather than treating ocean, rock, and forest equally.

The initial benchmark is a 20 km x 20 km cell. Candidate 10 km and 30 km grids will establish whether finer localization adds real skill or only more zero cells.

## Forecast Targets

For cell `i` and local calendar day `d`:

```text
expected_total_starts(i,d)
  = expected_lightning_arrivals(i,d)
  + expected_human_starts(i,d)
```

The target should use the best available discovery/first-report date. Ignition date is useful supporting information but is not interchangeable with report date. For the live BCWS feed, store snapshots and derive first-seen time so later source revisions do not erase the operational verification target.

## Lightning Component

Use two linked processes:

```text
expected_sustained_ignitions(i,d)
  = expected_flashes(i,d) * ignition_survival_probability(i,d)

expected_lightning_arrivals(i,d)
  = sum over lag k of:
      expected_sustained_ignitions(i,d-k)
      * reporting_delay_probability(k | weather and fuels)
```

`expected_flashes` comes from calibrated HRDPS LPI and, during retrospective evaluation, observed ECCC lightning. The ignition/survival probability should use continuous DMC and DC, FFMC, BUI, storm rainfall, VPD/RH, temperature, forest/fuel coverage, season, and geographic baseline risk. DMC should not be converted into a hard yes/no threshold.

Maintain daily ignition cohorts for up to 21 days. Subsequent rain, FFMC, VPD, wind, and drying influence whether a holdover becomes a reported fire on a later day.

Start with an interpretable hierarchical hurdle/count model: a cell-day occurrence probability plus a zero-truncated negative-binomial count conditional on occurrence. Compare it with a single negative-binomial model and a carefully regularized tree-based benchmark.

## Human Component

Fit a separate cell-day count model using:

- long-term spatial baseline and burnable area;
- roads, communities, recreation/access, and population exposure where licensing permits;
- season, weekday, and holidays;
- FFMC, VPD/RH, temperature, wind/gust, recent precipitation, and BUI/DMC/DC;
- regional partial pooling so data-poor northern cells do not get unstable coefficients.

Cause labels can be imperfect. Keep unknown-cause records as a separately audited group and test probabilistic allocation rather than silently dropping them.

## Regional Counts and Ranges

Sum cell expectations for each of the seven regions and BC. Generate the likely range by Monte Carlo simulation from the fitted count models while sampling shared day-level and spatial error. Report rounded operational bins, for example:

```text
Southwest and South-Central Interior
Expected: 14
Likely range: 10-20
Lightning component: 11
Human component: 3
```

The likely range should be calibrated to a declared coverage level, initially 70%, and tested for observed coverage by region and season.

## Validation

- Hold out entire years, not random cell-days, to prevent weather-event leakage.
- Include a leave-one-severe-season-out test.
- Score cell occurrence, regional daily count, BC daily count, and outbreak-day classification.
- Use reliability, mean absolute error, negative-binomial deviance/log score, interval coverage, and peak-day bias.
- Compare against climatology, persistence, lightning-only, and fire-weather-only baselines.
- Treat the recent 29-start day as a retrospective case study, never as a tuning target by itself.

## Implementation Phases

1. Keep the ECCC lightning and 00Z/12Z HRDPS feature archives healthy and measurable.
2. Finalize and validate the seven region polygons.
3. Build a historical fire table with cause, location, first-report/discovery date, and data-quality flags.
4. Join historical lightning, daily fuel-moisture codes, weather, fuels, and exposure to candidate grids.
5. Fit simple climatology and count baselines before adding LPI or holdovers.
6. Calibrate LPI to observed lightning, then add the lightning ignition and reporting-delay stages.
7. Fit the separate human-start component.
8. Validate cell size and models with blocked years.
9. Run prospectively without publishing for at least several weeks, then add a webpage table and cell-risk map.

## LPI Readiness Trigger

The scheduled LPI verification job now considers the initial tuning dataset ready when:

- the observation archive spans at least 21 days;
- at least 95% of expected three-hour blocks are present; and
- at least 12 blocks contain observed lightning.

It records these metrics in `logs/state/lpi_verification.status.json` and sends a one-time macOS notification to continue the LPI tuning and regional fire-start project. More data will still be required for stable final coefficients and independent validation.

## Deferred Related Product

Keep a separate future product for lightning near transmission corridors. It should buffer each named line/corridor, calculate observed and forecast lightning exposure within configurable distances, and report first/last strike time, density, and confidence. It should reuse the calibrated lightning layer but remain separate from the fire-start count model.

## Method References

- Magnussen and Taylor (2012), BC cell-day lightning fire occurrence models: <https://doi.org/10.1071/WF11088>
- Nadeem et al. (2020), observed- and forecast-lightning fire occurrence prediction in BC: <https://doi.org/10.1071/WF19058>
- Canadian Forest Service fire occurrence prediction guide: <https://publications.gc.ca/collections/collection_2021/rncan-nrcan/Fo123-2-26-2021-eng.pdf>
- Canadian Forest Service lightning fire occurrence system and holdover treatment: <https://publications.gc.ca/collections/collection_2013/rncan-nrcan/Fo122-1-60-2012-eng.pdf>
- Wotton and Wheatley (2025), continuous DMC relationship to lightning ignition probability: <https://doi.org/10.1071/WF24164>
