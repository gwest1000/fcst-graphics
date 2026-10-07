# LPI Verification and Tuning Plan

Revised 2026-10-05. This is the execution plan for Codex to evaluate the 2026 HRDPS Continental lightning archive, investigate timing and location errors, and assess whether seasonal or solar information improves guidance. The next work is objective verification and controlled experiments. Forecaster labeling and category calibration are deferred until labels are available.

The main questions are whether high LPI precedes observed lightning by one or more three-hour blocks, whether day two benefits from more spatial smoothing or a different displacement tolerance, and whether a seasonal or solar signal remains after accounting for LPI, forecast lead and local time. Treat these as hypotheses to test. Candidate changes require improvement on observations excluded from fitting and selection.

## Data available and limits

Use the configured archive root, currently `/Volumes/Greg1_2tb/project-data/fcstGraphics/data/lightning_ml`. Resolve it through `project_paths` or an explicit archive-root argument. The August analysis summary refers to an older path that is no longer present.

The October 4 inventory provides the following reference snapshot. Refresh the inventory and freeze an observation cutoff before running the experiments, because archiving continues.

| Archive | October 4 inventory | Role in this study |
| --- | --- | --- |
| ECCC three-hour density grids | 704 of 709 expected blocks, July 8 03Z to October 4 15Z; 83 complete operational days | Observed occurrence and density, alternative distances, and three-hour lag targets |
| Issued LPI baseline | 5,056 fields across 300 runs; 295 runs contain all F000 to F048 fields at three-hour spacing | Full forecast-horizon timing and spatial experiments |
| Hourly LPI ingredients | 6,660 snapshots; 274 of 285 runs complete | Recompute formula candidates and three-hour maxima within the retained operational window |
| Pressure-profile predictors | 1,306 snapshots; 161 of 167 runs complete, from 00Z and 12Z cycles | Deeper ingredient diagnostics at the retained three-hour snapshots |

The October 4 sample selected by the existing ingredient-analysis rules contained 2,145 matched three-hour forecasts, 556 distinct observation blocks and 251 complete daily forecasts from individual runs. It covered 71 operational dates, including 57 dates with all four issue cycles and all eight verification blocks. Operational dates August 29 through September 1 had no samples under those complete-run rules. Missing observation blocks ended at 00Z on July 25, August 6, September 4, September 16 and September 27. Missing blocks remain missing rather than becoming zero-lightning cases.

A separate October 5 feasibility count, with observations available through 21Z, found 2,394 observed matches for F003 to F024 and 2,339 for F027 to F048 in the baseline archive. There were 2,257 pairs with the same observed block, issue cycle and leads separated by 24 hours. These are preliminary availability counts before formula-version, common-domain and final-cutoff exclusions. They establish that day-one versus day-two tests are feasible.

The baseline archive includes both `bc_lpi_v2` and `bc_lpi_v3_3hmax`. Use v3 for the primary comparisons and report older versions separately. The first complete 12Z-to-12Z operational day retained in the hourly ingredient archive is not equivalent to retaining the entire 48-hour horizon. Consequently, full-horizon tests of the issued field are feasible, while full-horizon reconstruction of every ingredient candidate is not guaranteed. Reuse additional retained inputs only where their provenance and completeness can be established.

The observations are aggregated three-hour flash-density grids. They support three-hour lag tests and flash occurrence within a distance, but do not provide individual strike coordinates or exact strike times. Subhour timing tests require a separate complete archive of finer-time observations; the aggregates cannot be disaggregated. July through October covers part of one season, not an annual cycle.

## Execution order

1. Freeze and audit the usable data, then build a shared sample table and fixed validation splits.
2. Establish the current LPI benchmark and quantify support by lead, time, region and season.
3. Investigate timing lags while holding spatial treatment fixed.
4. Test spatial smoothing and displacement tolerance by forecast horizon on matched samples.
5. Test a small set of combined timing and spatial candidates selected from the earlier stages.
6. Investigate seasonal and solar residuals, then test limited probability-calibration extensions.
7. Revisit ingredient tuning on the subset with complete ingredients and report decisions with uncertainty.

Each stage must produce its results even if later stages lack sufficient data. Record unsupported comparisons explicitly and continue those supported by the archive. Do not infer missing second-day ingredients from first-day fields or combine hours from different issued runs to manufacture one forecast.

## Shared sample and benchmark

Create one inventory row per issued run and forecast field, with initialization time, cycle, forecast hour, valid interval, formula version, grid identity, archive path and data-quality flags. Link observed blocks separately for each lag and target definition. Record missing fields, missing observations, out-of-domain cells and incomplete daily windows independently. Check sidecars, timestamp conventions, field packing, grid compatibility and masks before reading values into a common sample.

Define the forecast interval for a field valid at `v` as `(v minus 3 hours, v]`. Confirm from the cache and archiving code that F003 and later contain the trailing maximum of three instantaneous hourly LPI calculations. Exclude F000 from three-hour interval scores because it is instantaneous.

Use forecast day one as F003 to F024, covering `(initialization, initialization plus 24 hours]`, and day two as F027 to F048, covering the subsequent 24 hours. Also report lead bands F003 to F012, F015 to F024, F027 to F036 and F039 to F048. Keep the 12Z-to-12Z operational daily analysis as a separate aggregation; it must not be labeled interchangeably with initialization-relative day one or day two.

The primary domain is BC grid cells within 30 km of BCH transmission lines. Verify all BC as a secondary domain and stratify maritime versus Interior, local solar time, issue cycle and lead. Keep geographic variables diagnostic at this stage. Missing observation coverage must be masked rather than assigned no lightning. Audit the existing observation reader before reusing it: it currently assigns zero to some invalid or out-of-grid locations.

The primary event is at least one observed flash within 30 km during the specified three-hour interval. Preserve the undilated density for distance experiments. Record the exact archive field and its spatial treatment. The current two-panel display uses 10 km Gaussian smoothing after the temporal maximum; archive-field scores and current-display scores need explicit, separate reference labels. Determine whether smoothing is already present before applying any additional operation, and do not smooth twice. Historical display settings may differ from current code.

Benchmark raw LPI discrimination, training-fitted monotone probabilities, climatology and a climatology conditioned on local time and horizon. Report Brier score and skill relative to the appropriate climatology, ROC AUC, precision-recall performance, reliability, event frequency, and hits, misses and false alarms at thresholds chosen using training or validation data. Report flash density as a supporting diagnostic. Include sample counts and active BCH-corridor days with every stratum; the readiness counter's BC-wide active blocks are not independent storm-day counts.

## Timing lag experiments

For an LPI interval ending at `v`, compare it with observed three-hour intervals ending at `v + delta`, using `delta = -6, -3, 0, +3, +6 hours`. A positive delta means lightning occurs later than the modeled LPI interval. Negative deltas diagnose forecasts that develop convection too late. They are diagnostic comparisons, not predictors based on future observations.

Keep the observation duration, 30 km target radius and spatial smoothing fixed during the first lag sweep. Use the common set of forecast intervals and cells with valid observations for all five offsets, alongside a coverage table for each offset. Missing observations near archive boundaries or gaps must not give one offset an easier sample. Calculate lag relationships separately by horizon, local solar time and maritime versus Interior regime. Inspect cases with LPI rising before the first observed active block, and cases with sustained lightning spanning several blocks.

Use lagged correlations and distributions of first-active-block offsets as diagnostics. Test forecast usefulness with the same probability and event scores used for the benchmark. Correlation alone cannot select a timing correction: storms persist across blocks and lightning has a strong diurnal cycle. Compare with time-conditioned climatology and, as a diagnostic control, suitably separated observation days matched by region and local time.

Translate a selected delay into an operational candidate explicitly. For example, to predict the observed block ending at `v` with a three-hour delay, use the LPI field ending at `v minus 3 hours` from the same issued model run. Evaluate zero delay, a fixed three-hour delay, a fixed six-hour delay, and at most one simple blend of current and preceding LPI fields. Specify the blend before scoring test data and fit its probability calibration on training data. Exclude leads whose required predecessor is unavailable; F000 cannot substitute for a three-hour field.

Store both the original field lead and the shifted target lead. Horizon-specific operational scores use the observed target's lead and stay within the 48-hour horizon. Diagnostic offsets extending beyond F048 may be shown separately when observations exist, but do not count them as day-two operational forecasts. Compare operational timing candidates on the same eligible target intervals and report the loss of early-lead coverage caused by unavailable predecessor fields.

As a separate sensitivity test, evaluate occurrence over the union of the current and next observed three-hour blocks. Label this a six-hour event target and compare against benchmarks predicting the identical six-hour target. Its higher event frequency must not be presented as better three-hour skill. Any 24-hour lag-adjusted product must retain a declared fixed daily window and account for overlapping shifted blocks.

## Spatial tests by forecast horizon

Test Gaussian smoothing after the temporal maximum at sigma values of 0, 5, 10, 15 and 20 km. Here sigma is the Gaussian standard deviation, not a neighborhood radius. Retain the existing 30 and 40 km settings only as sensitivity tests where the actual retained grid and finite-value buffer provide adequate support. Record kernel truncation and boundary treatment. Hourly fields are limited to BC plus 50 km, so larger kernels can extend beyond the available buffer. Use a common domain with adequate kernel support for every compared setting, report excluded edge areas, and check sensitivity to boundary handling. Preserve valid zeros and exclude nodata from smoothing.

At the fixed 30 km target, compare one smoothing scale shared by both horizons with independently selected day-one and day-two scales. Fit probability calibration for each candidate on the same training dates. Select settings on validation dates, then evaluate them once on test dates. Day two may require more smoothing, but the zero-change and shared-scale candidates remain eligible.

For the primary horizon comparison, pair forecasts verifying the same observed three-hour block, using the same issue cycle from runs initialized 24 hours apart: F003 pairs with F027, F006 with F030, and so on. Keep formula version, target, cells and observation coverage identical. This controls the actual storm and local-time distribution. Also report all usable forecasts to show operational coverage and score changes beyond the paired subset.

Separately evaluate occurrence radii of 10, 20, 30 and 40 km and neighborhood scores such as fractions skill score as functions of scale. Estimate how useful displacement tolerance changes with lead. Increasing the occurrence radius changes the event and its base rate, so raw Brier values across radii cannot select an operational radius. Compare skill against each target's own climatology and report event frequencies, discrimination and localization. Keep 30 km as the primary corridor target. Any proposed horizon-dependent radius must state whether it changes the operational exposure definition or merely the displayed location uncertainty.

Review representative quiet, marginal and organized-storm maps for each finalist. Track affected area, neighboring-cell variation, disconnected high-LPI regions and loss of narrow risk features. Broadening every field can improve apparent overlap while reducing useful location detail. A selected scale should improve held-out skill and remain useful on maps.

## Seasonal and solar experiments

Compute solar geometry directly from valid time, latitude and longitude. Use solar elevation at the block midpoint, hourly daylight fraction within the block, local solar time, daily maximum solar elevation, day length and solar declination. These deterministic features require no additional observation feed. Handle civil-time and daylight-saving conversions explicitly when showing local clock time; use solar time for the physical diurnal comparison.

First stratify the baseline calibration residuals by month, a few predefined solar-elevation regimes, lead and region. Include uncertainty and counts of independent active days. Compare early versus late parts of the archive at similar LPI values and local solar times. Where ingredients are available, also condition diagnostics on MU-LI, CAPE and moisture to assess whether the model already captures the solar response.

Then compare a small sequence of probability models: LPI alone; LPI plus local-time and horizon effects; and that model plus one low-complexity seasonal or solar term. Compare a solar-only or seasonal-climatology benchmark as well. Use training-fitted regularization and validation to choose among alternatives. Add at most one simple interaction with LPI if the residual diagnostics justify it. Preserve probability monotonicity with LPI within each specified regime. Avoid a broad unconstrained search over strongly correlated calendar and solar variables.

Judge added information by improvement beyond LPI and diurnal/horizon calibration on later dates and repeated chronological validation windows. Solar elevation, day length, declination and calendar date move together across this single July-to-October archive; it cannot establish causation or a stable annual multiplier. If a signal does not repeat, retain it as a diagnostic. Recheck any provisional seasonal calibration with 2027 data before treating it as stable.

## Validation and candidate selection

Freeze the inventory, candidate definitions, primary scores and chronological date assignments before selecting candidates. Repeated issue cycles verifying one observed block receive weights so they do not multiply the event count. Bootstrap paired score differences by verifying operational day, with a sensitivity analysis using consecutive multi-day blocks for persistent storms. Pixels and line segments are not independent samples.

Use an initial chronological 60/20/20 split by verifying operational day and additional rolling chronological development splits. Preserve the final test dates across timing, spatial, solar and ingredient experiments. Keep all forecasts verifying the same observed period in the same split. Expanded or shifted target intervals crossing a split boundary are excluded, and purge overlapping forecast and target windows at boundaries so the same issued field or observed event does not enter fitting and evaluation. Record the resulting exclusions.

Audit active corridor days in each split before fitting. A quiet autumn holdout can make some skill measures unidentifiable. Report that limitation and retain the chronological holdout; do not search for a more favorable final test period. If repeated development splits are too sparse, reduce model complexity and report descriptive results. The August reports and their previously inspected test dates serve as development evidence in this revised study.

Select candidates using validation Brier score while requiring ROC AUC to stay within 0.02 of the reference. Report precision-recall and reliability changes, not only the selection score. A candidate qualifies for a parallel operational trial only if its test Brier improvement has a day-block bootstrap 95 percent interval below zero, discrimination shows no material degradation, and the same candidate family improves on at least two chronological development splits. Where there is insufficient event support or unstable behavior across horizons and regimes, retain the reference and list the additional evidence required.

Select timing and spatial settings in stages, then test only a small declared set of combined finalists. All probability mappings are trained on training data. Changes to the meteorological ingredient formula follow those diagnostics, using the available ingredient subset and the same final test dates. A candidate passing archive-grid evaluation must be reconstructed through the native-grid operational processing and reviewed on case maps before a deployment proposal.

## Implementation and outputs

Build a shared sample loader that supports both the full F003-to-F048 baseline archive and the shorter hourly ingredient windows. The current `analyze_lpi_calibration.py` and `analyze_lpi_smoothing.py` select only complete hourly ingredient runs; using them unchanged would omit much of the intended day-two study. Reuse their target, calibration and bootstrap routines only after auditing masking, grid alignment and time definitions. Add focused checks for interval alignment, lag sign, same-valid-time lead pairs, missing-data behavior, Gaussian scale units and split exclusions.

Write the revised analysis to `output/lpi_calibration/review_20261005/` with a manifest containing the archive cutoff, source-file identities or checksums, code revision, configuration, random seeds, formula versions and split assignments. Preserve the August outputs. Keep archive inputs immutable; use local output and temporary directories for derived work.

Required deliverables are:

- A refreshed inventory and machine-readable sample table, with exclusions and data support for every experiment.
- Baseline metrics and reliability plots by horizon, local time and region, including independent active-day counts.
- Lag score curves and event-timing case plots, with a clear sign convention and equal-duration targets.
- Paired day-one/day-two smoothing results, scale-dependent displacement diagnostics and finalist maps.
- Seasonal and solar residual plots and scores showing whether added terms improve later-date predictions.
- Candidate comparisons with day-block confidence intervals and a report stating whether each hypothesis is supported, unsupported or inconclusive.
- A concrete recommendation for retaining current guidance, running a parallel trial, or collecting more data. Forecaster labels and operational category thresholds remain deferred.

Complete the objective study before proposing operational changes. Use the existing archive-safety and retry behavior in [the archive documentation](lightning_ml_archive.md). Keep the future expert-labeling workflow in [the labeling protocol](lpi_meteorologist_labeling_protocol.md) for later use. LPI-to-strike timing in this study is distinct from ignition-to-fire-report delays in [the regional fire-start plan](regional_fire_start_forecast_plan.md).

## October 5 execution

The archive study is complete. See the [readable HTML results](../output/lpi_calibration/review_20261005/report.html) and its machine-readable tables. Inputs were frozen at October 5, 12Z: 711 observation blocks, 4,565 v3 forecast fields before fold exclusions, 4,021 matched records after exclusions, and 1,881 same-valid-time horizon pairs. Six focused checks passed.

An equal-weight current/preceding-block blend with 20 km smoothing passed the statistical trial gate on both horizons, reducing final-period Brier scores by 6.6% and 8.5% on its common eligible sample. Validation selected the same smoothing scale for both days; a distinct day-two setting is unsupported within the tested range. Solar/calendar and ingredient changes did not improve both earlier chronological windows. Keep current operations while reproducing the finalist through native processing and reviewing its location-detail tradeoff before a parallel trial. Define early-lead treatment and verify a fixed-window daily aggregation before using it for a 24-hour display. The final test contains only eight to nine active corridor days. Dynamic missing LPI coverage makes scores conditional on valid common cells: the benchmark retains about 62% and 60% of corridor cells per observed block in the final period. Twilight probability errors worsen on an event-free test stratum and require further storm examples. The single partial season cannot establish an annual adjustment. Forecaster labels remain deferred.

## October 6 revision: daily target and seasonal evaluation

The primary optimisation target is now occurrence over a complete 12Z-to-12Z
24-hour day across eligible BC grid cells. Independently calibrated transmission
corridor scores remain a secondary application. Use three-hour skill to distinguish
candidates only when daily validation Brier is within 1% of the best and the
three-hour Brier gain is at least 2%; inspect 0.5% and 2% daily tolerances too.
These are decision tolerances, not significance thresholds.

The revised study retains the October 5 archive freeze and uses 12Z issuances,
eight consecutive three-hour forecasts/observations per daily target, same-weather-day
pairs of day-one and day-two forecasts, common spatial support, and whole-run
boundary exclusion. Evaluate August 13–24, August 27–September 9, and September
21–October 4 nominal test periods, each with earlier training and validation windows.
Actual retained dates and sample counts are in the evaluation inventory. A run's
two 24-hour windows must stay in the same stage; both horizons must remain eligible
on the same verifying date. All calendar splits are UTC operational-day dates.

Test daily maximum followed by Gaussian smoothing at 0/5/10/15/20 km, with and
without a current/previous-three-hour mean. Day one's first block is unchanged,
because F000 is instantaneous. Require complete forecast values for the primary
comparison, and repeat daily comparisons with the available-input maximum used
by production as a coverage sensitivity. All eight forecast files and observation
blocks must still exist. The smoothing-kernel coverage rule remains conservative;
this is an archived-grid study, not an exact native production reproduction.

Reconstruct all retained ingredient formulas for complete day-one hourly windows;
do not manufacture day-two ingredients from the 24-hour archive. Evaluate daily
LPI-only logistic calibration versus one added daylight-duration, solar-declination,
or calendar feature. Keep solar effects separate from formula and smoothing tuning.

Assess Brier changes with 2,000 whole-day and consecutive-available-three-day
bootstrap draws and an AUC loss guard of 0.02. A candidate's research screening
requires improvement across all three test periods and both forecast-coverage
treatments. Intervals condition on fixed calibration; selection multiplicity,
retraining uncertainty, missing support and reused historical data still limit claims.
These are retrospective comparisons. Lock a candidate before future confirmation
and extend collection to spring/early summer and all 48 hourly ingredient leads.
Forecaster-labelled evaluation remains deferred.

Reproducible implementation: `scripts/run_lpi_daily_review_20261006.py`;
report: `scripts/build_lpi_daily_report_20261006.py`;
focused checks: `tests/check_lpi_daily_review_20261006.py`.
Results and portable HTML: `output/lpi_calibration/review_20261006_24h/`.
The Studio report URL is preserved; the original report remains available there.

## Creative formulation experiment requested October 6

Compare substantially different formula architectures against the reconstructed
current recipe with 20 km smoothing: additive evidence, soft ingredient gates,
generalised means, fuzzy bottlenecks, OR-combined storm routes, ascent-first and
instability-first recipes, dry elevated-storm context, stratiform-rain discrimination,
lagged ingredients, daily envelopes, accumulated evidence and peak/persistence blends.
Also fit regularised additive and storm-route models to earlier observations;
include signed-weight and nonnegative-weight variants. Archived surface/subcloud
humidity can enter these recipes alongside the existing ingredients.

The initial strict requirement of 24 complete ingredient hours and 95% valid
neighbourhood support leaves only about 0.13% of BC in the autumn test. Preserve
that audit, but do not use that sample to support a province-wide seasonal claim.
For the main experiment, compare recipes using the same available hours with
all required fields finite, preserving hourly gaps. Use a common daily mask and
common masks within each three-hour companion block. Normalise persistence
summaries by available hours. Separately restrict verification centres to at least
12 available ingredient hours, keeping primary fitted pipelines fixed.

Use training-only probability calibration or learned coefficients. Screen families
on daily validation improvement across the existing three rolling periods, then
refine up to two promising distinct families through seeded parameter trials or
additional regularisation settings. Record the final validation selection before
assessing test scores. Retain the daily-first/three-hour-tie preference. No day-two
recipe claim is supported by the 24-hour-only ingredient archive. Findings remain
retrospective and require native-grid and prospective confirmation.

Implementation: `scripts/run_lpi_radical_review_20261006.py`; coverage sensitivity:
`scripts/finalize_lpi_radical_sensitivity_20261006.py`; report:
`scripts/build_lpi_radical_report_20261006.py`.

### Creative experiment outcome

The initial learned model with signed ingredient weights achieved about 17.6%
mean relative daily validation improvement against the reconstructed current recipe
at 20 km smoothing. Refinement around its regularisation boundary selected ridge
0.00005 after testing 0.000025 and 0.00005 in a final local extension. The selected
model uses 20 daily peak, mean, favourable-hour-fraction and co-occurrence features.
Its mean validation gain is 18.7%; pooled daily test Brier improvement is 17.3%
across the scored BC sample, with individual test-period gains of 13.9%, 17.6%,
and 27.0%. Three-hour companion gains are 10.3%, 4.3%, and 15.5%.

Both whole-day and three-day Brier intervals are negative in all three primary
periods, with improved AUC. Restricting verification centres to at least 12 usable
ingredient hours also yields negative intervals in every period. Main test coverage
is about 80%, 61%, and 44% of the eligible BC grid; half-day support coverage is
about 65%, 46%, and 30%. These percentages cannot be added to the earlier 4%
smoothing headline because the ingredient and baseline-only studies use different
available-input populations. Recommend a day-one native-grid prospective research
trial. Production settings are unchanged; day-two recipe verification remains
unsupported until the full second-day hourly ingredients are retained.

Reproduction order: run the creative study, the adaptive boundary refinement,
the half-day sensitivity, the focused scientific checks, and the report builder.
The existing Studio report URL now serves the creative study. The prior daily
report is retained as `daily_review_report.html`.

### Additional formulation round — 7 October 2026

Completed 18 candidates in six families: piecewise-linear ingredient responses,
quadratic and pairwise interactions, direct Brier optimisation, Gaussian radial
basis prototype similarities, a fixed random hidden-layer network, and a
class-conditional Gaussian classifier. Reused the frozen 62-case cohort, common
masks, 20 km feature smoothing and chronological folds. The previous learned
winner was the primary comparator. All new models were fitted over BC; corridor
scores are secondary and compare against the older corridor-specific fit.

Validation refined the nonlinear network and interaction families with three
additional penalty settings each. The daily-first rule selected the initial
32-unit random network at penalty 0.001; its mean relative daily validation gain
was 0.772%. Refinement did not improve the network, and barely changed interaction
skill. The selected network improved pooled daily test Brier by 2.754% versus the
previous learned model, equivalent to 19.531% versus the original smoothed recipe
on the identical cohort. Period gains were 2.979%, 3.235% and 1.210%. August and
autumn uncertainty intervals cross zero, including the 12-hour support check, so
this is a promising challenger rather than an established replacement. Retain the
previous winner as the preferred research candidate and freeze the network for
future independent evaluation. No production settings changed.

Scripts: `scripts/run_lpi_experiments_20261007.py`,
`tests/check_lpi_experiments_20261007.py`, and
`scripts/build_lpi_experiments_report_20261007.py`. Results are in
`output/lpi_calibration/review_20261007/`. The Studio report now serves this round;
`radical_report.html` preserves the preceding radical-formulation report.

### Deployment decision — 7 October 2026

The user authorised implementing the new nonlinear network after reviewing its
positive gains in all three periods and the remaining uncertainty about its
incremental advantage. It is now the provisional preferred formulation. The
strict gate remains recorded as not passed; implementation does not change that
statistical result. The previous learned formulation is the benchmark/fallback.

Implemented `bc_lpi_v4_random32` as a portable NumPy/SciPy inference module with
JSON coefficients. Refit the frozen architecture on all 62 eligible archived
cases for separate daily and three-hour outputs. Activated in the existing
forecast code for 12Z continental HRDPS, first 24 hours. Added a direct daily
cache and forecast map; daily verification consumes that cache without taking
a block-probability maximum. Kept the 5 km feature grid and 20 km Gaussian sigma
used in the evaluated model. Native-grid interpolation is for display only.

Today’s 12Z run was processed for a three-hour example and the direct daily
forecast. Feature and probability parity checks, missing-data and scope checks,
and daily-cache integration checks passed. The daily example has limited valid
coverage because derived charging-layer ingredients are undefined across much
of the quiet-season forecast. Do not convert those missing areas to zero.

Created a one-page forecaster PDF describing only the new LPI, its inputs,
probability meaning and the dry-lightning heuristic. Preserved the humidity,
rainfall and marker thresholds, applying the screening score to the new
three-hour probability. Dry-lightning thresholds still require independent
verification. See `docs/lpi_model_v4.md` and `docs/reference/`.

### 7 October 2026: all-horizon live rollout

Extended the frozen formulation to 00Z, 06Z, 12Z and 18Z HRDPS cycles at
F003-F048. F000 is explicitly an initialization diagnostic. The active public
continental Fire Weather run issued 7 October at 12Z was regenerated at all
17 displayed horizons, installed into the operational archive and published.
Daily verification for 6 October's 00Z, 06Z and 12Z runs was rebuilt using the
new direct 24-hour probabilities and published after observations were complete.
The public manifest identifies the new version. Remote F048 and verification
PNG hashes matched the publisher's uploaded-file hashes; older forecast images
retain their original description even when verification caches are backfilled.

Future runs retain hourly inference inputs across the full 48 hours. Each full
12Z-12Z daily window receives its own direct probability cache; 12Z runs have
both F024 and F048 verification frames. Pruning preserves both. Day two and
other cycles transfer the frozen day-one coefficients, so independent evidence
for these extensions remains outstanding. The historical skill results have
not been relabelled as tests of the extensions or final production refit.

The quick reference keeps its contour, dry-lightning, rain, humidity and gust
instructions. Its scope statement now covers all cycles through F048 and its
reading section explains the F000 initialization exception. Existing diagnostic,
verification, two-panel and publication checks passed, alongside feature and
probability parity and direct day-two window checks. The revised report and PDF
are served from the existing private Studio report URL.
