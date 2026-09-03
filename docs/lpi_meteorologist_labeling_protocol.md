# BCH Lightning-Risk Labeling Protocol

## Purpose

The objective archive can calibrate flash occurrence and density, but it cannot by itself define what BCH meteorologists mean operationally by low, moderate, and high lightning risk. This exercise will connect those subjective categories to observed lightning density near transmission lines and then to forecast probabilities.

The primary label is the 24-hour lightning risk for a 12Z-to-12Z operational day. Eight three-hour panels are supporting evidence and receive separate labels so the daily category can be translated back to the existing three-hour workflow.

## Cases

- Use all complete 12Z-to-12Z days from approximately 1.5 convective-season months.
- Keep dates and forecast LPI hidden during independent labeling to reduce anchoring and hindsight effects.
- Randomize cases with stable case IDs. Reveal dates only during adjudication.
- Include quiet, marginal, organized, and widespread-lightning days; do not sample only active days.
- Use the same map extent, line layer, legend, and symbol scale for every case.

## Observation Maps

Each case package should contain:

1. One 24-hour map of BC transmission lines and observed flash-density shading.
2. A deterministic synthetic point rendering of the gridded observations.
3. Eight three-hour maps covering the same 12Z-to-12Z period.
4. Stable transmission-line segment identifiers for recording corridor-specific labels.

The point locations are display aids, not observed strike coordinates. For each 2.5 km grid cell, estimate the number of flashes as `flash density x cell area`, stochastically round using a case-seeded random number generator, and place that many points uniformly within the cell. Use the same seed whenever a case is regenerated. When density is too high for a legible map, one point may represent multiple flashes, but the legend must state the multiplier.

## Independent Labels

Three or four BCH meteorologists should label each case independently before discussion.

- `none`: no operationally meaningful lightning risk.
- `low / isolated`: isolated strikes with limited line exposure.
- `moderate / scattered`: scattered strikes or a meaningful portion of one or more line corridors exposed.
- `high / clustered`: clustered, repeated, or broad corridor exposure.
- `uncertain`: insufficient confidence to force a category.

For every case, record:

- overall 24-hour category;
- polygons enclosing low, moderate, and high areas;
- category for each materially affected transmission-line segment;
- maximum three-hour category in each of the eight blocks;
- confidence from 1 to 3;
- a short reason when terrain, storm motion, timing, or data quality drove the rating.

Use nested polygons where appropriate: high areas sit inside moderate and low envelopes. Ratings describe risk within roughly 30 km of a line, not the visual density at an isolated pixel.

## Adjudication

After independent labeling, calculate inter-rater agreement using weighted kappa for the ordinal categories and display disagreements without meteorologist names. Discuss cases differing by two categories or cases with poor polygon overlap. Retain both the individual labels and a consensus/adjudicated label; do not replace the originals.

If agreement is weak, refine the written definitions using example cases and relabel a small common set before continuing. The goal is a reproducible operational scale, not forced consensus.

## Quantitative Translation

For each labeled polygon and line segment, calculate objective summaries from the original gridded density:

- total estimated flashes within 10, 20, and 30 km of the segment;
- flash density per square kilometre in the corridor buffer;
- maximum local density after 10, 20, and 30 km smoothing;
- fraction and kilometres of line exposed above candidate density thresholds;
- number of active three-hour blocks and the maximum three-hour density;
- 24-hour total and 24-hour occurrence.

Estimate category boundaries on training cases with an ordinal monotone model. Select thresholds on validation cases to reproduce consensus labels, then report accuracy, weighted kappa, and one-category error on untouched test cases. Bootstrap by day, not by grid cell or line segment.

The objective 24-hour forecast should remain a calibrated probability of lightning near each line segment. Low/moderate/high are operational display categories derived from meteorologist labels and should not replace the probability in the verification archive.

For the three-hour product, fit a separate calibration to three-hour occurrence/density. Do not simply divide a 24-hour threshold by eight. Compare the daily probability built from the eight three-hour probabilities with a separately calibrated daily maximum/aggregate and retain whichever verifies better on held-out days.

## Data Template

Store labels in a long-form CSV or GeoPackage with these fields:

```text
case_id,met_id,period_start_utc,period_hours,geometry_id,line_segment_id,
rating,confidence,notes,label_version,created_at_utc
```

Polygon geometries should be kept in a companion GeoPackage/GeoJSON keyed by `geometry_id`. Transmission lines should be split into stable 25 km segments so exposure statistics and labels refer to the same units across cases.

## Guardrails

- Do not show forecasts during the observation-labeling pass.
- Do not tune thresholds on the final chronological test period.
- Keep synthetic point generation deterministic and clearly labeled.
- Preserve all raw labels, case maps, code version, density grids, and train/validate/test assignments.
- Recheck thresholds with the 2027 season before treating 2026 calibration as stable climatology.
