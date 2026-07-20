# Lightning and Gust Diagnostic Review

## Lightning Potential Index

The operational name remains **Lightning Potential Index (LPI)**. It is a
relative ingredients index, not a calibrated lightning probability.

The `bc_lpi_v2` formulation is intended for BC's terrain-driven and often
modest-CAPE convection:

1. MU lifted index is the primary instability gate. CAPE can strengthen an
   already unstable signal by at most 20%; CAPE cannot create LPI by itself.
2. The mixed-phase charging layer is limited to 0 to -20 C. Temperature and RH
   contributions are integrated by represented pressure thickness, with
   below-ground portions removed using surface pressure.
3. Pressure vertical velocity is converted from Pa/s to upward-positive
   geometric velocity using local pressure-level temperature and density.
4. The trigger is the maximum of preceding 3-hour precipitation, current
   precipitation rate, and an updraft/charging realization term.
5. The final field is smoothed once. The plotted field and verification cache
   are now the same field.

In compact form:

```text
I = LI_factor * (0.80 + 0.20 * CAPE_factor)
M = charge_factor * (0.85 + 0.15 * midlevel_RH_factor)
w = max(0, -omega_500 / (rho_500 g), -omega_700 / (rho_700 g))
T = max(precip_3h_factor, precip_rate_factor, w_factor * charge_factor)
LPI = 100 * I * M * sqrt(T)
```

The first full-run comparison retained nearly the same LPI 20 footprint while
reducing the saturated upper tail. This is a conservative structural revision;
the new upper bands still need multiple convective seasons of verification.

## Gust Products

The remembered all-cause **Forecast Gust Estimate (FGE)** was designed in an
earlier task but was never implemented. Before 2026-07-10, the fire-weather map
read the native HRDPS `GUST-Max` field at the plotted forecast hour. It now uses
the regular HRDPS `GUST` field and colors unit vectors pointing in the
instantaneous 10 m wind direction. It does not currently combine native gust,
PCGE, or mixed-down momentum.

This field is not the best native baseline for a most-likely gust. ECCC's
WEonG documentation identifies `GUST` as WGE, the estimated instantaneous gust
from parcels with the potential to descend to the surface. `GUST-Max` is WGX,
the maximum instantaneous momentum that can reach the surface. ECCC only uses
WGX in its post-processed gust after masks for cross-barrier flow, stability,
an inversion over the barrier, and sufficient terrain slope all pass.

The proposed FGE was:

```text
G_native_3h = max(hourly native GUST over the three-hour plot window)
z_agl(p) = HGT_ISBL(p) - HGT_SFC
U_mix = strongest wind where 40 m <= z_agl <= min(HPBL + 250 m, 2500 m)
G_mix = U10 + M * max(0, U_mix - U10)
G_conv = G_native_3h + conv_weight * max(0, PCGE - G_native_3h)
FGE = calibrated_blend(G_native_3h, G_downslope, G_conv)
```

### Initial BC case verification

A properly time-aligned stress test used the 2026-07-09 12 UTC HRDPS-West run,
240 BCWS stations, five forecast times, and 1,198 station-time matches. This is
one warm-season case, not a climatological score.

| Predictor and target | Bias | MAE | RMSE | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Endpoint `GUST`, observed hourly peak | -0.1 km/h | 7.8 km/h | 10.0 km/h | 0.66 |
| Endpoint `GUST-Max`, observed hourly peak | +8.7 km/h | 12.5 km/h | 15.7 km/h | 0.58 |
| Three-hour max of hourly `GUST`, observed three-hour peak | -0.1 km/h | 7.6 km/h | 9.9 km/h | 0.68 |
| Endpoint `GUST-Max`, observed three-hour peak | +6.1 km/h | 11.8 km/h | 14.8 km/h | 0.55 |

For the hourly 40 km/h threshold, `GUST-Max` forecast 236 events when 53 were
observed, with a false-alarm ratio of 0.85. The central `GUST` forecast 71,
with a false-alarm ratio of 0.59. The result was nearly unchanged when limited
to the 214 stations whose latest available historical metadata places the
anemometer in the 9.0-11.9 m band.

### What is sound

- Native HRDPS `GUST`, rather than `GUST-Max`, is the correct anchor for
  resolved synoptic, frontal, gap, and terrain-accelerated winds.
- Computing pressure-level height AGL avoids using underground 850 hPa winds or
  mixing free-atmosphere winds down through a shallow coastal boundary layer.
- VPD, mixed-layer depth, and ventilation are reasonable modifiers of momentum
  transfer. They should not be added directly to a wind speed.
- PCGE should remain a conditional convective ceiling, not the default gust.

### What needs correction or calibration

- Taking the maximum of three candidates is deliberately high-biased and is not
  compatible with a "most likely point gust" label.
- `U_mix = max(wind)` is sensitive to sparse pressure levels and HPBL error.
  Interpolation to height AGL and a high layer percentile are more stable than
  choosing one pressure-level maximum.
- A hand-tuned mixing efficiency from HPBL, VPD, HDWI, and ventilation can
  double-count the same daytime-mixing signal. It also misses nocturnal jets,
  downslope windstorms, mountain waves, and unresolved gap exposure.
- Using `max(native, mixed, convective)` guarantees that model errors only move
  the result upward. A calibrated conditional-median blend is preferable once
  enough observations are available.
- PCGE must be trigger-weighted independently. Feeding LPI directly back into
  both products risks circular reinforcement because they share instability,
  moisture, and vertical-motion ingredients.

### Observation limitations

BCWS `HOURLY_WIND_GUST` is documented as the highest gust during the preceding
hour, so three consecutive reports do represent a three-hour peak. BCWS clocks
are fixed PST (UTC-8) year-round; interpreting them as daylight time shifts
summer verification by one hour. The current station layer does not expose wind
height, but the 2023 historical metadata covered 224 of the 240 active stations
in this case and put 214 in the 9.0-11.9 m band. The explicitly low 1.5 m sites
are portable/QD stations. Historical metadata also identifies an applied
roughness adjustment.

ECCC SWOB gust reports are not homogeneous: source fields can be a past-10-
minute gust, a past-hour peak gust, or a past-hour maximum wind. Verification
must retain and filter the source field rather than pooling them blindly.

A station peak is a point and exposure measurement, not the maximum anywhere in
a 1 km or 2.5 km grid cell. It can evaluate station-conditioned point forecasts,
but it cannot directly prove a grid-cell P90 or grid-cell maximum.

### Recommended next gust product

Keep three concepts distinct during development:

1. The maximum of the three hourly native `GUST` fields as the most-likely
   three-hour baseline.
2. A tightly gated terrain branch modeled after ECCC's downslope diagnosis,
   using `GUST-Max` only where cross-barrier flow, stable-layer/inversion, and
   slope tests support momentum descent.
3. A separately gated convective increment from PCGE, activated by independent
   evidence that convection and precipitation-loaded downdrafts are expected.

Train the final blend against the observed conditional median or an MAE/Huber
objective, not P90 quantile loss. Verify it with entire years held out and with
stations held out, stratified by regime, network, anemometer height,
station/model elevation mismatch, season, and local time. Keep the raw PCGE and
`GUST-Max` available as potential/ceiling guidance rather than labeling either
one the most-likely gust.

## Primary Observation References

- [BCWS Weather Data Sharing Guide](https://www2.gov.bc.ca/assets/gov/public-safety-and-emergency-services/wildfire-status/prepare/bcws_datamart_and_api_v2_1.pdf)
- [BC Weather Archive field definitions](https://wa.th.gov.bc.ca/)
- [ECCC SWOB-ML Product User Guide](https://dd.weather.gc.ca/today/observations/doc/SWOB-ML_Product_User_Guide_v8.16_e.pdf)
- [ECCC WEonG-HRDPS Technical Note](https://collaboration.cmc.ec.gc.ca/cmc/cmoi/product_guide/docs/tech_notes/technote_weong-hrdps_e.pdf)
