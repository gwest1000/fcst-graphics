# HRDPS All-Cause Gust

The fire-weather graphic uses an intentionally upper-envelope gust:

```text
G_all = max(G_regular, G_downslope, G_pcge)
```

The vectors retain the instantaneous HRDPS 10 m wind direction. Their colour
is the all-cause gust magnitude.

## Regular branch

`G_regular` is the native instantaneous HRDPS `GUST` field. It remains the
result where neither conditional branch raises it. F000 uses this field alone;
the downslope and PCGE branches begin at F003 so analysis-time `GUST-Max` and
zero-hour convective diagnostics cannot create a spin-up artifact.

## Downslope branch

This branch adapts the four-mask ECCC WEonG technique:

1. Find the highest terrain within 10 km and vertically interpolate the wind
   profile to that ridge height.
2. Compute `SLX = U * dH/dx + V * dH/dy`; require `SLX < -0.1 m/s`.
3. Require terrain slope magnitude of at least `0.01` (100 m per 10 km).
4. Find the deepest inversion in the first 3000 m AGL. Uniformly average the
   lapse rate below its top over 10 km and require at least `-0.006 K/m`.
5. Identify terrain maxima with the ECCC Hessian test and require an inversion
   over a qualifying maximum within 10 km.

All four masks must pass. The adjusted value follows ECCC's ratio rule:

```text
if GUST-Max / GUST < 1.5:
    G_downslope = GUST-Max
else:
    G_downslope = mean(GUST, GUST-Max)
```

The implementation uses 1000, 950, 900, 850, 800, 750, and 700 hPa heights,
temperatures, and winds. This is an ECCC-method adaptation to the pressure-level
fields in the public HRDPS feed, not the proprietary operational WEonG code.

## PCGE branch

The previous PCGE added scaled DCAPE velocity and pressure-level momentum, then
multiplied the whole sum by LI and PBL factors. That could double-count kinetic
energy and allowed an environmental instability gate to stand in for evidence
that convection was active.

The revised potential is:

```text
M850 = (0.70 + 0.15 * PBL_factor) * wind850
M700 = (0.45 + 0.20 * PBL_factor) * wind700
M = max(G_regular, M850, M700)
D = 0.35 * sqrt(2 * DCAPE)
G_potential = sqrt(M^2 + D^2)
```

The continuous convection trigger is independent of LPI:

```text
environment = LI_factor * (0.85 + 0.15 * CAPE_factor)
active = max(precipitation_rate_factor, strong_700hPa_ascent_factor)
trigger = environment * active * DCAPE_factor
G_pcge = G_regular + trigger * max(0, G_potential - G_regular)
```

LI is the primary BC instability input; CAPE is only a small modifier. Modeled
precipitation activates the branch directly, while dry convection requires much
stronger 700 hPa ascent than ordinary synoptic lift. DCAPE begins contributing
at 150 J/kg and reaches full trigger weight at 750 J/kg.

## Status

This is `all_cause_gust_v1`. It is deterministic conditional guidance, not a
calibrated percentile. Verification should retain each branch separately so
false alarms and misses can be attributed to the regular, downslope, or PCGE
component.

Primary method reference: [ECCC WEonG-HRDPS technical note](https://collaboration.cmc.ec.gc.ca/cmc/cmoi/product_guide/docs/tech_notes/technote_weong-hrdps_e.pdf).
