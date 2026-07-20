#!/usr/bin/env python3
"""Vectorized core calculations for the NRCan FWI2025 hourly system.

The equations follow the Canadian Forest Service FWI2025 information report and
the reference implementation at https://github.com/nrcan-cfs-fire/cffdrs-ng.
Only the standard forest-fuel components needed by these graphics are included:
FFMC, DMC, DC, ISI, BUI, and FWI. Grass-fuel components are intentionally out of
scope.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass

import numpy as np


FFMC_INTERCEPT_MM = 0.5
DMC_INTERCEPT_MM = 1.5
DC_INTERCEPT_MM = 2.8
RAIN_EVENT_RESET_HOURS = 5.0
_C_FFMC = 14875.0 / 101.0
CALC_DTYPE = np.float64


@dataclass
class FWI2025State:
    """Moisture and rain-event state required between hourly calculations."""

    mcffmc: np.ndarray
    mcdmc: np.ndarray
    mcdc: np.ndarray
    rain_total_mm: np.ndarray
    canopy_drying_hours: np.ndarray

    @classmethod
    def from_codes(cls, ffmc: np.ndarray, dmc: np.ndarray, dc: np.ndarray) -> "FWI2025State":
        ffmc = np.asarray(ffmc, dtype=CALC_DTYPE)
        dmc = np.asarray(dmc, dtype=CALC_DTYPE)
        dc = np.asarray(dc, dtype=CALC_DTYPE)
        valid = np.isfinite(ffmc) & np.isfinite(dmc) & np.isfinite(dc)
        with np.errstate(over="ignore", invalid="ignore"):
            mcffmc = _C_FFMC * (101.0 - ffmc) / (59.5 + ffmc)
            mcdmc = 280.0 / np.exp(dmc / 43.43) + 20.0
            mcdc = 400.0 * np.exp(-dc / 400.0)
        nan = CALC_DTYPE(np.nan)
        return cls(
            np.where(valid, mcffmc, nan).astype(CALC_DTYPE),
            np.where(valid, mcdmc, nan).astype(CALC_DTYPE),
            np.where(valid, mcdc, nan).astype(CALC_DTYPE),
            np.where(valid, 0.0, nan).astype(CALC_DTYPE),
            np.where(valid, 0.0, nan).astype(CALC_DTYPE),
        )

    def copy(self) -> "FWI2025State":
        return FWI2025State(*(field.copy() for field in self.as_tuple()))

    def as_tuple(self) -> tuple[np.ndarray, ...]:
        return self.mcffmc, self.mcdmc, self.mcdc, self.rain_total_mm, self.canopy_drying_hours


@dataclass(frozen=True)
class FWI2025Output:
    state: FWI2025State
    ffmc: np.ndarray
    dmc: np.ndarray
    dc: np.ndarray
    isi: np.ndarray
    bui: np.ndarray
    fwi: np.ndarray


def codes_from_state(state: FWI2025State) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        ffmc = 59.5 * (250.0 - state.mcffmc) / (_C_FFMC + state.mcffmc)
        dmc = 43.43 * np.log(280.0 / np.maximum(state.mcdmc - 20.0, 1.0e-6))
        dc = 400.0 * np.log(400.0 / np.maximum(state.mcdc, 1.0e-6))
    return (
        np.clip(ffmc, 0.0, 101.0).astype(CALC_DTYPE),
        np.maximum(dmc, 0.0).astype(CALC_DTYPE),
        np.maximum(dc, 0.0).astype(CALC_DTYPE),
    )


def daylight_mask(
    valid_utc: dt.datetime,
    lat: np.ndarray,
    lon: np.ndarray,
    local_tz: dt.tzinfo,
) -> np.ndarray:
    """Return the FWI2025 sunrise-to-sunset mask for a gridded valid time.

    A common Pacific clock is used across the map. Longitude remains in the
    solar calculation, so this does not shift the physical sunrise/sunset time
    in eastern BC; it only provides a common clock for evaluating that time.
    """

    if valid_utc.tzinfo is None:
        raise ValueError("valid_utc must be timezone-aware")
    local = valid_utc.astimezone(local_tz)
    day = local.timetuple().tm_yday
    days_in_year = 366.0 if calendar.isleap(local.year) else 365.0
    gamma = 2.0 * np.pi * (day - 1.0) / days_in_year
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.001480 * np.sin(3.0 * gamma)
    )
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float32))
    lon = np.asarray(lon, dtype=np.float32)
    zenith = np.deg2rad(90.833)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_halfday = np.cos(zenith) / (np.cos(lat_rad) * np.cos(decl)) - np.tan(lat_rad) * np.tan(decl)
    halfday = np.rad2deg(np.arccos(np.clip(cos_halfday, -1.0, 1.0)))
    offset = local.utcoffset()
    if offset is None:
        raise ValueError("local_tz did not provide a UTC offset")
    timezone_hours = offset.total_seconds() / 3600.0
    sunrise = (720.0 - 4.0 * (lon + halfday) - eqtime) / 60.0 + timezone_hours
    sunset = (720.0 - 4.0 * (lon - halfday) - eqtime) / 60.0 + timezone_hours
    hour = local.hour + local.minute / 60.0 + local.second / 3600.0
    return ((sunrise <= hour) & (hour <= sunset)) | ((hour < 6.0) & (sunrise <= hour + 24.0) & (hour + 24.0 <= sunset))


def _ffmc_moisture_step(
    last_mc: np.ndarray,
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind_kmh: np.ndarray,
    rain_after_intercept_mm: np.ndarray,
) -> np.ndarray:
    mo = last_mc.astype(CALC_DTYPE, copy=True)
    rain = np.maximum(np.asarray(rain_after_intercept_mm, dtype=CALC_DTYPE), 0.0)
    wet = rain > 0.0
    safe_rain = np.where(wet, rain, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = 42.5 * rain * np.exp(-100.0 / (251.0 - last_mc)) * (1.0 - np.exp(-6.93 / safe_rain))
        delta += np.where(last_mc > 150.0, 0.0015 * np.square(last_mc - 150.0) * np.sqrt(rain), 0.0)
    mo = np.where(wet, np.minimum(mo + delta, 250.0), mo)

    rh = np.clip(np.asarray(rh, dtype=CALC_DTYPE), 0.0, 100.0)
    temp_c = np.asarray(temp_c, dtype=CALC_DTYPE)
    wind_kmh = np.maximum(np.asarray(wind_kmh, dtype=CALC_DTYPE), 0.0)
    e1 = 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh))
    ed = 0.942 * np.power(rh, 0.679) + 11.0 * np.exp((rh - 100.0) / 10.0) + e1
    ew = 0.618 * np.power(rh, 0.753) + 10.0 * np.exp((rh - 100.0) / 10.0) + e1
    equilibrium = np.where(mo < ed, ew, ed)
    a1 = np.where(mo > ed, rh / 100.0, (100.0 - rh) / 100.0)
    k0 = 0.424 * (1.0 - np.power(a1, 1.7)) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - np.power(a1, 8.0))
    rate = 2.0 * 0.0579 * k0 * np.exp(0.0365 * temp_c)
    updated = equilibrium + (mo - equilibrium) * np.power(10.0, -rate)
    return np.where(mo == ed, mo, updated).astype(CALC_DTYPE)


def _dmc_moisture_step(
    last_mc: np.ndarray,
    local_hour: float,
    temp_c: np.ndarray,
    rh: np.ndarray,
    precip_mm: np.ndarray,
    daylight: np.ndarray,
    rain_total_before_mm: np.ndarray,
) -> np.ndarray:
    del local_hour  # Daylight has already been evaluated on the grid.
    total_after = rain_total_before_mm + precip_mm
    wet = total_after > DMC_INTERCEPT_MM
    first_wet_hour = rain_total_before_mm <= DMC_INTERCEPT_MM
    rw = np.where(first_wet_hour, 0.92 * total_after - 1.27, 0.92 * precip_mm)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        last_dmc = 43.43 * np.log(280.0 / np.maximum(last_mc - 20.0, 1.0e-6))
        b = np.where(last_dmc <= 33.0, 100.0 / (0.3 * last_dmc + 0.5), 0.0)
        b = np.where((last_dmc > 33.0) & (last_dmc <= 65.0), -1.3 * np.log(last_dmc) + 14.0, b)
        b = np.where(last_dmc > 65.0, 6.2 * np.log(last_dmc) - 17.2, b)
        rain_mc = last_mc + 1000.0 * rw / (b * rw + 48.77)
    rain_mc = np.where(wet, rain_mc, last_mc)
    rain_mc = np.minimum(rain_mc, 300.0)
    drying_temp = np.maximum(temp_c, 0.0)
    rate = 2.22e-4 * drying_temp * (100.0 - rh) / 43.43
    dried = (rain_mc - 20.0) * np.exp(-rate) + 20.0
    return np.minimum(np.where(daylight, dried, rain_mc), 300.0).astype(CALC_DTYPE)


def _dc_moisture_step(
    last_mc: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    daylight: np.ndarray,
    rain_total_before_mm: np.ndarray,
) -> np.ndarray:
    total_after = rain_total_before_mm + precip_mm
    wet = total_after > DC_INTERCEPT_MM
    first_wet_hour = rain_total_before_mm <= DC_INTERCEPT_MM
    rw = np.where(first_wet_hour, 0.83 * total_after - 1.27, 0.83 * precip_mm)
    rain_mc = np.where(wet, last_mc + 3.937 * rw / 2.0, last_mc)
    rain_mc = np.minimum(rain_mc, 400.0)
    pe = np.where(temp_c > 0.0, 1.5e-2 * temp_c + 3.0 / 16.0, 0.0)
    dried = rain_mc * np.exp(-pe / 400.0)
    return np.minimum(np.where(daylight, dried, rain_mc), 400.0).astype(CALC_DTYPE)


def initial_spread_index(ffmc: np.ndarray, wind_kmh: np.ndarray) -> np.ndarray:
    ffmc = np.asarray(ffmc, dtype=CALC_DTYPE)
    wind = np.maximum(np.asarray(wind_kmh, dtype=CALC_DTYPE), 0.0)
    mc = _C_FFMC * (101.0 - ffmc) / (59.5 + ffmc)
    wind_factor = np.where(wind >= 40.0, 12.0 * (1.0 - np.exp(-0.0818 * (wind - 28.0))), np.exp(0.05039 * wind))
    fuel_factor = 91.9 * np.exp(-0.1386 * mc) * (1.0 + np.power(mc, 5.31) / 4.93e7)
    return np.maximum(0.0, 0.208 * wind_factor * fuel_factor).astype(CALC_DTYPE)


def buildup_index(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    dmc = np.maximum(np.asarray(dmc, dtype=CALC_DTYPE), 0.0)
    dc = np.maximum(np.asarray(dc, dtype=CALC_DTYPE), 0.0)
    denom = dmc + 0.4 * dc
    base = np.divide(0.8 * dc * dmc, denom, out=np.zeros_like(dmc), where=denom > 0.0)
    fraction = np.divide(dmc - base, dmc, out=np.zeros_like(dmc), where=dmc > 0.0)
    corrected = dmc - (0.92 + np.power(0.0114 * dmc, 1.7)) * fraction
    return np.maximum(0.0, np.where(base < dmc, corrected, base)).astype(CALC_DTYPE)


def fire_weather_index(isi: np.ndarray, bui: np.ndarray) -> np.ndarray:
    isi = np.maximum(np.asarray(isi, dtype=CALC_DTYPE), 0.0)
    bui = np.maximum(np.asarray(bui, dtype=CALC_DTYPE), 0.0)
    fuel = np.where(bui > 80.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)), 0.626 * np.power(bui, 0.809) + 2.0)
    base = 0.1 * isi * fuel
    high = np.exp(2.72 * np.power(0.434 * np.log(np.maximum(base, 1.0)), 0.647))
    return np.maximum(0.0, np.where(base <= 1.0, base, high)).astype(CALC_DTYPE)


def step(
    state: FWI2025State,
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind_kmh: np.ndarray,
    precip_mm: np.ndarray,
    daylight: np.ndarray,
    local_hour: float = 0.0,
) -> FWI2025Output:
    """Advance all standard FWI2025 forest components by one hour."""

    temp_c = np.asarray(temp_c, dtype=CALC_DTYPE)
    rh = np.clip(np.asarray(rh, dtype=CALC_DTYPE), 0.0, 100.0)
    wind_kmh = np.maximum(np.asarray(wind_kmh, dtype=CALC_DTYPE), 0.0)
    precip_mm = np.maximum(np.asarray(precip_mm, dtype=CALC_DTYPE), 0.0)
    daylight = np.asarray(daylight, dtype=bool)
    valid = np.isfinite(state.mcffmc) & np.isfinite(temp_c) & np.isfinite(rh) & np.isfinite(wind_kmh) & np.isfinite(precip_mm)

    rain_total = state.rain_total_mm.copy()
    drying = state.canopy_drying_hours.copy()
    drying = np.where((precip_mm > 0.0) | (rain_total == 0.0), 0.0, drying + 1.0)
    reset = drying >= RAIN_EVENT_RESET_HOURS
    rain_total = np.where(reset, 0.0, rain_total)
    drying = np.where(reset, 0.0, drying)

    total_after = rain_total + precip_mm
    rain_ffmc = np.where(
        total_after <= FFMC_INTERCEPT_MM,
        0.0,
        np.where(rain_total > FFMC_INTERCEPT_MM, precip_mm, total_after - FFMC_INTERCEPT_MM),
    ).astype(CALC_DTYPE)
    mcffmc = _ffmc_moisture_step(state.mcffmc, temp_c, rh, wind_kmh, rain_ffmc)
    mcdmc = _dmc_moisture_step(state.mcdmc, local_hour, temp_c, rh, precip_mm, daylight, rain_total)
    mcdc = _dc_moisture_step(state.mcdc, temp_c, precip_mm, daylight, rain_total)
    rain_total = rain_total + precip_mm

    nan = CALC_DTYPE(np.nan)
    next_state = FWI2025State(
        np.where(valid, mcffmc, nan).astype(CALC_DTYPE),
        np.where(valid, mcdmc, nan).astype(CALC_DTYPE),
        np.where(valid, mcdc, nan).astype(CALC_DTYPE),
        np.where(valid, rain_total, nan).astype(CALC_DTYPE),
        np.where(valid, drying, nan).astype(CALC_DTYPE),
    )
    ffmc, dmc, dc = codes_from_state(next_state)
    isi = initial_spread_index(ffmc, wind_kmh)
    bui = buildup_index(dmc, dc)
    fwi = fire_weather_index(isi, bui)
    isi = np.where(valid, isi, nan).astype(CALC_DTYPE)
    bui = np.where(valid, bui, nan).astype(CALC_DTYPE)
    fwi = np.where(valid, fwi, nan).astype(CALC_DTYPE)
    return FWI2025Output(next_state, ffmc, dmc, dc, isi, bui, fwi)


def adjective_class(fwi: np.ndarray) -> np.ndarray:
    """Return NRCan's proposed national FWI2025 adjective class (1 through 5)."""

    fwi = np.asarray(fwi, dtype=np.float32)
    out = np.digitize(fwi, np.asarray([6.0, 16.0, 23.0, 30.0], dtype=np.float32), right=False) + 1
    return np.where(np.isfinite(fwi), out, np.nan).astype(np.float32)
