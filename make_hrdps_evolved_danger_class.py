#!/usr/bin/env python3
"""Create native-grid HRDPS fire-danger graphics using hourly FWI2025 evolution.

Daily CWFIS FFMC/DMC/DC grids provide the latest observed anchor. The state is
bridged to model initialization with hourly fields from the preceding HRDPS run
or a checkpoint created by that run, then advanced with every hourly model field.
Only three-hourly forecast frames are rendered.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy

import fwi2025
import fire_danger_peak
import fire_danger_verification
import make_experimental_danger_class as danger_base
import make_hrdps_west_convective as hrdps
import plot_style
from make_hrdps_west_fourpanel import crop, hour_file
from make_hrdps_west_lightning import (
    DATA_CRS,
    FWI_CACHE_DIR,
    FWI_TO_3978,
    PLOT_CRS,
    add_transmission_lines,
    load_transmission_lines,
    relative_humidity_from_t_td,
)


FORECAST_HOURS = hrdps.FORECAST_HOURS
SURFACE_FIELDS = (
    ("TMP", "TGL", "2"),
    ("DPT", "TGL", "2"),
    ("UGRD", "TGL", "10"),
    ("VGRD", "TGL", "10"),
)
STATE_VERSION = 4
DANGER_DISPLAY_SMOOTHING_KM = 2.0
DANGER_CONTOUR_LEVELS = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)


@dataclass(frozen=True)
class ModelGrid:
    lat: np.ndarray
    lon: np.ndarray
    yslice: slice
    xslice: slice


@dataclass(frozen=True)
class EvolvedFields:
    fhour: int
    valid_utc: dt.datetime
    anchor_date: dt.date
    fwi: np.ndarray
    bui: np.ndarray
    danger: np.ndarray


@dataclass
class EvolutionStart:
    state: fwi2025.FWI2025State
    anchor_date: dt.date
    start_fhour: int
    bootstrap: bool = False


def log(message: str) -> None:
    print(message, flush=True)


def anchor_time_utc(anchor_date: dt.date) -> dt.datetime:
    """CWFIS daily codes represent noon local standard time in Pacific BC."""

    return dt.datetime.combine(anchor_date, dt.time(20, 0), tzinfo=dt.timezone.utc)


def candidate_anchor_dates(init_time: dt.datetime, count: int = 4) -> Iterator[dt.date]:
    local_date = init_time.astimezone(plot_style.LOCAL_TZ).date()
    for offset in range(count):
        candidate = local_date - dt.timedelta(days=offset)
        if anchor_time_utc(candidate) <= init_time:
            yield candidate


def required_names(stamp: str, fhour: int) -> list[str]:
    names = [hrdps.field_name(variable, level_type, level, stamp, fhour) for variable, level_type, level in SURFACE_FIELDS]
    if fhour > 0:
        names.append(hrdps.field_name("APCP", "SFC", "0", stamp, fhour))
    return names


def prerequisite_hours(hours: Iterable[int], start_hour: int = 0) -> tuple[int, ...]:
    requested = tuple(sorted(set(int(hour) for hour in hours)))
    if not requested:
        return ()
    return tuple(range(max(0, int(start_hour)), max(requested) + 1))


def download_hourly_fields(run: hrdps.RunInfo, data_dir: Path, hours: Iterable[int], workers: int = 8) -> None:
    run_dir = data_dir / run.stamp
    jobs: list[tuple[str, Path]] = []
    for fhour in sorted(set(int(hour) for hour in hours)):
        for name in required_names(run.stamp, fhour):
            dest = run_dir / f"{fhour:03d}" / name
            jobs.append((f"{hrdps.model_config().base_url}/{run.cycle}/{fhour:03d}/{name}", dest))
    missing = [(url, path) for url, path in jobs if not (path.exists() and path.stat().st_size > 0)]
    if not missing:
        return
    log(f"Downloading {len(missing)} hourly FWI2025 forcing files for {run.stamp}.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(hrdps.download_one, url, path) for url, path in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def load_model_grid(run: hrdps.RunInfo, data_dir: Path, extent: tuple[float, float, float, float]) -> ModelGrid:
    run_dir = data_dir / run.stamp
    sample_path = hour_file(run_dir, run, 0, "TMP", "TGL", "2")
    _, lat, lon = hrdps.read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read HRDPS coordinates.")
    yslice, xslice = hrdps.subset_slices(lat, lon, extent)
    return ModelGrid(lat[yslice, xslice], lon[yslice, xslice], yslice, xslice)


def interpolate_cwfis_to_model(
    grid: danger_base.CwfisGrid,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour sample a regular CWFIS raster after edge infilling."""

    source_x, source_y = FWI_TO_3978.transform(grid.lon, grid.lat)
    target_x, target_y = FWI_TO_3978.transform(target_lon, target_lat)
    x_axis = source_x[0, :]
    y_axis = source_y[:, 0]
    dx = float(np.nanmedian(np.diff(x_axis)))
    dy = float(np.nanmedian(np.diff(y_axis)))
    if not (np.isfinite(dx) and np.isfinite(dy) and abs(dx) > 0.0 and abs(dy) > 0.0):
        raise RuntimeError("CWFIS raster coordinates are not a regular projected grid.")
    cols = np.rint((target_x - x_axis[0]) / dx).astype(np.int64)
    rows = np.rint((target_y - y_axis[0]) / dy).astype(np.int64)
    inside = (rows >= 0) & (rows < grid.data.shape[0]) & (cols >= 0) & (cols < grid.data.shape[1])
    source_data = fill_nearest_valid(grid.data)
    output = np.full(target_lon.shape, np.nan, dtype=np.float32)
    output[inside] = source_data[rows[inside], cols[inside]]
    return output


def fill_nearest_valid(data: np.ndarray) -> np.ndarray:
    """Extend the CWFIS analyzed state across raster no-data gaps and edges."""

    valid = np.isfinite(data)
    if np.all(valid):
        return data.astype(np.float32, copy=False)
    if not np.any(valid):
        raise RuntimeError("CWFIS anchor raster contains no valid values.")
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return data[tuple(nearest)].astype(np.float32, copy=False)


def smooth_danger_for_display(danger: np.ndarray) -> np.ndarray:
    """Lightly smooth categorical danger for rendering while preserving its validity mask."""

    valid = np.isfinite(danger)
    smoothed = hrdps.smooth_nan(
        danger,
        sigma=hrdps.sigma_for_km(DANGER_DISPLAY_SMOOTHING_KM),
    )
    return np.where(valid, smoothed, np.nan).astype(np.float32)


def anchor_cache_path(
    model_key: str,
    anchor_date: dt.date,
    cache_dir: Path,
    extent: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> Path:
    west, east, south, north = extent
    domain = f"{west:.1f}_{east:.1f}_{south:.1f}_{north:.1f}".replace("-", "m").replace(".", "p")
    return cache_dir / "regridded_anchor_v3" / model_key / f"cwfis_ffmc_dmc_dc_{anchor_date:%Y%m%d}_{domain}_{shape[0]}x{shape[1]}.npz"


def load_anchor_state(
    model_key: str,
    anchor_date: dt.date,
    cache_dir: Path,
    extent: tuple[float, float, float, float],
    model_lon: np.ndarray,
    model_lat: np.ndarray,
) -> fwi2025.FWI2025State:
    cache_path = anchor_cache_path(model_key, anchor_date, cache_dir, extent, model_lat.shape)
    if cache_path.exists():
        with np.load(cache_path) as data:
            return fwi2025.FWI2025State.from_codes(data["ffmc"], data["dmc"], data["dc"])
    start = time.perf_counter()
    grids = {name: danger_base.read_cwfis_grid(name, anchor_date, cache_dir, extent) for name in ("ffmc", "dmc", "dc")}
    ffmc = np.clip(interpolate_cwfis_to_model(grids["ffmc"], model_lon, model_lat), 0.0, 101.0)
    dmc = np.maximum(interpolate_cwfis_to_model(grids["dmc"], model_lon, model_lat), 0.0)
    dc = np.maximum(interpolate_cwfis_to_model(grids["dc"], model_lon, model_lat), 0.0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ffmc=ffmc.astype(np.float32), dmc=dmc.astype(np.float32), dc=dc.astype(np.float32))
    log(f"Cached native-grid CWFIS anchor in {time.perf_counter() - start:.1f}s: {cache_path}")
    return fwi2025.FWI2025State.from_codes(ffmc, dmc, dc)


def state_root(cache_dir: Path, model_key: str) -> Path:
    return cache_dir / "fwi2025_state" / model_key


def handoff_checkpoint_hour() -> int:
    """Forecast hour that aligns with the next cycle of the active model."""

    cycles = sorted(int(cycle) for cycle in hrdps.model_config().cycles)
    intervals = [
        (cycles[(index + 1) % len(cycles)] - cycle) % 24
        for index, cycle in enumerate(cycles)
    ]
    return min(interval for interval in intervals if interval > 0)


def checkpoint_path(cache_dir: Path, model_key: str, run_stamp: str, fhour: int) -> Path:
    return state_root(cache_dir, model_key) / run_stamp / f"f{fhour:03d}.npz"


def save_checkpoint(
    cache_dir: Path,
    model_key: str,
    run: hrdps.RunInfo,
    fhour: int,
    anchor_date: dt.date,
    state: fwi2025.FWI2025State,
) -> Path:
    path = checkpoint_path(cache_dir, model_key, run.stamp, fhour)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                version=np.asarray([STATE_VERSION], dtype=np.int16),
                anchor_date=np.asarray(anchor_date.isoformat()),
                valid_utc=np.asarray((run.init_time + dt.timedelta(hours=fhour)).isoformat()),
                mcffmc=state.mcffmc,
                mcdmc=state.mcdmc,
                mcdc=state.mcdc,
                rain_total_mm=state.rain_total_mm,
                canopy_drying_hours=state.canopy_drying_hours,
            )
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path, shape: tuple[int, int], anchor_date: dt.date | None = None) -> tuple[dt.date, fwi2025.FWI2025State] | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if int(data["version"][0]) != STATE_VERSION:
                return None
            saved_anchor = dt.date.fromisoformat(str(data["anchor_date"]))
            if anchor_date is not None and saved_anchor != anchor_date:
                return None
            fields = tuple(
                data[name].astype(fwi2025.CALC_DTYPE)
                for name in ("mcffmc", "mcdmc", "mcdc", "rain_total_mm", "canopy_drying_hours")
            )
        if any(field.shape != shape for field in fields):
            return None
        return saved_anchor, fwi2025.FWI2025State(*fields)
    except (OSError, KeyError, ValueError):
        return None


def prune_checkpoints(cache_dir: Path, model_key: str, run: hrdps.RunInfo, keep_fhours: set[int]) -> None:
    directory = state_root(cache_dir, model_key) / run.stamp
    if not directory.exists():
        return
    for path in directory.glob("f*.npz"):
        try:
            fhour = int(path.stem[1:])
        except ValueError:
            continue
        if fhour not in keep_fhours:
            path.unlink()
    cutoff = run.init_time - dt.timedelta(hours=24)
    for child in state_root(cache_dir, model_key).iterdir():
        if not child.is_dir() or child.name == run.stamp:
            continue
        try:
            init = hrdps.parse_stamp(child.name)
        except ValueError:
            continue
        if init < cutoff:
            __import__("shutil").rmtree(child)


def run_from_stamp(stamp: str) -> hrdps.RunInfo:
    init = hrdps.parse_stamp(stamp)
    return hrdps.RunInfo(cycle=f"{init.hour:02d}", stamp=stamp, init_time=init)


def surface_weather_hourly(
    run_dir: Path,
    run: hrdps.RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temp_c = crop(hour_file(run_dir, run, fhour, "TMP", "TGL", "2"), yslice, xslice) - 273.15
    dpt_c = crop(hour_file(run_dir, run, fhour, "DPT", "TGL", "2"), yslice, xslice) - 273.15
    rh = relative_humidity_from_t_td(temp_c, dpt_c)
    u10 = crop(hour_file(run_dir, run, fhour, "UGRD", "TGL", "10"), yslice, xslice)
    v10 = crop(hour_file(run_dir, run, fhour, "VGRD", "TGL", "10"), yslice, xslice)
    wind_kmh = 3.6 * np.hypot(u10, v10)
    if fhour <= 0:
        precip = np.zeros_like(temp_c, dtype=np.float32)
    else:
        current = crop(hour_file(run_dir, run, fhour, "APCP", "SFC", "0"), yslice, xslice)
        if fhour == 1:
            precip = np.maximum(current, 0.0)
        else:
            previous = crop(hour_file(run_dir, run, fhour - 1, "APCP", "SFC", "0"), yslice, xslice)
            precip = np.maximum(current - previous, 0.0)
    # Remove sub-micrometre differencing noise before rain-event threshold tests.
    precip = np.round(precip, 3)
    return temp_c.astype(np.float32), rh.astype(np.float32), wind_kmh.astype(np.float32), precip.astype(np.float32)


def advance_state(
    state: fwi2025.FWI2025State,
    source_run: hrdps.RunInfo,
    source_dir: Path,
    grid: ModelGrid,
    start_valid: dt.datetime,
    end_valid: dt.datetime,
) -> fwi2025.FWI2025State:
    if end_valid < start_valid:
        raise ValueError("Cannot advance FWI2025 state backwards.")
    current = start_valid
    while current < end_valid:
        current += dt.timedelta(hours=1)
        fhour = int((current - source_run.init_time).total_seconds() // 3600)
        if fhour < 1 or fhour > 48:
            raise RuntimeError(f"Bridge time {current:%Y-%m-%d %HZ} is outside {source_run.stamp}.")
        temp, rh, wind, precip = surface_weather_hourly(source_dir, source_run, fhour, grid.yslice, grid.xslice)
        daylight = fwi2025.daylight_mask(current, grid.lat, grid.lon, plot_style.LOCAL_TZ)
        local_hour = current.astimezone(plot_style.LOCAL_TZ).hour
        state = fwi2025.step(state, temp, rh, wind, precip, daylight, float(local_hour)).state
    return state


def find_bridge_run(data_dir: Path, anchor_time: dt.datetime, target_time: dt.datetime) -> hrdps.RunInfo | None:
    candidates = []
    if not data_dir.exists():
        return None
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            run = run_from_stamp(child.name)
        except ValueError:
            continue
        if run.init_time <= anchor_time and run.init_time + dt.timedelta(hours=48) >= target_time:
            candidates.append(run)
    return max(candidates, key=lambda item: item.init_time, default=None)


def previous_checkpoint(
    run: hrdps.RunInfo,
    cache_dir: Path,
    model_key: str,
    shape: tuple[int, int],
    anchor_date: dt.date,
) -> fwi2025.FWI2025State | None:
    for delta_hours in (6, 12, 18, 24):
        previous_init = run.init_time - dt.timedelta(hours=delta_hours)
        previous_stamp = previous_init.strftime("%Y%m%dT%HZ")
        loaded = load_checkpoint(checkpoint_path(cache_dir, model_key, previous_stamp, delta_hours), shape, anchor_date)
        if loaded is not None:
            log(f"Starting from verified checkpoint {previous_stamp} F{delta_hours:03d}.")
            return loaded[1]
    return None


def initialize_evolution(
    run: hrdps.RunInfo,
    data_dir: Path,
    cache_dir: Path,
    extent: tuple[float, float, float, float],
    grid: ModelGrid,
    allow_bootstrap: bool,
) -> EvolutionStart:
    model_key = hrdps.model_config().key
    for anchor_date in candidate_anchor_dates(run.init_time):
        current_checkpoint = load_checkpoint(checkpoint_path(cache_dir, model_key, run.stamp, 0), grid.lat.shape, anchor_date)
        if current_checkpoint is not None:
            return EvolutionStart(current_checkpoint[1], anchor_date, 0)
        prior = previous_checkpoint(run, cache_dir, model_key, grid.lat.shape, anchor_date)
        if prior is not None:
            save_checkpoint(cache_dir, model_key, run, 0, anchor_date, prior)
            return EvolutionStart(prior, anchor_date, 0)
        try:
            anchor_state = load_anchor_state(model_key, anchor_date, cache_dir, extent, grid.lon, grid.lat)
        except Exception as exc:
            log(f"CWFIS anchor {anchor_date:%Y-%m-%d} is unavailable: {exc}")
            continue
        bridge_run = find_bridge_run(data_dir, anchor_time_utc(anchor_date), run.init_time)
        if bridge_run is not None:
            log(f"Bridging CWFIS {anchor_date:%Y-%m-%d} to {run.stamp} with hourly {bridge_run.stamp} fields.")
            state = advance_state(
                anchor_state,
                bridge_run,
                data_dir / bridge_run.stamp,
                grid,
                anchor_time_utc(anchor_date),
                run.init_time,
            )
            save_checkpoint(cache_dir, model_key, run, 0, anchor_date, state)
            return EvolutionStart(state, anchor_date, 0)
        latest_date = dt.datetime.now(dt.timezone.utc).astimezone(plot_style.LOCAL_TZ).date()
        latest_lead = int((anchor_time_utc(latest_date) - run.init_time).total_seconds() // 3600)
        if allow_bootstrap and 0 <= latest_lead <= 48:
            break
        log(f"No hourly bridge is retained for CWFIS {anchor_date:%Y-%m-%d}; trying an older verified anchor.")

    if allow_bootstrap:
        latest_date = dt.datetime.now(dt.timezone.utc).astimezone(plot_style.LOCAL_TZ).date()
        bootstrap_time = anchor_time_utc(latest_date)
        lead = int((bootstrap_time - run.init_time).total_seconds() // 3600)
        if 0 <= lead <= 48:
            log(f"Bootstrapping at CWFIS {latest_date:%Y-%m-%d} ({bootstrap_time:%HZ}); earlier forecast frames will be skipped.")
            state = load_anchor_state(model_key, latest_date, cache_dir, extent, grid.lon, grid.lat)
            save_checkpoint(cache_dir, model_key, run, lead, latest_date, state)
            return EvolutionStart(state, latest_date, lead, bootstrap=True)

    raise RuntimeError(
        "No valid CWFIS-to-initialization bridge exists. Keep the preceding run's hourly surface data/checkpoint, "
        "or use --allow-bootstrap once to start at the next observed anchor without publishing earlier frames."
    )


def diagnose_state(state: fwi2025.FWI2025State, wind_kmh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ffmc, dmc, dc = fwi2025.codes_from_state(state)
    isi = fwi2025.initial_spread_index(ffmc, wind_kmh)
    bui = fwi2025.buildup_index(dmc, dc)
    fwi = fwi2025.fire_weather_index(isi, bui)
    return fwi, bui, isi


def iter_evolved_fields(
    run: hrdps.RunInfo,
    data_dir: Path,
    cache_dir: Path,
    extent: tuple[float, float, float, float],
    hours: Iterable[int],
    grid: ModelGrid,
    classification: str = "fwi2025",
    allow_bootstrap: bool = False,
) -> Iterator[EvolvedFields]:
    requested = tuple(sorted(set(int(hour) for hour in hours)))
    if not requested:
        return
    start = initialize_evolution(run, data_dir, cache_dir, extent, grid, allow_bootstrap)
    requested = tuple(hour for hour in requested if hour >= start.start_fhour)
    if not requested:
        return
    model_key = hrdps.model_config().key
    state = start.state
    start_hour = start.start_fhour
    for candidate in sorted((state_root(cache_dir, model_key) / run.stamp).glob("f*.npz"), reverse=True):
        try:
            fhour = int(candidate.stem[1:])
        except ValueError:
            continue
        if start_hour < fhour <= min(requested):
            loaded = load_checkpoint(candidate, grid.lat.shape, start.anchor_date)
            if loaded is not None:
                state = loaded[1]
                start_hour = fhour
                break

    bc_geometry = danger_base.load_bc_geometry()
    in_bc = contains_xy(bc_geometry, grid.lon, grid.lat)
    regions = danger_base.danger_regions(grid.lon, grid.lat)
    peak_accumulator = fire_danger_peak.PeakBurnAccumulator(plot_style.LOCAL_TZ)
    saved_peak_dates: list[dt.date] = []
    station_mapping = None
    run_dir = data_dir / run.stamp
    max_hour = max(requested)
    latest_saved = start_hour
    handoff_hour = handoff_checkpoint_hour()

    def cache_peak_day(
        peak_day: fire_danger_peak.PeakBurnDay,
        *,
        allow_partial: bool,
    ) -> tuple[np.ndarray, Path]:
        peak_danger = danger_base.classify_danger(peak_day.fwi, peak_day.bui, regions)
        peak_danger = np.where(in_bc, peak_danger, np.nan).astype(np.float32)
        peak_path = fire_danger_peak.save_peak_danger_grid(
            cache_dir,
            model_key,
            run.stamp,
            run.init_time,
            peak_day,
            peak_danger,
            allow_partial=allow_partial,
        )
        saved_peak_dates.append(peak_day.fire_date)
        return peak_danger, peak_path

    for fhour in range(start_hour, max_hour + 1):
        valid = run.init_time + dt.timedelta(hours=fhour)
        temp, rh, wind, precip = surface_weather_hourly(run_dir, run, fhour, grid.yslice, grid.xslice)
        if fhour > start_hour:
            daylight = fwi2025.daylight_mask(valid, grid.lat, grid.lon, plot_style.LOCAL_TZ)
            local_hour = valid.astimezone(plot_style.LOCAL_TZ).hour
            output = fwi2025.step(state, temp, rh, wind, precip, daylight, float(local_hour))
            state = output.state
            ffmc = output.ffmc
            fwi, bui = output.fwi, output.bui
        else:
            fwi, bui, _ = diagnose_state(state, wind)
            ffmc, _, _ = fwi2025.codes_from_state(state)

        for peak_day in peak_accumulator.push(valid, wind, ffmc, fwi, bui):
            if not peak_day.complete:
                continue
            peak_danger, peak_path = cache_peak_day(peak_day, allow_partial=False)
            log(f"Cached complete peak-burn fire day {peak_day.fire_date:%Y-%m-%d}: {peak_path}")
            try:
                if station_mapping is None:
                    station_mapping = fire_danger_verification.station_grid_mapping(
                        model_key,
                        grid.lat,
                        grid.lon,
                        cache_dir,
                    )
                station_path = fire_danger_verification.archive_station_peak_forecast(
                    run.stamp,
                    run.init_time,
                    model_key,
                    peak_day,
                    peak_danger,
                    station_mapping,
                )
                log(f"Archived BCWS-station peak forecast sample: {station_path}")
            except Exception as exc:
                log(f"BCWS station forecast archive unavailable: {exc}")

        if fhour in requested:
            if classification == "schedule2":
                danger = danger_base.classify_danger(fwi, bui, regions)
            else:
                danger = fwi2025.adjective_class(fwi)
            danger = np.where(in_bc, danger, np.nan).astype(np.float32)
            frame = EvolvedFields(
                fhour,
                valid,
                start.anchor_date,
                fwi.astype(np.float32),
                bui.astype(np.float32),
                danger,
            )
            save_checkpoint(cache_dir, model_key, run, fhour, start.anchor_date, state)
            latest_saved = fhour
            incremental_keep = {fhour}
            if fhour >= handoff_hour:
                incremental_keep.add(handoff_hour)
            prune_checkpoints(cache_dir, model_key, run, incremental_keep)
            yield frame

    terminal_peak_day = peak_accumulator.finish()
    if terminal_peak_day is not None and not terminal_peak_day.complete:
        if fire_danger_peak.reaches_peak_guidance_cutoff(terminal_peak_day, plot_style.LOCAL_TZ):
            _, peak_path = cache_peak_day(terminal_peak_day, allow_partial=True)
            log(
                f"Cached partial peak-burn fire day through at least 17:00 local "
                f"{terminal_peak_day.fire_date:%Y-%m-%d} "
                f"({terminal_peak_day.hour_count} hourly fields): {peak_path}"
            )
        else:
            stale_path = fire_danger_peak.peak_cache_path(
                cache_dir,
                model_key,
                terminal_peak_day.fire_date,
                run.stamp,
            )
            stale_path.unlink(missing_ok=True)
            coverage_end = terminal_peak_day.last_valid_utc.astimezone(plot_style.LOCAL_TZ)
            log(
                f"Skipped under-covered peak-burn fire day "
                f"{terminal_peak_day.fire_date:%Y-%m-%d}; run ends at "
                f"{coverage_end:%H:%M %Z}, before the 17:00 local cutoff."
            )

    keep = {latest_saved}
    if latest_saved >= handoff_hour:
        keep.add(handoff_hour)
    prune_checkpoints(cache_dir, model_key, run, keep)
    if set(FORECAST_HOURS).issubset(requested) and saved_peak_dates:
        fire_danger_peak.mark_peak_run_complete(cache_dir, model_key, run.stamp, saved_peak_dates)
        fire_danger_peak.prune_peak_cache(cache_dir, model_key)


def plot_evolved_danger(
    out_path: Path,
    run: hrdps.RunInfo,
    grid: ModelGrid,
    fields: EvolvedFields,
    watersheds: list,
    transmission_lines: list,
    extent: tuple[float, float, float, float],
    classification: str,
    plot_stride: int = 1,
) -> None:
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=PLOT_CRS)
    ax.set_extent(extent, crs=DATA_CRS)
    ax.set_facecolor("#dbeaf0")
    hrdps.add_map_features(ax)
    cmap, norm = danger_base.danger_cmap()
    sample = slice(None, None, max(1, int(plot_stride)))
    display_danger = smooth_danger_for_display(fields.danger)
    shaded = ax.contourf(
        grid.lon[sample, sample],
        grid.lat[sample, sample],
        display_danger[sample, sample],
        levels=DANGER_CONTOUR_LEVELS,
        cmap=cmap,
        norm=norm,
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    hrdps.add_hydro_features(ax)
    add_transmission_lines(ax, transmission_lines)
    hrdps.add_watersheds(ax, watersheds)
    if classification == "schedule2":
        bc_geometry = danger_base.load_bc_geometry()
        in_bc = contains_xy(bc_geometry, grid.lon, grid.lat)
        regions = danger_base.danger_regions(grid.lon, grid.lat)
        danger_base.draw_region_boundaries(ax, regions, grid.lon, grid.lat, in_bc)
    hrdps.add_city_labels(ax, fontsize=7.0, marker_size=2.0, path_width=2.3, zorder=30)
    tick_labels = ["Very low", "Low", "Moderate", "High", "Extreme"] if classification == "schedule2" else ["Low", "Moderate", "High", "Very high", "Extreme"]
    colorbar_label = "Experimental hourly fire danger" if classification == "schedule2" else "Experimental FWI2025 fire danger"
    plot_style.add_internal_colorbar(
        fig,
        ax,
        shaded,
        ticks=[1, 2, 3, 4, 5],
        label=colorbar_label,
        fmt="%d",
        tick_labels=tick_labels,
        extend=None,
        backdrop=(0.920, 0.102, 0.070, 0.760),
        cax_bounds=[0.948, 0.135, 0.020, 0.680],
    )
    valid_local = fields.valid_utc.astimezone(plot_style.LOCAL_TZ)
    title = "FWI2025 FIRE DANGER" if classification == "fwi2025" else "EXPERIMENTAL HOURLY BC FIRE DANGER"
    method = "NRCan proposed FWI2025 classes" if classification == "fwi2025" else "BC Schedule 2 classes from hourly FWI2025 FWI+BUI"
    plot_style.add_single_panel_text(
        ax,
        (
            f"{hrdps.model_config().label} {title}  |  F{fields.fhour:03d}  "
            f"{valid_local:%a %H:%M%Z %d%b%Y}"
        ).upper(),
        f"Shaded: {method}; CWFIS FFMC/DMC/DC anchor {fields.anchor_date:%d%b%Y}; hourly HRDPS evolution; grey: BC transmission; not an official BC danger rating",
        run,
        source_label="ECCC HRDPS + CWFIS",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_plots(
    run: hrdps.RunInfo,
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    hours: Iterable[int],
    classification: str = "schedule2",
    plot_stride: int = 1,
    watersheds: list | None = None,
    allow_bootstrap: bool = False,
) -> list[Path]:
    requested = tuple(sorted(set(int(hour) for hour in hours)))
    grid = load_model_grid(run, data_dir, hrdps.model_config().extent)
    watersheds = watersheds if watersheds is not None else hrdps.load_watersheds(hrdps.WATERSHED_CACHE)
    transmission_lines = load_transmission_lines()
    run_output = output_dir / run.stamp
    prefix = f"{hrdps.model_config().output_prefix}_fwi2025_danger"
    paths = []
    for fields in iter_evolved_fields(
        run,
        data_dir,
        cache_dir,
        hrdps.model_config().extent,
        requested,
        grid,
        classification,
        allow_bootstrap,
    ):
        out_path = run_output / f"{prefix}_{run.stamp}_f{fields.fhour:03d}.png"
        plot_evolved_danger(
            out_path,
            run,
            grid,
            fields,
            watersheds,
            transmission_lines,
            hrdps.model_config().extent,
            classification,
            plot_stride,
        )
        paths.append(out_path)
        log(f"Wrote {out_path}")
    return paths


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stamp", help="HRDPS run stamp, e.g. 20260709T12Z")
    parser.add_argument("--model", choices=sorted(hrdps.MODEL_CONFIGS), default="west")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/experimental_fwi2025_danger"))
    parser.add_argument("--cache-dir", type=Path, default=FWI_CACHE_DIR)
    parser.add_argument("--hours", default=",".join(str(hour) for hour in FORECAST_HOURS))
    parser.add_argument("--classification", choices=("fwi2025", "schedule2"), default="schedule2")
    parser.add_argument("--plot-stride", type=int, default=1, help="Native-grid shading is the default; larger values are diagnostic only.")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--allow-bootstrap", action="store_true", help="Initialize at a later observed anchor if no historical bridge exists; skips earlier frames.")
    parser.add_argument("--no-watersheds", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = hrdps.set_model(args.model)
    data_dir = args.data_dir or Path(config.default_data_dir)
    run = run_from_stamp(args.stamp)
    hours = tuple(int(item) for item in args.hours.split(",") if item.strip())
    if args.download_missing:
        download_hourly_fields(run, data_dir, prerequisite_hours(hours), args.download_workers)
    watersheds = [] if args.no_watersheds else None
    paths = make_plots(
        run,
        data_dir,
        args.output_dir,
        args.cache_dir,
        hours,
        args.classification,
        args.plot_stride,
        watersheds,
        args.allow_bootstrap,
    )
    if args.metadata:
        metadata = {
            "run": run.stamp,
            "model": config.label,
            "method": "CWFIS daily FFMC/DMC/DC anchor; sequential hourly vectorized FWI2025 evolution; three-hourly graphics.",
            "classification": args.classification,
            "frames": [path.name for path in paths],
        }
        meta_path = args.output_dir / run.stamp / f"{config.output_prefix}_fwi2025_danger_{run.stamp}.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
