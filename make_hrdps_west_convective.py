#!/usr/bin/env python3
"""Download HRDPS data and make 3-hourly convective composites."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import functools
import hashlib
import json
import math
import multiprocessing
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
import requests
from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release
from metpy.calc import downdraft_cape
from metpy.units import units
from scipy.ndimage import gaussian_filter
from shapely import make_valid
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

import plot_style
import gust_diagnostics

WEST_EXTENT = (-138.7, -109.0, 46.0, 58.45)
WATERSHED_CACHE = Path("data/bc_watersheds/bch/AllWatershedsUTM.shp")
WATERSHED_EDGE_COLOR = "#173f73"
WATERSHED_LINEWIDTH = 0.80
WATERSHED_HALO_LINEWIDTH = 1.18


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    source_label: str
    base_url: str
    grid_tag: str
    filename_style: str
    cycles: tuple[str, ...]
    output_prefix: str
    resolution_km: float
    default_data_dir: str
    default_output_dir: str
    extent: tuple[float, float, float, float] = WEST_EXTENT


MODEL_CONFIGS = {
    "west": ModelConfig(
        key="west",
        label="HRDPS-West 1 km",
        source_label="ECCC HRDPS-West 1 km",
        base_url="https://dd.alpha.weather.gc.ca/model_hrdps/west/1km/grib2",
        grid_tag="rotated_latlon0.009x0.009",
        filename_style="legacy",
        cycles=("00", "12"),
        output_prefix="hrdps_west",
        resolution_km=1.0,
        default_data_dir="data/hrdps_west",
        default_output_dir="plots/hrdps_west",
    ),
    "continental": ModelConfig(
        key="continental",
        label="HRDPS 2.5 km",
        source_label="ECCC HRDPS continental 2.5 km",
        base_url="https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km",
        grid_tag="RLatLon0.0225",
        filename_style="modern",
        cycles=("00", "06", "12", "18"),
        output_prefix="hrdps_continental",
        resolution_km=2.5,
        default_data_dir="data/hrdps_continental",
        default_output_dir="plots/hrdps_continental",
    ),
}

ACTIVE_MODEL = MODEL_CONFIGS["west"]
BASE_URL = ACTIVE_MODEL.base_url
GRID_TAG = ACTIVE_MODEL.grid_tag
AVAILABLE_CYCLES = ACTIVE_MODEL.cycles
FORECAST_HOURS = tuple(range(0, 49, 3))
TERRAIN_FHOUR = 3
DCAPE_LEVELS_HPA = (1015, 1000, 985, 970, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500)
DIAGNOSTIC_CACHE_VERSION = "dcape_pcge_v5"
EXTENT = ACTIVE_MODEL.extent
LOCAL_TZ = plot_style.LOCAL_TZ
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RunInfo:
    cycle: str
    stamp: str
    init_time: dt.datetime


def set_model(model_key: str) -> ModelConfig:
    global ACTIVE_MODEL, BASE_URL, GRID_TAG, AVAILABLE_CYCLES, EXTENT
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported HRDPS model: {model_key}")
    ACTIVE_MODEL = MODEL_CONFIGS[model_key]
    BASE_URL = ACTIVE_MODEL.base_url
    GRID_TAG = ACTIVE_MODEL.grid_tag
    AVAILABLE_CYCLES = ACTIVE_MODEL.cycles
    EXTENT = ACTIVE_MODEL.extent
    return ACTIVE_MODEL


def model_config() -> ModelConfig:
    return ACTIVE_MODEL


def model_output_prefix(product: str) -> str:
    return f"{ACTIVE_MODEL.output_prefix}_{product}"


def grid_stride(target_km: float, minimum: int = 1) -> int:
    return max(minimum, int(round(target_km / ACTIVE_MODEL.resolution_km)))


def sigma_for_km(target_km: float, minimum: float = 0.55) -> float:
    return max(minimum, target_km / ACTIVE_MODEL.resolution_km)


def log(message: str) -> None:
    print(message, flush=True)


def parse_links(html: str) -> list[str]:
    return re.findall(r'href="([^"]+)"', html)


def fetch_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def run_stamp_from_listing(cycle: str) -> str | None:
    html = fetch_text(f"{BASE_URL}/{cycle}/000/")
    if ACTIVE_MODEL.filename_style == "modern":
        match = re.search(r"(\d{8}T\d{2}Z)_MSC_HRDPS_", html)
        return match.group(1) if match else None
    match = re.search(r"_(\d{8}T\d{2}Z)_P000-00\.grib2", html)
    return match.group(1) if match else None


def parse_stamp(stamp: str) -> dt.datetime:
    return dt.datetime.strptime(stamp, "%Y%m%dT%HZ").replace(tzinfo=dt.timezone.utc)


def field_name(variable: str, level_type: str, level: str | int, stamp: str, fhour: int) -> str:
    if ACTIVE_MODEL.filename_style == "modern":
        return f"{stamp}_MSC_HRDPS_{variable}_{modern_level(variable, level_type, level)}_{GRID_TAG}_PT{fhour:03d}H.grib2"
    return f"CMC_hrdps_west_{variable}_{level_type}_{level}_{GRID_TAG}_{stamp}_P{fhour:03d}-00.grib2"


def modern_level(variable: str, level_type: str, level: str | int) -> str:
    level_type = level_type.upper()
    if variable in {"HLCY", "HPBL", "CAPE"} or level_type == "SFC":
        return "Sfc"
    if level_type == "MSL":
        return "MSL"
    if level_type == "TGL":
        return f"AGL-{int(level)}m"
    if level_type == "ISBL":
        return f"ISBL_{int(level):04d}"
    if level_type in {"ETAL", "EATM"}:
        return "Sfc"
    raise ValueError(f"Unsupported modern HRDPS level: {variable} {level_type} {level}")


def required_names(stamp: str, fhour: int, include_static: bool = False) -> list[str]:
    names = [
        field_name("MU-VT-LI", "ISBL", "500", stamp, fhour),
        field_name("HLCY", "ETAL", "10000", stamp, fhour),
        field_name("HPBL", "SFC", "0", stamp, fhour),
        field_name("CAPE", "ETAL", "10000", stamp, fhour),
        field_name("GUST", "TGL", "10", stamp, fhour),
        field_name("WIND", "ISBL", "0850", stamp, fhour),
        field_name("WIND", "ISBL", "0700", stamp, fhour),
        field_name("VVEL", "ISBL", "0700", stamp, fhour),
        field_name("PRES", "SFC", "0", stamp, fhour),
        field_name("TMP", "TGL", "2", stamp, fhour),
        field_name("DPT", "TGL", "2", stamp, fhour),
    ]
    for level in DCAPE_LEVELS_HPA:
        names.append(field_name("TMP", "ISBL", f"{level:04d}", stamp, fhour))
        names.append(field_name("DEPR", "ISBL", f"{level:04d}", stamp, fhour))
    if fhour > 0:
        names.append(field_name("PRATE", "SFC", "0", stamp, fhour))
    if include_static:
        names.append(field_name("HGT", "SFC", "0", stamp, fhour))
    return names


def run_is_complete(run: RunInfo) -> bool:
    for fhour in FORECAST_HOURS:
        html = fetch_text(f"{BASE_URL}/{run.cycle}/{fhour:03d}/")
        links = set(parse_links(html))
        needed = set(required_names(run.stamp, fhour, include_static=(fhour == TERRAIN_FHOUR)))
        missing = sorted(needed - links)
        if missing:
            log(f"Skipping {run.stamp}: missing {len(missing)} files at F{fhour:03d}.")
            return False
    return True


def latest_complete_run() -> RunInfo:
    candidates: list[RunInfo] = []
    for cycle in AVAILABLE_CYCLES:
        try:
            stamp = run_stamp_from_listing(cycle)
        except Exception as exc:
            log(f"Could not inspect cycle {cycle}Z: {exc}")
            continue
        if stamp:
            candidates.append(RunInfo(cycle=cycle, stamp=stamp, init_time=parse_stamp(stamp)))

    for run in sorted(candidates, key=lambda item: item.init_time, reverse=True):
        log(f"Checking completeness for {ACTIVE_MODEL.label} {run.stamp}.")
        if run_is_complete(run):
            return run

    raise RuntimeError(f"No complete {ACTIVE_MODEL.label} {'/'.join(AVAILABLE_CYCLES)} run was found in the live feed.")


def is_retryable_download_error(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(
        exc,
        (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    )


def retry_delay_seconds(attempt: int) -> float:
    return min(90.0, (2 ** max(0, attempt - 1)) + random.uniform(0.0, 1.0))


def download_one(url: str, dest: Path, retries: int = 4) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file: {url}")
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt >= retries or not is_retryable_download_error(exc):
                break
            time.sleep(retry_delay_seconds(attempt))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_run(run: RunInfo, data_dir: Path, workers: int) -> None:
    jobs: list[tuple[str, Path]] = []
    run_dir = data_dir / run.stamp
    for fhour in FORECAST_HOURS:
        names = required_names(run.stamp, fhour, include_static=(fhour == TERRAIN_FHOUR))
        for name in names:
            url = f"{BASE_URL}/{run.cycle}/{fhour:03d}/{name}"
            jobs.append((url, run_dir / f"{fhour:03d}" / name))

    log(f"Downloading or reusing {len(jobs)} GRIB2 files into {run_dir}.")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, url, dest) for url, dest in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                log(f"  files ready: {completed}/{len(jobs)}")


def read_grib(path: Path, coords: bool = False) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    with path.open("rb") as handle:
        gid = codes_grib_new_from_file(handle)
        try:
            nx = int(codes_get(gid, "Ni"))
            ny = int(codes_get(gid, "Nj"))
            values = codes_get_array(gid, "values").reshape(ny, nx).astype(np.float32)
            values[np.abs(values) > 1e20] = np.nan
            if coords:
                lat = codes_get_array(gid, "latitudes").reshape(ny, nx).astype(np.float32)
                lon = codes_get_array(gid, "longitudes").reshape(ny, nx).astype(np.float32)
                lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
                return values, lat, lon
            return values, None, None
        finally:
            codes_release(gid)


def subset_slices(lat: np.ndarray, lon: np.ndarray, extent: tuple[float, float, float, float]) -> tuple[slice, slice]:
    west, east, south, north = extent
    mask = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    if not np.any(mask):
        raise RuntimeError("Requested extent does not overlap the model grid.")
    y, x = np.where(mask)
    pad = 8
    return (
        slice(max(int(y.min()) - pad, 0), min(int(y.max()) + pad + 1, lat.shape[0])),
        slice(max(int(x.min()) - pad, 0), min(int(x.max()) + pad + 1, lat.shape[1])),
    )


def thin_indices(size: int, stride: int) -> np.ndarray:
    indices = np.arange(0, size, stride, dtype=int)
    if indices[-1] != size - 1:
        indices = np.r_[indices, size - 1]
    return indices


def smooth_nan(data: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    valid = np.isfinite(data)
    if not np.any(valid):
        return data
    filled = np.where(valid, data, 0.0)
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    smoothed = gaussian_filter(filled.astype(np.float32), sigma=sigma, mode="nearest")
    out = smoothed / np.where(weights == 0, np.nan, weights)
    out[~np.isfinite(out)] = np.nan
    return out


def contour_sample(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    stride: int = 6,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = smooth_nan(data, sigma=sigma)
    return lat[::stride, ::stride], lon[::stride, ::stride], smoothed[::stride, ::stride]


def profile_dcape(pressure_hpa: np.ndarray, temp_c: np.ndarray, dewpoint_c: np.ndarray) -> float:
    order = np.argsort(pressure_hpa)[::-1]
    pressure_hpa = pressure_hpa[order]
    temp_c = temp_c[order]
    dewpoint_c = dewpoint_c[order]

    keep = np.r_[True, np.diff(pressure_hpa) < -0.05]
    pressure_hpa = pressure_hpa[keep]
    temp_c = temp_c[keep]
    dewpoint_c = dewpoint_c[keep]

    finite = np.isfinite(pressure_hpa) & np.isfinite(temp_c) & np.isfinite(dewpoint_c)
    pressure_hpa = pressure_hpa[finite]
    temp_c = temp_c[finite]
    dewpoint_c = dewpoint_c[finite]

    if len(pressure_hpa) < 6 or np.nanmax(pressure_hpa) < 700.0 or np.nanmin(pressure_hpa) > 500.0:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dcape = downdraft_cape(
                pressure_hpa * units.hPa,
                temp_c * units.degC,
                dewpoint_c * units.degC,
            )[0].magnitude
    except Exception:
        return np.nan

    if not math.isfinite(dcape):
        return np.nan
    return max(0.0, float(dcape))


def _dcape_rows(
    row_start: int,
    psfc_hpa: np.ndarray,
    tmp2_c: np.ndarray,
    dpt2_c: np.ndarray,
    levels: np.ndarray,
    temp_levels: np.ndarray,
    dpt_levels: np.ndarray,
) -> tuple[int, np.ndarray]:
    output = np.full(psfc_hpa.shape, np.nan, dtype=np.float32)
    for jj in range(output.shape[0]):
        for ii in range(output.shape[1]):
            p0 = float(psfc_hpa[jj, ii])
            if not math.isfinite(p0) or p0 < 705.0:
                continue
            above_ground = levels <= p0
            pressure_profile = np.r_[p0, levels[above_ground]]
            temp_profile = np.r_[tmp2_c[jj, ii], temp_levels[above_ground, jj, ii]]
            dpt_profile = np.r_[dpt2_c[jj, ii], dpt_levels[above_ground, jj, ii]]
            output[jj, ii] = profile_dcape(pressure_profile, temp_profile, dpt_profile)
    return row_start, output


def compute_dcape(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    stride: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hour_dir = run_dir / f"{fhour:03d}"
    jidx = thin_indices(lat[yslice, xslice].shape[0], stride)
    iidx = thin_indices(lat[yslice, xslice].shape[1], stride)
    coarse_lat = lat[yslice, xslice][np.ix_(jidx, iidx)]
    coarse_lon = lon[yslice, xslice][np.ix_(jidx, iidx)]

    def crop_sample(path: Path) -> np.ndarray:
        data, _, _ = read_grib(path)
        return data[yslice, xslice][np.ix_(jidx, iidx)]

    psfc_hpa = crop_sample(hour_dir / field_name("PRES", "SFC", "0", run.stamp, fhour)) / 100.0
    tmp2_c = crop_sample(hour_dir / field_name("TMP", "TGL", "2", run.stamp, fhour)) - 273.15
    dpt2_c = crop_sample(hour_dir / field_name("DPT", "TGL", "2", run.stamp, fhour)) - 273.15

    temp_levels = []
    dpt_levels = []
    levels = np.asarray(DCAPE_LEVELS_HPA, dtype=np.float32)
    for level in DCAPE_LEVELS_HPA:
        tmp = crop_sample(hour_dir / field_name("TMP", "ISBL", f"{level:04d}", run.stamp, fhour)) - 273.15
        depr = crop_sample(hour_dir / field_name("DEPR", "ISBL", f"{level:04d}", run.stamp, fhour))
        temp_levels.append(tmp)
        dpt_levels.append(tmp - depr)
    temp_levels_arr = np.stack(temp_levels)
    dpt_levels_arr = np.stack(dpt_levels)

    configured_workers = max(1, int(os.environ.get("FCSTGRAPHICS_DCAPE_WORKERS", "4")))
    worker_count = min(configured_workers, psfc_hpa.shape[0])
    if worker_count == 1:
        _, out = _dcape_rows(0, psfc_hpa, tmp2_c, dpt2_c, levels, temp_levels_arr, dpt_levels_arr)
    else:
        out = np.full(psfc_hpa.shape, np.nan, dtype=np.float32)
        row_groups = [group for group in np.array_split(np.arange(psfc_hpa.shape[0]), worker_count) if group.size]
        log(f"    DCAPE using {len(row_groups)} row workers for F{fhour:03d}")
        # Fork keeps the already-read profile arrays copy-on-write and avoids
        # repeatedly importing Cartopy/MetPy in every short-lived row worker.
        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(row_groups), mp_context=context) as executor:
            futures = [
                executor.submit(
                    _dcape_rows,
                    int(group[0]),
                    psfc_hpa[group],
                    tmp2_c[group],
                    dpt2_c[group],
                    levels,
                    temp_levels_arr[:, group, :],
                    dpt_levels_arr[:, group, :],
                )
                for group in row_groups
            ]
            for future in concurrent.futures.as_completed(futures):
                start_row, values = future.result()
                out[start_row : start_row + values.shape[0]] = values

    return coarse_lat, coarse_lon, smooth_nan(out)


def compute_pcge(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    stride: int,
    lat: np.ndarray,
    lon: np.ndarray,
    dcape: np.ndarray,
) -> np.ndarray:
    """Experimental potential convective gust estimate in km/h.

    PCGE is intended as a threshold-oriented "if convection occurs" diagnostic,
    not as a deterministic 10 m gust forecast.
    """
    hour_dir = run_dir / f"{fhour:03d}"
    jidx = thin_indices(lat[yslice, xslice].shape[0], stride)
    iidx = thin_indices(lat[yslice, xslice].shape[1], stride)

    def crop_sample(path: Path) -> np.ndarray:
        data, _, _ = read_grib(path)
        return data[yslice, xslice][np.ix_(jidx, iidx)]

    regular_gust_kmh = 3.6 * crop_sample(hour_dir / field_name("GUST", "TGL", "10", run.stamp, fhour))
    li = crop_sample(hour_dir / field_name("MU-VT-LI", "ISBL", "500", run.stamp, fhour))
    cape = crop_sample(hour_dir / field_name("CAPE", "ETAL", "10000", run.stamp, fhour))
    hpbl_m = crop_sample(hour_dir / field_name("HPBL", "SFC", "0", run.stamp, fhour))
    wind850_ms = crop_sample(hour_dir / field_name("WIND", "ISBL", "0850", run.stamp, fhour))
    wind700_ms = crop_sample(hour_dir / field_name("WIND", "ISBL", "0700", run.stamp, fhour))
    omega700_pa_s = crop_sample(hour_dir / field_name("VVEL", "ISBL", "0700", run.stamp, fhour))
    temp700_k = crop_sample(hour_dir / field_name("TMP", "ISBL", "0700", run.stamp, fhour))
    if fhour == 0:
        precip_rate_mm_h = np.zeros(li.shape, dtype=np.float32)
    else:
        precip_rate_mm_h = 3600.0 * crop_sample(
            hour_dir / field_name("PRATE", "SFC", "0", run.stamp, fhour)
        )

    li = np.where(li > 50.0, np.nan, li)
    hpbl_m = np.where((hpbl_m < 0.0) | (hpbl_m > 6000.0), np.nan, hpbl_m)

    result = gust_diagnostics.pcge_gust(
        regular_gust_kmh,
        dcape,
        li,
        cape,
        hpbl_m,
        wind850_ms,
        wind700_ms,
        precip_rate_mm_h,
        omega700_pa_s,
        temp700_k,
    )
    increment_kmh = np.maximum(0.0, result.gust_kmh - regular_gust_kmh)
    return regular_gust_kmh + smooth_nan(increment_kmh, sigma=sigma_for_km(0.7))


def diagnostic_cache_key(
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    stride: int,
    lat_shape: tuple[int, ...],
) -> str:
    payload = {
        "version": DIAGNOSTIC_CACHE_VERSION,
        "model": model_config().key,
        "stamp": run.stamp,
        "fhour": int(fhour),
        "extent": list(model_config().extent),
        "stride": int(stride),
        "yslice": [yslice.start, yslice.stop, yslice.step],
        "xslice": [xslice.start, xslice.stop, xslice.step],
        "lat_shape": list(lat_shape),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def diagnostic_cache_path(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    stride: int,
    lat: np.ndarray,
) -> Path:
    key = diagnostic_cache_key(run, fhour, yslice, xslice, stride, lat.shape)
    return run_dir / "derived" / f"{model_config().key}_dcape_pcge_f{fhour:03d}_s{stride}_{key}.npz"


def compute_dcape_pcge_cached(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    stride: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = diagnostic_cache_path(run_dir, run, fhour, yslice, xslice, stride, lat)
    if cache_path.exists():
        try:
            with np.load(cache_path) as cached:
                log(f"  using cached DCAPE/PCGE diagnostics for F{fhour:03d}.")
                return (
                    cached["dcape_lat"],
                    cached["dcape_lon"],
                    cached["dcape"],
                    cached["pcge_kmh"],
                )
        except Exception as exc:
            log(f"  ignoring unreadable diagnostic cache for F{fhour:03d}: {exc}")
            cache_path.unlink(missing_ok=True)

    dcape_lat, dcape_lon, dcape = compute_dcape(run_dir, run, fhour, yslice, xslice, stride, lat, lon)
    pcge_kmh = compute_pcge(run_dir, run, fhour, yslice, xslice, stride, lat, lon, dcape)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                dcape_lat=dcape_lat.astype(np.float32, copy=False),
                dcape_lon=dcape_lon.astype(np.float32, copy=False),
                dcape=dcape.astype(np.float32, copy=False),
                pcge_kmh=pcge_kmh.astype(np.float32, copy=False),
            )
        tmp_path.replace(cache_path)
        log(f"  cached DCAPE/PCGE diagnostics for F{fhour:03d}.")
    finally:
        tmp_path.unlink(missing_ok=True)

    return dcape_lat, dcape_lon, dcape, pcge_kmh


def make_dcape_cmap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm, list[int]]:
    levels = [100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500]
    colors = [
        "#fff7bc",
        "#fee391",
        "#fec44f",
        "#fe9929",
        "#ec7014",
        "#cc4c02",
        "#9e0142",
        "#5e3c99",
    ]
    cmap = mcolors.ListedColormap(colors, name="dcape")
    cmap.set_under((1.0, 1.0, 1.0, 0.0))
    cmap.set_over("#3b0f70")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return cmap, norm, levels


PANEL_PROJ = ccrs.LambertConformal(central_longitude=-123.0, central_latitude=53.0)
DATA_CRS = ccrs.PlateCarree()
_PROJECTED_WATERSHEDS: dict[int, tuple[BaseGeometry, ...]] = {}


@functools.lru_cache(maxsize=24)
def projected_natural_earth(
    category: str,
    name: str,
    scale: str,
    extent: tuple[float, float, float, float],
) -> tuple[BaseGeometry, ...]:
    """Project static map geometry once per worker instead of once per frame."""

    feature = cfeature.NaturalEarthFeature(category, name, scale)
    return tuple(PANEL_PROJ.project_geometry(geom, DATA_CRS) for geom in feature.intersecting_geometries(extent))


def add_projected_feature(
    ax: plt.Axes,
    category: str,
    name: str,
    scale: str,
    extent: tuple[float, float, float, float] | None = None,
    **style,
) -> None:
    plot_extent = tuple(extent or model_config().extent)
    ax.add_geometries(
        projected_natural_earth(category, name, scale, plot_extent),
        crs=PANEL_PROJ,
        **style,
    )


def add_base_features(
    ax: plt.Axes,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    plot_extent = extent or model_config().extent
    ax.set_extent(plot_extent, crs=DATA_CRS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#ffffff")
    add_projected_feature(ax, "physical", "land", "50m", plot_extent, facecolor="#f5f4ef", edgecolor="none", zorder=0)
    add_projected_feature(ax, "physical", "ocean", "50m", plot_extent, facecolor="#ffffff", edgecolor="none", zorder=0)
    add_projected_feature(ax, "physical", "coastline", "10m", plot_extent, facecolor="none", edgecolor="black", linewidth=0.75, zorder=20)
    add_projected_feature(ax, "cultural", "admin_0_boundary_lines_land", "50m", plot_extent, facecolor="none", edgecolor="black", linewidth=0.65, zorder=21)
    add_projected_feature(ax, "cultural", "admin_1_states_provinces_lines", "50m", plot_extent, facecolor="none", edgecolor="black", linewidth=0.55, zorder=21)
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        ax.spines["geo"].set_linewidth(1.5)
        ax.spines["geo"].set_edgecolor("black")
    except Exception:
        pass


def read_watershed_source(source_path: Path) -> list[BaseGeometry]:
    suffix = source_path.suffix.lower()
    if suffix == ".shp":
        projection_path = source_path.with_suffix(".prj")
        if not projection_path.exists():
            raise FileNotFoundError(f"Watershed projection file is missing: {projection_path}")
        source_crs = CRS.from_wkt(projection_path.read_text())
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        reader = shapereader.Reader(source_path)
        try:
            return [transform_geometry(transformer.transform, geom) for geom in reader.geometries()]
        finally:
            reader.close()
    if suffix in {".geojson", ".json"}:
        data = json.loads(source_path.read_text())
        return [shape(feature["geometry"]) for feature in data.get("features", []) if feature.get("geometry")]
    raise ValueError(f"Unsupported watershed boundary format: {source_path}")


def load_watersheds(
    cache_path: Path,
    refresh: bool = False,
    extent: tuple[float, float, float, float] | None = None,
) -> list[BaseGeometry]:
    try:
        source_path = Path(cache_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Watershed boundary source is missing: {source_path}")
        source_geoms = read_watershed_source(source_path)
    except Exception as exc:
        log(f"Could not load watershed overlay: {exc}")
        return []

    plot_extent = extent or model_config().extent
    extent_poly = box(plot_extent[0], plot_extent[2], plot_extent[1], plot_extent[3])
    geoms: list[BaseGeometry] = []
    for geom in source_geoms:
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty or not geom.intersects(extent_poly):
            continue
        clipped = geom.intersection(extent_poly)
        if not clipped.is_valid:
            clipped = make_valid(clipped)
        geoms.append(clipped.simplify(0.015, preserve_topology=True))
    if refresh:
        log("Watershed boundaries were re-read from the local source.")
    log(f"Loaded {len(geoms)} BC Hydro watershed outlines from {source_path}.")
    return geoms


def add_watersheds(ax: plt.Axes, watersheds: list[BaseGeometry]) -> None:
    if not watersheds:
        return
    key = id(watersheds)
    projected = _PROJECTED_WATERSHEDS.get(key)
    if projected is None:
        projected = tuple(PANEL_PROJ.project_geometry(geom, DATA_CRS) for geom in watersheds)
        _PROJECTED_WATERSHEDS[key] = projected
    ax.add_geometries(
        projected,
        crs=PANEL_PROJ,
        facecolor="none",
        edgecolor="white",
        linewidth=WATERSHED_HALO_LINEWIDTH,
        alpha=0.82,
        zorder=24,
    )
    ax.add_geometries(
        projected,
        crs=PANEL_PROJ,
        facecolor="none",
        edgecolor=WATERSHED_EDGE_COLOR,
        linewidth=WATERSHED_LINEWIDTH,
        alpha=0.96,
        zorder=25,
    )


def label_contours(contours, fontsize: float = 6.2, fmt: str = "%g", colors: str | None = None) -> None:
    labels = contours.axes.clabel(contours, inline=True, inline_spacing=4, fmt=fmt, fontsize=fontsize, colors=colors)
    for label in labels:
        label.set_path_effects([path_effects.withStroke(linewidth=1.7, foreground="white", alpha=0.9)])


def add_map_features(ax: plt.Axes) -> None:
    add_projected_feature(ax, "physical", "land", "50m", facecolor="#f3f1ec", edgecolor="none", zorder=0)
    add_projected_feature(ax, "physical", "coastline", "10m", facecolor="none", edgecolor="#48525a", linewidth=0.65, zorder=4)
    add_projected_feature(ax, "cultural", "admin_0_boundary_lines_land", "50m", facecolor="none", edgecolor="#343a40", linewidth=0.7, zorder=5)
    add_projected_feature(ax, "cultural", "admin_1_states_provinces_lines", "50m", facecolor="none", edgecolor="#59636b", linewidth=0.55, zorder=5)


def add_hydro_features(ax: plt.Axes) -> None:
    water_color = "#2f8aa7"
    add_projected_feature(ax, "physical", "lakes", "10m", facecolor="none", edgecolor=water_color, linewidth=0.42, zorder=6.9)
    add_projected_feature(ax, "physical", "rivers_lake_centerlines", "10m", facecolor="none", edgecolor=water_color, linewidth=0.34, zorder=6.9)


def add_city_labels(
    ax: plt.Axes,
    fontsize: float = 5.9,
    marker_size: float = 1.9,
    path_width: float = 2.0,
    x_offset: float = 0.13,
    y_offset: float = 0.08,
    zorder: int = 10,
) -> None:
    cities = [
        ("Vancouver", -123.12, 49.28),
        ("Victoria", -123.37, 48.43),
        ("Nanaimo", -123.94, 49.17),
        ("Campbell River", -125.24, 50.02),
        ("Port Hardy", -127.50, 50.72),
        ("Tofino", -125.91, 49.15),
        ("Abbotsford", -122.30, 49.05),
        ("Whistler", -122.96, 50.12),
        ("Hope", -121.44, 49.38),
        ("Kelowna", -119.49, 49.89),
        ("Penticton", -119.59, 49.50),
        ("Vernon", -119.27, 50.27),
        ("Kamloops", -120.33, 50.67),
        ("Merritt", -120.79, 50.11),
        ("Lillooet", -121.94, 50.69),
        ("Revelstoke", -118.20, 50.99),
        ("Golden", -116.96, 51.30),
        ("Cranbrook", -115.77, 49.51),
        ("Nelson", -117.29, 49.49),
        ("Castlegar", -117.66, 49.32),
        ("Williams Lake", -122.14, 52.13),
        ("Quesnel", -122.49, 52.98),
        ("Prince George", -122.75, 53.92),
        ("Smithers", -127.17, 54.78),
        ("Terrace", -128.60, 54.52),
        ("Prince Rupert", -130.32, 54.32),
        ("Dease Lake", -130.02, 58.44),
        ("Fort Nelson", -122.70, 58.81),
        ("Dawson Creek", -120.24, 55.76),
        ("Fort St. John", -120.85, 56.25),
        ("Grande Prairie", -118.79, 55.17),
        ("Jasper", -118.08, 52.88),
        ("Banff", -115.57, 51.18),
        ("Calgary", -114.07, 51.05),
        ("Edmonton", -113.49, 53.55),
        ("Bellingham", -122.48, 48.75),
        ("Seattle", -122.33, 47.61),
        ("Spokane", -117.43, 47.66),
    ]
    transform = ccrs.PlateCarree()
    west, east, south, north = ax.get_extent(transform)
    x_pad = (east - west) * 0.01
    y_pad = (north - south) * 0.01
    for name, x, y in cities:
        if not (west - x_pad <= x <= east + x_pad and south - y_pad <= y <= north + y_pad):
            continue
        ax.plot(x, y, "o", ms=marker_size, color="#232323", transform=transform, zorder=zorder, clip_on=True)
        txt = ax.text(
            x + x_offset,
            y + y_offset,
            name,
            transform=transform,
            fontsize=fontsize,
            color="#202020",
            zorder=zorder,
            clip_on=True,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=path_width, foreground="white", alpha=0.9)])


def plot_forecast(
    out_path: Path,
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    lat: np.ndarray,
    lon: np.ndarray,
    terrain_m: np.ndarray,
    dcape_lat: np.ndarray,
    dcape_lon: np.ndarray,
    dcape: np.ndarray,
    pcge_kmh: np.ndarray,
    watersheds: list[BaseGeometry],
) -> None:
    hour_dir = run_dir / f"{fhour:03d}"
    li, _, _ = read_grib(hour_dir / field_name("MU-VT-LI", "ISBL", "500", run.stamp, fhour))
    hlcy, _, _ = read_grib(hour_dir / field_name("HLCY", "ETAL", "10000", run.stamp, fhour))
    li = li[yslice, xslice]
    hlcy = hlcy[yslice, xslice]
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    li = np.where(li > 50.0, np.nan, li)
    hlcy = np.where(np.abs(hlcy) > 5000.0, np.nan, hlcy)

    cmap, norm, dcape_levels = make_dcape_cmap()

    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=PANEL_PROJ)
    add_base_features(ax)

    dcape_plot = ax.contourf(
        dcape_lon,
        dcape_lat,
        dcape,
        levels=dcape_levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=DATA_CRS,
        transform_first=True,
        antialiased=True,
        zorder=3,
    )
    with plt.rc_context({"hatch.color": "#555555", "hatch.linewidth": 0.6}):
        ax.contourf(
            dcape_lon,
            dcape_lat,
            pcge_kmh,
            levels=[60, 90, 1000],
            colors="none",
            hatches=["////", "xxxx"],
            transform=DATA_CRS,
            zorder=19,
        )

    contour_stride = grid_stride(6.0)
    contour_lat, contour_lon, li_contour_data = contour_sample(
        plot_lat,
        plot_lon,
        li,
        stride=contour_stride,
        sigma=sigma_for_km(2.4),
    )
    _, _, hlcy_contour_data = contour_sample(
        plot_lat,
        plot_lon,
        hlcy,
        stride=contour_stride,
        sigma=sigma_for_km(2.0),
    )

    li_levels = [-6, -4, -2, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        li_contours = ax.contour(
            contour_lon,
            contour_lat,
            li_contour_data,
            levels=li_levels,
            colors="#171717",
            linewidths=[1.8, 1.35, 0.85, 1.05],
            linestyles=["solid", "solid", "solid", "dashed"],
            transform=DATA_CRS,
            zorder=22,
        )
        label_contours(li_contours, fontsize=5.8, fmt="%d")

        hlcy_levels = [150, 250, 400]
        hlcy_contours = ax.contour(
            contour_lon,
            contour_lat,
            hlcy_contour_data,
            levels=hlcy_levels,
            colors="#006d77",
            linewidths=[1.05, 1.25, 1.45],
            transform=DATA_CRS,
            zorder=23,
        )
        label_contours(hlcy_contours, fontsize=5.6, fmt="%d", colors="#005b63")

    add_watersheds(ax, watersheds)
    plot_style.add_internal_colorbar(fig, ax, dcape_plot, ticks=dcape_levels, label="J kg$^{-1}$", fmt="%g")
    plot_style.add_single_panel_text(
        ax,
        plot_style.valid_header(run, fhour),
        "DCAPE(shaded,J kg$^{-1}$), LI(cntrd,K), SRH(cntrd,m$^2$s$^{-2}$), PCGE(gray hatch 60/90km/h)",
        run,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def make_plots(
    run: RunInfo,
    data_dir: Path,
    output_dir: Path,
    stride: int,
    hours: Iterable[int] = FORECAST_HOURS,
    watershed_cache: Path = WATERSHED_CACHE,
    refresh_watersheds: bool = False,
    no_watersheds: bool = False,
) -> list[Path]:
    hours = tuple(int(hour) for hour in hours)
    if not hours:
        return []
    watersheds = [] if no_watersheds else load_watersheds(watershed_cache, refresh=refresh_watersheds)
    run_dir = data_dir / run.stamp
    first_hour_dir = run_dir / f"{FORECAST_HOURS[0]:03d}"
    sample_path = first_hour_dir / field_name("MU-VT-LI", "ISBL", "500", run.stamp, FORECAST_HOURS[0])
    _, lat, lon = read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read model coordinates.")
    yslice, xslice = subset_slices(lat, lon, EXTENT)

    terrain_hour_dir = run_dir / f"{TERRAIN_FHOUR:03d}"
    terrain_path = terrain_hour_dir / field_name("HGT", "SFC", "0", run.stamp, TERRAIN_FHOUR)
    terrain, _, _ = read_grib(terrain_path)
    terrain = terrain[yslice, xslice]

    out_paths: list[Path] = []
    plot_dir = output_dir / run.stamp
    for fhour in hours:
        log(f"Processing F{fhour:03d}.")
        dcape_lat, dcape_lon, dcape, pcge_kmh = compute_dcape_pcge_cached(
            run_dir, run, fhour, yslice, xslice, stride, lat, lon
        )
        out_path = plot_dir / f"{model_output_prefix('convective')}_{run.stamp}_f{fhour:03d}.png"
        plot_forecast(
            out_path,
            run_dir,
            run,
            fhour,
            yslice,
            xslice,
            lat,
            lon,
            terrain,
            dcape_lat,
            dcape_lon,
            dcape,
            pcge_kmh,
            watersheds,
        )
        log(f"  wrote {out_path}")
        out_paths.append(out_path)

    return out_paths


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="west", help="HRDPS model domain to use.")
    parser.add_argument("--data-dir", type=Path, default=None, help="GRIB2 cache directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG output directory.")
    parser.add_argument("--cycle", choices=["latest", "00", "06", "12", "18"], default="latest", help="Cycle to use.")
    parser.add_argument("--stride", type=int, default=None, help="Grid stride for DCAPE profile calculation.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent download workers.")
    parser.add_argument("--hours", default=None, help="Comma-separated forecast hours to plot, e.g. 0,3,6.")
    parser.add_argument(
        "--watershed-cache",
        type=Path,
        default=WATERSHED_CACHE,
        help="Local BC Hydro watershed boundary shapefile.",
    )
    parser.add_argument("--refresh-watersheds", action="store_true", help="Re-read the local watershed overlay.")
    parser.add_argument("--no-watersheds", action="store_true", help="Skip watershed overlays.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = set_model(args.model)
    data_dir = args.data_dir or Path(config.default_data_dir)
    output_dir = args.output_dir or Path(config.default_output_dir)
    stride = args.stride or grid_stride(18.0)
    if args.cycle != "latest" and args.cycle not in AVAILABLE_CYCLES:
        raise RuntimeError(f"{config.label} does not provide a {args.cycle}Z cycle.")
    if args.cycle == "latest":
        run = latest_complete_run()
    else:
        stamp = run_stamp_from_listing(args.cycle)
        if not stamp:
            raise RuntimeError(f"No files found for cycle {args.cycle}Z.")
        run = RunInfo(cycle=args.cycle, stamp=stamp, init_time=parse_stamp(stamp))
        if not run_is_complete(run):
            raise RuntimeError(f"Cycle {run.stamp} is not complete for the required fields.")

    log(f"Using {config.label} run {run.stamp}.")
    download_run(run, data_dir, args.workers)
    hours = FORECAST_HOURS if args.hours is None else tuple(int(item) for item in args.hours.split(",") if item.strip())
    make_plots(run, data_dir, output_dir, stride, hours, args.watershed_cache, args.refresh_watersheds, args.no_watersheds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
