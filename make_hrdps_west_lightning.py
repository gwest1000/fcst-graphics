#!/usr/bin/env python3
"""Make HRDPS fire-weather graphics from model-derived ingredients."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

import gust_diagnostics
import fire_danger_peak
import plot_style
import make_hrdps_west_convective as hrdps
from make_hrdps_west_convective import (
    MODEL_CONFIGS,
    RunInfo,
    download_one,
    fetch_text,
    field_name,
    grid_stride,
    model_config,
    model_output_prefix,
    parse_links,
    parse_stamp,
    read_grib,
    run_stamp_from_listing,
    set_model,
    sigma_for_km,
    subset_slices,
)
from make_hrdps_west_fourpanel import compute_precip_3h, crop, hour_file, smooth_nan

FORECAST_HOURS = tuple(range(0, 49, 3))
PLOT_WINDOW_HOURS = 3
PROFILE_LEVELS_HPA = (
    1000,
    985,
    970,
    950,
    925,
    900,
    875,
    850,
    800,
    750,
    700,
    650,
    600,
    550,
    500,
    450,
    400,
    350,
    300,
    250,
)
SUBCLOUD_RH_LEVELS_HPA = (850, 800, 750, 700)
DOWNSLOPE_PROFILE_LEVELS_HPA = (1000, 950, 900, 850, 800, 750, 700)
DATA_CRS = ccrs.PlateCarree()
PLOT_CRS = ccrs.LambertConformal(central_longitude=-123.0, central_latitude=53.0)
LPI_CACHE_VERSION = 3
LPI_FORMULA_VERSION = "bc_lpi_v3_3hmax"
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
GRAVITY_MS2 = 9.80665
FWI_WCS_URL = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
FWI_COVERAGE = "public:fwi"
FWI_CACHE_DIR = Path("data/cwfis_fwi")
FWI_NATIVE_BBOX_3978 = (-2378164.081065, -707617.771124, 3039835.918935, 3854382.228876)
FWI_CACHE_RESOLUTION_M = 5000.0
FWI_MAX_DIMENSION = 900
FWI_TO_3978 = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)
FWI_FROM_3978 = Transformer.from_crs("EPSG:3978", "EPSG:4326", always_xy=True)
TRANSMISSION_LINES_URL = (
    "https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/"
    "bcgw_pub_whse_basemapping/MapServer/77/query"
)
TRANSMISSION_LINES_CACHE = Path("data/bc_transmission_lines.geojson")
TRANSMISSION_LINES_PAGE_SIZE = 1000
TRANSMISSION_LINES_COLOR = "#62676d"
TRANSMISSION_LINES_WIDTH = 1.45
TRANSMISSION_LINES_ALPHA = 0.86
TRANSMISSION_LINES_HALO_COLOR = "#ffffff"
TRANSMISSION_LINES_HALO_WIDTH = 2.55
TRANSMISSION_LINES_HALO_ALPHA = 0.92
_PROJECTED_TRANSMISSION_LINES: dict[int, tuple[BaseGeometry, ...]] = {}


@dataclass(frozen=True)
class RegionConfig:
    key: str
    label: str
    extent: tuple[float, float, float, float] | None = None


FIRE_WEATHER_REGIONS = {
    "bc": RegionConfig("bc", "BC", (-138.2, -109.5, 46.0, 58.45)),
    "sw": RegionConfig("sw", "Southwest BC", (-128.5, -120.0, 48.0, 55.0)),
    "se": RegionConfig("se", "Southeast BC", (-121.25, -114.2, 48.0, 53.7)),
    "ne": RegionConfig("ne", "Northeast BC", (-130.0, -118.5, 51.0, 59.2)),
}

BASE_GUST_VECTOR_DENSITY = 1.625
BASE_DRY_LIGHTNING_MARKER_AREA = 9.0
REGIONAL_DRY_LIGHTNING_MARKER_AREA = 14.0
DRY_LIGHTNING_MARKER = (5, 2, 0)
DRY_LIGHTNING_COLOR = "#161616"
LOW_RH_HATCH_COLOR = "#c3935f"
VERY_LOW_RH_HATCH_COLOR = "#77451f"
GOOD_RECOVERY_HATCH_COLOR = "#87cbdc"
EXCELLENT_RECOVERY_HATCH_COLOR = "#4fa9c6"
RH_HATCH_LINEWIDTH = 0.55
RH_HATCH_ALPHA = 0.50
RECOVERY_RH_HATCH_ALPHA = 0.55
RH_BOUNDARY_GRID_KM = 2.5
RH_SMOOTHING_KM = 6.0
PEAK_DANGER_CONTOUR_LEVELS = (1.5, 2.5, 3.5, 4.5)
PEAK_DANGER_CONTOUR_COLORS = ("#666666", "#252525", "#d36b00", "#a20d18")
PEAK_DANGER_CONTOUR_LINEWIDTHS = (1.65, 1.9, 2.3, 2.75)
PEAK_DANGER_CONTOUR_LABELS = {1.5: "LOW", 2.5: "MODERATE", 3.5: "HIGH", 4.5: "EXTREME"}
GUST_VECTOR_EDGE_COLOR = "#30363a"
GUST_VECTOR_EDGE_ALPHA = 0.55
GUST_VECTOR_EDGE_WIDTH = 0.22
def fire_weather_footer(fhour: int) -> str:
    period = "3-h max" if fhour > 0 else "Init-time"
    return (
        f"{period} LPI(shaded), {period} Gust(vectors), Peak Daily Fire Danger(cntr), "
        "valid-time 10m RH crosshatch(brown 20-30%, dark <20%; blue 60-80%, dark blue >80%), "
        f"{period} dry lightning(black *), 3-h precip, BC transmission(grey)"
    )


def region_config(region_key: str) -> RegionConfig:
    try:
        return FIRE_WEATHER_REGIONS[region_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported fire-weather region: {region_key}") from exc


def region_extent(region: RegionConfig) -> tuple[float, float, float, float]:
    return region.extent or model_config().extent


def region_output_prefix(region: RegionConfig) -> str:
    prefix = model_output_prefix("lightning")
    return prefix if region.key == "bc" else f"{prefix}_{region.key}"


def download_transmission_lines() -> dict[str, object]:
    """Fetch the public GeoBC transmission-line layer as GeoJSON."""
    features: list[dict[str, object]] = []
    offset = 0
    while True:
        response = requests.get(
            TRANSMISSION_LINES_URL,
            params={
                "where": "1=1",
                "outFields": "TRANSMISSION_LINE_ID,CIRCUIT_NAME",
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": TRANSMISSION_LINES_PAGE_SIZE,
                "f": "geojson",
            },
            timeout=(20, 180),
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("features", [])
        if not isinstance(page, list):
            raise RuntimeError("GeoBC transmission-line response did not contain GeoJSON features.")
        features.extend(page)
        if len(page) < TRANSMISSION_LINES_PAGE_SIZE:
            break
        offset += len(page)
    if not features:
        raise RuntimeError("GeoBC transmission-line service returned no features.")
    return {"type": "FeatureCollection", "features": features}


def load_transmission_lines(
    cache_path: Path = TRANSMISSION_LINES_CACHE,
    extent: tuple[float, float, float, float] | None = None,
) -> list[BaseGeometry]:
    if cache_path.exists():
        collection = json.loads(cache_path.read_text())
    else:
        collection = download_transmission_lines()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(collection, separators=(",", ":")))
        temporary.replace(cache_path)
        log(f"Cached {len(collection['features'])} public BC transmission-line features at {cache_path}.")

    west, east, south, north = extent or model_config().extent
    extent_polygon = box(west, south, east, north)
    lines: list[BaseGeometry] = []
    for feature in collection.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        line = shape(geometry)
        if line.is_empty or not line.intersects(extent_polygon):
            continue
        clipped = line.intersection(extent_polygon)
        if not clipped.is_empty:
            lines.append(clipped.simplify(0.004, preserve_topology=True))
    if not lines:
        raise RuntimeError("No public BC transmission lines intersect the Fire Weather plot extent.")
    return lines


def add_transmission_lines(ax: plt.Axes, lines: list[BaseGeometry]) -> None:
    key = id(lines)
    projected = _PROJECTED_TRANSMISSION_LINES.get(key)
    if projected is None:
        projected = tuple(PLOT_CRS.project_geometry(line, DATA_CRS) for line in lines)
        _PROJECTED_TRANSMISSION_LINES[key] = projected
    ax.add_geometries(
        projected,
        crs=PLOT_CRS,
        facecolor="none",
        edgecolor=TRANSMISSION_LINES_HALO_COLOR,
        linewidth=TRANSMISSION_LINES_HALO_WIDTH,
        alpha=TRANSMISSION_LINES_HALO_ALPHA,
        zorder=7.9,
    )
    ax.add_geometries(
        projected,
        crs=PLOT_CRS,
        facecolor="none",
        edgecolor=TRANSMISSION_LINES_COLOR,
        linewidth=TRANSMISSION_LINES_WIDTH,
        alpha=TRANSMISSION_LINES_ALPHA,
        zorder=8,
    )


def gust_vector_density(region_key: str) -> tuple[float, float]:
    """Return row/column gust-vector density tuned by model and fire-weather region."""
    if model_config().key != "west":
        return BASE_GUST_VECTOR_DENSITY, BASE_GUST_VECTOR_DENSITY
    if region_key == "bc":
        density = BASE_GUST_VECTOR_DENSITY * 0.75
    else:
        density = BASE_GUST_VECTOR_DENSITY * 1.25
    return density, density


def dry_lightning_marker_area(region_key: str) -> float:
    """Return dry-lightning marker area, with modestly larger regional symbols."""
    if region_key != "bc":
        return REGIONAL_DRY_LIGHTNING_MARKER_AREA
    return BASE_DRY_LIGHTNING_MARKER_AREA


@dataclass(frozen=True)
class LightningFields:
    potential: np.ndarray
    dry_potential: np.ndarray
    li: np.ndarray
    cape: np.ndarray
    precip_3h: np.ndarray
    charge_rh: np.ndarray
    trigger: np.ndarray
    subcloud_rh: np.ndarray
    surface_rh: np.ndarray
    gust_kmh: np.ndarray
    u10_ms: np.ndarray
    v10_ms: np.ndarray


def subset_lightning_fields(fields: LightningFields, yslice: slice, xslice: slice) -> LightningFields:
    return LightningFields(*(getattr(fields, name)[yslice, xslice] for name in fields.__dataclass_fields__))


@dataclass(frozen=True)
class FwiGrid:
    date: dt.date
    lat: np.ndarray
    lon: np.ndarray
    fwi: np.ndarray


def log(message: str) -> None:
    print(message, flush=True)


def ramp(data: np.ndarray, low: float, high: float) -> np.ndarray:
    if high == low:
        raise ValueError("ramp high and low must differ.")
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def pressure_layer_edges_hpa(levels_hpa: tuple[int, ...]) -> np.ndarray:
    """Return midpoint pressure-layer edges for descending pressure levels."""
    levels = np.asarray(levels_hpa, dtype=np.float32)
    if levels.ndim != 1 or levels.size < 2 or not np.all(np.diff(levels) < 0.0):
        raise ValueError("Pressure levels must be a descending one-dimensional sequence.")
    edges = np.empty(levels.size + 1, dtype=np.float32)
    edges[1:-1] = 0.5 * (levels[:-1] + levels[1:])
    edges[0] = levels[0] + 0.5 * (levels[0] - levels[1])
    edges[-1] = levels[-1] - 0.5 * (levels[-2] - levels[-1])
    return edges


def pressure_layer_thickness_hpa(
    psfc_hpa: np.ndarray,
    upper_edge_hpa: float,
    lower_edge_hpa: float,
) -> np.ndarray:
    """Return the above-ground pressure thickness represented by one level."""
    nominal_depth = upper_edge_hpa - lower_edge_hpa
    return np.clip(np.minimum(psfc_hpa, upper_edge_hpa) - lower_edge_hpa, 0.0, nominal_depth)


def geometric_vertical_velocity_ms(
    omega_pa_s: np.ndarray,
    temp_k: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    """Convert pressure vertical velocity to upward-positive geometric velocity."""
    valid = np.isfinite(omega_pa_s) & np.isfinite(temp_k) & (temp_k > 150.0)
    density_kg_m3 = (pressure_hpa * 100.0) / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * np.where(valid, temp_k, np.nan)
    )
    return np.where(valid, -omega_pa_s / (density_kg_m3 * GRAVITY_MS2), np.nan)


def saturation_vapor_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def relative_humidity_from_t_td(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    rh = 100.0 * saturation_vapor_pressure_hpa(dewpoint_c) / saturation_vapor_pressure_hpa(temp_c)
    return np.clip(rh, 0.0, 100.0)


def gust_field_name(stamp: str, fhour: int) -> str:
    return field_name("GUST", "TGL", "10", stamp, fhour)


def gust_max_field_name(stamp: str, fhour: int) -> str:
    variable = "GUST-Max" if model_config().filename_style == "modern" else "GUST_MAX"
    return field_name(variable, "TGL", "10", stamp, fhour)


def required_names(stamp: str, fhour: int) -> list[str]:
    names = [
        field_name("MU-VT-LI", "ISBL", "500", stamp, fhour),
        field_name("CAPE", "ETAL", "10000", stamp, fhour),
        field_name("PRES", "SFC", "0", stamp, fhour),
        field_name("VVEL", "ISBL", "0500", stamp, fhour),
        field_name("VVEL", "ISBL", "0700", stamp, fhour),
        field_name("TMP", "TGL", "2", stamp, fhour),
        field_name("DPT", "TGL", "2", stamp, fhour),
        gust_field_name(stamp, fhour),
        gust_max_field_name(stamp, fhour),
        field_name("UGRD", "TGL", "10", stamp, fhour),
        field_name("VGRD", "TGL", "10", stamp, fhour),
    ]
    if fhour > 0:
        names.append(field_name("PRATE", "SFC", "0", stamp, fhour))
    if fhour > 0:
        names.append(field_name("APCP", "SFC", "0", stamp, fhour))
    if fhour == 0:
        names.append(field_name("PRMSL", "MSL", "0", stamp, fhour))
    for level in PROFILE_LEVELS_HPA:
        names.append(field_name("TMP", "ISBL", f"{level:04d}", stamp, fhour))
        names.append(field_name("RH", "ISBL", f"{level:04d}", stamp, fhour))
    for level in DOWNSLOPE_PROFILE_LEVELS_HPA:
        names.append(field_name("HGT", "ISBL", f"{level:04d}", stamp, fhour))
        names.append(field_name("UGRD", "ISBL", f"{level:04d}", stamp, fhour))
        names.append(field_name("VGRD", "ISBL", f"{level:04d}", stamp, fhour))
    if fhour == hrdps.TERRAIN_FHOUR:
        names.append(field_name("HGT", "SFC", "0", stamp, fhour))
    names.extend(
        hrdps.required_names(
            stamp,
            fhour,
            include_static=(fhour == hrdps.TERRAIN_FHOUR),
        )
    )
    return names


def diagnostic_window_hours(fhour: int) -> tuple[int, ...]:
    """Return hourly valid times represented by a three-hour plot ending at fhour."""
    fhour = int(fhour)
    if fhour <= 0:
        return (0,)
    return tuple(range(max(1, fhour - PLOT_WINDOW_HOURS + 1), fhour + 1))


def diagnostic_snapshot_hours(hours: Iterable[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                snapshot_hour
                for fhour in hours
                for snapshot_hour in diagnostic_window_hours(int(fhour))
            }
        )
    )


def precipitation_history_hours(hours: Iterable[int]) -> tuple[int, ...]:
    snapshots = diagnostic_snapshot_hours(hours)
    return tuple(sorted({hour - PLOT_WINDOW_HOURS for hour in snapshots if hour > PLOT_WINDOW_HOURS}))


def required_names_by_hour(stamp: str, hours: Iterable[int]) -> dict[int, set[str]]:
    hours = tuple(int(hour) for hour in hours)
    requirements: dict[int, set[str]] = {}
    for fhour in diagnostic_snapshot_hours(hours):
        requirements.setdefault(fhour, set()).update(required_names(stamp, fhour))
    for fhour in precipitation_history_hours(hours):
        requirements.setdefault(fhour, set()).add(field_name("APCP", "SFC", "0", stamp, fhour))
    requirements.setdefault(hrdps.TERRAIN_FHOUR, set()).add(
        field_name("HGT", "SFC", "0", stamp, hrdps.TERRAIN_FHOUR)
    )
    return requirements


def prerequisite_hours(hours: Iterable[int]) -> tuple[int, ...]:
    hours = tuple(int(hour) for hour in hours)
    needed = set(diagnostic_snapshot_hours(hours))
    needed.update(precipitation_history_hours(hours))
    needed.add(hrdps.TERRAIN_FHOUR)
    return tuple(sorted(needed))


def run_is_complete(run: RunInfo, hours: Iterable[int] = FORECAST_HOURS) -> bool:
    for fhour, names in required_names_by_hour(run.stamp, hours).items():
        html = fetch_text(f"{model_config().base_url}/{run.cycle}/{fhour:03d}/")
        links = set(parse_links(html))
        missing = sorted(names - links)
        if missing:
            log(f"Skipping {run.stamp}: missing {len(missing)} lightning files at F{fhour:03d}.")
            return False
    return True


def latest_complete_run(hours: Iterable[int] = FORECAST_HOURS) -> RunInfo:
    candidates: list[RunInfo] = []
    for cycle in model_config().cycles:
        try:
            stamp = run_stamp_from_listing(cycle)
        except Exception as exc:
            log(f"Could not inspect cycle {cycle}Z: {exc}")
            continue
        if stamp:
            candidates.append(RunInfo(cycle=cycle, stamp=stamp, init_time=parse_stamp(stamp)))

    for run in sorted(candidates, key=lambda item: item.init_time, reverse=True):
        log(f"Checking completeness for {model_config().label} {run.stamp}.")
        if run_is_complete(run, hours):
            return run

    raise RuntimeError(f"No complete {model_config().label} run was found for the lightning fields.")


def download_run(run: RunInfo, data_dir: Path, workers: int, hours: Iterable[int] = FORECAST_HOURS) -> None:
    jobs: dict[Path, str] = {}
    run_dir = data_dir / run.stamp
    for fhour, names in required_names_by_hour(run.stamp, hours).items():
        for name in names:
            dest = run_dir / f"{fhour:03d}" / name
            jobs[dest] = f"{model_config().base_url}/{run.cycle}/{fhour:03d}/{name}"

    log(f"Downloading or reusing {len(jobs)} lightning GRIB2 files into {run_dir}.")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, url, dest) for dest, url in jobs.items()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                log(f"  files ready: {completed}/{len(futures)}")


def compute_charging_layer(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    psfc_hpa: np.ndarray,
    yslice: slice,
    xslice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted_rh = np.zeros(psfc_hpa.shape, dtype=np.float32)
    weights = np.zeros(psfc_hpa.shape, dtype=np.float32)
    mid_rh_sum = np.zeros(psfc_hpa.shape, dtype=np.float32)
    mid_rh_weights = np.zeros(psfc_hpa.shape, dtype=np.float32)
    layer_edges_hpa = pressure_layer_edges_hpa(PROFILE_LEVELS_HPA)

    for index, level in enumerate(PROFILE_LEVELS_HPA):
        tmp_c = crop(hour_file(run_dir, run, fhour, "TMP", "ISBL", f"{level:04d}"), yslice, xslice) - 273.15
        rh = crop(hour_file(run_dir, run, fhour, "RH", "ISBL", f"{level:04d}"), yslice, xslice)
        layer_depth_hpa = pressure_layer_thickness_hpa(
            psfc_hpa,
            float(layer_edges_hpa[index]),
            float(layer_edges_hpa[index + 1]),
        )
        valid = np.isfinite(tmp_c) & np.isfinite(rh) & (layer_depth_hpa > 0.0)

        thermal_weight = np.exp(-0.5 * ((tmp_c + 15.0) / 5.5) ** 2)
        charge_weight = np.where(valid & (tmp_c <= 0.0) & (tmp_c >= -20.0), thermal_weight * layer_depth_hpa, 0.0)
        finite_rh = np.where(np.isfinite(rh), rh, 0.0)
        weighted_rh += finite_rh * charge_weight
        weights += charge_weight

        mid_weight = np.where(valid & (tmp_c <= -5.0) & (tmp_c >= -30.0), layer_depth_hpa, 0.0)
        mid_rh_sum += finite_rh * mid_weight
        mid_rh_weights += mid_weight

    charge_rh = weighted_rh / np.where(weights == 0.0, np.nan, weights)
    mid_rh = mid_rh_sum / np.where(mid_rh_weights == 0.0, np.nan, mid_rh_weights)
    charge_depth = ramp(weights, 35.0, 150.0)
    charge_rh_factor = ramp(charge_rh, 45.0, 80.0)
    charge_factor = np.sqrt(np.clip(charge_rh_factor * charge_depth, 0.0, 1.0))
    return np.clip(charge_factor, 0.0, 1.0), np.clip(charge_rh, 0.0, 100.0), np.clip(mid_rh, 0.0, 100.0)


def compute_subcloud_rh(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    psfc_hpa: np.ndarray,
    yslice: slice,
    xslice: slice,
    tmp2_c: np.ndarray | None = None,
    dpt2_c: np.ndarray | None = None,
) -> np.ndarray:
    if tmp2_c is None:
        tmp2_c = crop(hour_file(run_dir, run, fhour, "TMP", "TGL", "2"), yslice, xslice) - 273.15
    if dpt2_c is None:
        dpt2_c = crop(hour_file(run_dir, run, fhour, "DPT", "TGL", "2"), yslice, xslice) - 273.15
    fields = [relative_humidity_from_t_td(tmp2_c, dpt2_c)]

    for level in SUBCLOUD_RH_LEVELS_HPA:
        rh = crop(hour_file(run_dir, run, fhour, "RH", "ISBL", f"{level:04d}"), yslice, xslice)
        fields.append(np.where(psfc_hpa >= level, rh, np.nan))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(fields), axis=0)


def compute_surface_rh(run_dir: Path, run: RunInfo, fhour: int, yslice: slice, xslice: slice) -> np.ndarray:
    tmp2_c = crop(hour_file(run_dir, run, fhour, "TMP", "TGL", "2"), yslice, xslice) - 273.15
    dpt2_c = crop(hour_file(run_dir, run, fhour, "DPT", "TGL", "2"), yslice, xslice) - 273.15
    return relative_humidity_from_t_td(tmp2_c, dpt2_c)


def read_gust_kmh(run_dir: Path, run: RunInfo, fhour: int, yslice: slice, xslice: slice) -> np.ndarray:
    gust_ms = crop(run_dir / f"{fhour:03d}" / gust_field_name(run.stamp, fhour), yslice, xslice)
    gust_ms = np.where((gust_ms >= 0.0) & (gust_ms < 100.0), gust_ms, np.nan)
    return 3.6 * gust_ms


def upsample_coarse_field(field: np.ndarray, output_shape: tuple[int, int], stride: int) -> np.ndarray:
    """Interpolate a thinned diagnostic back to the native cropped grid."""
    jidx = hrdps.thin_indices(output_shape[0], stride).astype(np.float64)
    iidx = hrdps.thin_indices(output_shape[1], stride).astype(np.float64)
    if field.shape != (jidx.size, iidx.size):
        raise ValueError(f"Coarse field shape {field.shape} does not match stride coordinates.")

    valid = np.isfinite(field).astype(np.float32)
    values = np.where(np.isfinite(field), field, 0.0).astype(np.float32)
    value_interp = RegularGridInterpolator(
        (jidx, iidx), values, bounds_error=False, fill_value=None
    )
    weight_interp = RegularGridInterpolator(
        (jidx, iidx), valid, bounds_error=False, fill_value=None
    )
    jj, ii = np.meshgrid(
        np.arange(output_shape[0], dtype=np.float32),
        np.arange(output_shape[1], dtype=np.float32),
        indexing="ij",
    )
    points = np.column_stack((jj.ravel(), ii.ravel()))
    weights = np.clip(weight_interp(points).reshape(output_shape), 0.0, 1.0)
    interpolated = value_interp(points).reshape(output_shape)
    return np.where(weights > 0.20, interpolated / np.maximum(weights, 1.0e-6), np.nan).astype(np.float32)


def compute_all_cause_gust(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    lat: np.ndarray,
    lon: np.ndarray,
    terrain_m: np.ndarray,
    surface_temp_k: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    dcape_stride: int,
) -> np.ndarray:
    """Maximize regular, ECCC-style downslope, and triggered PCGE gusts."""
    hour_dir = run_dir / f"{fhour:03d}"
    regular_gust_kmh = read_gust_kmh(run_dir, run, fhour, yslice, xslice)
    if fhour == 0:
        log("  gust branches disabled at F000; using regular HRDPS gust.")
        return regular_gust_kmh.astype(np.float32)
    gust_max_kmh = 3.6 * crop(
        hour_dir / gust_max_field_name(run.stamp, fhour), yslice, xslice
    )

    profile_heights: list[np.ndarray] = []
    profile_temps: list[np.ndarray] = []
    profile_u: list[np.ndarray] = []
    profile_v: list[np.ndarray] = []
    for level in DOWNSLOPE_PROFILE_LEVELS_HPA:
        profile_heights.append(
            crop(hour_dir / field_name("HGT", "ISBL", f"{level:04d}", run.stamp, fhour), yslice, xslice)
            - terrain_m
        )
        profile_temps.append(
            crop(hour_dir / field_name("TMP", "ISBL", f"{level:04d}", run.stamp, fhour), yslice, xslice)
        )
        profile_u.append(
            crop(hour_dir / field_name("UGRD", "ISBL", f"{level:04d}", run.stamp, fhour), yslice, xslice)
        )
        profile_v.append(
            crop(hour_dir / field_name("VGRD", "ISBL", f"{level:04d}", run.stamp, fhour), yslice, xslice)
        )

    height_profile = np.stack(profile_heights).astype(np.float32)
    temp_profile = np.stack(profile_temps).astype(np.float32)
    wind_height_profile = np.concatenate(
        [np.full((1, *terrain_m.shape), 10.0, dtype=np.float32), height_profile], axis=0
    )
    u_profile = np.concatenate([u10_ms[None, ...], np.stack(profile_u)], axis=0).astype(np.float32)
    v_profile = np.concatenate([v10_ms[None, ...], np.stack(profile_v)], axis=0).astype(np.float32)

    grid_spacing_m = model_config().resolution_km * 1000.0
    radius_points = max(
        1,
        int(np.ceil(gust_diagnostics.DOWNSLOPE_RADIUS_KM * 1000.0 / grid_spacing_m)),
    )
    ridge_terrain_m = maximum_filter(
        terrain_m,
        size=2 * radius_points + 1,
        mode="nearest",
    )
    ridge_height_agl_m = np.maximum(10.0, ridge_terrain_m - terrain_m)
    ridge_u_ms = gust_diagnostics.interpolate_profile_to_height(
        wind_height_profile, u_profile, ridge_height_agl_m
    )
    ridge_v_ms = gust_diagnostics.interpolate_profile_to_height(
        wind_height_profile, v_profile, ridge_height_agl_m
    )
    downslope = gust_diagnostics.downslope_adjusted_gust(
        regular_gust_kmh,
        gust_max_kmh,
        terrain_m,
        ridge_u_ms,
        ridge_v_ms,
        surface_temp_k,
        temp_profile,
        height_profile,
        grid_spacing_m,
    )

    _, _, _, pcge_coarse_kmh = hrdps.compute_dcape_pcge_cached(
        run_dir,
        run,
        fhour,
        yslice,
        xslice,
        dcape_stride,
        lat,
        lon,
    )
    coarse_jidx = hrdps.thin_indices(terrain_m.shape[0], dcape_stride)
    coarse_iidx = hrdps.thin_indices(terrain_m.shape[1], dcape_stride)
    regular_coarse_kmh = regular_gust_kmh[np.ix_(coarse_jidx, coarse_iidx)]
    pcge_increment_coarse_kmh = np.maximum(0.0, pcge_coarse_kmh - regular_coarse_kmh)
    pcge_increment_kmh = upsample_coarse_field(
        pcge_increment_coarse_kmh,
        terrain_m.shape,
        dcape_stride,
    )
    pcge_kmh = regular_gust_kmh + np.maximum(0.0, pcge_increment_kmh)
    synoptic_gust_kmh = np.fmax(regular_gust_kmh, downslope.gust_kmh)
    combined_gust_kmh = np.fmax(synoptic_gust_kmh, pcge_kmh).astype(np.float32)
    valid_count = max(1, int(np.isfinite(regular_gust_kmh).sum()))
    material_increment_kmh = 5.0
    downslope_count = int(
        (downslope.gust_kmh > regular_gust_kmh + material_increment_kmh).sum()
    )
    pcge_count = int((pcge_kmh > synoptic_gust_kmh + material_increment_kmh).sum())
    log(
        f"  gust branches (+{material_increment_kmh:g} km/h): "
        f"downslope raised {100.0 * downslope_count / valid_count:.2f}% of grid; "
        f"PCGE raised {100.0 * pcge_count / valid_count:.2f}%."
    )
    return combined_gust_kmh


def compute_instantaneous_lightning_fields(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    lat: np.ndarray,
    lon: np.ndarray,
    terrain_m: np.ndarray,
    dcape_stride: int,
) -> LightningFields:
    psfc_pa = crop(hour_file(run_dir, run, fhour, "PRES", "SFC", "0"), yslice, xslice)
    psfc_hpa = psfc_pa / 100.0
    li = crop(hour_file(run_dir, run, fhour, "MU-VT-LI", "ISBL", "500"), yslice, xslice)
    cape = crop(hour_file(run_dir, run, fhour, "CAPE", "ETAL", "10000"), yslice, xslice)
    if fhour == 0:
        prate = np.zeros(psfc_pa.shape, dtype=np.float32)
    else:
        prate = crop(hour_file(run_dir, run, fhour, "PRATE", "SFC", "0"), yslice, xslice)
    omega500 = crop(hour_file(run_dir, run, fhour, "VVEL", "ISBL", "0500"), yslice, xslice)
    omega700 = crop(hour_file(run_dir, run, fhour, "VVEL", "ISBL", "0700"), yslice, xslice)
    tmp500_k = crop(hour_file(run_dir, run, fhour, "TMP", "ISBL", "0500"), yslice, xslice)
    tmp700_k = crop(hour_file(run_dir, run, fhour, "TMP", "ISBL", "0700"), yslice, xslice)
    tmp2_c = crop(hour_file(run_dir, run, fhour, "TMP", "TGL", "2"), yslice, xslice) - 273.15
    dpt2_c = crop(hour_file(run_dir, run, fhour, "DPT", "TGL", "2"), yslice, xslice) - 273.15

    li = np.where(np.abs(li) > 50.0, np.nan, li)
    cape = np.where((cape >= 0.0) & (cape < 20000.0), cape, np.nan)
    prate_mm_h = np.maximum(0.0, prate * 3600.0)
    precip_3h = compute_precip_3h(run_dir, run, fhour, yslice, xslice)

    charge_factor, charge_rh, mid_rh = compute_charging_layer(run_dir, run, fhour, psfc_hpa, yslice, xslice)
    subcloud_rh = compute_subcloud_rh(run_dir, run, fhour, psfc_hpa, yslice, xslice, tmp2_c, dpt2_c)
    surface_rh = relative_humidity_from_t_td(tmp2_c, dpt2_c)
    u10_ms = crop(hour_file(run_dir, run, fhour, "UGRD", "TGL", "10"), yslice, xslice)
    v10_ms = crop(hour_file(run_dir, run, fhour, "VGRD", "TGL", "10"), yslice, xslice)
    gust_kmh = compute_all_cause_gust(
        run_dir,
        run,
        fhour,
        yslice,
        xslice,
        lat,
        lon,
        terrain_m,
        tmp2_c + 273.15,
        u10_ms,
        v10_ms,
        dcape_stride,
    )

    li_factor = np.clip((1.0 - li) / 6.0, 0.0, 1.0)
    cape_factor = ramp(cape, 75.0, 800.0)
    # Negative MU-LI is the primary BC instability signal; CAPE only strengthens it.
    instability = np.clip(li_factor * (0.80 + 0.20 * cape_factor), 0.0, 1.0)

    mid_rh_factor = ramp(mid_rh, 35.0, 75.0)
    moisture = np.clip(charge_factor * (0.85 + 0.15 * mid_rh_factor), 0.0, 1.0)

    upward_w = np.fmax.reduce(
        [
            np.zeros_like(omega500),
            geometric_vertical_velocity_ms(omega500, tmp500_k, 500.0),
            geometric_vertical_velocity_ms(omega700, tmp700_k, 700.0),
        ]
    )
    upward_w = smooth_nan(np.where(np.isfinite(upward_w), upward_w, np.nan), sigma=sigma_for_km(1.4))
    updraft_factor = ramp(upward_w, 0.005, 0.050)
    precip_signal = np.maximum(ramp(precip_3h, 0.05, 1.5), ramp(prate_mm_h, 0.02, 0.8))

    dry_realization = updraft_factor * charge_factor
    trigger = np.maximum(precip_signal, dry_realization)
    cap_break_proxy = np.sqrt(np.clip(trigger, 0.0, 1.0))

    potential = 100.0 * instability * moisture * cap_break_proxy
    potential = np.where((li_factor < 0.05) & (cape_factor < 0.08), 0.0, potential)
    potential = np.where(moisture < 0.08, 0.0, potential)
    potential = np.where(np.isfinite(potential), potential, np.nan)
    potential = np.clip(smooth_nan(potential.astype(np.float32), sigma=sigma_for_km(1.2)), 0.0, 100.0)

    dry_rh = np.minimum(surface_rh, subcloud_rh)
    dry_factor = ramp(55.0 - dry_rh, 0.0, 25.0) * (1.0 - ramp(precip_3h, 0.25, 2.5))
    # The LPI already includes the convective trigger; dry lightning should not be re-gated on pressure-level omega.
    dry_potential = np.where(potential >= 25.0, potential * dry_factor, 0.0)

    return LightningFields(
        potential=potential.astype(np.float32),
        dry_potential=dry_potential.astype(np.float32),
        li=li.astype(np.float32),
        cape=cape.astype(np.float32),
        precip_3h=precip_3h.astype(np.float32),
        charge_rh=charge_rh.astype(np.float32),
        trigger=trigger.astype(np.float32),
        subcloud_rh=subcloud_rh.astype(np.float32),
        surface_rh=surface_rh.astype(np.float32),
        gust_kmh=gust_kmh.astype(np.float32),
        u10_ms=u10_ms.astype(np.float32),
        v10_ms=v10_ms.astype(np.float32),
    )


def finite_window_max(fields: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.stack(tuple(fields)).astype(np.float32, copy=False)
    valid = np.isfinite(stack)
    maximum = np.max(np.where(valid, stack, -np.inf), axis=0)
    return np.where(np.any(valid, axis=0), maximum, np.nan).astype(np.float32)


def gust_window_max(
    snapshots: tuple[LightningFields, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gust_stack = np.stack([snapshot.gust_kmh for snapshot in snapshots]).astype(np.float32, copy=False)
    valid = np.isfinite(gust_stack)
    safe_gust = np.where(valid, gust_stack, -np.inf)
    maximum_index = np.argmax(safe_gust, axis=0)[None, ...]
    maximum_gust = np.take_along_axis(safe_gust, maximum_index, axis=0)[0]
    u_stack = np.stack([snapshot.u10_ms for snapshot in snapshots]).astype(np.float32, copy=False)
    v_stack = np.stack([snapshot.v10_ms for snapshot in snapshots]).astype(np.float32, copy=False)
    maximum_u = np.take_along_axis(u_stack, maximum_index, axis=0)[0]
    maximum_v = np.take_along_axis(v_stack, maximum_index, axis=0)[0]
    any_valid = np.any(valid, axis=0)
    return (
        np.where(any_valid, maximum_gust, np.nan).astype(np.float32),
        np.where(any_valid, maximum_u, np.nan).astype(np.float32),
        np.where(any_valid, maximum_v, np.nan).astype(np.float32),
    )


def compute_lightning_fields(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
    lat: np.ndarray,
    lon: np.ndarray,
    terrain_m: np.ndarray,
    dcape_stride: int,
) -> LightningFields:
    """Compute hourly diagnostics and aggregate hazards over the plotted three-hour window."""
    snapshot_hours = diagnostic_window_hours(fhour)
    snapshots = tuple(
        compute_instantaneous_lightning_fields(
            run_dir,
            run,
            snapshot_hour,
            yslice,
            xslice,
            lat,
            lon,
            terrain_m,
            dcape_stride,
        )
        for snapshot_hour in snapshot_hours
    )
    current = snapshots[-1]
    if len(snapshots) == 1:
        return current

    potential = finite_window_max(snapshot.potential for snapshot in snapshots)
    trigger = finite_window_max(snapshot.trigger for snapshot in snapshots)
    dry_candidates: list[np.ndarray] = []
    for snapshot in snapshots:
        dry_rh = np.minimum(snapshot.surface_rh, snapshot.subcloud_rh)
        dry_rh_factor = ramp(55.0 - dry_rh, 0.0, 25.0)
        dry_candidates.append(
            np.where(snapshot.potential >= 25.0, snapshot.potential * dry_rh_factor, 0.0)
        )
    dry_core = finite_window_max(dry_candidates)
    window_precip_factor = 1.0 - ramp(current.precip_3h, 0.25, 2.5)
    dry_potential = np.clip(dry_core * window_precip_factor, 0.0, 100.0).astype(np.float32)
    gust_kmh, u10_ms, v10_ms = gust_window_max(snapshots)

    log(
        f"  aggregated F{snapshot_hours[0]:03d}-F{snapshot_hours[-1]:03d}: "
        "3-h max LPI/dry-lightning/gust; valid-time RH; ending-window precipitation."
    )
    return LightningFields(
        potential=potential,
        dry_potential=dry_potential,
        li=current.li,
        cape=current.cape,
        precip_3h=current.precip_3h,
        charge_rh=current.charge_rh,
        trigger=trigger,
        subcloud_rh=current.subcloud_rh,
        surface_rh=current.surface_rh,
        gust_kmh=gust_kmh,
        u10_ms=u10_ms,
        v10_ms=v10_ms,
    )


def lightning_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, list[int]]:
    levels = [5, 10, 20, 30, 45, 60, 75, 90, 100]
    colors = [
        "#f1f1f1",
        "#ded7ec",
        "#c2afe2",
        "#9b78ce",
        "#b447a6",
        "#d71e72",
        "#f05289",
        "#ff9fc2",
    ]
    cmap = mcolors.ListedColormap(colors, name="bc_lightning")
    cmap.set_under((1.0, 1.0, 1.0, 0.0))
    cmap.set_over("#ff9fc2")
    return cmap, mcolors.BoundaryNorm(levels, cmap.N), levels


def gust_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, list[int]]:
    levels = list(range(10, 121, 10))
    colors = [
        "#42e650",
        "#00c93b",
        "#fff45a",
        "#f4df16",
        "#c9ba00",
        "#ff5a4f",
        "#ff150d",
        "#a90000",
        "#ffd2ff",
        "#ff00e6",
        "#b000b8",
    ]
    cmap = mcolors.ListedColormap(colors, name="fire_gust")
    cmap.set_under("#5ee466")
    cmap.set_over("#050505")
    return cmap, mcolors.BoundaryNorm(levels, cmap.N), levels


def fwi_date_for_valid(run: RunInfo, fhour: int) -> dt.date:
    valid = run.init_time + dt.timedelta(hours=int(fhour))
    return valid.astimezone(plot_style.LOCAL_TZ).date()


def fwi_bbox_3978(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, east, south, north = extent
    lons = np.linspace(west, east, 9)
    lats = np.linspace(south, north, 9)
    lon2d, lat2d = np.meshgrid(lons, lats)
    x, y = FWI_TO_3978.transform(lon2d, lat2d)
    pad = 35000.0
    xmin = max(float(np.nanmin(x)) - pad, FWI_NATIVE_BBOX_3978[0])
    xmax = min(float(np.nanmax(x)) + pad, FWI_NATIVE_BBOX_3978[2])
    ymin = max(float(np.nanmin(y)) - pad, FWI_NATIVE_BBOX_3978[1])
    ymax = min(float(np.nanmax(y)) + pad, FWI_NATIVE_BBOX_3978[3])
    if xmin >= xmax or ymin >= ymax:
        raise RuntimeError("CWFIS FWI grid does not overlap the plot extent.")
    return xmin, ymin, xmax, ymax


def fwi_cache_path(fwi_dir: Path, valid_date: dt.date, extent: tuple[float, float, float, float]) -> Path:
    west, east, south, north = extent
    domain = f"{west:.1f}_{east:.1f}_{south:.1f}_{north:.1f}".replace("-", "m").replace(".", "p")
    return fwi_dir / f"{valid_date:%Y%m%d}" / f"cwfis_fwi_{valid_date:%Y%m%d}_{domain}.tif"


def fwi_wcs_dimensions(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    width = max(2, int(math.ceil((xmax - xmin) / FWI_CACHE_RESOLUTION_M)))
    height = max(2, int(math.ceil((ymax - ymin) / FWI_CACHE_RESOLUTION_M)))
    largest = max(width, height)
    if largest > FWI_MAX_DIMENSION:
        scale = largest / FWI_MAX_DIMENSION
        width = max(2, int(round(width / scale)))
        height = max(2, int(round(height / scale)))
    return width, height


def fetch_fwi_geotiff(valid_date: dt.date, fwi_dir: Path, extent: tuple[float, float, float, float]) -> Path:
    dest = fwi_cache_path(fwi_dir, valid_date, extent)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    bbox = fwi_bbox_3978(extent)
    width, height = fwi_wcs_dimensions(bbox)
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": FWI_COVERAGE,
        "BBOX": ",".join(f"{value:.1f}" for value in bbox),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "CRS": "EPSG:3978",
        "FORMAT": "geotiff",
        "time": valid_date.isoformat(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
    try:
        response = requests.get(FWI_WCS_URL, params=params, timeout=(20, 180))
        response.raise_for_status()
        tmp_path.write_bytes(response.content)
        with Image.open(tmp_path) as image:
            image.verify()
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def geotiff_projected_lon_lat(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    tags = image.tag_v2
    width, height = image.size
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    transform = tags.get(34264)
    if transform:
        m = [float(value) for value in transform]
        x = m[0] * cols + m[1] * rows + m[3]
        y = m[4] * cols + m[5] * rows + m[7]
    else:
        tiepoint = tags.get(33922)
        scale = tags.get(33550)
        if not tiepoint or not scale:
            raise RuntimeError("CWFIS FWI GeoTIFF is missing georeferencing metadata.")
        x0 = float(tiepoint[3])
        y0 = float(tiepoint[4])
        dx = float(scale[0])
        dy = float(scale[1])
        x = x0 + cols * dx
        y = y0 - rows * dy
    lon, lat = FWI_FROM_3978.transform(x, y)
    return lon.astype(np.float32), lat.astype(np.float32)


def read_fwi_grid(valid_date: dt.date, fwi_dir: Path, extent: tuple[float, float, float, float]) -> FwiGrid:
    path = fetch_fwi_geotiff(valid_date, fwi_dir, extent)
    with Image.open(path) as image:
        lon, lat = geotiff_projected_lon_lat(image)
        data = np.asarray(image, dtype=np.float32)
        nodata_tag = image.tag_v2.get(42113)
    if nodata_tag is not None:
        nodata = float(nodata_tag)
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.isfinite(data) & (data >= 0.0) & (data <= 150.0), data, np.nan)
    return FwiGrid(date=valid_date, lat=lat, lon=lon, fwi=data.astype(np.float32))


def lpi_cache_path(out_path: Path) -> Path:
    return out_path.parent / "lpi_cache" / f"{out_path.stem}_lpi.npz"


def save_lpi_cache(
    out_path: Path,
    run: RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    potential: np.ndarray,
    shade_stride: int,
) -> Path:
    """Store a compact plot-ready LPI grid for later observation verification."""
    cache_path = lpi_cache_path(out_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sample = (slice(None, None, shade_stride), slice(None, None, shade_stride))
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray([LPI_CACHE_VERSION], dtype=np.int16),
                formula_version=np.asarray(LPI_FORMULA_VERSION),
                model_key=np.asarray(model_config().key),
                model_label=np.asarray(model_config().label),
                source_label=np.asarray(model_config().source_label),
                run_stamp=np.asarray(run.stamp),
                init_iso=np.asarray(run.init_time.isoformat().replace("+00:00", "Z")),
                fhour=np.asarray([int(fhour)], dtype=np.int16),
                window_fhours=np.asarray(diagnostic_window_hours(fhour), dtype=np.int16),
                temporal_aggregation=np.asarray("three_hour_max" if fhour > 0 else "initial_snapshot"),
                lat=lat[sample].astype(np.float32, copy=False),
                lon=lon[sample].astype(np.float32, copy=False),
                potential=potential[sample].astype(np.float32, copy=False),
            )
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return cache_path


def label_contours(contours, fontsize: float = 6.8, fmt: str = "%g", colors: str | None = None) -> None:
    labels = contours.axes.clabel(contours, inline=True, inline_spacing=4, fmt=fmt, fontsize=fontsize, colors=colors)
    for label in labels:
        label.set_path_effects([path_effects.withStroke(linewidth=1.9, foreground="white", alpha=0.9)])


def rh_hatch_masks(surface_rh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return non-overlapping masks for dry and moist 10 m RH categories."""
    finite = np.isfinite(surface_rh)
    return (
        finite & (surface_rh >= 20.0) & (surface_rh <= 30.0),
        finite & (surface_rh < 20.0),
        finite & (surface_rh >= 60.0) & (surface_rh <= 80.0),
        finite & (surface_rh > 80.0),
    )


def add_rh_hatching(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    surface_rh: np.ndarray,
    _contour_stride: int,
) -> None:
    """Overlay categorical low- and high-RH hatching on the fire-weather map."""
    rh_stride = grid_stride(RH_BOUNDARY_GRID_KM)
    sample = (slice(None, None, rh_stride), slice(None, None, rh_stride))
    rh = smooth_nan(surface_rh, sigma=sigma_for_km(RH_SMOOTHING_KM))[sample]
    layers = (
        ((-0.1, 20.0), VERY_LOW_RH_HATCH_COLOR),
        ((20.0, 30.0), LOW_RH_HATCH_COLOR),
        ((80.0, 100.1), EXCELLENT_RECOVERY_HATCH_COLOR),
        ((60.0, 80.0), GOOD_RECOVERY_HATCH_COLOR),
    )
    for levels, color in layers:
        if not np.any(np.isfinite(rh) & (rh >= levels[0]) & (rh <= levels[1])):
            continue
        with plt.rc_context({"hatch.color": color, "hatch.linewidth": RH_HATCH_LINEWIDTH}):
            hatched = ax.contourf(
                lon[sample],
                lat[sample],
                rh,
                levels=levels,
                colors="none",
                hatches=["xx"],
                transform=DATA_CRS,
                transform_first=True,
                zorder=10,
            )
        for collection in hatched.collections:
            style_rh_hatch_collection(collection, color)


def style_rh_hatch_collection(collection, color: str) -> None:
    collection.set_facecolor((0.0, 0.0, 0.0, 0.0))
    alpha = (
        RECOVERY_RH_HATCH_ALPHA
        if color in {GOOD_RECOVERY_HATCH_COLOR, EXCELLENT_RECOVERY_HATCH_COLOR}
        else RH_HATCH_ALPHA
    )
    collection.set_edgecolor(mcolors.to_rgba(color, alpha))


def plot_gust_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    gust_kmh: np.ndarray,
    shade_stride: int,
    row_density: float = BASE_GUST_VECTOR_DENSITY,
    column_density: float = BASE_GUST_VECTOR_DENSITY,
):
    sample = plot_style.vector_sample_slices(
        ax,
        lon.shape,
        minimum=max(1, shade_stride),
        spacing_px=27.0,
        row_density=row_density,
        column_density=column_density,
    )
    u = u10_ms[sample]
    v = v10_ms[sample]
    gust = gust_kmh[sample]
    speed = np.hypot(u, v)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(gust) & (speed > 0.05)
    unit_u = np.divide(u, speed, out=np.zeros_like(u, dtype=np.float32), where=speed > 0.05)
    unit_v = np.divide(v, speed, out=np.zeros_like(v, dtype=np.float32), where=speed > 0.05)
    cmap, norm, levels = gust_cmap()
    vectors = ax.quiver(
        lon[sample][finite],
        lat[sample][finite],
        unit_u[finite],
        unit_v[finite],
        gust[finite],
        cmap=cmap,
        norm=norm,
        transform=DATA_CRS,
        scale_units="width",
        scale=68,
        width=0.00155,
        headwidth=3.5,
        headlength=4.2,
        headaxislength=3.9,
        minlength=0.05,
        pivot="middle",
        edgecolors=mcolors.to_rgba(GUST_VECTOR_EDGE_COLOR, GUST_VECTOR_EDGE_ALPHA),
        linewidths=GUST_VECTOR_EDGE_WIDTH,
        zorder=11,
    )
    return vectors, levels


def plot_lightning(
    out_path: Path,
    run: RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    yslice: slice,
    xslice: slice,
    fields: LightningFields,
    transmission_lines: list[BaseGeometry],
    shade_stride: int,
    contour_stride: int,
    peak_danger_grid: fire_danger_peak.PeakDangerGrid | None = None,
    extent: tuple[float, float, float, float] | None = None,
    region_key: str = "bc",
    region_label: str = "BC",
) -> None:
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]

    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=PLOT_CRS)
    ax.set_extent(extent or model_config().extent, crs=DATA_CRS)
    ax.set_facecolor("#dbeaf0")
    hrdps.add_map_features(ax)

    lpi_cmap, lpi_norm, lpi_levels = lightning_cmap()
    shade_sample = (slice(None, None, shade_stride), slice(None, None, shade_stride))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpi_shaded = ax.contourf(
            plot_lon[shade_sample],
            plot_lat[shade_sample],
            fields.potential[shade_sample],
            levels=lpi_levels,
            cmap=lpi_cmap,
            norm=lpi_norm,
            extend="max",
            alpha=0.78,
            transform=DATA_CRS,
            transform_first=True,
            zorder=3,
        )

    hrdps.add_hydro_features(ax)
    add_transmission_lines(ax, transmission_lines)
    row_density, column_density = gust_vector_density(region_key)
    gust_vectors, gust_levels = plot_gust_vectors(
        ax,
        plot_lon,
        plot_lat,
        fields.u10_ms,
        fields.v10_ms,
        fields.gust_kmh,
        shade_stride,
        row_density=row_density,
        column_density=column_density,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if peak_danger_grid is not None:
            peak_danger = peak_danger_grid.danger[yslice, xslice]
            peak_danger_s = smooth_nan(peak_danger, sigma=sigma_for_km(5.0))
            danger_ct = ax.contour(
                plot_lon,
                plot_lat,
                peak_danger_s,
                levels=PEAK_DANGER_CONTOUR_LEVELS,
                colors=PEAK_DANGER_CONTOUR_COLORS,
                linewidths=PEAK_DANGER_CONTOUR_LINEWIDTHS,
                transform=DATA_CRS,
                transform_first=True,
                zorder=12,
            )
            label_contours(danger_ct, fontsize=6.6, fmt=PEAK_DANGER_CONTOUR_LABELS)

        add_rh_hatching(ax, plot_lon, plot_lat, fields.surface_rh, contour_stride)

        # DKP-style filled dry-potential hatching is intentionally disabled for now.
        dry_star_stride = max(contour_stride, shade_stride * 2)
        dry_star_mask = fields.dry_potential[::dry_star_stride, ::dry_star_stride] >= 15.0
        if np.any(dry_star_mask):
            ax.scatter(
                plot_lon[::dry_star_stride, ::dry_star_stride][dry_star_mask],
                plot_lat[::dry_star_stride, ::dry_star_stride][dry_star_mask],
                marker=DRY_LIGHTNING_MARKER,
                s=dry_lightning_marker_area(region_key),
                color=DRY_LIGHTNING_COLOR,
                linewidths=0.45,
                transform=DATA_CRS,
                zorder=14,
            )

    hrdps.add_city_labels(ax, fontsize=7.1, marker_size=2.2, path_width=2.35, zorder=30)

    plot_style.add_internal_colorbar(
        fig,
        ax,
        lpi_shaded,
        ticks=lpi_levels,
        label="Lightning potential index",
        title="LPI",
        fmt="%g",
        extend="max",
        backdrop=(0.000, 0.060, 0.054, 0.875),
        cax_bounds=[0.011, 0.078, 0.018, 0.800],
        tick_position="right",
    )
    plot_style.add_internal_colorbar(
        fig,
        ax,
        gust_vectors,
        ticks=gust_levels,
        label="All-cause gust (km h$^{-1}$)",
        title="GUST km/h",
        fmt="%g",
        extend="both",
        backdrop=(0.945, 0.060, 0.055, 0.875),
        cax_bounds=[0.971, 0.078, 0.018, 0.800],
    )
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
        spine.set_zorder(60)
    plot_style.add_single_panel_text(
        ax,
        plot_style.valid_header(run, fhour, f"{model_config().label} {region_label}"),
        fire_weather_footer(fhour),
        run,
        source_label="ECCC HRDPS + CWFIS",
        header_y=0.998,
        source_x=0.999,
        source_y=0.998,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def contact_sheet(images: list[Path], out_path: Path, run: RunInfo) -> None:
    thumbs: list[Image.Image] = []
    for path in images:
        img = Image.open(path).convert("RGB")
        img.thumbnail((520, 380), Image.Resampling.LANCZOS)
        thumbs.append(img.copy())
        img.close()

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    tile_w, tile_h = 540, 420
    header_h = 64
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h + header_h), "#f7f7f4")
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("Arial Bold.ttf", 30)
        label_font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((24, 18), f"{model_config().label} fire weather - run {run.init_time:%Y-%m-%d %HZ}", fill="#1d1d1d", font=title_font)

    for idx, img in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * tile_w + (tile_w - img.width) // 2
        y = header_h + row * tile_h + 12
        sheet.paste(img, (x, y))
        fhour = int(images[idx].stem.rsplit("_f", 1)[-1])
        valid = run.init_time + dt.timedelta(hours=fhour)
        draw.text((col * tile_w + 16, header_h + row * tile_h + tile_h - 28), f"F{fhour:03d} valid {valid:%d %HZ}", fill="#262626", font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_plots(
    run: RunInfo,
    data_dir: Path,
    output_dir: Path,
    shade_stride: int,
    contour_stride: int,
    dcape_stride: int,
    hours: Iterable[int] = FORECAST_HOURS,
    watershed_cache: Path = hrdps.WATERSHED_CACHE,
    refresh_watersheds: bool = False,
    no_watersheds: bool = False,
    fwi_dir: Path = FWI_CACHE_DIR,
    no_fwi: bool = False,
    region_key: str = "bc",
) -> list[Path]:
    return make_region_plots(
        run,
        data_dir,
        output_dir,
        shade_stride,
        contour_stride,
        dcape_stride,
        hours,
        watershed_cache,
        refresh_watersheds,
        no_watersheds,
        fwi_dir,
        no_fwi,
        (region_key,),
    )


def make_region_plots(
    run: RunInfo,
    data_dir: Path,
    output_dir: Path,
    shade_stride: int,
    contour_stride: int,
    dcape_stride: int,
    hours: Iterable[int] = FORECAST_HOURS,
    watershed_cache: Path = hrdps.WATERSHED_CACHE,
    refresh_watersheds: bool = False,
    no_watersheds: bool = False,
    fwi_dir: Path = FWI_CACHE_DIR,
    no_fwi: bool = False,
    region_keys: Iterable[str] = ("bc",),
) -> list[Path]:
    """Compute each forecast hour once, then render all requested regional crops."""

    hours = tuple(int(hour) for hour in hours)
    if not hours:
        return []
    regions = tuple(region_config(key) for key in region_keys)
    render_bc_twopanel = model_config().key == "continental"
    render_regional_twopanel = model_config().key == "west"
    watersheds: list[BaseGeometry] = []
    transmission_lines = load_transmission_lines()
    run_dir = data_dir / run.stamp
    sample_path = hour_file(run_dir, run, hours[0], "MU-VT-LI", "ISBL", "500")
    _, lat, lon = read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read model coordinates.")
    base_yslice, base_xslice = subset_slices(lat, lon, model_config().extent)
    base_lat = lat[base_yslice, base_xslice]
    base_lon = lon[base_yslice, base_xslice]
    terrain_path = (
        run_dir
        / f"{hrdps.TERRAIN_FHOUR:03d}"
        / field_name("HGT", "SFC", "0", run.stamp, hrdps.TERRAIN_FHOUR)
    )
    terrain_full, _, _ = read_grib(terrain_path)
    terrain_m = terrain_full[base_yslice, base_xslice]
    region_slices = {
        region.key: (slice(None), slice(None))
        if region.key == "bc"
        else subset_slices(base_lat, base_lon, region_extent(region))
        for region in regions
    }

    out_paths: list[Path] = []
    plot_dir = output_dir / run.stamp
    peak_danger_by_date: dict[dt.date, fire_danger_peak.PeakDangerGrid | None] = {}
    for fhour in hours:
        log(f"Processing lightning fields once for {len(regions)} region(s) at F{fhour:03d}.")
        fields = compute_lightning_fields(
            run_dir,
            run,
            fhour,
            base_yslice,
            base_xslice,
            lat,
            lon,
            terrain_m,
            dcape_stride,
        )
        cache_source_path = plot_dir / f"{model_output_prefix('lightning')}_{run.stamp}_f{fhour:03d}.png"
        cache_path = save_lpi_cache(
            cache_source_path,
            run,
            fhour,
            base_lat,
            base_lon,
            fields.potential,
            shade_stride,
        )
        log(f"  cached LPI verification grid {cache_path}")
        peak_danger_grid: fire_danger_peak.PeakDangerGrid | None = None
        if not no_fwi:
            valid = run.init_time + dt.timedelta(hours=fhour)
            fire_date = fire_danger_peak.fire_date_for_valid(valid, plot_style.LOCAL_TZ)
            if fire_date not in peak_danger_by_date:
                peak_danger_by_date[fire_date] = fire_danger_peak.load_peak_danger_for_display(
                    fwi_dir,
                    model_config().key,
                    fire_date,
                    run.init_time,
                    base_lat.shape,
                )
                source = peak_danger_by_date[fire_date]
                if source is None:
                    log(f"  no qualified peak-daily fire-danger guidance for {fire_date:%Y-%m-%d}.")
                else:
                    coverage = (
                        "complete"
                        if source.complete
                        else f"best-available partial ({source.hour_count} hourly fields)"
                    )
                    if source.fire_date == fire_date:
                        log(
                            f"  {coverage} peak-daily fire danger for {fire_date:%Y-%m-%d} "
                            f"from {source.source_run_stamp}."
                        )
                    else:
                        log(
                            f"  retaining {coverage} peak-daily fire danger for "
                            f"{source.fire_date:%Y-%m-%d} from {source.source_run_stamp}; "
                            f"{fire_date:%Y-%m-%d} does not reach 17:00 local."
                        )
            peak_danger_grid = peak_danger_by_date[fire_date]
        for region in regions:
            if render_bc_twopanel and region.key == "bc":
                continue
            yslice, xslice = region_slices[region.key]
            region_fields = subset_lightning_fields(fields, yslice, xslice)
            out_path = plot_dir / f"{region_output_prefix(region)}_{run.stamp}_f{fhour:03d}.png"
            if render_regional_twopanel and region.key != "bc":
                from make_hrdps_fire_weather_twopanel import plot_regional_twopanel

                regional_peak_danger = (
                    peak_danger_grid.danger[yslice, xslice] if peak_danger_grid is not None else None
                )
                plot_regional_twopanel(
                    out_path,
                    run,
                    fhour,
                    base_lat[yslice, xslice],
                    base_lon[yslice, xslice],
                    region_fields,
                    regional_peak_danger,
                    transmission_lines,
                    shade_stride,
                    contour_stride,
                    region.key,
                    region.label,
                    region_extent(region),
                )
            else:
                plot_lightning(
                    out_path,
                    run,
                    fhour,
                    base_lat,
                    base_lon,
                    yslice,
                    xslice,
                    region_fields,
                    transmission_lines,
                    shade_stride,
                    contour_stride,
                    peak_danger_grid,
                    region_extent(region),
                    region.key,
                    region.label,
                )
            log(f"  wrote {out_path}")
            out_paths.append(out_path)
        if render_bc_twopanel:
            from make_hrdps_fire_weather_twopanel import OUTPUT_PREFIX, plot_twopanel

            out_path = plot_dir / f"{OUTPUT_PREFIX}_{run.stamp}_f{fhour:03d}.png"
            plot_twopanel(
                out_path,
                run,
                fhour,
                base_lat,
                base_lon,
                fields,
                transmission_lines,
                shade_stride,
                contour_stride,
                peak_danger_grid,
                watersheds,
            )
            out_paths.append(out_path)
    return out_paths


def parse_hours(text: str | None) -> tuple[int, ...]:
    if not text:
        return FORECAST_HOURS
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="west", help="HRDPS model domain to use.")
    parser.add_argument("--data-dir", type=Path, default=None, help="GRIB2 cache directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG output directory.")
    parser.add_argument("--cycle", choices=["latest", "00", "06", "12", "18"], default="latest", help="Cycle to use.")
    parser.add_argument("--hours", default=None, help="Comma-separated forecast hours to plot, e.g. 0,3,6.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent download workers.")
    parser.add_argument("--shade-stride", type=int, default=None, help="Grid stride for shaded fields.")
    parser.add_argument("--contour-stride", type=int, default=None, help="Grid stride for contours.")
    parser.add_argument("--dcape-stride", type=int, default=None, help="Grid stride for DCAPE/PCGE profile calculations.")
    parser.add_argument(
        "--watershed-cache",
        type=Path,
        default=hrdps.WATERSHED_CACHE,
        help="Local BC Hydro watershed boundary shapefile.",
    )
    parser.add_argument("--fwi-dir", type=Path, default=FWI_CACHE_DIR, help="Cached FWI2025 state and peak-danger directory.")
    parser.add_argument("--no-fwi", action="store_true", help="Skip experimental peak-daily fire-danger contours.")
    parser.add_argument("--refresh-watersheds", action="store_true", help="Re-read the local watershed overlay.")
    parser.add_argument("--no-watersheds", action="store_true", help="Skip watershed overlays.")
    parser.add_argument(
        "--region",
        choices=sorted(FIRE_WEATHER_REGIONS),
        default="bc",
        help="Fire-weather plot region to render.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = set_model(args.model)
    data_dir = args.data_dir or Path(config.default_data_dir)
    output_dir = args.output_dir or Path(f"{config.default_output_dir}_lightning")
    hours = parse_hours(args.hours)
    shade_stride = args.shade_stride or grid_stride(5.0)
    contour_stride = args.contour_stride or grid_stride(12.0)
    dcape_stride = args.dcape_stride or grid_stride(18.0)

    if args.cycle != "latest" and args.cycle not in model_config().cycles:
        raise RuntimeError(f"{config.label} does not provide a {args.cycle}Z cycle.")
    if args.cycle == "latest":
        run = latest_complete_run(hours)
    else:
        stamp = run_stamp_from_listing(args.cycle)
        if not stamp:
            raise RuntimeError(f"No files found for cycle {args.cycle}Z.")
        run = RunInfo(cycle=args.cycle, stamp=stamp, init_time=parse_stamp(stamp))
        if not run_is_complete(run, hours):
            raise RuntimeError(f"Cycle {run.stamp} is not complete for the required lightning fields.")

    log(f"Using {config.label} run {run.stamp}.")
    download_run(run, data_dir, args.workers, hours)
    make_plots(
        run,
        data_dir,
        output_dir,
        shade_stride,
        contour_stride,
        dcape_stride,
        hours,
        args.watershed_cache,
        args.refresh_watersheds,
        args.no_watersheds,
        args.fwi_dir,
        args.no_fwi,
        args.region,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
