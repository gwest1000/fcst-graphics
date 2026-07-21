#!/usr/bin/env python3
"""Make an RDPS-style four-panel diagnostic sheet from HRDPS-West data."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry.base import BaseGeometry

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

FORECAST_HOURS = tuple(range(0, 49, 3))
SPFH_LEVELS_HPA = (
    50,
    100,
    150,
    175,
    200,
    225,
    250,
    275,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    875,
    900,
    925,
    950,
    970,
    985,
    1000,
    1015,
)
RH_LAYER_HPA = (850, 800, 750, 700)
WIND_850_TERRAIN_MARGIN_HPA = 35.0

PANEL_PROJ = ccrs.LambertConformal(central_longitude=-123.0, central_latitude=53.0)
DATA_CRS = ccrs.PlateCarree()
ABSV_SHADE_SMOOTHING_KM = {
    "west": 3.0,
    "continental": 1.4,
}
MSLP_SMOOTHING_KM = 12.0
MSLP_LEVELS_KPA = np.round(np.arange(95.2, 104.8, 0.4), 1)
MSLP_HIGH_THRESHOLD_KPA = 102.4
MSLP_STANDARD_COLOR = "#5f5f5f"
MSLP_BLUE = "#0046ff"
TEMP850_SMOOTHING_KM = 8.0
TEMP850_LEVELS_C = np.arange(-34, 36, 2)
TEMP850_STANDARD_LINEWIDTH = 1.40
TEMP850_ZERO_LINEWIDTH = 1.90
TEMP850_WARM_LINEWIDTH = 1.75
TEMP850_HOT_LINEWIDTH = 1.90
TEMP850_ZERO_COLOR = "#0057ff"
TEMP850_WARM_COLOR = "#ff8c00"
TEMP850_HOT_COLOR = "#d7191c"
IPW_SMOOTHING_KM = 7.5
LI_SMOOTHING_KM = 8.0
CAPE_SMOOTHING_KM = 10.0
LI_LINEWIDTHS = (2.15, 2.00, 1.85, 1.75)
HGT500_LINEWIDTH = 1.65
HGT500_HALO_LINEWIDTH = 2.65
HGT500_LEVELS_KM = np.round(np.arange(4.80, 6.30, 0.06), 2)
TERRAIN_LEVELS_M = [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3500]
TERRAIN_COLORS = [
    "#ead8b8",
    "#ddc298",
    "#cfaa79",
    "#bd9160",
    "#aa784c",
    "#96623f",
    "#814e35",
    "#6b3d2d",
    "#552f26",
    "#3e211d",
    "#281411",
]
TRANSMISSION_PANEL_INDICES = (1, 3)
CONTINENTAL_FOURPANEL_EXTENT = (-141.0, -106.7, 45.5, 59.8)


def fourpanel_extent() -> tuple[float, float, float, float]:
    if model_config().key == "continental":
        return CONTINENTAL_FOURPANEL_EXTENT
    return model_config().extent


def log(message: str) -> None:
    print(message, flush=True)


def required_names(stamp: str, fhour: int) -> list[str]:
    names = [
        field_name("ABSV", "ISBL", "0500", stamp, fhour),
        field_name("HGT", "ISBL", "0500", stamp, fhour),
        field_name("UGRD", "ISBL", "0250", stamp, fhour),
        field_name("VGRD", "ISBL", "0250", stamp, fhour),
        field_name("PRES", "SFC", "0", stamp, fhour),
        field_name("MU-VT-LI", "ISBL", "500", stamp, fhour),
        field_name("CAPE", "ETAL", "10000", stamp, fhour),
        field_name("TMP", "ISBL", "0850", stamp, fhour),
        field_name("UGRD", "ISBL", "0850", stamp, fhour),
        field_name("VGRD", "ISBL", "0850", stamp, fhour),
        field_name("UGRD", "ISBL", "0700", stamp, fhour),
        field_name("VGRD", "ISBL", "0700", stamp, fhour),
        field_name("PRMSL", "MSL", "0", stamp, fhour),
        field_name("UGRD", "TGL", "10", stamp, fhour),
        field_name("VGRD", "TGL", "10", stamp, fhour),
    ]
    if fhour > 0:
        names.append(field_name("APCP", "SFC", "0", stamp, fhour))
    if fhour == hrdps.TERRAIN_FHOUR:
        names.append(field_name("HGT", "SFC", "0", stamp, fhour))
    for level in SPFH_LEVELS_HPA:
        names.append(field_name("SPFH", "ISBL", f"{level:04d}", stamp, fhour))
    for level in RH_LAYER_HPA:
        names.append(field_name("RH", "ISBL", f"{level:04d}", stamp, fhour))
    return names


def run_is_complete(run: RunInfo) -> bool:
    for fhour in FORECAST_HOURS:
        html = fetch_text(f"{model_config().base_url}/{run.cycle}/{fhour:03d}/")
        links = set(parse_links(html))
        missing = sorted(set(required_names(run.stamp, fhour)) - links)
        if missing:
            log(f"Skipping {run.stamp}: missing {len(missing)} files at F{fhour:03d}.")
            return False
    return True


def latest_complete_run() -> RunInfo:
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
        if run_is_complete(run):
            return run

    raise RuntimeError(f"No complete {model_config().label} run was found for the four-panel fields.")


def download_run(run: RunInfo, data_dir: Path, workers: int) -> None:
    jobs: list[tuple[str, Path]] = []
    run_dir = data_dir / run.stamp
    for fhour in FORECAST_HOURS:
        for name in required_names(run.stamp, fhour):
            jobs.append((f"{model_config().base_url}/{run.cycle}/{fhour:03d}/{name}", run_dir / f"{fhour:03d}" / name))

    log(f"Downloading or reusing {len(jobs)} GRIB2 files into {run_dir}.")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, url, dest) for url, dest in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                log(f"  files ready: {completed}/{len(jobs)}")


def smooth_nan(data: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    valid = np.isfinite(data)
    if not np.any(valid):
        return data
    filled = np.where(valid, data, 0.0)
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    smoothed = gaussian_filter(filled.astype(np.float32), sigma=sigma, mode="nearest")
    out = smoothed / np.where(weights == 0, np.nan, weights)
    out[~np.isfinite(out)] = np.nan
    return out


def crop(path: Path, yslice: slice, xslice: slice) -> np.ndarray:
    data, _, _ = read_grib(path)
    return data[yslice, xslice]


def hour_file(run_dir: Path, run: RunInfo, fhour: int, variable: str, level_type: str, level: str | int) -> Path:
    return run_dir / f"{fhour:03d}" / field_name(variable, level_type, level, run.stamp, fhour)


def compute_precip_3h(run_dir: Path, run: RunInfo, fhour: int, yslice: slice, xslice: slice) -> np.ndarray:
    if fhour == 0:
        sample = crop(hour_file(run_dir, run, fhour, "PRMSL", "MSL", "0"), yslice, xslice)
        return np.zeros(sample.shape, dtype=np.float32)

    current = crop(hour_file(run_dir, run, fhour, "APCP", "SFC", "0"), yslice, xslice)
    previous_path = hour_file(run_dir, run, fhour - 3, "APCP", "SFC", "0")
    if fhour == 3 or not previous_path.exists():
        return np.maximum(current, 0.0)
    previous = crop(previous_path, yslice, xslice)
    return np.maximum(current - previous, 0.0)


def compute_layer_rh(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    psfc_hpa: np.ndarray,
    yslice: slice,
    xslice: slice,
) -> np.ndarray:
    layers = []
    for level in RH_LAYER_HPA:
        rh = crop(hour_file(run_dir, run, fhour, "RH", "ISBL", f"{level:04d}"), yslice, xslice)
        layers.append(np.where(psfc_hpa >= level, rh, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(layers), axis=0)


def terrain_adjusted_850_wind(
    u850: np.ndarray,
    v850: np.ndarray,
    u700: np.ndarray,
    v700: np.ndarray,
    psfc_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    use700 = np.isfinite(psfc_hpa) & (psfc_hpa <= 850.0 + WIND_850_TERRAIN_MARGIN_HPA)
    return np.where(use700, u700, u850), np.where(use700, v700, v850)


def compute_ipw(
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    psfc_pa: np.ndarray,
    yslice: slice,
    xslice: slice,
) -> np.ndarray:
    ipw = np.zeros(psfc_pa.shape, dtype=np.float32)
    last_p = np.full(psfc_pa.shape, np.nan, dtype=np.float32)
    last_q = np.full(psfc_pa.shape, np.nan, dtype=np.float32)
    prev_p: float | None = None
    prev_q: np.ndarray | None = None
    g = 9.80665

    for level in SPFH_LEVELS_HPA:
        p_pa = float(level * 100.0)
        q = crop(hour_file(run_dir, run, fhour, "SPFH", "ISBL", f"{level:04d}"), yslice, xslice)
        q = np.where(np.isfinite(q), np.maximum(q, 0.0), np.nan).astype(np.float32)
        under_surface = p_pa <= psfc_pa

        if prev_p is not None and prev_q is not None:
            segment = 0.5 * (q + prev_q) * (p_pa - prev_p) / g
            ipw = np.where(under_surface & np.isfinite(segment), ipw + segment, ipw)

        last_p = np.where(under_surface, p_pa, last_p)
        last_q = np.where(under_surface, q, last_q)
        prev_p = p_pa
        prev_q = q

    surface_layer = last_q * np.maximum(psfc_pa - last_p, 0.0) / g
    ipw = np.where(np.isfinite(surface_layer), ipw + surface_layer, ipw)
    ipw[~np.isfinite(psfc_pa)] = np.nan
    return np.maximum(ipw, 0.0)


def decimate(data: np.ndarray, factor: int) -> np.ndarray:
    return data[::factor, ::factor]


def contour_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    stride: int = 8,
    sigma: float = 1.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return lat[::stride, ::stride], lon[::stride, ::stride], smooth_nan(data, sigma=sigma)[::stride, ::stride]


def label_contours(contours, fontsize: float = 6.2, fmt: str = "%g", colors=None) -> None:
    labels = contours.axes.clabel(contours, inline=True, inline_spacing=4, fmt=fmt, fontsize=fontsize, colors=colors)
    for label in labels:
        label.set_path_effects([path_effects.withStroke(linewidth=1.7, foreground="white", alpha=0.9)])


def make_absv_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, list[float]]:
    levels = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    colors = [
        "#001071",
        "#0026d9",
        "#7081ff",
        "#d7dcff",
        "#ffffff",
        "#ffd6d6",
        "#ff9f9f",
        "#ff6969",
        "#ff3434",
        "#ff0000",
        "#d60000",
        "#aa0000",
        "#790000",
        "#4a0000",
    ]
    cmap = mcolors.ListedColormap(colors, name="absv_rdps")
    cmap.set_under("#00004d")
    cmap.set_over("#270000")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return cmap, norm, levels


def absv_shade_sigma() -> float:
    return sigma_for_km(ABSV_SHADE_SMOOTHING_KM.get(model_config().key, 1.4))


def make_ipw_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, np.ndarray]:
    levels = np.arange(10, 52, 2)
    colors = [
        "#f2f2f2",
        "#c9cbff",
        "#8f95ff",
        "#4f59ff",
        "#152bd9",
        "#0052ff",
        "#008dff",
        "#00c0ff",
        "#00e260",
        "#4cf24d",
        "#a5f64b",
        "#ffff35",
        "#c9c62a",
        "#9a811c",
        "#d8650c",
        "#ff2f00",
        "#ff5757",
        "#ff9c9c",
        "#d474ff",
        "#ff7dff",
    ]
    cmap = mcolors.ListedColormap(colors, name="ipw_rdps")
    cmap.set_under("#ffffff")
    cmap.set_over("#ffb2ff")
    return cmap, mcolors.BoundaryNorm(levels, cmap.N), levels


def make_rh_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, list[int]]:
    levels = [10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 100]
    colors = [
        "#b29400",
        "#dec400",
        "#fff03d",
        "#fff99b",
        "#ffffff",
        "#caffb9",
        "#80e468",
        "#39bf46",
        "#168d30",
        "#005d1e",
    ]
    cmap = mcolors.ListedColormap(colors, name="rh_split")
    cmap.set_under((1.0, 1.0, 1.0, 0.0))
    cmap.set_over("#003f15")
    return cmap, mcolors.BoundaryNorm(levels, cmap.N), levels


def make_precip_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, list[float]]:
    levels = [0.25, 1, 2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 60, 80, 100]
    colors = [
        "#f0f0ff",
        "#c9c7ff",
        "#7e73ff",
        "#263cff",
        "#00d24a",
        "#42ee45",
        "#c7ff8a",
        "#ffff2f",
        "#c4b51e",
        "#7f6a19",
        "#bd6b00",
        "#f28a00",
        "#ff5a5a",
        "#ff0000",
    ]
    cmap = mcolors.ListedColormap(colors, name="precip_rdps")
    cmap.set_under((1.0, 1.0, 1.0, 0.0))
    cmap.set_over("#5a0000")
    return cmap, mcolors.BoundaryNorm(levels, cmap.N), levels


def make_terrain_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm, list[int]]:
    cmap = mcolors.ListedColormap(TERRAIN_COLORS, name="model_topography")
    cmap.set_under((1.0, 1.0, 1.0, 0.0))
    cmap.set_over("#1b0c09")
    return cmap, mcolors.BoundaryNorm(TERRAIN_LEVELS_M, cmap.N), TERRAIN_LEVELS_M


def mslp_contour_groups() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    major = np.isclose(np.mod(MSLP_LEVELS_KPA - MSLP_LEVELS_KPA[0], 0.8), 0.0, atol=0.05)
    high = MSLP_LEVELS_KPA > MSLP_HIGH_THRESHOLD_KPA
    threshold = np.isclose(MSLP_LEVELS_KPA, MSLP_HIGH_THRESHOLD_KPA)
    below = MSLP_LEVELS_KPA < MSLP_HIGH_THRESHOLD_KPA
    return (
        MSLP_LEVELS_KPA[below & ~major],
        MSLP_LEVELS_KPA[below & major],
        MSLP_LEVELS_KPA[threshold],
        MSLP_LEVELS_KPA[high],
    )


def temp850_contour_groups() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    standard = TEMP850_LEVELS_C[(TEMP850_LEVELS_C < 16) & ~np.isclose(TEMP850_LEVELS_C, 0)]
    zero = TEMP850_LEVELS_C[np.isclose(TEMP850_LEVELS_C, 0)]
    warm = TEMP850_LEVELS_C[(TEMP850_LEVELS_C >= 16) & (TEMP850_LEVELS_C < 20)]
    hot = TEMP850_LEVELS_C[TEMP850_LEVELS_C >= 20]
    return standard, zero, warm, hot


def add_base_features(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    hrdps.add_base_features(ax, extent=extent)


def load_watersheds(
    cache_path: Path,
    refresh: bool = False,
    extent: tuple[float, float, float, float] | None = None,
) -> list[BaseGeometry]:
    return hrdps.load_watersheds(cache_path, refresh=refresh, extent=extent)


def add_watersheds(ax: plt.Axes, watersheds: list[BaseGeometry]) -> None:
    hrdps.add_watersheds(ax, watersheds)


def load_transmission_lines(extent: tuple[float, float, float, float]) -> list[BaseGeometry]:
    # Lazy imports avoid the lightning/four-panel diagnostic dependency cycle.
    from make_hrdps_west_lightning import load_transmission_lines as load_lines

    return load_lines(extent=extent)


def add_transmission_lines(ax: plt.Axes, lines: list[BaseGeometry]) -> None:
    from make_hrdps_west_lightning import add_transmission_lines as add_lines

    add_lines(ax, lines)


def plot_barbs(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    stride: int,
    color: str = "black",
    row_density: float = 1.0,
    column_density: float = 1.0,
) -> None:
    sample = plot_style.vector_sample_slices(
        ax,
        lon.shape,
        minimum=stride,
        row_density=row_density,
        column_density=column_density,
    )
    u = u_ms[sample] * 3.6
    v = v_ms[sample] * 3.6
    finite = np.isfinite(u) & np.isfinite(v)
    ax.barbs(
        lon[sample][finite],
        lat[sample][finite],
        u[finite],
        v[finite],
        transform=DATA_CRS,
        length=4.3,
        linewidth=0.48,
        color=color,
        pivot="middle",
        barb_increments={"half": 10, "full": 20, "flag": 100},
        sizes={"emptybarb": 0.035, "spacing": 0.16, "height": 0.35},
        zorder=23,
    )


def plot_fourpanel(
    out_path: Path,
    run_dir: Path,
    run: RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    extent: tuple[float, float, float, float],
    yslice: slice,
    xslice: slice,
    terrain_m: np.ndarray,
    watersheds: list[BaseGeometry],
    transmission_lines: list[BaseGeometry],
    shade_stride: int,
    contour_stride: int,
    barb_stride: int,
) -> None:
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    psfc_pa = crop(hour_file(run_dir, run, fhour, "PRES", "SFC", "0"), yslice, xslice)
    psfc_hpa = psfc_pa / 100.0

    header = plot_style.valid_header(run, fhour)
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    axes = [fig.add_axes(position, projection=PANEL_PROJ) for position in plot_style.FOURPANEL_POSITIONS]

    for ax in axes:
        add_base_features(ax, extent)
    for panel_index in TRANSMISSION_PANEL_INDICES:
        add_transmission_lines(axes[panel_index], transmission_lines)

    # 1) 500 hPa absolute vorticity, 500 hPa height, 250 hPa wind.
    ax = axes[0]
    absv = crop(hour_file(run_dir, run, fhour, "ABSV", "ISBL", "0500"), yslice, xslice) * 1.0e5
    hgt500_km = crop(hour_file(run_dir, run, fhour, "HGT", "ISBL", "0500"), yslice, xslice) / 1000.0
    u250 = crop(hour_file(run_dir, run, fhour, "UGRD", "ISBL", "0250"), yslice, xslice)
    v250 = crop(hour_file(run_dir, run, fhour, "VGRD", "ISBL", "0250"), yslice, xslice)
    cmap, norm, levels = make_absv_cmap()
    cf = ax.contourf(
        decimate(plot_lon, shade_stride),
        decimate(plot_lat, shade_stride),
        decimate(smooth_nan(absv, absv_shade_sigma()), shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    clat, clon, chgt = contour_grid(plot_lat, plot_lon, hgt500_km, stride=contour_stride, sigma=sigma_for_km(3.0))
    hgt_ct = ax.contour(
        clon,
        clat,
        chgt,
        levels=HGT500_LEVELS_KM,
        colors="black",
        linewidths=HGT500_LINEWIDTH,
        transform=DATA_CRS,
        zorder=22,
    )
    for collection in hgt_ct.collections:
        collection.set_path_effects(
            [
                path_effects.Stroke(linewidth=HGT500_HALO_LINEWIDTH, foreground="white", alpha=0.55),
                path_effects.Normal(),
            ]
        )
    label_contours(hgt_ct, fontsize=5.4, fmt="%.2f")
    plot_barbs(ax, plot_lon, plot_lat, u250, v250, barb_stride, color="black", row_density=2.0, column_density=2.0)
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=[-4, 0, 4, 8, 12, 16, 20, 24], label="$10^{-5}$ s$^{-1}$", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, "50.0kPa AbsVort(s$^{-1}$,shaded), HgtThk(cntrd,km), 25.0kPa Wind(hlf brb=10km/h)", run)

    # 2) Integrated precipitable water, lifted index, CAPE.
    ax = axes[1]
    ipw = compute_ipw(run_dir, run, fhour, psfc_pa, yslice, xslice)
    li = crop(hour_file(run_dir, run, fhour, "MU-VT-LI", "ISBL", "500"), yslice, xslice)
    cape = crop(hour_file(run_dir, run, fhour, "CAPE", "ETAL", "10000"), yslice, xslice)
    li = np.where(np.abs(li) > 50.0, np.nan, li)
    cape = np.where((cape >= 0.0) & (cape < 20000.0), cape, np.nan)
    cmap, norm, levels = make_ipw_cmap()
    cf = ax.contourf(
        decimate(plot_lon, shade_stride),
        decimate(plot_lat, shade_stride),
        decimate(smooth_nan(ipw, sigma_for_km(IPW_SMOOTHING_KM)), shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    clat, clon, ccape = contour_grid(
        plot_lat,
        plot_lon,
        cape,
        stride=contour_stride,
        sigma=sigma_for_km(CAPE_SMOOTHING_KM),
    )
    with plt.rc_context({"hatch.color": "#aaaaaa", "hatch.linewidth": 0.26}):
        ax.contourf(
            clon,
            clat,
            ccape,
            levels=[500, 1000],
            colors="none",
            hatches=["/"],
            transform=DATA_CRS,
            zorder=20,
        )
    with plt.rc_context({"hatch.color": "#555555", "hatch.linewidth": 0.30}):
        ax.contourf(
            clon,
            clat,
            ccape,
            levels=[1000, 20000],
            colors="none",
            hatches=["xx"],
            transform=DATA_CRS,
            zorder=20,
        )
    clat, clon, cli = contour_grid(
        plot_lat,
        plot_lon,
        li,
        stride=contour_stride,
        sigma=sigma_for_km(LI_SMOOTHING_KM),
    )
    li_levels = [-6, -4, -2, 0]
    li_colors = ["#7b3294", "#d7191c", "#f28e2b", "black"]
    li_ct = ax.contour(
        clon,
        clat,
        cli,
        levels=li_levels,
        colors=li_colors,
        linewidths=LI_LINEWIDTHS,
        linestyles=["solid", "solid", "solid", "solid"],
        transform=DATA_CRS,
        zorder=22,
    )
    label_contours(li_ct, fontsize=5.8, fmt="%d", colors=li_colors)
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=np.arange(10, 52, 2), label="mm", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, "IPW(shaded,mm), LI(cntrd 0/-2/-4/-6), CAPE(hatch 500/1000J/kg)", run)

    # 3) 850-700 hPa RH, 850 hPa temperature, 850 hPa wind.
    ax = axes[2]
    rh = compute_layer_rh(run_dir, run, fhour, psfc_hpa, yslice, xslice)
    tmp850_c = crop(hour_file(run_dir, run, fhour, "TMP", "ISBL", "0850"), yslice, xslice) - 273.15
    u850 = crop(hour_file(run_dir, run, fhour, "UGRD", "ISBL", "0850"), yslice, xslice)
    v850 = crop(hour_file(run_dir, run, fhour, "VGRD", "ISBL", "0850"), yslice, xslice)
    u700 = crop(hour_file(run_dir, run, fhour, "UGRD", "ISBL", "0700"), yslice, xslice)
    v700 = crop(hour_file(run_dir, run, fhour, "VGRD", "ISBL", "0700"), yslice, xslice)
    u_panel, v_panel = terrain_adjusted_850_wind(u850, v850, u700, v700, psfc_hpa)
    cmap, norm, levels = make_rh_cmap()
    cf = ax.contourf(
        decimate(plot_lon, shade_stride),
        decimate(plot_lat, shade_stride),
        decimate(smooth_nan(rh, sigma_for_km(1.5)), shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    clat, clon, ctmp = contour_grid(
        plot_lat,
        plot_lon,
        tmp850_c,
        stride=contour_stride,
        sigma=sigma_for_km(TEMP850_SMOOTHING_KM),
    )
    standard_levels, zero_levels, warm_levels, hot_levels = temp850_contour_groups()
    temp_ct = ax.contour(
        clon,
        clat,
        ctmp,
        levels=standard_levels,
        colors="black",
        linewidths=TEMP850_STANDARD_LINEWIDTH,
        transform=DATA_CRS,
        zorder=22,
    )
    label_contours(temp_ct, fontsize=5.8, fmt="%d")
    zero_ct = ax.contour(
        clon,
        clat,
        ctmp,
        levels=zero_levels,
        colors=TEMP850_ZERO_COLOR,
        linewidths=TEMP850_ZERO_LINEWIDTH,
        transform=DATA_CRS,
        zorder=23,
    )
    label_contours(zero_ct, fontsize=5.8, fmt="%d", colors=TEMP850_ZERO_COLOR)
    warm_ct = ax.contour(
        clon,
        clat,
        ctmp,
        levels=warm_levels,
        colors=TEMP850_WARM_COLOR,
        linewidths=TEMP850_WARM_LINEWIDTH,
        transform=DATA_CRS,
        zorder=23,
    )
    label_contours(warm_ct, fontsize=5.8, fmt="%d", colors=TEMP850_WARM_COLOR)
    hot_ct = ax.contour(
        clon,
        clat,
        ctmp,
        levels=hot_levels,
        colors=TEMP850_HOT_COLOR,
        linewidths=TEMP850_HOT_LINEWIDTH,
        transform=DATA_CRS,
        zorder=23,
    )
    label_contours(hot_ct, fontsize=5.8, fmt="%d", colors=TEMP850_HOT_COLOR)
    plot_barbs(ax, plot_lon, plot_lat, u_panel, v_panel, barb_stride, color="black", row_density=2.0, column_density=2.0)
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=[10, 15, 20, 25, 30, 70, 75, 80, 85, 90], label="%", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, "85.0-70.0kPa RH(%,shaded), 85.0kPa Temp(C,cntrd), 85/70kPa Wind(hlf brb=10km/h)", run)

    # 4) Three-hour precipitation, MSLP, 10 m wind.
    ax = axes[3]
    precip = compute_precip_3h(run_dir, run, fhour, yslice, xslice)
    mslp_kpa = crop(hour_file(run_dir, run, fhour, "PRMSL", "MSL", "0"), yslice, xslice) / 1000.0
    u10 = crop(hour_file(run_dir, run, fhour, "UGRD", "TGL", "10"), yslice, xslice)
    v10 = crop(hour_file(run_dir, run, fhour, "VGRD", "TGL", "10"), yslice, xslice)
    terrain_cmap, terrain_norm, terrain_levels = make_terrain_cmap()
    terrain_smoothed = smooth_nan(terrain_m, sigma_for_km(1.5))
    terrain_land = np.where(terrain_m > 0.5, terrain_smoothed, np.nan)
    ax.contourf(
        decimate(plot_lon, shade_stride),
        decimate(plot_lat, shade_stride),
        decimate(terrain_land, shade_stride),
        levels=terrain_levels,
        cmap=terrain_cmap,
        norm=terrain_norm,
        extend="max",
        transform=DATA_CRS,
        transform_first=True,
        zorder=1,
    )
    cmap, norm, levels = make_precip_cmap()
    cf = ax.contourf(
        decimate(plot_lon, shade_stride),
        decimate(plot_lat, shade_stride),
        decimate(smooth_nan(precip, sigma_for_km(1.0)), shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    clat, clon, cmslp = contour_grid(
        plot_lat,
        plot_lon,
        mslp_kpa,
        stride=contour_stride,
        sigma=sigma_for_km(MSLP_SMOOTHING_KM),
    )
    minor_levels, major_levels, threshold_levels, high_levels = mslp_contour_groups()
    minor_mslp = ax.contour(
        clon,
        clat,
        cmslp,
        levels=minor_levels,
        colors="black",
        linewidths=1.0,
        transform=DATA_CRS,
        zorder=22,
    )
    label_contours(minor_mslp, fontsize=5.6, fmt="%.1f")
    major_mslp = ax.contour(
        clon,
        clat,
        cmslp,
        levels=major_levels,
        colors="black",
        linewidths=1.3,
        transform=DATA_CRS,
        zorder=23,
    )
    label_contours(major_mslp, fontsize=5.8, fmt="%.1f", colors="black")
    threshold_mslp = ax.contour(
        clon,
        clat,
        cmslp,
        levels=threshold_levels,
        colors=MSLP_BLUE,
        linewidths=1.3,
        transform=DATA_CRS,
        zorder=23,
    )
    label_contours(threshold_mslp, fontsize=5.8, fmt="%.1f", colors=MSLP_BLUE)
    high_mslp = ax.contour(
        clon,
        clat,
        cmslp,
        levels=high_levels,
        colors=MSLP_BLUE,
        linewidths=1.15,
        transform=DATA_CRS,
        zorder=23,
    )
    label_contours(high_mslp, fontsize=5.8, fmt="%.1f", colors=MSLP_BLUE)
    plot_barbs(ax, plot_lon, plot_lat, u10, v10, barb_stride, color="black", row_density=2.0, column_density=2.0)
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=[0.25, 2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 60, 80, 100], label="mm", fmt="%g")
    plot_style.add_fourpanel_text(
        ax,
        header,
        "Topo(brown), 3h Precip(shaded,mm), MSLP(cntrd,kPa), 10m Wind(hlf brb=10km/h)",
        run,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def make_plots(
    run: RunInfo,
    data_dir: Path,
    output_dir: Path,
    watershed_cache: Path,
    refresh_watersheds: bool,
    no_watersheds: bool,
    shade_stride: int,
    contour_stride: int,
    barb_stride: int,
    hours: Iterable[int] = FORECAST_HOURS,
) -> list[Path]:
    hours = tuple(int(hour) for hour in hours)
    if not hours:
        return []
    run_dir = data_dir / run.stamp
    sample_path = hour_file(run_dir, run, FORECAST_HOURS[0], "ABSV", "ISBL", "0500")
    _, lat, lon = read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read model coordinates.")
    extent = fourpanel_extent()
    yslice, xslice = subset_slices(lat, lon, extent)
    watersheds = (
        []
        if no_watersheds
        else load_watersheds(watershed_cache, refresh=refresh_watersheds, extent=extent)
    )
    transmission_lines = load_transmission_lines(extent)
    terrain_m = crop(
        hour_file(run_dir, run, hrdps.TERRAIN_FHOUR, "HGT", "SFC", "0"),
        yslice,
        xslice,
    )

    out_paths: list[Path] = []
    plot_dir = output_dir / run.stamp
    for fhour in hours:
        log(f"Plotting four-panel F{fhour:03d}.")
        out_path = plot_dir / f"{model_output_prefix('fourpanel')}_{run.stamp}_f{fhour:03d}.png"
        plot_fourpanel(
            out_path,
            run_dir,
            run,
            fhour,
            lat,
            lon,
            extent,
            yslice,
            xslice,
            terrain_m,
            watersheds,
            transmission_lines,
            shade_stride,
            contour_stride,
            barb_stride,
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
    parser.add_argument("--workers", type=int, default=10, help="Concurrent download workers.")
    parser.add_argument("--shade-stride", type=int, default=None, help="Grid stride for shaded fields.")
    parser.add_argument("--contour-stride", type=int, default=None, help="Grid stride for contours.")
    parser.add_argument("--barb-stride", type=int, default=None, help="Grid stride for wind barbs.")
    parser.add_argument("--hours", default=None, help="Comma-separated forecast hours to plot, e.g. 0,3,6.")
    parser.add_argument(
        "--watershed-cache",
        type=Path,
        default=hrdps.WATERSHED_CACHE,
        help="Local BC Hydro watershed boundary shapefile.",
    )
    parser.add_argument("--refresh-watersheds", action="store_true", help="Re-read the local watershed overlay.")
    parser.add_argument("--no-watersheds", action="store_true", help="Skip watershed overlays.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = set_model(args.model)
    data_dir = args.data_dir or Path(config.default_data_dir)
    output_dir = args.output_dir or Path(f"{config.default_output_dir}_fourpanel")
    shade_stride = args.shade_stride or grid_stride(5.0)
    contour_stride = args.contour_stride or grid_stride(12.0)
    barb_stride = args.barb_stride or grid_stride(27.0)
    if args.cycle != "latest" and args.cycle not in model_config().cycles:
        raise RuntimeError(f"{config.label} does not provide a {args.cycle}Z cycle.")
    if args.cycle == "latest":
        run = latest_complete_run()
    else:
        stamp = run_stamp_from_listing(args.cycle)
        if not stamp:
            raise RuntimeError(f"No files found for cycle {args.cycle}Z.")
        run = RunInfo(cycle=args.cycle, stamp=stamp, init_time=parse_stamp(stamp))
        if not run_is_complete(run):
            raise RuntimeError(f"Cycle {run.stamp} is not complete for the required four-panel fields.")

    log(f"Using {config.label} run {run.stamp}.")
    download_run(run, data_dir, args.workers)
    hours = FORECAST_HOURS if args.hours is None else tuple(int(item) for item in args.hours.split(",") if item.strip())
    make_plots(
        run,
        data_dir,
        output_dir,
        args.watershed_cache,
        args.refresh_watersheds,
        args.no_watersheds,
        shade_stride,
        contour_stride,
        barb_stride,
        hours,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
