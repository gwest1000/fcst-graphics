#!/usr/bin/env python3
"""Plot ECMWF ENS 500 hPa ensemble-mean height and ensemble spread."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from eccodes import codes_get, codes_grib_new_from_file, codes_release
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

import ecmwf_ensemble_stats_data as stats_data
import plot_style
import project_paths
from make_ensemble_control_fourpanel import (
    Field,
    grib_field_matches,
    grib_values,
    latest_cycle_stamp,
)
from make_hrdps_west_convective import RunInfo, parse_stamp


ECMWF_THREE_HOURLY_END = 144
ECMWF_ENSEMBLE_FORECAST_END = 360
FORECAST_HOURS = (
    tuple(range(0, ECMWF_THREE_HOURLY_END + 1, 3))
    + tuple(range(ECMWF_THREE_HOURLY_END + 6, ECMWF_ENSEMBLE_FORECAST_END + 1, 6))
)
DATA_CRS = ccrs.PlateCarree()
REFERENCE_EXTENT = (-172.0, -42.0, 9.0, 82.0)
DOMAIN_SCALE = 0.85
DOMAIN_EAST_LONGITUDE = -52.0
DOMAIN_LONGITUDE_SPAN = (REFERENCE_EXTENT[1] - REFERENCE_EXTENT[0]) * DOMAIN_SCALE
DOMAIN_LATITUDE_SPAN = (REFERENCE_EXTENT[3] - REFERENCE_EXTENT[2]) * DOMAIN_SCALE
DOMAIN_LATITUDE_CENTER = (REFERENCE_EXTENT[2] + REFERENCE_EXTENT[3]) / 2.0
EXTENT = (
    DOMAIN_EAST_LONGITUDE - DOMAIN_LONGITUDE_SPAN,
    DOMAIN_EAST_LONGITUDE,
    DOMAIN_LATITUDE_CENTER - DOMAIN_LATITUDE_SPAN / 2.0,
    DOMAIN_LATITUDE_CENTER + DOMAIN_LATITUDE_SPAN / 2.0,
)
PLOT_CRS = ccrs.LambertConformal(
    central_longitude=(EXTENT[0] + EXTENT[1]) / 2.0,
    central_latitude=(EXTENT[2] + EXTENT[3]) / 2.0,
    standard_parallels=(30.0, 60.0),
)
PROJECTION_EDGE_SAMPLES = 721
SOURCE_COVERAGE_MARGIN_DEGREES = 2.0
MEAN_HEIGHT_SMOOTHING_SIGMA = 1.25
SPREAD_CONTOUR_LINEWIDTH = 0.48
HEIGHT_CONTOUR_LINEWIDTH = 1.50
GREEN_BLUE_BOUNDARY_KM = 0.05
GREEN_BLUE_BOUNDARY_LINEWIDTH = 1.80
HEIGHT_LEVELS_KM = np.arange(4.20, 6.421, 0.06)
SPREAD_LEVELS_KM = np.asarray(
    [0.005, *[value / 100.0 for value in range(1, 31)]],
    dtype=np.float64,
)
SPREAD_TICKS_KM = np.asarray(
    [0.005, *[value / 100.0 for value in range(2, 31, 2)]],
    dtype=np.float64,
)
SPREAD_TICK_LABELS = (
    "0.005",
    "0.02",
    "0.04",
    "0.06",
    "0.08",
    "0.1",
    "0.12",
    "0.14",
    "0.16",
    "0.18",
    "0.2",
    "0.22",
    "0.24",
    "0.26",
    "0.28",
    "0.3",
)
SPREAD_COLORS = (
    "#c9ffc9",
    "#7cff7c",
    "#64ff64",
    "#4aff4a",
    "#32ff32",
    "#7878ff",
    "#8d8dff",
    "#a1a1ff",
    "#b5b5ff",
    "#c9c9ff",
    "#ff97ff",
    "#ff7cff",
    "#ff64ff",
    "#ff4aff",
    "#ff32ff",
    "#ff4a4a",
    "#ff6464",
    "#ff7c7c",
    "#ff9797",
    "#ffafaf",
    "#ffff64",
    "#ffff4a",
    "#ffff32",
    "#ffff18",
    "#ffff00",
    "#b0b0b0",
    "#cdcdce",
    "#d8d9d8",
    "#e9e9e9",
    "#efefef",
)


def centered_longitudes(longitudes: np.ndarray, center: float) -> np.ndarray:
    return center + np.mod(longitudes - center + 180.0, 360.0) - 180.0


def source_coverage_extent() -> tuple[float, float, float, float]:
    """Return the geographic data bounds needed to fill the projected plot box."""
    west, east, south, north = EXTENT
    vertical = np.linspace(south, north, PROJECTION_EDGE_SAMPLES)
    horizontal = np.linspace(west, east, PROJECTION_EDGE_SAMPLES)
    boundary_lon = np.concatenate(
        (
            np.full_like(vertical, west),
            np.full_like(vertical, east),
            horizontal,
            horizontal,
        )
    )
    boundary_lat = np.concatenate(
        (
            vertical,
            vertical,
            np.full_like(horizontal, south),
            np.full_like(horizontal, north),
        )
    )
    projected = PLOT_CRS.transform_points(DATA_CRS, boundary_lon, boundary_lat)
    projected = projected[np.isfinite(projected).all(axis=1)]
    x0, x1 = np.min(projected[:, 0]), np.max(projected[:, 0])
    y0, y1 = np.min(projected[:, 1]), np.max(projected[:, 1])

    edge = np.linspace(0.0, 1.0, PROJECTION_EDGE_SAMPLES)
    projected_x = np.concatenate(
        (
            x0 + (x1 - x0) * edge,
            x0 + (x1 - x0) * edge,
            np.full_like(edge, x0),
            np.full_like(edge, x1),
        )
    )
    projected_y = np.concatenate(
        (
            np.full_like(edge, y0),
            np.full_like(edge, y1),
            y0 + (y1 - y0) * edge,
            y0 + (y1 - y0) * edge,
        )
    )
    geographic = DATA_CRS.transform_points(PLOT_CRS, projected_x, projected_y)
    geographic = geographic[np.isfinite(geographic).all(axis=1)]
    center = (west + east) / 2.0
    longitude = centered_longitudes(geographic[:, 0], center)
    latitude = geographic[:, 1]
    margin = SOURCE_COVERAGE_MARGIN_DEGREES
    return (
        float(np.min(longitude) - margin),
        float(np.max(longitude) + margin),
        float(max(-90.0, np.min(latitude) - margin)),
        float(min(90.0, np.max(latitude) + margin)),
    )


SOURCE_EXTENT = source_coverage_extent()
OUTPUT_PREFIX = "ecmwf_ensemble_spread_500"
DEFAULT_OUTPUT_DIR = project_paths.plot_path("ecmwf_ensemble_spread")


def log(message: str) -> None:
    print(message, flush=True)


def spread_cmap_norm() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(SPREAD_COLORS, name="reference_ensemble_spread")
    cmap.set_under("white")
    cmap.set_over("white")
    return cmap, BoundaryNorm(SPREAD_LEVELS_KM, cmap.N)


def crop_field(field: Field) -> Field:
    domain_center = (EXTENT[0] + EXTENT[1]) / 2.0
    centered_lon = centered_longitudes(field.lon, domain_center)
    longitude_order = np.argsort(centered_lon[0, :])
    data = field.data[:, longitude_order]
    lat = field.lat[:, longitude_order]
    lon = centered_lon[:, longitude_order]
    west, east, south, north = SOURCE_EXTENT
    row_indices = np.flatnonzero(
        (lat[:, 0] >= south)
        & (lat[:, 0] <= north)
    )
    column_indices = np.flatnonzero(
        (lon[0, :] >= west)
        & (lon[0, :] <= east)
    )
    if not row_indices.size or not column_indices.size:
        raise ValueError("ECMWF field does not overlap the ensemble-spread plotting extent.")
    yslice = slice(int(row_indices[0]), int(row_indices[-1]) + 1)
    xslice = slice(int(column_indices[0]), int(column_indices[-1]) + 1)
    return Field(
        data=data[yslice, xslice],
        lat=lat[yslice, xslice],
        lon=lon[yslice, xslice],
        step_range=field.step_range,
    )


def smooth_mean_height(data: np.ndarray) -> np.ndarray:
    valid = np.isfinite(data)
    if valid.all():
        return gaussian_filter(data, sigma=MEAN_HEIGHT_SMOOTHING_SIGMA, mode="nearest")
    weights = gaussian_filter(valid.astype(np.float32), sigma=MEAN_HEIGHT_SMOOTHING_SIGMA, mode="nearest")
    values = gaussian_filter(np.where(valid, data, 0.0), sigma=MEAN_HEIGHT_SMOOTHING_SIGMA, mode="nearest")
    return np.where(weights > 0.0, values / weights, np.nan)


def read_field_set(
    path: Path,
    hours: Iterable[int],
    *,
    smooth: bool,
) -> dict[int, Field]:
    requested = {int(hour) for hour in hours}
    fields: dict[int, Field] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                step = int(codes_get(gid, "step"))
                if step not in requested or not grib_field_matches(
                    gid,
                    "gh",
                    "isobaricInhPa",
                    500,
                    step,
                ):
                    continue
                data, lat, lon = grib_values(gid)
                field = crop_field(
                    Field(
                        data=data,
                        lat=lat,
                        lon=lon,
                        step_range=str(codes_get(gid, "stepRange")),
                    )
                )
                field.data = field.data / 1000.0
                if smooth:
                    field.data = smooth_mean_height(field.data)
                fields[step] = field
            finally:
                codes_release(gid)
    missing = requested.difference(fields)
    if missing:
        raise KeyError(f"Missing ECMWF 500 hPa fields at forecast hours {sorted(missing)} in {path}")
    return fields


def read_fields(paths: stats_data.ArchivePaths, fhour: int) -> tuple[Field, Field]:
    mean = read_field_set(paths.mean, (fhour,), smooth=True)[fhour]
    spread = read_field_set(paths.spread, (fhour,), smooth=False)[fhour]
    return mean, spread


def add_map_base(ax: plt.Axes) -> None:
    ax.set_extent(EXTENT, crs=DATA_CRS)
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.75, zorder=7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.75, zorder=7)
    provinces = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_1_states_provinces_lines",
        "50m",
        facecolor="none",
    )
    ax.add_feature(provinces, edgecolor="black", linewidth=0.55, zorder=7)
    gridlines = ax.gridlines(
        crs=DATA_CRS,
        xlocs=np.arange(-160.0, -39.0, 20.0),
        ylocs=np.arange(20.0, 81.0, 10.0),
        draw_labels=False,
        linewidth=0.65,
        color="black",
        alpha=0.72,
        linestyle="-",
        zorder=6,
    )
    gridlines.x_inline = False
    gridlines.y_inline = False


def add_spread_colorbar(fig: plt.Figure, ax: plt.Axes, mappable) -> None:
    ax.add_patch(
        Rectangle(
            (0.925, 0.045),
            0.074,
            0.91,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=34,
        )
    )
    cax = ax.inset_axes([0.946, 0.083, 0.017, 0.83])
    cax.set_zorder(50)
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        ticks=SPREAD_TICKS_KM,
        extend="both",
    )
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.tick_params(labelsize=10.0, length=3.0, width=0.7, pad=2.4)
    cbar.ax.set_yticklabels(SPREAD_TICK_LABELS)
    cbar.ax.text(
        1.55,
        -0.055,
        "km",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="black",
    )


def image_name(stamp: str, fhour: int) -> str:
    return f"{OUTPUT_PREFIX}_{stamp}_f{fhour:03d}.png"


def make_plot(
    run: RunInfo,
    fhour: int,
    mean: Field,
    spread: Field,
    output_dir: Path,
) -> Path:
    cmap, norm = spread_cmap_norm()

    fig = plt.figure(
        figsize=plot_style.PLOT_FIGSIZE,
        dpi=plot_style.PLOT_DPI,
        facecolor="white",
    )
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=PLOT_CRS)
    add_map_base(ax)
    shading = ax.contourf(
        spread.lon,
        spread.lat,
        spread.data,
        levels=SPREAD_LEVELS_KM,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        antialiased=False,
        zorder=1,
    )
    ax.contour(
        spread.lon,
        spread.lat,
        spread.data,
        levels=SPREAD_LEVELS_KM[1:],
        colors="white",
        linewidths=SPREAD_CONTOUR_LINEWIDTH,
        transform=DATA_CRS,
        zorder=3,
    )
    ax.contour(
        spread.lon,
        spread.lat,
        spread.data,
        levels=[GREEN_BLUE_BOUNDARY_KM],
        colors="#707070",
        linewidths=GREEN_BLUE_BOUNDARY_LINEWIDTH,
        transform=DATA_CRS,
        zorder=4,
    )
    heights = ax.contour(
        mean.lon,
        mean.lat,
        mean.data,
        levels=HEIGHT_LEVELS_KM,
        colors="black",
        linewidths=HEIGHT_CONTOUR_LINEWIDTH,
        transform=DATA_CRS,
        zorder=8,
    )
    ax.clabel(
        heights,
        levels=heights.levels,
        fmt="%.2f",
        inline=True,
        inline_spacing=2,
        fontsize=10.5,
        colors="black",
        zorder=9,
    )
    add_spread_colorbar(fig, ax, shading)
    plot_style.add_text_bands(
        ax,
        plot_style.valid_header(run, fhour, model_label="ECMWF ENS"),
        "500 hPa Ensemble-Mean GeoHgt (contoured, km) and Std Dev (shaded, km)",
        f"Data: ECMWF | Init:{run.init_time:%Y%m%d%H}",
        header_fontsize=14.0,
        footer_fontsize=12.0,
        source_fontsize=9.5,
        header_height=0.048,
        footer_height=0.041,
    )

    target = output_dir / run.stamp / image_name(run.stamp, fhour)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    log(f"Wrote {target}.")
    return target


def make_plots(
    run: RunInfo,
    archive_root: Path,
    output_dir: Path,
    hours: Iterable[int],
) -> list[Path]:
    requested = tuple(int(fhour) for fhour in hours)
    paths = stats_data.archive_paths(
        f"{run.init_time:%Y%m%d}",
        run.cycle,
        archive_root,
    )
    mean_fields = read_field_set(paths.mean, requested, smooth=True)
    spread_fields = read_field_set(paths.spread, requested, smooth=False)
    return [
        make_plot(
            run,
            fhour,
            mean_fields[fhour],
            spread_fields[fhour],
            output_dir,
        )
        for fhour in requested
    ]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=None, help="Run stamp, e.g. 20260817T00Z.")
    parser.add_argument("--cycle", type=int, default=0, choices=[0, 12])
    parser.add_argument(
        "--hours",
        default=None,
        help="Comma-separated hours; defaults to the operational schedule through F360.",
    )
    parser.add_argument("--archive-root", type=Path, default=stats_data.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    stamp = args.stamp or latest_cycle_stamp(args.cycle)
    init_time = parse_stamp(stamp)
    run = RunInfo(cycle=f"{init_time.hour:02d}", stamp=stamp, init_time=init_time)
    hours = (
        FORECAST_HOURS
        if args.hours is None
        else tuple(int(value) for value in args.hours.split(",") if value.strip())
    )
    if args.download or args.force_download:
        stats_data.ensure_archives(
            f"{init_time:%Y%m%d}",
            run.cycle,
            hours,
            archive_root=args.archive_root,
            force=args.force_download,
        )
    make_plots(run, args.archive_root, args.output_dir, hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
