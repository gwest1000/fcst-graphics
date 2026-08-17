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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

import ecmwf_ensemble_stats_data as stats_data
import plot_style
import project_paths
from make_ensemble_control_fourpanel import (
    ECMWF_FORECAST_HOURS,
    Field,
    latest_cycle_stamp,
    read_matching_grib,
)
from make_hrdps_west_convective import RunInfo, parse_stamp


FORECAST_HOURS = ECMWF_FORECAST_HOURS
DATA_CRS = ccrs.PlateCarree()
PLOT_CRS = ccrs.LambertConformal(
    central_longitude=-105.0,
    central_latitude=48.0,
    standard_parallels=(30.0, 60.0),
)
EXTENT = (-172.0, -42.0, 9.0, 82.0)
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
    longitude_order = np.argsort(field.lon[0, :])
    return Field(
        data=field.data[:, longitude_order],
        lat=field.lat[:, longitude_order],
        lon=field.lon[:, longitude_order],
        step_range=field.step_range,
    )


def read_fields(paths: stats_data.ArchivePaths, fhour: int) -> tuple[Field, Field]:
    mean = crop_field(
        read_matching_grib(paths.mean, "gh", "isobaricInhPa", 500, fhour)
    )
    spread = crop_field(
        read_matching_grib(paths.spread, "gh", "isobaricInhPa", 500, fhour)
    )
    mean.data = mean.data / 1000.0
    spread.data = spread.data / 1000.0
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
    paths: stats_data.ArchivePaths,
    output_dir: Path,
) -> Path:
    mean, spread = read_fields(paths, fhour)
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
        linewidths=0.40,
        transform=DATA_CRS,
        zorder=3,
    )
    heights = ax.contour(
        mean.lon,
        mean.lat,
        mean.data,
        levels=HEIGHT_LEVELS_KM,
        colors="black",
        linewidths=1.25,
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
    paths = stats_data.archive_paths(
        f"{run.init_time:%Y%m%d}",
        run.cycle,
        archive_root,
    )
    return [make_plot(run, int(fhour), paths, output_dir) for fhour in hours]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=None, help="Run stamp, e.g. 20260817T00Z.")
    parser.add_argument("--cycle", type=int, default=0, choices=[0, 12])
    parser.add_argument(
        "--hours",
        default=None,
        help="Comma-separated hours; defaults to the operational schedule through F252.",
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
