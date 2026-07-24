#!/usr/bin/env python3
"""Render three-hourly HRDPS 2 m temperature graphics."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.base import BaseGeometry

import make_hrdps_west_convective as hrdps
import plot_style
from make_hrdps_west_fourpanel import smooth_nan

FORECAST_HOURS = tuple(range(0, 49, 3))
DATA_CRS = hrdps.DATA_CRS
PLOT_CRS = hrdps.PANEL_PROJ
WATERSHED_CACHE = hrdps.WATERSHED_CACHE

# Reproduces the operational reference palette at 2 C intervals. The unusual
# cold-end warm colors are intentional and are part of that established scale.
TEMPERATURE_LEVELS_C = tuple(range(-58, 50, 2))
TEMPERATURE_COLORS = (
    "#000000",
    "#444343",
    "#787877",
    "#b0b0b0",
    "#ffc300",
    "#ffa500",
    "#ff8700",
    "#f16400",
    "#c93c00",
    "#e10000",
    "#ff1818",
    "#fe4848",
    "#fe7b7b",
    "#e9b1aa",
    "#fd7bfd",
    "#ff4aff",
    "#fd1afd",
    "#e100e1",
    "#ac00ac",
    "#0505b4",
    "#1717f0",
    "#1717f0",
    "#7979ff",
    "#7979ff",
    "#7cff7c",
    "#4bfd4b",
    "#1df218",
    "#00e100",
    "#00ab00",
    "#adad06",
    "#e2e20a",
    "#ffff18",
    "#ffff4a",
    "#ffff7c",
    "#ffc300",
    "#ffa500",
    "#ff8700",
    "#f16400",
    "#c93c00",
    "#af0000",
    "#e10000",
    "#ff1818",
    "#fe4848",
    "#fe7b7b",
    "#fd7bfd",
    "#ff4aff",
    "#fd1afd",
    "#e100e1",
    "#ac00ac",
    "#444343",
    "#787877",
    "#b0b0b0",
    "#dadada",
)
TEMPERATURE_TICKS_C = tuple(range(-56, 49, 4))
ISOTHERM_LEVELS_C = tuple(range(-60, 51, 10))
ISOTHERM_COLOR = "#555555"
ISOTHERM_LINEWIDTH = 0.78
ISOTHERM_SMOOTHING_KM = 1.5
SHADE_TARGET_KM = 1.0
CONTOUR_TARGET_KM = 2.5

REGION_KEYS_BY_MODEL = {
    "west": ("south", "north"),
    "continental": ("bc",),
}


@dataclass(frozen=True)
class RegionConfig:
    key: str
    label: str
    extent: tuple[float, float, float, float]


TEMPERATURE_REGIONS = {
    "bc": RegionConfig("bc", "BC", (-138.2, -108.03, 47.55, 60.0)),
    "south": RegionConfig("south", "Southern BC", (-129.5, -113.1, 48.0, 54.08)),
    "north": RegionConfig("north", "Northern BC", (-133.06, -112.44, 51.7, 59.2)),
}


def log(message: str) -> None:
    print(message, flush=True)


def set_model(model_key: str) -> hrdps.ModelConfig:
    return hrdps.set_model(model_key)


def region_keys_for_model(model_key: str) -> tuple[str, ...]:
    try:
        return REGION_KEYS_BY_MODEL[model_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported temperature model: {model_key}") from exc


def output_prefix(region_key: str) -> str:
    prefix = hrdps.model_output_prefix("temperature")
    return prefix if region_key == "bc" else f"{prefix}_{region_key}"


def region_config(region_key: str) -> RegionConfig:
    try:
        return TEMPERATURE_REGIONS[region_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported temperature region: {region_key}") from exc


def required_names(stamp: str, fhour: int) -> tuple[str, ...]:
    return (hrdps.field_name("TMP", "TGL", "2", stamp, fhour),)


def temperature_cmap() -> tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    if len(TEMPERATURE_COLORS) != len(TEMPERATURE_LEVELS_C) - 1:
        raise RuntimeError("Temperature palette must have exactly one color per interval.")
    cmap = mcolors.ListedColormap(TEMPERATURE_COLORS, name="operational_temperature")
    cmap.set_under("#000000")
    cmap.set_over("#ffffff")
    return cmap, mcolors.BoundaryNorm(TEMPERATURE_LEVELS_C, cmap.N)


def add_full_canvas_text(
    fig: plt.Figure,
    run: hrdps.RunInfo,
    fhour: int,
    region_label: str,
) -> None:
    overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    overlay.patch.set_alpha(0.0)
    overlay.set_axis_off()
    overlay.set_zorder(100)
    plot_style.add_single_panel_text(
        overlay,
        plot_style.valid_header(run, fhour, model_label=hrdps.model_config().label),
        f"2 m temperature (shaded, °C); grey isotherms every 10°C; blue BCH watersheds; {region_label}",
        run,
        source_label=hrdps.model_config().source_label,
    )


def render_temperature(
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    temperature_c: np.ndarray,
    region_key: str,
    watersheds: list[BaseGeometry],
    shade_stride: int,
    contour_stride: int,
) -> None:
    region = region_config(region_key)
    extent = region.extent
    yslice, xslice = hrdps.subset_slices(lat, lon, extent)
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    plot_temperature = temperature_c[yslice, xslice]

    cmap, norm = temperature_cmap()
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=PLOT_CRS)
    hrdps.add_base_features(ax, extent=extent)

    shade_sample = (slice(None, None, shade_stride), slice(None, None, shade_stride))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shaded = ax.contourf(
            plot_lon[shade_sample],
            plot_lat[shade_sample],
            plot_temperature[shade_sample],
            levels=TEMPERATURE_LEVELS_C,
            cmap=cmap,
            norm=norm,
            extend="both",
            antialiased=False,
            transform=DATA_CRS,
            transform_first=True,
            zorder=3,
        )

    hrdps.add_hydro_features(ax)
    hrdps.add_watersheds(ax, watersheds)

    contour_sample = (slice(None, None, contour_stride), slice(None, None, contour_stride))
    contour_temperature = smooth_nan(
        plot_temperature[contour_sample],
        sigma=hrdps.sigma_for_km(ISOTHERM_SMOOTHING_KM) / contour_stride,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isotherms = ax.contour(
            plot_lon[contour_sample],
            plot_lat[contour_sample],
            contour_temperature,
            levels=ISOTHERM_LEVELS_C,
            colors=ISOTHERM_COLOR,
            linewidths=ISOTHERM_LINEWIDTH,
            transform=DATA_CRS,
            transform_first=True,
            zorder=26,
        )
        hrdps.label_contours(isotherms, fontsize=6.8, fmt="%d", colors=ISOTHERM_COLOR)

    city_fontsize = 7.2 if region_key != "bc" else 6.8
    hrdps.add_city_labels(
        ax,
        fontsize=city_fontsize,
        marker_size=2.2,
        path_width=2.35,
        zorder=30,
    )
    plot_style.add_internal_colorbar(
        fig,
        ax,
        shaded,
        ticks=TEMPERATURE_TICKS_C,
        label="°C",
        title="2 m TEMP",
        fmt="%g",
        extend="both",
    )
    add_full_canvas_text(fig, run, fhour, region.label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def make_plots(
    run: hrdps.RunInfo,
    data_dir: Path,
    output_dir: Path,
    hours: Iterable[int] = FORECAST_HOURS,
    region_keys: Iterable[str] | None = None,
    shade_stride: int | None = None,
    contour_stride: int | None = None,
    watershed_cache: Path = WATERSHED_CACHE,
    no_watersheds: bool = False,
) -> list[Path]:
    hours = tuple(sorted(set(int(hour) for hour in hours)))
    region_keys = tuple(region_keys or region_keys_for_model(hrdps.model_config().key))
    if not hours or not region_keys:
        return []

    shade_stride = shade_stride or hrdps.grid_stride(SHADE_TARGET_KM)
    contour_stride = contour_stride or hrdps.grid_stride(CONTOUR_TARGET_KM)
    regions = tuple(region_config(key) for key in region_keys)
    extents = tuple(region.extent for region in regions)
    union_extent = (
        min(extent[0] for extent in extents),
        max(extent[1] for extent in extents),
        min(extent[2] for extent in extents),
        max(extent[3] for extent in extents),
    )
    watersheds = (
        []
        if no_watersheds
        else hrdps.load_watersheds(watershed_cache, extent=union_extent)
    )

    run_dir = data_dir / run.stamp
    coordinate_hour = hours[0]
    sample_path = run_dir / f"{coordinate_hour:03d}" / required_names(run.stamp, coordinate_hour)[0]
    _, lat, lon = hrdps.read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError(f"Could not read HRDPS coordinates from {sample_path}.")

    plot_dir = output_dir / run.stamp
    outputs: list[Path] = []
    for fhour in hours:
        source_path = run_dir / f"{fhour:03d}" / required_names(run.stamp, fhour)[0]
        temperature_k, _, _ = hrdps.read_grib(source_path)
        temperature_c = temperature_k.astype(np.float32, copy=False) - 273.15
        temperature_c = np.where(
            np.isfinite(temperature_c) & (temperature_c >= -100.0) & (temperature_c <= 70.0),
            temperature_c,
            np.nan,
        ).astype(np.float32)
        for region in regions:
            out_path = plot_dir / f"{output_prefix(region.key)}_{run.stamp}_f{fhour:03d}.png"
            log(f"Rendering {hrdps.model_config().label} {region.label} temperature F{fhour:03d}.")
            render_temperature(
                out_path,
                run,
                fhour,
                lat,
                lon,
                temperature_c,
                region.key,
                watersheds,
                shade_stride,
                contour_stride,
            )
            outputs.append(out_path)
    return outputs


def parse_hours(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return FORECAST_HOURS
    hours = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    invalid = [hour for hour in hours if hour not in FORECAST_HOURS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported forecast hours: {invalid}")
    return hours


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(REGION_KEYS_BY_MODEL), required=True)
    parser.add_argument("--run", required=True, help="Run stamp, for example 20260724T12Z.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hours", type=parse_hours, default=FORECAST_HOURS)
    parser.add_argument("--regions", default=None, help="Comma-separated region keys; defaults by model.")
    parser.add_argument("--shade-stride", type=int, default=None)
    parser.add_argument("--contour-stride", type=int, default=None)
    parser.add_argument("--no-watersheds", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = set_model(args.model)
    run = hrdps.RunInfo(
        cycle=args.run[9:11],
        stamp=args.run,
        init_time=hrdps.parse_stamp(args.run),
    )
    region_keys = (
        tuple(item.strip() for item in args.regions.split(",") if item.strip())
        if args.regions
        else region_keys_for_model(args.model)
    )
    outputs = make_plots(
        run,
        args.data_dir or Path(config.default_data_dir),
        args.output_dir or Path(f"{config.default_output_dir}_temperature"),
        args.hours,
        region_keys,
        args.shade_stride,
        args.contour_stride,
        no_watersheds=args.no_watersheds,
    )
    log(f"Rendered {len(outputs)} temperature frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
