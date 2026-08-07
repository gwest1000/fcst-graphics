#!/usr/bin/env python3
"""Render three-hourly HRDPS-West 10 m sustained-wind and gust graphics."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

import make_hrdps_fire_weather_twopanel as firewx
import make_hrdps_west_convective as hrdps
import make_hrdps_west_lightning as lightning
import plot_style


FORECAST_HOURS = tuple(range(0, 49, 3))
OUTPUT_PREFIX = "hrdps_west_wind_sw"
REGION_KEY = "sw"
REGION_LABEL = "SOUTHWEST BC"
REGION_EXTENT = firewx.REGIONAL_EXTENTS[REGION_KEY]
PLOT_FIGSIZE = (9.0, 9.0)

# Preserve the Fire Weather vector density and apparent pixel size on a
# single full-width panel. Fire Weather's vectors occupy a half-width panel.
ROW_DENSITY_MULTIPLIER = firewx.REGIONAL_VECTOR_ROW_DENSITY_MULTIPLIER
COLUMN_DENSITY_MULTIPLIER = firewx.REGIONAL_VECTOR_COLUMN_DENSITY_MULTIPLIER
VECTOR_SCALE = 2.0 * 68.0 / firewx.REGIONAL_VECTOR_SIZE_MULTIPLIER
SUSTAINED_VECTOR_WIDTH = 0.00145
GUST_VECTOR_WIDTH = 0.00380
VECTOR_EDGE_COLOR = "#202020"

COLORBAR_LAYOUT = {
    "backdrop": (0.951, 0.160, 0.048, 0.680),
    "cax_bounds": (0.974, 0.200, 0.015, 0.600),
    "tick_position": "left",
    "backdrop_edgecolor": "black",
    "backdrop_linewidth": 0.65,
}


def log(message: str) -> None:
    print(message, flush=True)


def set_model(model_key: str) -> hrdps.ModelConfig:
    if model_key != "west":
        raise ValueError("The 10 m wind product is only configured for HRDPS-West.")
    return hrdps.set_model(model_key)


def required_names(stamp: str, fhour: int) -> tuple[str, ...]:
    names = [
        hrdps.field_name("UGRD", "TGL", "10", stamp, fhour),
        hrdps.field_name("VGRD", "TGL", "10", stamp, fhour),
    ]
    names.extend(
        lightning.gust_field_name(stamp, hour)
        for hour in lightning.diagnostic_window_hours(fhour)
    )
    return tuple(names)


def _read_native_field(run_dir: Path, run: hrdps.RunInfo, fhour: int, variable: str) -> np.ndarray:
    path = lightning.hour_file(run_dir, run, fhour, variable, "TGL", "10")
    field, _, _ = hrdps.read_grib(path)
    return field.astype(np.float32, copy=False)


def read_wind_fields(
    run_dir: Path,
    run: hrdps.RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read valid-time sustained wind and maximum native gust for its 3-h block."""
    u10_ms = _read_native_field(run_dir, run, fhour, "UGRD")[yslice, xslice]
    v10_ms = _read_native_field(run_dir, run, fhour, "VGRD")[yslice, xslice]
    valid_wind = (
        np.isfinite(u10_ms)
        & np.isfinite(v10_ms)
        & (np.abs(u10_ms) < 150.0)
        & (np.abs(v10_ms) < 150.0)
    )
    u10_ms = np.where(valid_wind, u10_ms, np.nan).astype(np.float32)
    v10_ms = np.where(valid_wind, v10_ms, np.nan).astype(np.float32)
    sustained_kmh = (3.6 * np.hypot(u10_ms, v10_ms)).astype(np.float32)

    gust_stack = np.stack(
        [
            lightning.read_gust_kmh(run_dir, run, hour, yslice, xslice)
            for hour in lightning.diagnostic_window_hours(fhour)
        ]
    )
    finite_gust = np.isfinite(gust_stack)
    gust_kmh = np.max(np.where(finite_gust, gust_stack, -np.inf), axis=0)
    gust_kmh = np.where(np.any(finite_gust, axis=0), gust_kmh, np.nan).astype(np.float32)
    return u10_ms, v10_ms, sustained_kmh, gust_kmh


def plot_dual_wind_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    sustained_kmh: np.ndarray,
    gust_kmh: np.ndarray,
):
    """Draw gust-coloured outer arrows beneath sustained-coloured inner arrows."""
    row_density, column_density = lightning.gust_vector_density("bc")
    sample = plot_style.vector_sample_slices(
        ax,
        lon.shape,
        minimum=1,
        spacing_px=27.0,
        row_density=row_density * ROW_DENSITY_MULTIPLIER,
        column_density=column_density * COLUMN_DENSITY_MULTIPLIER,
    )
    u = u10_ms[sample]
    v = v10_ms[sample]
    speed_ms = np.hypot(u, v)
    sustained = sustained_kmh[sample]
    gust = gust_kmh[sample]
    finite = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(sustained)
        & np.isfinite(gust)
        & (speed_ms > 0.05)
    )
    unit_u = np.divide(u, speed_ms, out=np.zeros_like(u), where=speed_ms > 0.05)
    unit_v = np.divide(v, speed_ms, out=np.zeros_like(v), where=speed_ms > 0.05)
    x = lon[sample][finite]
    y = lat[sample][finite]
    unit_u = unit_u[finite]
    unit_v = unit_v[finite]
    cmap, norm, levels = lightning.gust_cmap()

    common = {
        "cmap": cmap,
        "norm": norm,
        "transform": lightning.DATA_CRS,
        "scale_units": "width",
        "scale": VECTOR_SCALE,
        "minlength": 0.05,
        "pivot": "middle",
    }
    outer = ax.quiver(
        x,
        y,
        unit_u,
        unit_v,
        gust[finite],
        width=GUST_VECTOR_WIDTH,
        headwidth=3.8,
        headlength=4.6,
        headaxislength=4.2,
        edgecolors=mcolors.to_rgba(VECTOR_EDGE_COLOR, 0.82),
        linewidths=0.20,
        zorder=11,
        **common,
    )
    ax.quiver(
        x,
        y,
        unit_u,
        unit_v,
        sustained[finite],
        width=SUSTAINED_VECTOR_WIDTH,
        headwidth=3.35,
        headlength=4.0,
        headaxislength=3.7,
        edgecolors="none",
        linewidths=0.0,
        zorder=12,
        **common,
    )
    return outer, levels


def render_wind(
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    sustained_kmh: np.ndarray,
    gust_kmh: np.ndarray,
    transmission_lines,
) -> Path:
    fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=lightning.PLOT_CRS)
    firewx.add_regional_panel_base(ax, REGION_EXTENT)
    gust_vectors, levels = plot_dual_wind_vectors(
        ax,
        lon,
        lat,
        u10_ms,
        v10_ms,
        sustained_kmh,
        gust_kmh,
    )
    lightning.add_transmission_lines(ax, transmission_lines)
    hrdps.add_city_labels(ax, fontsize=7.3, marker_size=2.2, path_width=2.35, zorder=30)

    # Use a standalone mappable so the shared scale is independent of which
    # arrow layer Matplotlib happens to register as the colourbar source.
    cmap, norm, _ = lightning.gust_cmap()
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    plot_style.add_internal_colorbar(
        fig,
        ax,
        mappable,
        ticks=levels,
        label="Wind speed (km h$^{-1}$)",
        title="km/h",
        fmt="%g",
        extend="both",
        **COLORBAR_LAYOUT,
    )
    del gust_vectors

    period = "valid-time gust" if fhour == 0 else "3-h maximum gust ending valid time"
    plot_style.add_single_panel_text(
        ax,
        plot_style.valid_header(run, fhour, "SW BC 10 m WIND"),
        f"Fill: valid-time sustained 10 m wind; border: HRDPS {period}; grey: BC transmission",
        run,
        source_text=f"ECCC | Init:{run.init_time:%Y%m%d%H}",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log(f"Wrote {out_path}.")
    return out_path


def make_plots(
    run: hrdps.RunInfo,
    data_dir: Path,
    output_dir: Path,
    hours: Iterable[int] = FORECAST_HOURS,
) -> list[Path]:
    if hrdps.model_config().key != "west":
        return []
    hours = tuple(sorted(set(int(hour) for hour in hours)))
    if not hours:
        return []

    run_dir = data_dir / run.stamp
    sample_path = lightning.hour_file(run_dir, run, hours[0], "UGRD", "TGL", "10")
    _, lat, lon = hrdps.read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError(f"Could not read HRDPS coordinates from {sample_path}.")
    yslice, xslice = hrdps.subset_slices(lat, lon, REGION_EXTENT)
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    transmission_lines = lightning.load_transmission_lines()
    plot_dir = output_dir / run.stamp

    outputs: list[Path] = []
    for fhour in hours:
        u10_ms, v10_ms, sustained_kmh, gust_kmh = read_wind_fields(
            run_dir,
            run,
            fhour,
            yslice,
            xslice,
        )
        out_path = plot_dir / f"{OUTPUT_PREFIX}_{run.stamp}_f{fhour:03d}.png"
        render_wind(
            out_path,
            run,
            fhour,
            plot_lat,
            plot_lon,
            u10_ms,
            v10_ms,
            sustained_kmh,
            gust_kmh,
            transmission_lines,
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--hours", type=parse_hours, default=FORECAST_HOURS)
    parser.add_argument("--data-dir", type=Path, default=Path(hrdps.MODEL_CONFIGS["west"].default_data_dir))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/hrdps_west_lightning"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    set_model("west")
    run = hrdps.RunInfo(
        cycle=args.stamp[9:11],
        stamp=args.stamp,
        init_time=hrdps.parse_stamp(args.stamp),
    )
    make_plots(run, args.data_dir, args.output_dir, args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
