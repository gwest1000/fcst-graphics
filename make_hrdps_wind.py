#!/usr/bin/env python3
"""Render three-hourly HRDPS-West South Coast wind and gust graphics."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import make_hrdps_fire_weather_twopanel as firewx
import make_hrdps_west_convective as hrdps
import make_hrdps_west_fourpanel as fourpanel
import make_hrdps_west_lightning as lightning
import plot_style


FORECAST_HOURS = tuple(range(0, 49, 3))
OUTPUT_PREFIX = "hrdps_west_wind_south_coast"
REGION_LABEL = "SOUTH COAST"
REGION_EXTENT = (-129.3, -119.5, 47.6, 51.2)
PLOT_FIGSIZE = (14.4, 9.0)

# Preserve the regional Fire Weather spacing, with 50% more rows for the
# detail requested on this narrower South Coast domain.
ROW_DENSITY_MULTIPLIER = firewx.REGIONAL_VECTOR_ROW_DENSITY_MULTIPLIER * 1.50
COLUMN_DENSITY_MULTIPLIER = firewx.REGIONAL_VECTOR_COLUMN_DENSITY_MULTIPLIER
VECTOR_SPACING_PX = 27.0

# Both polygons are defined in display pixels. The inner polygon leaves an
# exact two-pixel gust border around the sustained-wind arrow.
OUTER_ARROW_SHAPE = np.array(
    [
        (-8.5, -2.5),
        (0.5, -2.5),
        (0.5, -4.0),
        (8.5, 0.0),
        (0.5, 4.0),
        (0.5, 2.5),
        (-8.5, 2.5),
    ],
    dtype=np.float32,
)
INNER_ARROW_SHAPE = np.array(
    [
        (-6.5, -0.5),
        (1.5, -0.5),
        (1.5, -2.0),
        (6.5, 0.0),
        (1.5, 2.0),
        (1.5, 0.5),
        (-6.5, 0.5),
    ],
    dtype=np.float32,
)

# The accepted lower-left colorbar, with a slightly narrower backdrop and
# exactly 50% more vertical extent than the preceding preview.
COLORBAR_LAYOUT = {
    "backdrop": (0.001, 0.028, 0.070, 0.632),
    "cax_bounds": (0.018, 0.073, 0.017, 0.503),
    "tick_position": "right",
    "backdrop_edgecolor": "black",
    "backdrop_linewidth": 0.65,
}


def log(message: str) -> None:
    print(message, flush=True)


def set_model(model_key: str) -> hrdps.ModelConfig:
    if model_key != "west":
        raise ValueError("The 10 m wind product is only configured for HRDPS-West.")
    return hrdps.set_model(model_key)


def _read_native_field(
    run_dir: Path,
    run: hrdps.RunInfo,
    fhour: int,
    variable: str,
) -> np.ndarray:
    path = lightning.hour_file(run_dir, run, fhour, variable, "TGL", "10")
    field, _, _ = hrdps.read_grib(path)
    return field.astype(np.float32, copy=False)


def read_valid_wind_fields(
    run_dir: Path,
    run: hrdps.RunInfo,
    fhour: int,
    yslice: slice,
    xslice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read valid-time 10 m components and sustained speed."""
    u10_ms = _read_native_field(run_dir, run, fhour, "UGRD")[yslice, xslice]
    v10_ms = _read_native_field(run_dir, run, fhour, "VGRD")[yslice, xslice]
    valid = (
        np.isfinite(u10_ms)
        & np.isfinite(v10_ms)
        & (np.abs(u10_ms) < 150.0)
        & (np.abs(v10_ms) < 150.0)
    )
    u10_ms = np.where(valid, u10_ms, np.nan).astype(np.float32)
    v10_ms = np.where(valid, v10_ms, np.nan).astype(np.float32)
    sustained_kmh = (3.6 * np.hypot(u10_ms, v10_ms)).astype(np.float32)
    return u10_ms, v10_ms, sustained_kmh


def _sampled_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    sustained_kmh: np.ndarray,
    gust_kmh: np.ndarray,
) -> tuple[np.ndarray, ...]:
    row_density, column_density = lightning.gust_vector_density("bc")
    sample = plot_style.vector_sample_slices(
        ax,
        lon.shape,
        minimum=1,
        spacing_px=VECTOR_SPACING_PX,
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
    return (
        lon[sample][finite],
        lat[sample][finite],
        unit_u[finite],
        unit_v[finite],
        sustained[finite],
        gust[finite],
    )


def plot_dual_wind_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    sustained_kmh: np.ndarray,
    gust_kmh: np.ndarray,
) -> None:
    """Draw fixed-size sustained arrows inside all-cause gust borders."""
    x, y, unit_u, unit_v, sustained, gust = _sampled_vectors(
        ax,
        lon,
        lat,
        u10_ms,
        v10_ms,
        sustained_kmh,
        gust_kmh,
    )
    ax.figure.canvas.draw()
    centres_data = lightning.PLOT_CRS.transform_points(lightning.DATA_CRS, x, y)[:, :2]
    cos_lat = np.maximum(0.2, np.cos(np.deg2rad(y)))
    direction_data = lightning.PLOT_CRS.transform_points(
        lightning.DATA_CRS,
        x + 0.02 * unit_u / cos_lat,
        y + 0.02 * unit_v,
    )[:, :2]
    centres_px = ax.transData.transform(centres_data)
    direction_px = ax.transData.transform(direction_data) - centres_px
    lengths = np.hypot(direction_px[:, 0], direction_px[:, 1])
    direction_px = np.divide(
        direction_px,
        lengths[:, None],
        out=np.zeros_like(direction_px),
        where=lengths[:, None] > 0.0,
    )

    inverse = ax.transData.inverted()
    outer_patches: list[Polygon] = []
    inner_patches: list[Polygon] = []
    for centre, direction in zip(centres_px, direction_px, strict=True):
        perpendicular = np.array((-direction[1], direction[0]))

        def vertices(shape: np.ndarray) -> np.ndarray:
            pixels = (
                centre[None, :]
                + shape[:, :1] * direction[None, :]
                + shape[:, 1:] * perpendicular[None, :]
            )
            return inverse.transform(pixels)

        outer_patches.append(Polygon(vertices(OUTER_ARROW_SHAPE), closed=True))
        inner_patches.append(Polygon(vertices(INNER_ARROW_SHAPE), closed=True))

    cmap, norm, _ = lightning.gust_cmap()
    outer_collection = PatchCollection(
        outer_patches,
        facecolors=cmap(norm(gust)),
        edgecolors="none",
        linewidths=0.0,
        zorder=11,
    )
    inner_collection = PatchCollection(
        inner_patches,
        facecolors=cmap(norm(sustained)),
        edgecolors="none",
        linewidths=0.0,
        zorder=12,
    )
    outer_collection.set_transform(ax.transData)
    inner_collection.set_transform(ax.transData)
    ax.add_collection(outer_collection)
    ax.add_collection(inner_collection)


def add_terrain_background(
    ax: plt.Axes,
    lat: np.ndarray,
    lon: np.ndarray,
    terrain_m: np.ndarray,
) -> None:
    """Use the lower-right convective four-panel terrain treatment."""
    cmap, norm, levels = fourpanel.make_terrain_cmap()
    terrain_smoothed = fourpanel.smooth_nan(terrain_m, fourpanel.sigma_for_km(1.5))
    terrain_land = np.where(terrain_m > 0.5, terrain_smoothed, np.nan)
    stride = max(1, hrdps.grid_stride(2.0))
    sample = (slice(None, None, stride), slice(None, None, stride))
    ax.contourf(
        lon[sample],
        lat[sample],
        terrain_land[sample],
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=1,
    )


def _wind_header(run: hrdps.RunInfo, fhour: int) -> str:
    valid = run.init_time + dt.timedelta(hours=fhour)
    valid_local = valid.astimezone(plot_style.LOCAL_TZ)
    local_text = valid_local.strftime("%a %H:%M%Z %d%b%Y").upper()
    utc_text = valid.strftime("%H:%MUTC %d%b%Y").upper()
    return (
        f"HRDPS-West 1-km  |  F{fhour:03d}  |  "
        f"{local_text}  |  {utc_text}"
    )


def render_wind(
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    terrain_m: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    sustained_kmh: np.ndarray,
    gust_kmh: np.ndarray,
    transmission_lines,
) -> Path:
    fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=lightning.PLOT_CRS)
    firewx.add_regional_panel_base(ax, REGION_EXTENT)
    add_terrain_background(ax, lat, lon, terrain_m)
    plot_dual_wind_vectors(
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
    ax.plot(
        -122.20,
        47.98,
        "o",
        ms=2.2,
        color="#232323",
        transform=lightning.DATA_CRS,
        zorder=30,
    )
    everett = ax.text(
        -122.07,
        48.06,
        "Everett",
        transform=lightning.DATA_CRS,
        fontsize=7.3,
        color="#202020",
        zorder=30,
    )
    everett.set_path_effects(
        [path_effects.withStroke(linewidth=2.35, foreground="white", alpha=0.9)]
    )

    cmap, norm, levels = lightning.gust_cmap()
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
    plot_style.add_single_panel_text(
        ax,
        _wind_header(run, fhour),
        "10-m Wind (arrow fill) and Gust (arrow outline)",
        run,
        source_text="Data: ECCC",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log(f"Wrote {out_path}.")
    return out_path


def render_from_all_cause(
    run: hrdps.RunInfo,
    fhour: int,
    run_dir: Path,
    output_dir: Path,
    base_lat: np.ndarray,
    base_lon: np.ndarray,
    base_yslice: slice,
    base_xslice: slice,
    terrain_m: np.ndarray,
    all_cause_gust_kmh: np.ndarray,
    transmission_lines,
) -> Path:
    """Render one frame from fields already computed by the Fire Weather worker."""
    yslice, xslice = hrdps.subset_slices(base_lat, base_lon, REGION_EXTENT)
    u10_ms, v10_ms, sustained_kmh = read_valid_wind_fields(
        run_dir,
        run,
        fhour,
        base_yslice,
        base_xslice,
    )
    out_path = output_dir / run.stamp / f"{OUTPUT_PREFIX}_{run.stamp}_f{fhour:03d}.png"
    return render_wind(
        out_path,
        run,
        fhour,
        base_lat[yslice, xslice],
        base_lon[yslice, xslice],
        terrain_m[yslice, xslice],
        u10_ms[yslice, xslice],
        v10_ms[yslice, xslice],
        sustained_kmh[yslice, xslice],
        all_cause_gust_kmh[yslice, xslice],
        transmission_lines,
    )


def make_plots(
    run: hrdps.RunInfo,
    data_dir: Path,
    output_dir: Path,
    hours: Iterable[int] = FORECAST_HOURS,
) -> list[Path]:
    """Standalone renderer; automation reuses Fire Weather calculations instead."""
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
    base_yslice, base_xslice = hrdps.subset_slices(lat, lon, hrdps.model_config().extent)
    base_lat = lat[base_yslice, base_xslice]
    base_lon = lon[base_yslice, base_xslice]
    terrain_full, _, _ = hrdps.read_grib(
        lightning.hour_file(run_dir, run, hrdps.TERRAIN_FHOUR, "HGT", "SFC", "0")
    )
    terrain_m = terrain_full[base_yslice, base_xslice]
    transmission_lines = lightning.load_transmission_lines()
    dcape_stride = hrdps.grid_stride(18.0)

    outputs: list[Path] = []
    for fhour in hours:
        fields = lightning.compute_lightning_fields(
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
        outputs.append(
            render_from_all_cause(
                run,
                fhour,
                run_dir,
                output_dir,
                base_lat,
                base_lon,
                base_yslice,
                base_xslice,
                terrain_m,
                fields.gust_kmh,
                transmission_lines,
            )
        )
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(hrdps.MODEL_CONFIGS["west"].default_data_dir),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/hrdps_west_lightning"),
    )
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
