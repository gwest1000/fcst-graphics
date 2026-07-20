#!/usr/bin/env python3
"""Render three alternative HRDPS two-panel fire-weather designs."""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.base import BaseGeometry

import fire_danger_peak
import make_hrdps_fire_weather_twopanel as production
import make_hrdps_west_convective as hrdps
import make_hrdps_west_lightning as lightning
import plot_style


DEFAULT_STAMP = "20260714T18Z"
DEFAULT_FHOUR = 30
DEFAULT_OUTPUT_DIR = Path("plots/test_fire_weather_twopanel/options_20260714T18Z_f030")
DANGER_LEVELS = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
DANGER_COLORS = ("#78add3", "#75bd74", "#f2df55", "#ed9418", "#c61d23")
DANGER_TICKS = (1, 2, 3, 4, 5)
DANGER_TICK_LABELS = ("VL", "L", "M", "H", "E")
RH_LEVELS = (-0.1, 20.0, 30.0, 60.0, 80.0, 100.1)
RH_FILL_COLORS = (
    mcolors.to_rgba("#743b16", 0.48),
    mcolors.to_rgba("#c47a3a", 0.36),
    (1.0, 1.0, 1.0, 0.0),
    mcolors.to_rgba("#74c4d7", 0.34),
    mcolors.to_rgba("#2f8fb5", 0.40),
)
RH_CONTOUR_LEVELS = (20.0, 30.0, 60.0, 80.0)
RH_CONTOUR_COLORS = ("#6f3215", "#b86c32", "#4a9eb7", "#176f98")
RH_CONTOUR_WIDTHS = (1.9, 1.25, 1.25, 1.9)
RH_CONTOUR_STYLES = ("solid", "dashed", "dashed", "solid")
LPI_CONTOUR_LEVELS = (20, 40, 60, 80)
LPI_CONTOUR_COLORS = ("#8064a2", "#70418f", "#9f277a", "#d31363")
GUST_CONTOUR_LEVELS = (40, 60, 80)


def add_panel_base(ax: plt.Axes) -> None:
    production.add_panel_base(ax)


def add_panel_text(ax: plt.Axes, title: str, footer: str) -> None:
    production.add_panel_label(ax, title, footer)


def sampled(fields: np.ndarray, stride: int) -> np.ndarray:
    return fields[::stride, ::stride]


def plot_gust_fill(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    gust_kmh: np.ndarray,
    stride: int,
):
    cmap, norm, levels = lightning.gust_cmap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shaded = ax.contourf(
            sampled(lon, stride),
            sampled(lat, stride),
            sampled(gust_kmh, stride),
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
            alpha=0.76,
            transform=lightning.DATA_CRS,
            transform_first=True,
            zorder=3,
        )
    return shaded, levels


def plot_lpi_fill(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    potential: np.ndarray,
    stride: int,
):
    cmap, norm, levels = lightning.lightning_cmap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shaded = ax.contourf(
            sampled(lon, stride),
            sampled(lat, stride),
            sampled(potential, stride),
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="max",
            alpha=0.78,
            transform=lightning.DATA_CRS,
            transform_first=True,
            zorder=3,
        )
    return shaded, levels


def plot_danger_fill(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, danger: np.ndarray):
    danger_s = lightning.smooth_nan(danger, sigma=lightning.sigma_for_km(5.0))
    cmap = mcolors.ListedColormap(DANGER_COLORS, name="twopanel_danger")
    norm = mcolors.BoundaryNorm(DANGER_LEVELS, cmap.N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ax.contourf(
            lon,
            lat,
            danger_s,
            levels=DANGER_LEVELS,
            cmap=cmap,
            norm=norm,
            alpha=0.44,
            transform=lightning.DATA_CRS,
            transform_first=True,
            zorder=3,
        )


def plot_danger_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, danger: np.ndarray) -> None:
    production.plot_peak_danger(ax, lon, lat, danger)


def plot_rh_fill(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, rh: np.ndarray) -> None:
    stride = lightning.grid_stride(lightning.RH_BOUNDARY_GRID_KM)
    rh_s = lightning.smooth_nan(rh, sigma=lightning.sigma_for_km(lightning.RH_SMOOTHING_KM))
    ax.contourf(
        sampled(lon, stride),
        sampled(lat, stride),
        sampled(rh_s, stride),
        levels=RH_LEVELS,
        colors=RH_FILL_COLORS,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=4,
    )


def plot_rh_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, rh: np.ndarray) -> None:
    stride = lightning.grid_stride(lightning.RH_BOUNDARY_GRID_KM)
    rh_s = lightning.smooth_nan(rh, sigma=lightning.sigma_for_km(lightning.RH_SMOOTHING_KM))
    contours = ax.contour(
        sampled(lon, stride),
        sampled(lat, stride),
        sampled(rh_s, stride),
        levels=RH_CONTOUR_LEVELS,
        colors=RH_CONTOUR_COLORS,
        linewidths=RH_CONTOUR_WIDTHS,
        linestyles=RH_CONTOUR_STYLES,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=12,
    )
    lightning.label_contours(
        contours,
        fontsize=6.4,
        fmt={level: f"{int(level)}%" for level in RH_CONTOUR_LEVELS},
    )


def plot_lpi_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, potential: np.ndarray) -> None:
    potential_s = lightning.smooth_nan(potential, sigma=lightning.sigma_for_km(5.0))
    ax.contour(
        lon,
        lat,
        potential_s,
        levels=LPI_CONTOUR_LEVELS,
        colors=LPI_CONTOUR_COLORS,
        linewidths=(1.0, 1.35, 1.7, 2.1),
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=12,
    )


def plot_gust_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, gust_kmh: np.ndarray) -> None:
    gust_s = lightning.smooth_nan(gust_kmh, sigma=lightning.sigma_for_km(5.0))
    contours = ax.contour(
        lon,
        lat,
        gust_s,
        levels=GUST_CONTOUR_LEVELS,
        colors=("#555555", "#a43b2d", "#770018"),
        linewidths=(1.0, 1.55, 2.05),
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=12,
    )
    lightning.label_contours(contours, fontsize=6.4, fmt="%g")


def plot_direction_arrows(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    minimum_stride: int,
) -> None:
    row_density, column_density = lightning.gust_vector_density("bc")
    sample = plot_style.vector_sample_slices(
        ax,
        lon.shape,
        minimum=minimum_stride,
        spacing_px=27.0,
        row_density=row_density * production.VECTOR_DENSITY_MULTIPLIER,
        column_density=column_density * production.VECTOR_DENSITY_MULTIPLIER,
    )
    u = u_ms[sample]
    v = v_ms[sample]
    speed = np.hypot(u, v)
    finite = np.isfinite(u) & np.isfinite(v) & (speed > 0.05)
    unit_u = np.divide(u, speed, out=np.zeros_like(u), where=speed > 0.05)
    unit_v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 0.05)
    ax.quiver(
        lon[sample][finite],
        lat[sample][finite],
        unit_u[finite],
        unit_v[finite],
        color="#202428",
        edgecolors=mcolors.to_rgba("white", 0.82),
        linewidths=0.35,
        transform=lightning.DATA_CRS,
        scale_units="width",
        scale=68,
        width=0.00155,
        headwidth=3.5,
        headlength=4.2,
        headaxislength=3.9,
        minlength=0.05,
        pivot="middle",
        zorder=13,
    )


def plot_colored_gust_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    fields: lightning.LightningFields,
    shade_stride: int,
):
    row_density, column_density = lightning.gust_vector_density("bc")
    return lightning.plot_gust_vectors(
        ax,
        lon,
        lat,
        fields.u10_ms,
        fields.v10_ms,
        fields.gust_kmh,
        shade_stride,
        row_density=row_density * production.VECTOR_DENSITY_MULTIPLIER,
        column_density=column_density * production.VECTOR_DENSITY_MULTIPLIER,
    )


def plot_dry_lightning(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    dry_potential: np.ndarray,
    contour_stride: int,
    marker_style: str,
) -> None:
    stride = max(contour_stride, 2)
    dry = sampled(dry_potential, stride)
    mask = dry >= 15.0
    if not np.any(mask):
        return
    marker = "." if marker_style == "stipple" else lightning.DRY_LIGHTNING_MARKER
    size = 7.0 if marker_style == "stipple" else lightning.dry_lightning_marker_area("bc")
    ax.scatter(
        sampled(lon, stride)[mask],
        sampled(lat, stride)[mask],
        marker=marker,
        s=size,
        color="#161616",
        alpha=0.68 if marker_style == "stipple" else 0.95,
        linewidths=0.25,
        transform=lightning.DATA_CRS,
        zorder=15,
    )


def add_common_overlays(
    axes: tuple[plt.Axes, plt.Axes],
    transmission_lines: list[BaseGeometry],
    watersheds: list[BaseGeometry],
) -> None:
    for ax in axes:
        lightning.add_transmission_lines(ax, transmission_lines)
        hrdps.add_watersheds(ax, watersheds)
        hrdps.add_city_labels(ax, fontsize=6.5, marker_size=2.0, path_width=2.2, zorder=30)


def add_left_colorbar(fig, ax, mappable, ticks, label, title, **kwargs) -> None:
    plot_style.add_internal_colorbar(
        fig,
        ax,
        mappable,
        ticks=ticks,
        label=label,
        title=title,
        backdrop=(0.040, 0.110, 0.080, 0.650),
        cax_bounds=(0.065, 0.135, 0.027, 0.600),
        tick_position="right",
        **kwargs,
    )


def add_right_colorbar(fig, ax, mappable, ticks, label, title, **kwargs) -> None:
    plot_style.add_internal_colorbar(
        fig,
        ax,
        mappable,
        ticks=ticks,
        label=label,
        title=title,
        backdrop=(0.895, 0.110, 0.080, 0.650),
        cax_bounds=(0.912, 0.135, 0.027, 0.600),
        **kwargs,
    )


def render_option(
    option: int,
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    fields: lightning.LightningFields,
    peak_danger: np.ndarray,
    transmission_lines: list[BaseGeometry],
    watersheds: list[BaseGeometry],
    shade_stride: int,
    contour_stride: int,
) -> Path:
    yslice, xslice = hrdps.subset_slices(lat, lon, production.DATA_EXTENT)
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    plot_fields = lightning.subset_lightning_fields(fields, yslice, xslice)
    plot_danger = peak_danger[yslice, xslice]
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    left = fig.add_axes(production.PANEL_POSITIONS[0], projection=production.PLOT_CRS)
    right = fig.add_axes(production.PANEL_POSITIONS[1], projection=production.PLOT_CRS)
    for ax in (left, right):
        add_panel_base(ax)

    if option == 1:
        gust_fill, gust_levels = plot_gust_fill(
            left, plot_lon, plot_lat, plot_fields.gust_kmh, shade_stride
        )
        plot_direction_arrows(
            left,
            plot_lon,
            plot_lat,
            plot_fields.u10_ms,
            plot_fields.v10_ms,
            shade_stride,
        )
        plot_rh_contours(left, plot_lon, plot_lat, plot_fields.surface_rh)
        lpi_fill, lpi_levels = plot_lpi_fill(
            right, plot_lon, plot_lat, plot_fields.potential, shade_stride
        )
        plot_dry_lightning(
            right, plot_lon, plot_lat, plot_fields.dry_potential, contour_stride, "stipple"
        )
        plot_danger_contours(right, plot_lon, plot_lat, plot_danger)
        add_left_colorbar(
            fig, left, gust_fill, gust_levels, "Gust (km h$^{-1}$)", "GUST", fmt="%g", extend="both"
        )
        add_right_colorbar(
            fig, right, lpi_fill, lpi_levels, "Lightning potential index", "LPI", fmt="%g", extend="max"
        )
        add_panel_text(
            left,
            "GUST MAGNITUDE + DIRECTION + RH THRESHOLDS",
            "Gust shaded | Direction arrows | RH: 20/30% brown, 60/80% blue contours",
        )
        add_panel_text(
            right,
            "LIGHTNING POTENTIAL + PEAK FIRE DANGER",
            "LPI shaded | Dry-lightning stipple | Peak fire danger contours",
        )
        option_name = "FIELD-NATIVE"
    elif option == 2:
        plot_rh_fill(left, plot_lon, plot_lat, plot_fields.surface_rh)
        gust_vectors, gust_levels = plot_colored_gust_vectors(
            left, plot_lon, plot_lat, plot_fields, shade_stride
        )
        danger_fill = plot_danger_fill(right, plot_lon, plot_lat, plot_danger)
        plot_lpi_contours(right, plot_lon, plot_lat, plot_fields.potential)
        plot_dry_lightning(
            right, plot_lon, plot_lat, plot_fields.dry_potential, contour_stride, "stars"
        )
        add_left_colorbar(
            fig,
            left,
            gust_vectors,
            gust_levels,
            "Gust (km h$^{-1}$)",
            "GUST",
            fmt="%g",
            extend="both",
        )
        add_right_colorbar(
            fig,
            right,
            danger_fill,
            DANGER_TICKS,
            "Peak daily fire danger",
            "DANGER",
            fmt="%g",
            tick_labels=DANGER_TICK_LABELS,
        )
        add_panel_text(
            left,
            "RH CATEGORIES + COLORED GUST VECTORS",
            "RH filled: brown <30%, blue >60% | Gust direction and magnitude vectors",
        )
        add_panel_text(
            right,
            "PEAK FIRE DANGER + LIGHTNING CONTOURS",
            "Danger filled: VL/L/M/H/E | LPI 20/40/60/80 contours (darker = higher) | Dry-lightning black *",
        )
        option_name = "CATEGORICAL"
    elif option == 3:
        danger_fill = plot_danger_fill(left, plot_lon, plot_lat, plot_danger)
        plot_gust_contours(left, plot_lon, plot_lat, plot_fields.gust_kmh)
        plot_direction_arrows(
            left,
            plot_lon,
            plot_lat,
            plot_fields.u10_ms,
            plot_fields.v10_ms,
            shade_stride,
        )
        lpi_fill, lpi_levels = plot_lpi_fill(
            right, plot_lon, plot_lat, plot_fields.potential, shade_stride
        )
        plot_rh_contours(right, plot_lon, plot_lat, plot_fields.surface_rh)
        plot_dry_lightning(
            right, plot_lon, plot_lat, plot_fields.dry_potential, contour_stride, "stipple"
        )
        add_left_colorbar(
            fig,
            left,
            danger_fill,
            DANGER_TICKS,
            "Peak daily fire danger",
            "DANGER",
            fmt="%g",
            tick_labels=DANGER_TICK_LABELS,
        )
        add_right_colorbar(
            fig, right, lpi_fill, lpi_levels, "Lightning potential index", "LPI", fmt="%g", extend="max"
        )
        add_panel_text(
            left,
            "SPREAD: PEAK FIRE DANGER + GUST",
            "Danger filled: VL/L/M/H/E | Gust 40/60/80 km/h contours + direction arrows",
        )
        add_panel_text(
            right,
            "IGNITION: LIGHTNING POTENTIAL + RH",
            "LPI shaded | RH: 20/30% brown, 60/80% blue contours | Dry-lightning stipple",
        )
        option_name = "DECISION-SPLIT"
    else:
        raise ValueError(f"Unsupported option: {option}")

    add_common_overlays((left, right), transmission_lines, watersheds)
    valid = run.init_time + dt.timedelta(hours=fhour)
    valid_local = valid.astimezone(plot_style.LOCAL_TZ)
    fig.text(
        0.5,
        0.997,
        (
            f"OPTION {option} - {option_name}  |  {hrdps.model_config().label}  |  "
            f"{valid_local:%a %H:%M%Z %d%b%Y}  |  {valid:%H:%MUTC %d%b%Y}"
        ).upper(),
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="black",
        zorder=80,
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    fig.text(
        0.998,
        0.997,
        f"Data:ECCC HRDPS + CWFIS | Init:{run.init_time:%Y%m%d%H}",
        ha="right",
        va="top",
        fontsize=6.4,
        color="black",
        zorder=80,
        bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    lightning.log(f"Wrote {out_path}.")
    return out_path


def load_fields(run: hrdps.RunInfo, fhour: int, data_dir: Path):
    run_dir = data_dir / run.stamp
    sample_path = lightning.hour_file(run_dir, run, fhour, "MU-VT-LI", "ISBL", "500")
    _, lat, lon = hrdps.read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read HRDPS coordinates.")
    base_yslice, base_xslice = hrdps.subset_slices(lat, lon, hrdps.model_config().extent)
    base_lat = lat[base_yslice, base_xslice]
    base_lon = lon[base_yslice, base_xslice]
    terrain_path = (
        run_dir
        / f"{hrdps.TERRAIN_FHOUR:03d}"
        / hrdps.field_name("HGT", "SFC", "0", run.stamp, hrdps.TERRAIN_FHOUR)
    )
    terrain_full, _, _ = hrdps.read_grib(terrain_path)
    terrain_m = terrain_full[base_yslice, base_xslice]
    fields = lightning.compute_lightning_fields(
        run_dir,
        run,
        fhour,
        base_yslice,
        base_xslice,
        lat,
        lon,
        terrain_m,
        hrdps.grid_stride(18.0),
    )
    valid = run.init_time + dt.timedelta(hours=fhour)
    fire_date = fire_danger_peak.fire_date_for_valid(valid, plot_style.LOCAL_TZ)
    peak_grid = fire_danger_peak.load_peak_danger_grid(
        lightning.FWI_CACHE_DIR,
        hrdps.model_config().key,
        fire_date,
        run.init_time,
        base_lat.shape,
    )
    if peak_grid is None:
        raise RuntimeError(f"No complete peak fire-danger grid is available for {fire_date:%Y-%m-%d}.")
    return base_lat, base_lon, fields, peak_grid.danger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--fhour", type=int, default=DEFAULT_FHOUR)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hrdps_continental"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hrdps.set_model("continental")
    lightning.set_model("continental")
    run = hrdps.RunInfo(
        cycle=args.stamp[9:11],
        stamp=args.stamp,
        init_time=hrdps.parse_stamp(args.stamp),
    )
    lat, lon, fields, peak_danger = load_fields(run, args.fhour, args.data_dir)
    transmission_lines = lightning.load_transmission_lines()
    watersheds = hrdps.load_watersheds(hrdps.WATERSHED_CACHE)
    shade_stride = hrdps.grid_stride(5.0)
    contour_stride = hrdps.grid_stride(12.0)
    for option in (1, 2, 3):
        render_option(
            option,
            args.output_dir / f"fire_weather_twopanel_option{option}_{run.stamp}_f{args.fhour:03d}.png",
            run,
            args.fhour,
            lat,
            lon,
            fields,
            peak_danger,
            transmission_lines,
            watersheds,
            shade_stride,
            contour_stride,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
