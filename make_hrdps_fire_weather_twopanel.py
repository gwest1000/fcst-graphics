#!/usr/bin/env python3
"""Render the HRDPS 2.5 km two-panel BC fire-weather product."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import Rectangle
from shapely.geometry.base import BaseGeometry

import fire_danger_peak
import fire_activity
import make_hrdps_west_convective as hrdps
import make_hrdps_west_lightning as lightning
import plot_style


DEFAULT_MODEL = "continental"
DEFAULT_STAMP = "20260714T12Z"
DEFAULT_FHOUR = 36
OUTPUT_PREFIX = "hrdps_continental_lightning_twopanel"
DATA_EXTENT = (-140.5, -111.5, 47.0, 60.5)
PLOT_CRS = ccrs.LambertConformal(central_longitude=-98.0, central_latitude=53.0)
# A projected crop fitted to mainland BC. Moving the Lambert central meridian
# east rotates BC about 20 degrees clockwise and uses the portrait panels well.
# The west edge follows Haida Gwaii, while the shorter north edge removes the
# data-free corner of the rotated HRDPS domain.
PROJECTED_X_LIMITS = (-2_335_000.0, -1_063_968.0)
PROJECTED_Y_LIMITS = (-383_441.0, 1_152_496.77)
VECTOR_DENSITY_MULTIPLIER = 1.875
VECTOR_SIZE_MULTIPLIER = 1.12
VECTOR_BOLD_MULTIPLIER = 1.45
PANEL_POSITIONS = (
    (0.0005, 0.003, 0.4985, 0.994),
    (0.5010, 0.003, 0.4985, 0.994),
)
EDGE_BAND_PANEL_POSITIONS = (
    (0.0, 0.0, 0.5, 1.0),
    (0.5, 0.0, 0.5, 1.0),
)
EDGE_HEADER_HEIGHT = 0.039
EDGE_FOOTER_HEIGHT = 0.031
EDGE_HEADER_FONTSIZE = 14.5 * 0.90
EDGE_HEADER_SOURCE_FONTSIZE = 8.8
EDGE_FOOTER_FONTSIZE = 16.0 * 0.90
BC_GUST_COLORBAR_LAYOUT = {
    "backdrop": (0.890, 0.180, 0.110, 0.560),
    "cax_bounds": (0.949, 0.210, 0.024, 0.500),
    "tick_position": "left",
    "backdrop_edgecolor": "black",
    "backdrop_linewidth": 0.65,
}
BC_DANGER_COLORBAR_LAYOUT = {
    "backdrop": (0.900, 0.180, 0.100, 0.560),
    "cax_bounds": (0.948, 0.210, 0.025, 0.500),
    "tick_position": "left",
    "backdrop_edgecolor": "black",
    "backdrop_linewidth": 0.65,
}
DANGER_LEVELS = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
DANGER_COLORS = ("#579dcc", "#54b35d", "#f2da32", "#ee8817", "#cf2730")
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
LPI_CONTOUR_LEVELS = (20, 40, 60, 80)
LPI_CONTOUR_COLORS = ("#8064a2", "#70418f", "#9f277a", "#d31363")
REGIONAL_EXTENTS = {
    "sw": (-128.5, -120.0, 48.0, 55.0),
    "se": (-120.75, -113.7, 48.0, 53.7),
    "ne": (-130.0, -118.5, 51.0, 59.2),
}
REGIONAL_LABELS = {
    "sw": "SOUTHWEST BC",
    "se": "SOUTHEAST BC",
    "ne": "NORTHEAST BC",
}
REGIONAL_VECTOR_ROW_DENSITY_MULTIPLIER = VECTOR_DENSITY_MULTIPLIER * 0.75
REGIONAL_VECTOR_COLUMN_DENSITY_MULTIPLIER = REGIONAL_VECTOR_ROW_DENSITY_MULTIPLIER * 1.20
REGIONAL_VECTOR_SIZE_MULTIPLIER = VECTOR_SIZE_MULTIPLIER * 1.25
REGIONAL_VECTOR_BOLD_MULTIPLIER = VECTOR_BOLD_MULTIPLIER * 1.25
REGIONAL_DRY_LIGHTNING_DENSITY_MULTIPLIER = 1.5 * 1.25
REGIONAL_DRY_LIGHTNING_AREA_MULTIPLIER = 1.25
DRY_LIGHTNING_AREA_MULTIPLIER = 1.25
PRECIP_DOT_GRID_KM = 12.0
PRECIP_DOT_MODERATE_MM = 2.5
PRECIP_DOT_HEAVY_MM = 10.0
PRECIP_DOT_MODERATE_COLOR = "#b8f1f2"
PRECIP_DOT_MODERATE_EDGE_COLOR = "#007f86"
PRECIP_DOT_HEAVY_COLOR = "#00a8ad"
REGIONAL_PRECIP_DOT_AREA_MULTIPLIER = 1.25
ACTIVE_FIRE_COLOR = "#ff815c"
ACTIVE_FIRE_OUTLINE_COLOR = "#111111"
ACTIVE_FIRE_AREA = 31.0
FIRE_OF_NOTE_AREA = 70.0
HOTSPOT_COLOR = "#f28e1c"
HOTSPOT_MARKER = "s"


# This is the Lucide Flame path used by wx_app, converted from its 24 px SVG
# coordinates into a centered Matplotlib marker.
ACTIVE_FIRE_MARKER = MatplotlibPath(
    [
        (0.0000, 1.0000),
        (0.1053, 0.5789),
        (0.4211, 0.3158),
        (0.7368, 0.0526),
        (0.7368, -0.2632),
        (0.7368, -0.6701),
        (0.4070, -1.0000),
        (0.0000, -1.0000),
        (-0.4070, -1.0000),
        (-0.7368, -0.6701),
        (-0.7368, -0.2632),
        (-0.7368, -0.1490),
        (-0.7001, -0.0387),
        (-0.6316, 0.0526),
        (-0.6316, -0.0927),
        (-0.5138, -0.2105),
        (-0.3684, -0.2105),
        (-0.2231, -0.2105),
        (-0.1053, -0.0927),
        (-0.1053, 0.0526),
        (-0.1053, 0.2632),
        (-0.2632, 0.3684),
        (-0.2632, 0.5789),
        (-0.2632, 0.7895),
        (0.0000, 1.0000),
        (0.0000, 1.0000),
    ],
    [
        MatplotlibPath.MOVETO,
        MatplotlibPath.CURVE3,
        MatplotlibPath.CURVE3,
        MatplotlibPath.CURVE3,
        MatplotlibPath.CURVE3,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE3,
        MatplotlibPath.CURVE3,
        MatplotlibPath.CLOSEPOLY,
    ],
)


def add_panel_base(ax: plt.Axes) -> None:
    ax.set_xlim(*PROJECTED_X_LIMITS)
    ax.set_ylim(*PROJECTED_Y_LIMITS)
    ax.set_facecolor("#dbeaf0")
    hrdps.add_map_features(ax)
    hrdps.add_hydro_features(ax)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.15)
        spine.set_zorder(60)


def add_regional_panel_base(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
) -> None:
    ax.set_extent(extent, crs=lightning.DATA_CRS)
    ax.set_facecolor("#dbeaf0")
    hrdps.add_map_features(ax)
    hrdps.add_hydro_features(ax)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.15)
        spine.set_zorder(60)


def add_panel_label(ax: plt.Axes, title: str, footer: str, edge_bands: bool = False) -> None:
    if edge_bands:
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                EDGE_FOOTER_HEIGHT,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="black",
                linewidth=0.75,
                zorder=76,
            )
        )
        ax.text(
            0.5,
            EDGE_FOOTER_HEIGHT / 2.0,
            footer,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=EDGE_FOOTER_FONTSIZE,
            fontweight="normal",
            color="black",
            zorder=77,
        )
        return
    if title:
        ax.text(
            0.5,
            0.974,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color="black",
            zorder=45,
            bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        )
    plot_style.add_footer(ax, footer, fontsize=6.7, stroke_width=2.0)


def add_figure_header(
    fig: plt.Figure,
    title: str,
    source: str,
    source_fontsize: float,
    source_pad: float = 0.08,
    edge_bands: bool = False,
) -> None:
    if edge_bands:
        fig.add_artist(
            Rectangle(
                (0.0, 1.0 - EDGE_HEADER_HEIGHT),
                1.0,
                EDGE_HEADER_HEIGHT,
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="black",
                linewidth=0.75,
                zorder=79,
            )
        )
        fig.text(
            0.006,
            1.0 - EDGE_HEADER_HEIGHT / 2.0,
            title,
            ha="left",
            va="center",
            fontsize=EDGE_HEADER_FONTSIZE,
            fontweight="bold",
            color="black",
            zorder=80,
        )
        fig.text(
            0.994,
            1.0 - EDGE_HEADER_HEIGHT / 2.0,
            source,
            ha="right",
            va="center",
            fontsize=max(EDGE_HEADER_SOURCE_FONTSIZE, source_fontsize),
            color="black",
            zorder=81,
        )
        return

    fig.text(
        0.5,
        0.997,
        title,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="black",
        zorder=80,
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    fig.text(
        0.998,
        0.997,
        source,
        ha="right",
        va="top",
        fontsize=source_fontsize,
        color="black",
        zorder=81,
        bbox={"boxstyle": f"square,pad={source_pad:.2f}", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def period_hazard_label(fhour: int) -> str:
    return "3-h max" if fhour > 0 else "Init-time"


def fire_activity_footer(activity: fire_activity.FireActivity | None) -> str:
    if activity is None:
        return ""
    if activity.is_active_fire_feed:
        return "Active fires"
    return "24-h hotspots orange squares"


def fire_activity_source(activity: fire_activity.FireActivity | None) -> str:
    if activity is None:
        return "ECCC HRDPS/CWFIS"
    if activity.is_active_fire_feed:
        cache_label = " CACHED" if activity.stale else ""
        return f"ECCC HRDPS/CWFIS/BCWS {activity.retrieved_at:%H}Z{cache_label}"
    return "ECCC HRDPS/CWFIS 24-H HOTSPOTS"


def edge_panel_footers(
    fhour: int,
    activity: fire_activity.FireActivity | None = None,
) -> tuple[str, str]:
    period = period_hazard_label(fhour)
    fire_label = fire_activity_footer(activity)
    fire_suffix = f" | {fire_label}" if fire_label else ""
    return (
        f"RH <30% brown / >60% blue | {period} gust | Transmission grey",
        f"Danger | {period} LPI/dry lightning * | Rain 2.5/10{fire_suffix}",
    )


def add_fire_activity(
    ax: plt.Axes,
    activity: fire_activity.FireActivity | None,
) -> None:
    if activity is None or not activity.observations:
        return
    if activity.is_active_fire_feed:
        regular = [observation for observation in activity.observations if not observation.fire_of_note]
        notes = [observation for observation in activity.observations if observation.fire_of_note]
        for observations, area in ((regular, ACTIVE_FIRE_AREA), (notes, FIRE_OF_NOTE_AREA)):
            if not observations:
                continue
            points = ax.scatter(
                [observation.longitude for observation in observations],
                [observation.latitude for observation in observations],
                marker=ACTIVE_FIRE_MARKER,
                s=area,
                facecolor=ACTIVE_FIRE_COLOR,
                edgecolors=ACTIVE_FIRE_OUTLINE_COLOR,
                linewidths=0.52,
                transform=lightning.DATA_CRS,
                zorder=36,
            )
        return

    observations = activity.observations
    sizes = [min(42.0, 10.0 + 2.8 * math.sqrt(observation.detection_count)) for observation in observations]
    points = ax.scatter(
        [observation.longitude for observation in observations],
        [observation.latitude for observation in observations],
        marker=HOTSPOT_MARKER,
        s=sizes,
        facecolor=HOTSPOT_COLOR,
        edgecolors="white",
        linewidths=0.75,
        transform=lightning.DATA_CRS,
        zorder=36,
    )
    points.set_path_effects(
        [
            path_effects.Stroke(linewidth=2.1, foreground="#6b3300"),
            path_effects.Normal(),
        ]
    )


def plot_peak_danger_fill(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    danger: np.ndarray,
) -> object:
    danger_s = lightning.smooth_nan(danger, sigma=lightning.sigma_for_km(5.0))
    cmap = mcolors.ListedColormap(DANGER_COLORS, name="fire_weather_danger")
    norm = mcolors.BoundaryNorm(DANGER_LEVELS, cmap.N)
    return ax.contourf(
        lon,
        lat,
        danger_s,
        levels=DANGER_LEVELS,
        cmap=cmap,
        norm=norm,
        alpha=0.68,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=3,
    )


def peak_danger_mappable() -> object:
    cmap = mcolors.ListedColormap(DANGER_COLORS, name="fire_weather_danger_legend")
    norm = mcolors.BoundaryNorm(DANGER_LEVELS, cmap.N)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_rh_categories(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    surface_rh: np.ndarray,
) -> None:
    stride = lightning.grid_stride(lightning.RH_BOUNDARY_GRID_KM)
    rh_s = lightning.smooth_nan(
        surface_rh,
        sigma=lightning.sigma_for_km(lightning.RH_SMOOTHING_KM),
    )
    sample = (slice(None, None, stride), slice(None, None, stride))
    ax.contourf(
        lon[sample],
        lat[sample],
        rh_s[sample],
        levels=RH_LEVELS,
        colors=RH_FILL_COLORS,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=4,
    )


def plot_lpi_contours(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    potential: np.ndarray,
) -> None:
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


def plot_gust_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    gust_kmh: np.ndarray,
    shade_stride: int,
    *,
    density_multiplier: float = VECTOR_DENSITY_MULTIPLIER,
    row_density_multiplier: float | None = None,
    column_density_multiplier: float | None = None,
    size_multiplier: float = VECTOR_SIZE_MULTIPLIER,
    bold_multiplier: float = VECTOR_BOLD_MULTIPLIER,
):
    row_density, column_density = lightning.gust_vector_density("bc")
    row_multiplier = density_multiplier if row_density_multiplier is None else row_density_multiplier
    column_multiplier = density_multiplier if column_density_multiplier is None else column_density_multiplier
    sample = plot_style.vector_sample_slices(
        ax,
        lon.shape,
        minimum=max(1, shade_stride),
        spacing_px=27.0,
        row_density=row_density * row_multiplier,
        column_density=column_density * column_multiplier,
    )
    u = u10_ms[sample]
    v = v10_ms[sample]
    gust = gust_kmh[sample]
    speed = np.hypot(u, v)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(gust) & (speed > 0.05)
    unit_u = np.divide(u, speed, out=np.zeros_like(u, dtype=np.float32), where=speed > 0.05)
    unit_v = np.divide(v, speed, out=np.zeros_like(v, dtype=np.float32), where=speed > 0.05)
    cmap, norm, levels = lightning.gust_cmap()
    vectors = ax.quiver(
        lon[sample][finite],
        lat[sample][finite],
        unit_u[finite],
        unit_v[finite],
        gust[finite],
        cmap=cmap,
        norm=norm,
        transform=lightning.DATA_CRS,
        scale_units="width",
        scale=68 / size_multiplier,
        width=0.00155 * bold_multiplier,
        headwidth=3.5,
        headlength=4.2,
        headaxislength=3.9,
        minlength=0.05,
        pivot="middle",
        edgecolors=mcolors.to_rgba(
            lightning.GUST_VECTOR_EDGE_COLOR,
            min(1.0, lightning.GUST_VECTOR_EDGE_ALPHA * bold_multiplier),
        ),
        linewidths=lightning.GUST_VECTOR_EDGE_WIDTH * bold_multiplier,
        zorder=11,
    )
    return vectors, levels


def plot_precipitation_dots(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    precip_mm: np.ndarray,
    area_multiplier: float = 1.0,
) -> None:
    stride = lightning.grid_stride(PRECIP_DOT_GRID_KM)
    sample = (slice(None, None, stride), slice(None, None, stride))
    sample_lon = lon[sample]
    sample_lat = lat[sample]
    precip = precip_mm[sample]
    moderate = np.isfinite(precip) & (precip >= PRECIP_DOT_MODERATE_MM) & (precip < PRECIP_DOT_HEAVY_MM)
    heavy = np.isfinite(precip) & (precip >= PRECIP_DOT_HEAVY_MM)
    if np.any(moderate):
        ax.scatter(
            sample_lon[moderate],
            sample_lat[moderate],
            marker="o",
            s=9.0 * area_multiplier,
            color=PRECIP_DOT_MODERATE_COLOR,
            edgecolors=PRECIP_DOT_MODERATE_EDGE_COLOR,
            linewidths=0.35,
            alpha=0.96,
            transform=lightning.DATA_CRS,
            zorder=9,
        )
    if np.any(heavy):
        ax.scatter(
            sample_lon[heavy],
            sample_lat[heavy],
            marker="o",
            s=14.0 * area_multiplier,
            color=PRECIP_DOT_HEAVY_COLOR,
            edgecolors="white",
            linewidths=0.35,
            transform=lightning.DATA_CRS,
            zorder=9,
        )


def add_regional_dry_lightning(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    dry_potential: np.ndarray,
    stride: int,
    region_key: str,
) -> None:
    sample = (slice(None, None, stride), slice(None, None, stride))
    sampled_dry = dry_potential[sample]
    mask = sampled_dry >= 15.0
    if not np.any(mask):
        return
    ax.scatter(
        lon[sample][mask],
        lat[sample][mask],
        marker=lightning.DRY_LIGHTNING_MARKER,
        s=(
            lightning.dry_lightning_marker_area(region_key)
            * REGIONAL_DRY_LIGHTNING_AREA_MULTIPLIER
            * DRY_LIGHTNING_AREA_MULTIPLIER
        ),
        color=lightning.DRY_LIGHTNING_COLOR,
        linewidths=0.40,
        transform=lightning.DATA_CRS,
        zorder=14,
    )


def regional_colorbar_layout(region_key: str) -> dict[str, object]:
    if region_key in {"sw", "ne"}:
        return {
            "backdrop": (0.001, 0.565, 0.083, 0.432),
            "cax_bounds": (0.029, 0.607, 0.024, 0.350),
            "tick_position": "right",
            "backdrop_edgecolor": "black",
            "backdrop_linewidth": 0.65,
        }
    if region_key == "se":
        return {
            "backdrop": (0.916, 0.565, 0.083, 0.432),
            "cax_bounds": (0.947, 0.607, 0.024, 0.350),
            "tick_position": "left",
            "backdrop_edgecolor": "black",
            "backdrop_linewidth": 0.65,
        }
    raise ValueError(f"Unsupported regional colorbar placement: {region_key}")


def plot_twopanel(
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    fields: lightning.LightningFields,
    transmission_lines: list[BaseGeometry],
    shade_stride: int,
    contour_stride: int,
    peak_danger_grid: fire_danger_peak.PeakDangerGrid | None = None,
    watersheds: list[BaseGeometry] | None = None,
    edge_bands: bool = True,
    fire_observations: fire_activity.FireActivity | None = None,
) -> Path:
    """Render one frame from diagnostics already computed for the single panel."""
    yslice, xslice = hrdps.subset_slices(lat, lon, DATA_EXTENT)
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    plot_fields = lightning.subset_lightning_fields(fields, yslice, xslice)
    valid = run.init_time + dt.timedelta(hours=fhour)
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    panel_positions = EDGE_BAND_PANEL_POSITIONS if edge_bands else PANEL_POSITIONS
    rh_ax = fig.add_axes(panel_positions[0], projection=PLOT_CRS)
    lpi_ax = fig.add_axes(panel_positions[1], projection=PLOT_CRS)
    for ax in (rh_ax, lpi_ax):
        add_panel_base(ax)

    plot_rh_categories(rh_ax, plot_lon, plot_lat, plot_fields.surface_rh)
    gust_vectors, gust_levels = plot_gust_vectors(
        rh_ax,
        plot_lon,
        plot_lat,
        plot_fields.u10_ms,
        plot_fields.v10_ms,
        plot_fields.gust_kmh,
        shade_stride,
    )

    danger_fill = peak_danger_mappable()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if peak_danger_grid is not None:
            danger_fill = plot_peak_danger_fill(
                lpi_ax,
                plot_lon,
                plot_lat,
                peak_danger_grid.danger[yslice, xslice],
            )
        plot_precipitation_dots(lpi_ax, plot_lon, plot_lat, plot_fields.precip_3h)
        plot_lpi_contours(lpi_ax, plot_lon, plot_lat, plot_fields.potential)
        dry_star_stride = max(contour_stride, shade_stride * 2)
        star_mask = plot_fields.dry_potential[::dry_star_stride, ::dry_star_stride] >= 15.0
        if np.any(star_mask):
            lpi_ax.scatter(
                plot_lon[::dry_star_stride, ::dry_star_stride][star_mask],
                plot_lat[::dry_star_stride, ::dry_star_stride][star_mask],
                marker=lightning.DRY_LIGHTNING_MARKER,
                s=lightning.dry_lightning_marker_area("bc") * DRY_LIGHTNING_AREA_MULTIPLIER,
                color=lightning.DRY_LIGHTNING_COLOR,
                linewidths=0.45,
                transform=lightning.DATA_CRS,
                zorder=14,
            )
    for ax in (rh_ax, lpi_ax):
        lightning.add_transmission_lines(ax, transmission_lines)
        hrdps.add_city_labels(ax, fontsize=6.5, marker_size=2.0, path_width=2.2, zorder=30)
    add_fire_activity(lpi_ax, fire_observations)

    plot_style.add_internal_colorbar(
        fig,
        rh_ax,
        gust_vectors,
        ticks=gust_levels,
        label="Gust (km h$^{-1}$)",
        title="GUST",
        fmt="%g",
        extend="both",
        labelpad=0.0,
        **BC_GUST_COLORBAR_LAYOUT,
    )
    plot_style.add_internal_colorbar(
        fig,
        lpi_ax,
        danger_fill,
        ticks=DANGER_TICKS,
        label="Peak daily fire danger",
        title="DANGER",
        fmt="%g",
        tick_labels=DANGER_TICK_LABELS,
        **BC_DANGER_COLORBAR_LAYOUT,
    )

    period = period_hazard_label(fhour)
    if edge_bands:
        left_footer, right_footer = edge_panel_footers(fhour, fire_observations)
    else:
        left_footer = f"Valid-time RH: brown <30%, blue >60% | {period} gust vectors | Grey: BC transmission"
        fire_suffix = fire_activity_footer(fire_observations)
        right_footer = (
            f"Danger VL/L/M/H/E | {period} LPI 20/40/60/80 | {period} dry lightning * | "
            "3-h rain dots: cyan 2.5/teal 10 mm | Transmission grey"
            + (f" | {fire_suffix}" if fire_suffix else "")
        )
    add_panel_label(
        rh_ax,
        "",
        left_footer,
        edge_bands=edge_bands,
    )
    add_panel_label(
        lpi_ax,
        "",
        right_footer,
        edge_bands=edge_bands,
    )

    valid_local = valid.astimezone(plot_style.LOCAL_TZ)
    add_figure_header(
        fig,
        (
            f"{hrdps.model_config().label} FIRE WEATHER  |  "
            f"{valid_local:%a %H:%M%Z %d%b%Y}  |  {valid:%H:%MUTC %d%b%Y}"
        ).upper(),
        f"{fire_activity_source(fire_observations)} | INIT {run.init_time:%Y%m%d%H}Z",
        source_fontsize=6.4,
        edge_bands=edge_bands,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    lightning.log(f"Wrote {out_path}.")
    return out_path


def plot_regional_twopanel(
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    plot_lat: np.ndarray,
    plot_lon: np.ndarray,
    fields: lightning.LightningFields,
    peak_danger: np.ndarray | None,
    transmission_lines: list[BaseGeometry],
    shade_stride: int,
    contour_stride: int,
    region_key: str,
    region_label: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    edge_bands: bool = True,
    fire_observations: fire_activity.FireActivity | None = None,
) -> Path:
    """Render the operational HRDPS-West regional two-panel fire-weather product."""
    if region_key not in REGIONAL_EXTENTS:
        raise ValueError(f"Unsupported regional two-panel area: {region_key}")
    extent = extent or REGIONAL_EXTENTS[region_key]
    region_label = (region_label or REGIONAL_LABELS[region_key]).upper()

    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    panel_positions = EDGE_BAND_PANEL_POSITIONS if edge_bands else PANEL_POSITIONS
    rh_ax = fig.add_axes(panel_positions[0], projection=lightning.PLOT_CRS)
    danger_ax = fig.add_axes(panel_positions[1], projection=lightning.PLOT_CRS)
    for ax in (rh_ax, danger_ax):
        add_regional_panel_base(ax, extent)

    plot_rh_categories(rh_ax, plot_lon, plot_lat, fields.surface_rh)
    gust_vectors, gust_levels = plot_gust_vectors(
        rh_ax,
        plot_lon,
        plot_lat,
        fields.u10_ms,
        fields.v10_ms,
        fields.gust_kmh,
        shade_stride,
        row_density_multiplier=REGIONAL_VECTOR_ROW_DENSITY_MULTIPLIER,
        column_density_multiplier=REGIONAL_VECTOR_COLUMN_DENSITY_MULTIPLIER,
        size_multiplier=REGIONAL_VECTOR_SIZE_MULTIPLIER,
        bold_multiplier=REGIONAL_VECTOR_BOLD_MULTIPLIER,
    )

    danger_fill = peak_danger_mappable()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if peak_danger is not None:
            danger_fill = plot_peak_danger_fill(danger_ax, plot_lon, plot_lat, peak_danger)
        plot_precipitation_dots(
            danger_ax,
            plot_lon,
            plot_lat,
            fields.precip_3h,
            area_multiplier=REGIONAL_PRECIP_DOT_AREA_MULTIPLIER,
        )
        plot_lpi_contours(danger_ax, plot_lon, plot_lat, fields.potential)
        base_dry_stride = max(contour_stride, shade_stride * 2)
        dry_stride = max(
            1,
            int(round(base_dry_stride / np.sqrt(REGIONAL_DRY_LIGHTNING_DENSITY_MULTIPLIER))),
        )
        add_regional_dry_lightning(
            danger_ax,
            plot_lon,
            plot_lat,
            fields.dry_potential,
            dry_stride,
            region_key,
        )

    for ax in (rh_ax, danger_ax):
        lightning.add_transmission_lines(ax, transmission_lines)
        hrdps.add_city_labels(ax, fontsize=7.1, marker_size=2.2, path_width=2.35, zorder=30)
    add_fire_activity(danger_ax, fire_observations)

    colorbar_layout = regional_colorbar_layout(region_key)
    plot_style.add_internal_colorbar(
        fig,
        rh_ax,
        gust_vectors,
        ticks=gust_levels,
        label="Gust (km h$^{-1}$)",
        title="GUST",
        fmt="%g",
        extend="both",
        **colorbar_layout,
    )
    plot_style.add_internal_colorbar(
        fig,
        danger_ax,
        danger_fill,
        ticks=DANGER_TICKS,
        label="Peak daily fire danger",
        title="DANGER",
        fmt="%g",
        tick_labels=DANGER_TICK_LABELS,
        **colorbar_layout,
    )

    period = period_hazard_label(fhour)
    if edge_bands:
        left_footer, right_footer = edge_panel_footers(fhour, fire_observations)
    else:
        left_footer = f"Valid-time RH: brown <30%, blue >60% | {period} gust vectors | Grey: BC transmission"
        fire_suffix = fire_activity_footer(fire_observations)
        right_footer = (
            f"Danger VL/L/M/H/E | {period} LPI 20/40/60/80 | {period} dry lightning * | "
            "3-h rain dots: cyan 2.5/teal 10 mm | Transmission grey"
            + (f" | {fire_suffix}" if fire_suffix else "")
        )
    add_panel_label(
        rh_ax,
        "" if edge_bands else "RH CATEGORIES + COLORED GUST VECTORS",
        left_footer,
        edge_bands=edge_bands,
    )
    add_panel_label(
        danger_ax,
        "" if edge_bands else "PEAK FIRE DANGER + LIGHTNING/RAIN",
        right_footer,
        edge_bands=edge_bands,
    )

    valid = run.init_time + dt.timedelta(hours=fhour)
    valid_local = valid.astimezone(plot_style.LOCAL_TZ)
    add_figure_header(
        fig,
        (
            f"{hrdps.model_config().label} {region_label} FIRE WEATHER  |  "
            f"{valid_local:%a %H:%M%Z %d%b%Y}  |  {valid:%H:%MUTC %d%b%Y}"
        ).upper(),
        f"{fire_activity_source(fire_observations)} | INIT {run.init_time:%Y%m%d%H}Z",
        source_fontsize=5.8,
        source_pad=0.06,
        edge_bands=edge_bands,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    lightning.log(f"Wrote {out_path}.")
    return out_path


def render_twopanel(run: hrdps.RunInfo, fhour: int, data_dir: Path, out_path: Path) -> Path:
    """Standalone wrapper used for manual regeneration and layout checks."""
    run_dir = data_dir / run.stamp
    shade_stride = hrdps.grid_stride(5.0)
    contour_stride = hrdps.grid_stride(12.0)
    dcape_stride = hrdps.grid_stride(18.0)
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
    lightning.log(f"Computing two-panel fire-weather fields for {run.stamp} F{fhour:03d}.")
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
    valid = run.init_time + dt.timedelta(hours=fhour)
    fire_date = fire_danger_peak.fire_date_for_valid(valid, plot_style.LOCAL_TZ)
    peak_danger_grid = fire_danger_peak.load_peak_danger_grid(
        lightning.FWI_CACHE_DIR,
        hrdps.model_config().key,
        fire_date,
        run.init_time,
        base_lat.shape,
    )
    transmission_lines = lightning.load_transmission_lines()
    return plot_twopanel(
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
        None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(hrdps.MODEL_CONFIGS), default=DEFAULT_MODEL)
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--fhour", type=int, default=DEFAULT_FHOUR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            f"plots/test_fire_weather_twopanel/{OUTPUT_PREFIX}_{DEFAULT_STAMP}_f{DEFAULT_FHOUR:03d}.png"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lightning.set_model(args.model)
    run = hrdps.RunInfo(
        cycle=args.stamp[9:11],
        stamp=args.stamp,
        init_time=hrdps.parse_stamp(args.stamp),
    )
    data_dir = args.data_dir or Path(hrdps.model_config().default_data_dir)
    render_twopanel(run, args.fhour, data_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
