#!/usr/bin/env python3
"""Render four alternative single-panel HRDPS regional fire-weather designs."""

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
import make_hrdps_west_convective as hrdps
import make_hrdps_west_lightning as lightning
import plot_style


DEFAULT_STAMP = "20260715T12Z"
DEFAULT_FHOUR = 12
DEFAULT_REGION = "se"
DEFAULT_OUTPUT_DIR = Path("plots/test_fire_weather_regional/options_20260715T12Z_f012_se")

DANGER_LEVELS = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
DANGER_COLORS = ("#579dcc", "#54b35d", "#f2da32", "#ee8817", "#cf2730")
DANGER_TICKS = (1, 2, 3, 4, 5)
DANGER_LABELS = ("VL", "L", "M", "H", "E")
LPI_CONTOUR_LEVELS = (20.0, 40.0, 60.0, 80.0)
LPI_CONTOUR_COLORS = ("#8064a2", "#70418f", "#9f277a", "#d31363")
RH_CONTOUR_LEVELS = (20.0, 30.0, 60.0, 80.0)
RH_CONTOUR_COLORS = ("#6f3215", "#b86c32", "#4a9eb7", "#176f98")
RH_CONTOUR_WIDTHS = (1.9, 1.3, 1.3, 1.9)
RH_CONTOUR_STYLES = ("solid", "dashed", "dashed", "solid")
RH_FILL_LEVELS = (-0.1, 20.0, 30.0, 60.0, 80.0, 100.1)
RH_FILL_COLORS = (
    mcolors.to_rgba("#743b16", 0.50),
    mcolors.to_rgba("#c47a3a", 0.38),
    (1.0, 1.0, 1.0, 0.0),
    mcolors.to_rgba("#74c4d7", 0.34),
    mcolors.to_rgba("#2f8fb5", 0.42),
)


def sampled(data: np.ndarray, stride: int) -> np.ndarray:
    return data[::stride, ::stride]


def add_base(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    ax.set_extent(extent, crs=lightning.DATA_CRS)
    ax.set_facecolor("#dbeaf0")
    hrdps.add_map_features(ax)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
        spine.set_zorder(60)


def danger_fill(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, danger: np.ndarray):
    cmap = mcolors.ListedColormap(DANGER_COLORS, name="regional_danger")
    norm = mcolors.BoundaryNorm(DANGER_LEVELS, cmap.N)
    smoothed = lightning.smooth_nan(danger, sigma=lightning.sigma_for_km(5.0))
    return ax.contourf(
        lon,
        lat,
        smoothed,
        levels=DANGER_LEVELS,
        cmap=cmap,
        norm=norm,
        alpha=0.42,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=3,
    )


def danger_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, danger: np.ndarray) -> None:
    smoothed = lightning.smooth_nan(danger, sigma=lightning.sigma_for_km(5.0))
    contours = ax.contour(
        lon,
        lat,
        smoothed,
        levels=lightning.PEAK_DANGER_CONTOUR_LEVELS,
        colors=lightning.PEAK_DANGER_CONTOUR_COLORS,
        linewidths=lightning.PEAK_DANGER_CONTOUR_LINEWIDTHS,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=12,
    )
    lightning.label_contours(contours, fontsize=7.0, fmt=lightning.PEAK_DANGER_CONTOUR_LABELS)


def lpi_fill(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, potential: np.ndarray, stride: int):
    cmap, norm, levels = lightning.lightning_cmap()
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


def lpi_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, potential: np.ndarray) -> None:
    smoothed = lightning.smooth_nan(potential, sigma=lightning.sigma_for_km(5.0))
    contours = ax.contour(
        lon,
        lat,
        smoothed,
        levels=LPI_CONTOUR_LEVELS,
        colors=LPI_CONTOUR_COLORS,
        linewidths=(1.15, 1.55, 1.95, 2.4),
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=13,
    )
    lightning.label_contours(
        contours,
        fontsize=6.8,
        fmt={20.0: "20", 40.0: "LPI 40", 60.0: "60", 80.0: "LPI 80"},
    )


def selective_lpi_zones(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    potential: np.ndarray,
    stride: int,
):
    levels = (20.0, 40.0, 60.0, 80.0, 100.1)
    colors = (
        mcolors.to_rgba("#b6a3d3", 0.18),
        mcolors.to_rgba("#8a65b2", 0.23),
        mcolors.to_rgba("#b22f8a", 0.28),
        mcolors.to_rgba("#e01768", 0.34),
    )
    cmap = mcolors.ListedColormap(colors, name="regional_selective_lpi")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    shaded = ax.contourf(
        sampled(lon, stride),
        sampled(lat, stride),
        sampled(potential, stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    lpi_contours(ax, lon, lat, potential)
    return shaded, levels[:-1]


def rh_contours(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, rh: np.ndarray) -> None:
    stride = lightning.grid_stride(lightning.RH_BOUNDARY_GRID_KM)
    smoothed = lightning.smooth_nan(rh, sigma=lightning.sigma_for_km(lightning.RH_SMOOTHING_KM))
    contours = ax.contour(
        sampled(lon, stride),
        sampled(lat, stride),
        sampled(smoothed, stride),
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
        fontsize=6.6,
        fmt={level: f"{int(level)}%" for level in RH_CONTOUR_LEVELS},
    )


def rh_fill(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, rh: np.ndarray) -> None:
    stride = lightning.grid_stride(lightning.RH_BOUNDARY_GRID_KM)
    smoothed = lightning.smooth_nan(rh, sigma=lightning.sigma_for_km(lightning.RH_SMOOTHING_KM))
    ax.contourf(
        sampled(lon, stride),
        sampled(lat, stride),
        sampled(smoothed, stride),
        levels=RH_FILL_LEVELS,
        colors=RH_FILL_COLORS,
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=4,
    )


def selective_rh(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    rh: np.ndarray,
    include_sixty_boundary: bool,
) -> None:
    stride = lightning.grid_stride(lightning.RH_BOUNDARY_GRID_KM)
    smoothed = lightning.smooth_nan(rh, sigma=lightning.sigma_for_km(lightning.RH_SMOOTHING_KM))
    sample_lon = sampled(lon, stride)
    sample_lat = sampled(lat, stride)
    sample_rh = sampled(smoothed, stride)
    layers = (
        ((-0.1, 20.0), "xx", lightning.VERY_LOW_RH_HATCH_COLOR, 0.60),
        ((20.0, 30.0), "//", lightning.LOW_RH_HATCH_COLOR, 0.55),
        ((80.0, 100.1), "xx", "#8ed4e4", 0.52),
    )
    for levels, hatch, color, alpha in layers:
        if not np.any(np.isfinite(sample_rh) & (sample_rh >= levels[0]) & (sample_rh <= levels[1])):
            continue
        with plt.rc_context({"hatch.color": color, "hatch.linewidth": 0.62}):
            hatched = ax.contourf(
                sample_lon,
                sample_lat,
                sample_rh,
                levels=levels,
                colors="none",
                hatches=[hatch],
                transform=lightning.DATA_CRS,
                transform_first=True,
                zorder=10,
            )
        for collection in hatched.collections:
            collection.set_facecolor((0.0, 0.0, 0.0, 0.0))
            collection.set_edgecolor(mcolors.to_rgba(color, alpha))
    if include_sixty_boundary:
        boundary = ax.contour(
            sample_lon,
            sample_lat,
            sample_rh,
            levels=(60.0,),
            colors=(RH_CONTOUR_COLORS[2],),
            linewidths=(1.2,),
            linestyles=("dashed",),
            transform=lightning.DATA_CRS,
            transform_first=True,
            zorder=12,
        )
        lightning.label_contours(boundary, fontsize=6.6, fmt={60.0: "60%"})


def vector_sample(
    ax: plt.Axes,
    shape: tuple[int, int],
    minimum_stride: int,
) -> tuple[slice, slice]:
    return plot_style.vector_sample_slices(
        ax,
        shape,
        minimum=max(1, minimum_stride),
        spacing_px=49.0,
        row_density=1.0,
        column_density=1.0,
    )


def emphasized_gust_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    fields: lightning.LightningFields,
    minimum_stride: int,
    hazards_only: bool = False,
) -> None:
    sample = vector_sample(ax, lon.shape, minimum_stride)
    u = fields.u10_ms[sample]
    v = fields.v10_ms[sample]
    gust = fields.gust_kmh[sample]
    speed = np.hypot(u, v)
    unit_u = np.divide(u, speed, out=np.zeros_like(u), where=speed > 0.05)
    unit_v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 0.05)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(gust) & (speed > 0.05)
    categories = (
        ((gust < 40.0), "#7e878c", 0.32, 0.00115, 0.15),
        (((gust >= 40.0) & (gust < 60.0)), "#3f464a", 0.80, 0.00145, 0.25),
        (((gust >= 60.0) & (gust < 80.0)), "#151719", 0.95, 0.00185, 0.38),
        ((gust >= 80.0), "#ffffff", 1.00, 0.00220, 0.80),
    )
    for index, (category, color, alpha, width, edge_width) in enumerate(categories):
        if hazards_only and index == 0:
            continue
        mask = finite & category
        if not np.any(mask):
            continue
        ax.quiver(
            lon[sample][mask],
            lat[sample][mask],
            unit_u[mask],
            unit_v[mask],
            color=mcolors.to_rgba(color, alpha),
            edgecolors=mcolors.to_rgba("#151719", 0.90 if index == 3 else 0.40),
            linewidths=edge_width,
            transform=lightning.DATA_CRS,
            scale_units="width",
            scale=67,
            width=width,
            headwidth=3.5,
            headlength=4.2,
            headaxislength=3.9,
            minlength=0.05,
            pivot="middle",
            zorder=11,
        )


def colored_gust_vectors(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    fields: lightning.LightningFields,
    minimum_stride: int,
):
    sample = vector_sample(ax, lon.shape, minimum_stride)
    u = fields.u10_ms[sample]
    v = fields.v10_ms[sample]
    gust = fields.gust_kmh[sample]
    speed = np.hypot(u, v)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(gust) & (speed > 0.05)
    unit_u = np.divide(u, speed, out=np.zeros_like(u), where=speed > 0.05)
    unit_v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 0.05)
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
        scale=67,
        width=0.00165,
        headwidth=3.5,
        headlength=4.2,
        headaxislength=3.9,
        minlength=0.05,
        pivot="middle",
        edgecolors=mcolors.to_rgba(lightning.GUST_VECTOR_EDGE_COLOR, 0.65),
        linewidths=0.28,
        zorder=11,
    )
    return vectors, levels


def dry_lightning(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    dry_potential: np.ndarray,
    stride: int,
) -> None:
    dry = sampled(dry_potential, stride)
    mask = dry >= 15.0
    if not np.any(mask):
        return
    ax.scatter(
        sampled(lon, stride)[mask],
        sampled(lat, stride)[mask],
        marker=lightning.DRY_LIGHTNING_MARKER,
        s=lightning.dry_lightning_marker_area("se"),
        color=lightning.DRY_LIGHTNING_COLOR,
        linewidths=0.40,
        transform=lightning.DATA_CRS,
        zorder=15,
    )


def precipitation_dots(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    precip_mm: np.ndarray,
) -> None:
    stride = lightning.grid_stride(12.0)
    sample_lon = sampled(lon, stride)
    sample_lat = sampled(lat, stride)
    precip = sampled(precip_mm, stride)
    moderate = np.isfinite(precip) & (precip >= 2.5) & (precip < 10.0)
    heavy = np.isfinite(precip) & (precip >= 10.0)
    if np.any(moderate):
        ax.scatter(
            sample_lon[moderate],
            sample_lat[moderate],
            marker="o",
            s=8.0,
            color="#65b9e8",
            edgecolors="none",
            alpha=0.82,
            transform=lightning.DATA_CRS,
            zorder=9,
        )
    if np.any(heavy):
        ax.scatter(
            sample_lon[heavy],
            sample_lat[heavy],
            marker="o",
            s=13.0,
            color="#174f9d",
            edgecolors="white",
            linewidths=0.20,
            transform=lightning.DATA_CRS,
            zorder=9,
        )


def projected_transmission_lines(lines: list[BaseGeometry]) -> tuple[BaseGeometry, ...]:
    return tuple(lightning.PLOT_CRS.project_geometry(line, lightning.DATA_CRS) for line in lines)


def add_transmission_lines(ax: plt.Axes, projected: tuple[BaseGeometry, ...]) -> None:
    ax.add_geometries(
        projected,
        crs=lightning.PLOT_CRS,
        facecolor="none",
        edgecolor=mcolors.to_rgba("white", 0.86),
        linewidth=2.5,
        zorder=7,
    )
    ax.add_geometries(
        projected,
        crs=lightning.PLOT_CRS,
        facecolor="none",
        edgecolor=lightning.TRANSMISSION_LINES_COLOR,
        linewidth=1.35,
        alpha=0.88,
        zorder=8,
    )


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    ticks,
    label: str,
    title: str,
    **kwargs,
) -> None:
    plot_style.add_internal_colorbar(
        fig,
        ax,
        mappable,
        ticks=ticks,
        label=label,
        title=title,
        backdrop=(0.010, 0.080, 0.052, 0.790),
        cax_bounds=(0.022, 0.100, 0.019, 0.750),
        tick_position="right",
        **kwargs,
    )


def render_option(
    option: int,
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    region: lightning.RegionConfig,
    lat: np.ndarray,
    lon: np.ndarray,
    fields: lightning.LightningFields,
    danger: np.ndarray,
    transmission_lines: tuple[BaseGeometry, ...],
    shade_stride: int,
    contour_stride: int,
) -> Path:
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=lightning.PLOT_CRS)
    add_base(ax, lightning.region_extent(region))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if option == 1:
            main_mappable, main_ticks = lpi_fill(ax, lon, lat, fields.potential, shade_stride)
            hrdps.add_hydro_features(ax)
            add_transmission_lines(ax, transmission_lines)
            precipitation_dots(ax, lon, lat, fields.precip_3h)
            rh_contours(ax, lon, lat, fields.surface_rh)
            emphasized_gust_vectors(ax, lon, lat, fields, shade_stride)
            danger_contours(ax, lon, lat, danger)
            dry_lightning(ax, lon, lat, fields.dry_potential, max(contour_stride, shade_stride * 2))
            add_colorbar(
                fig,
                ax,
                main_mappable,
                main_ticks,
                "Lightning potential index",
                "LPI",
                fmt="%g",
                extend="max",
            )
            option_name = "LPI-FIRST"
            footer = (
                "LPI shaded | Danger labelled contours | RH 20/30% brown, 60/80% blue contours | "
                "Gust arrows: faint <40, heavier 40/60/80+ | Dry lightning black * | Rain dots 2.5/10 mm"
            )
        elif option == 2:
            main_mappable = danger_fill(ax, lon, lat, danger)
            hrdps.add_hydro_features(ax)
            add_transmission_lines(ax, transmission_lines)
            precipitation_dots(ax, lon, lat, fields.precip_3h)
            selective_rh(ax, lon, lat, fields.surface_rh, include_sixty_boundary=True)
            emphasized_gust_vectors(ax, lon, lat, fields, shade_stride)
            lpi_contours(ax, lon, lat, fields.potential)
            dry_lightning(ax, lon, lat, fields.dry_potential, max(contour_stride, shade_stride * 2))
            add_colorbar(
                fig,
                ax,
                main_mappable,
                DANGER_TICKS,
                "Peak daily fire danger",
                "DANGER",
                fmt="%g",
                tick_labels=DANGER_LABELS,
            )
            option_name = "DANGER CONTEXT + WEATHER TRIGGERS"
            footer = (
                "Danger filled | LPI 20/40/60/80 purple contours | RH brown <30%, blue >80% hatch, 60% line | "
                "Gust arrows: faint <40, heavier 40/60/80+ | Dry lightning black * | Rain dots 2.5/10 mm"
            )
        elif option == 3:
            rh_fill(ax, lon, lat, fields.surface_rh)
            hrdps.add_hydro_features(ax)
            add_transmission_lines(ax, transmission_lines)
            precipitation_dots(ax, lon, lat, fields.precip_3h)
            main_mappable, main_ticks = colored_gust_vectors(ax, lon, lat, fields, shade_stride)
            lpi_contours(ax, lon, lat, fields.potential)
            danger_contours(ax, lon, lat, danger)
            dry_lightning(ax, lon, lat, fields.dry_potential, max(contour_stride, shade_stride * 2))
            add_colorbar(
                fig,
                ax,
                main_mappable,
                main_ticks,
                "All-cause gust (km h$^{-1}$)",
                "GUST",
                fmt="%g",
                extend="both",
            )
            option_name = "RH + WIND FIRST"
            footer = (
                "RH filled: brown <30%, blue >60% | Sparse gust vectors | LPI purple contours | "
                "Danger labelled contours | Dry lightning black * | Rain dots 2.5/10 mm"
            )
        elif option == 4:
            main_mappable, main_ticks = selective_lpi_zones(
                ax, lon, lat, fields.potential, shade_stride
            )
            hrdps.add_hydro_features(ax)
            add_transmission_lines(ax, transmission_lines)
            precipitation_dots(ax, lon, lat, fields.precip_3h)
            selective_rh(ax, lon, lat, fields.surface_rh, include_sixty_boundary=False)
            emphasized_gust_vectors(ax, lon, lat, fields, shade_stride, hazards_only=True)
            danger_contours(ax, lon, lat, danger)
            dry_lightning(ax, lon, lat, fields.dry_potential, max(contour_stride, shade_stride * 2))
            add_colorbar(
                fig,
                ax,
                main_mappable,
                main_ticks,
                "Lightning potential zones",
                "LPI",
                fmt="%g",
                extend="max",
            )
            option_name = "SELECTIVE HAZARDS"
            footer = (
                "Only LPI >=20 tinted | Gust arrows only >=40 km/h | RH hatch only <30% and >80% | "
                "Danger labelled contours | Dry lightning black * | Rain dots 2.5/10 mm"
            )
        else:
            raise ValueError(f"Unsupported option: {option}")

    hrdps.add_city_labels(ax, fontsize=7.3, marker_size=2.3, path_width=2.45, zorder=30)
    plot_style.add_single_panel_text(
        ax,
        plot_style.valid_header(
            run,
            fhour,
            f"OPTION {option}: {option_name} | {hrdps.model_config().label} {region.label}",
        ),
        footer,
        run,
        source_label="ECCC HRDPS + CWFIS",
        header_y=0.998,
        source_x=0.999,
        source_y=0.966,
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
    parser.add_argument("--region", choices=sorted(lightning.FIRE_WEATHER_REGIONS), default=DEFAULT_REGION)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hrdps_west"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hrdps.set_model("west")
    lightning.set_model("west")
    run = hrdps.RunInfo(
        cycle=args.stamp[9:11],
        stamp=args.stamp,
        init_time=hrdps.parse_stamp(args.stamp),
    )
    region = lightning.region_config(args.region)
    lat, lon, fields, peak_danger = load_fields(run, args.fhour, args.data_dir)
    yslice, xslice = hrdps.subset_slices(lat, lon, lightning.region_extent(region))
    regional_fields = lightning.subset_lightning_fields(fields, yslice, xslice)
    regional_danger = peak_danger[yslice, xslice]
    regional_lat = lat[yslice, xslice]
    regional_lon = lon[yslice, xslice]
    transmission = projected_transmission_lines(lightning.load_transmission_lines())
    shade_stride = hrdps.grid_stride(5.0)
    contour_stride = hrdps.grid_stride(12.0)
    for option in (1, 2, 3, 4):
        render_option(
            option,
            args.output_dir / f"fire_weather_regional_option{option}_{args.region}_{run.stamp}_f{args.fhour:03d}.png",
            run,
            args.fhour,
            region,
            regional_lat,
            regional_lon,
            regional_fields,
            regional_danger,
            transmission,
            shade_stride,
            contour_stride,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
