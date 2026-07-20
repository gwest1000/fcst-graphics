#!/usr/bin/env python3
"""Render alternative presentations of the convective four-panel diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import make_hrdps_west_convective as hrdps
import make_hrdps_west_fourpanel as fourpanel
import plot_style


OPTIONS = {
    "reduced_hatching": "Option A: Reduced CAPE hatching",
    "cape_contours": "Option B: CAPE contours",
    "cape_shaded": "Option C: CAPE-first shading",
}


def plot_ipw_shading(ax, lon, lat, ipw, shade_stride: int):
    cmap, norm, levels = fourpanel.make_ipw_cmap()
    return ax.contourf(
        fourpanel.decimate(lon, shade_stride),
        fourpanel.decimate(lat, shade_stride),
        fourpanel.decimate(fourpanel.smooth_nan(ipw, fourpanel.sigma_for_km(7.5)), shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=fourpanel.DATA_CRS,
        transform_first=True,
        zorder=3,
    )


def plot_li_contours(ax, lon, lat, li, contour_stride: int, levels=(-6, -4, -2, 0)) -> None:
    all_levels = (-6, -4, -2, 0)
    all_colors = ("#7b3294", "#d7191c", "#f28e2b", "black")
    all_widths = fourpanel.LI_LINEWIDTHS
    selected = [all_levels.index(level) for level in levels]
    clat, clon, cli = fourpanel.contour_grid(
        lat,
        lon,
        li,
        stride=contour_stride,
        sigma=fourpanel.sigma_for_km(8.0),
    )
    contours = ax.contour(
        clon,
        clat,
        cli,
        levels=[all_levels[index] for index in selected],
        colors=[all_colors[index] for index in selected],
        linewidths=[all_widths[index] for index in selected],
        transform=fourpanel.DATA_CRS,
        zorder=22,
    )
    fourpanel.label_contours(
        contours,
        fontsize=6.6,
        fmt="%d",
        colors=[all_colors[index] for index in selected],
    )


def plot_reduced_hatching(ax, lon, lat, cape, contour_stride: int) -> None:
    clat, clon, ccape = fourpanel.contour_grid(
        lat,
        lon,
        cape,
        stride=contour_stride,
        sigma=fourpanel.sigma_for_km(10.0),
    )
    with plt.rc_context({"hatch.color": "#aaaaaa", "hatch.linewidth": 0.26}):
        ax.contourf(
            clon,
            clat,
            ccape,
            levels=[500, 1000],
            colors="none",
            hatches=["/"],
            transform=fourpanel.DATA_CRS,
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
            transform=fourpanel.DATA_CRS,
            zorder=20,
        )


def plot_cape_contours(ax, lon, lat, cape, contour_stride: int) -> None:
    clat, clon, ccape = fourpanel.contour_grid(
        lat,
        lon,
        cape,
        stride=contour_stride,
        sigma=fourpanel.sigma_for_km(10.0),
    )
    contours = ax.contour(
        clon,
        clat,
        ccape,
        levels=[250, 500, 1000],
        colors=["#858585", "#555555", "#242424"],
        linewidths=[0.9, 1.25, 1.75],
        linestyles=["dotted", "dashed", "solid"],
        transform=fourpanel.DATA_CRS,
        zorder=21,
    )


def plot_cape_shading(ax, lon, lat, cape, shade_stride: int):
    levels = [0, 250, 500, 750, 1000, 1500, 2000, 3000]
    colors = [
        "#f7f7f7",
        "#fff2a8",
        "#ffd05b",
        "#f89b3d",
        "#eb5635",
        "#c52747",
        "#7d1f70",
    ]
    cmap = mcolors.ListedColormap(colors, name="cape_first")
    cmap.set_over("#4a174f")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return ax.contourf(
        fourpanel.decimate(lon, shade_stride),
        fourpanel.decimate(lat, shade_stride),
        fourpanel.decimate(fourpanel.smooth_nan(cape, fourpanel.sigma_for_km(8.0)), shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=fourpanel.DATA_CRS,
        transform_first=True,
        zorder=3,
    ), levels


def plot_ipw_contours(ax, lon, lat, ipw, contour_stride: int) -> None:
    clat, clon, cipw = fourpanel.contour_grid(
        lat,
        lon,
        ipw,
        stride=contour_stride,
        sigma=fourpanel.sigma_for_km(9.0),
    )
    contours = ax.contour(
        clon,
        clat,
        cipw,
        levels=[20, 30, 40],
        colors="#006d77",
        linewidths=1.15,
        transform=fourpanel.DATA_CRS,
        zorder=21,
    )
    fourpanel.label_contours(contours, fontsize=6.3, fmt="%d", colors="#00545b")


def render_option(
    out_path: Path,
    option: str,
    run: hrdps.RunInfo,
    fhour: int,
    lon: np.ndarray,
    lat: np.ndarray,
    ipw: np.ndarray,
    li: np.ndarray,
    cape: np.ndarray,
    watersheds,
    shade_stride: int,
    contour_stride: int,
) -> None:
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=fourpanel.PANEL_PROJ)
    fourpanel.add_base_features(ax)

    if option == "reduced_hatching":
        mappable = plot_ipw_shading(ax, lon, lat, ipw, shade_stride)
        plot_reduced_hatching(ax, lon, lat, cape, contour_stride)
        plot_li_contours(ax, lon, lat, li, contour_stride)
        ticks = np.arange(10, 52, 2)
        label = "IPW (mm)"
        footer = "IPW shaded; LI 0/-2/-4/-6; sparse CAPE hatch at 500 and cross-hatch at 1000 J kg$^{-1}$"
        extend = "both"
    elif option == "cape_contours":
        mappable = plot_ipw_shading(ax, lon, lat, ipw, shade_stride)
        plot_cape_contours(ax, lon, lat, cape, contour_stride)
        plot_li_contours(ax, lon, lat, li, contour_stride)
        ticks = np.arange(10, 52, 2)
        label = "IPW (mm)"
        footer = "IPW shaded; LI 0/-2/-4/-6; CAPE contours 250 dotted, 500 dashed, 1000 solid J kg$^{-1}$"
        extend = "both"
    elif option == "cape_shaded":
        mappable, ticks = plot_cape_shading(ax, lon, lat, cape, shade_stride)
        plot_ipw_contours(ax, lon, lat, ipw, contour_stride)
        plot_li_contours(ax, lon, lat, li, contour_stride, levels=(-4, 0))
        label = "CAPE (J kg$^{-1}$)"
        footer = "CAPE shaded; IPW contours 20/30/40 mm; only LI 0 (black) and -4 (red) retained"
        extend = "max"
    else:
        raise ValueError(f"Unknown option: {option}")

    fourpanel.add_watersheds(ax, watersheds)
    plot_style.add_internal_colorbar(
        fig,
        ax,
        mappable,
        ticks=ticks,
        label=label,
        fmt="%g",
        extend=extend,
    )
    plot_style.add_single_panel_text(
        ax,
        f"{OPTIONS[option]}  |  {plot_style.valid_header(run, fhour)}",
        footer,
        run,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(hrdps.MODEL_CONFIGS), default="continental")
    parser.add_argument("--stamp", default="20260720T06Z")
    parser.add_argument("--fhour", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/convective_panel_options"))
    args = parser.parse_args()

    config = hrdps.set_model(args.model)
    fourpanel.set_model(args.model)
    data_dir = args.data_dir or Path(config.default_data_dir)
    run = hrdps.RunInfo(
        cycle=args.stamp[9:11],
        stamp=args.stamp,
        init_time=hrdps.parse_stamp(args.stamp),
    )
    run_dir = data_dir / run.stamp
    sample_path = fourpanel.hour_file(run_dir, run, args.fhour, "PRES", "SFC", "0")
    psfc_pa, lat, lon = hrdps.read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read HRDPS coordinates.")
    yslice, xslice = hrdps.subset_slices(lat, lon, hrdps.model_config().extent)
    plot_lat = lat[yslice, xslice]
    plot_lon = lon[yslice, xslice]
    psfc_pa = psfc_pa[yslice, xslice]
    ipw = fourpanel.compute_ipw(run_dir, run, args.fhour, psfc_pa, yslice, xslice)
    li = fourpanel.crop(
        fourpanel.hour_file(run_dir, run, args.fhour, "MU-VT-LI", "ISBL", "500"),
        yslice,
        xslice,
    )
    cape = fourpanel.crop(
        fourpanel.hour_file(run_dir, run, args.fhour, "CAPE", "ETAL", "10000"),
        yslice,
        xslice,
    )
    li = np.where(np.abs(li) > 50.0, np.nan, li)
    cape = np.where((cape >= 0.0) & (cape < 20000.0), cape, np.nan)
    watersheds = fourpanel.load_watersheds(hrdps.WATERSHED_CACHE)
    shade_stride = hrdps.grid_stride(5.0)
    contour_stride = hrdps.grid_stride(12.0)

    for option in OPTIONS:
        out_path = args.output_dir / f"{args.stamp}_f{args.fhour:03d}_{option}.png"
        render_option(
            out_path,
            option,
            run,
            args.fhour,
            plot_lon,
            plot_lat,
            ipw,
            li,
            cape,
            watersheds,
            shade_stride,
            contour_stride,
        )
        print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
