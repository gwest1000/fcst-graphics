#!/usr/bin/env python3
"""Shared forecast plot canvas, text, and colorbar layout helpers."""

from __future__ import annotations

import datetime as dt
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LOCAL_TZ = ZoneInfo("America/Vancouver")

BC_PLOT_FIGSIZE = (14.4, 9.0)
PLOT_DPI = 100
BC_PLOT_ASPECT = BC_PLOT_FIGSIZE[0] / BC_PLOT_FIGSIZE[1]
BC_SINGLE_PANEL_AX_POS = [0.001, 0.003, 0.998, 0.994]
BC_FOURPANEL_POSITIONS = [
    [0.0005, 0.5045, 0.4990, 0.4910],
    [0.5005, 0.5045, 0.4990, 0.4910],
    [0.0005, 0.0045, 0.4990, 0.4910],
    [0.5005, 0.0045, 0.4990, 0.4910],
]

PLOT_FIGSIZE = BC_PLOT_FIGSIZE
PLOT_ASPECT = BC_PLOT_ASPECT
SINGLE_PANEL_AX_POS = BC_SINGLE_PANEL_AX_POS
FOURPANEL_POSITIONS = BC_FOURPANEL_POSITIONS

COLORBAR_BACKDROP = (0.926, 0.055, 0.066, 0.890)
COLORBAR_AX = [0.956, 0.075, 0.020, 0.840]

SINGLE_HEADER_FONTSIZE = 11.0 * 0.90
SINGLE_FOOTER_FONTSIZE = 9.0 * 0.90
SINGLE_SOURCE_FONTSIZE = 7.6

FOURPANEL_HEADER_FONTSIZE = 9.4 * 0.90
FOURPANEL_FOOTER_FONTSIZE = 7.7 * 0.90
FOURPANEL_SOURCE_FONTSIZE = 6.9
TEXT_BAND_EDGE_WIDTH = 0.75
SINGLE_HEADER_BAND_HEIGHT = 0.034
SINGLE_FOOTER_BAND_HEIGHT = 0.027
FOURPANEL_HEADER_BAND_HEIGHT = 0.050
FOURPANEL_FOOTER_BAND_HEIGHT = 0.042
FOURPANEL_COLORBAR_BACKDROP = (
    0.925,
    FOURPANEL_FOOTER_BAND_HEIGHT,
    0.075,
    1.0 - FOURPANEL_HEADER_BAND_HEIGHT - FOURPANEL_FOOTER_BAND_HEIGHT,
)
FOURPANEL_COLORBAR_AX = [
    0.972,
    FOURPANEL_FOOTER_BAND_HEIGHT,
    0.028,
    1.0 - FOURPANEL_HEADER_BAND_HEIGHT - FOURPANEL_FOOTER_BAND_HEIGHT,
]
FOURPANEL_BARB_COLUMNS = 16
FOURPANEL_BARB_ROWS = 12
VECTOR_TARGET_SPACING_PX = 42.0
VECTOR_MIN_COLUMNS = 6
VECTOR_MIN_ROWS = 5


def vector_stride_for_shape(
    shape: tuple[int, int],
    target_columns: int = FOURPANEL_BARB_COLUMNS,
    target_rows: int = FOURPANEL_BARB_ROWS,
    minimum: int = 1,
) -> int:
    """Return a grid stride that keeps vector plots near a visual target count."""
    if len(shape) != 2:
        return max(1, int(minimum))
    rows, columns = shape
    row_stride = max(1, (rows + max(1, target_rows) - 1) // max(1, target_rows))
    column_stride = max(1, (columns + max(1, target_columns) - 1) // max(1, target_columns))
    return max(1, int(minimum), row_stride, column_stride)


def axes_pixel_size(ax: plt.Axes) -> tuple[float, float]:
    fig = ax.figure
    width_in, height_in = fig.get_size_inches()
    bbox = ax.get_position(original=True)
    return width_in * fig.dpi * bbox.width, height_in * fig.dpi * bbox.height


def vector_target_count(
    ax: plt.Axes,
    spacing_px: float = VECTOR_TARGET_SPACING_PX,
    min_columns: int = VECTOR_MIN_COLUMNS,
    min_rows: int = VECTOR_MIN_ROWS,
    row_density: float = 1.0,
    column_density: float = 1.0,
) -> tuple[int, int]:
    width_px, height_px = axes_pixel_size(ax)
    spacing_px = max(1.0, float(spacing_px))
    columns = max(int(min_columns), int(round((width_px / spacing_px) * max(0.1, column_density))))
    rows = max(int(min_rows), int(round((height_px / spacing_px) * max(0.1, row_density))))
    return rows, columns


def vector_strides_for_axes(
    ax: plt.Axes,
    shape: tuple[int, int],
    minimum: int = 1,
    spacing_px: float = VECTOR_TARGET_SPACING_PX,
    row_density: float = 1.0,
    column_density: float = 1.0,
) -> tuple[int, int]:
    """Return row/column strides for stable vector density in rendered pixels."""
    if len(shape) != 2:
        minimum = max(1, int(minimum))
        return minimum, minimum
    rows, columns = shape
    target_rows, target_columns = vector_target_count(
        ax,
        spacing_px=spacing_px,
        row_density=row_density,
        column_density=column_density,
    )
    row_stride = max(1, (rows + target_rows - 1) // target_rows)
    column_stride = max(1, (columns + target_columns - 1) // target_columns)
    minimum = max(1, int(minimum))
    return max(minimum, row_stride), max(minimum, column_stride)


def vector_sample_slices(
    ax: plt.Axes,
    shape: tuple[int, int],
    minimum: int = 1,
    spacing_px: float = VECTOR_TARGET_SPACING_PX,
    row_density: float = 1.0,
    column_density: float = 1.0,
) -> tuple[slice, slice]:
    row_stride, column_stride = vector_strides_for_axes(
        ax,
        shape,
        minimum=minimum,
        spacing_px=spacing_px,
        row_density=row_density,
        column_density=column_density,
    )
    row_offset = max(row_stride // 2, 1)
    column_offset = max(column_stride // 2, 1)
    return slice(row_offset, None, row_stride), slice(column_offset, None, column_stride)


def valid_header(run, fhour: int, model_label: str = "HRDPS") -> str:
    valid = run.init_time + dt.timedelta(hours=fhour)
    valid_local = valid.astimezone(LOCAL_TZ)
    local_date = valid_local.strftime("%d%b%Y").upper()
    utc_date = valid.strftime("%d%b%Y").upper()
    return f"{model_label}  |  {valid_local:%a} {valid_local:%H:%M%Z} {local_date}  |  {valid:%H:%MUTC} {utc_date}"


def add_internal_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    ticks: Iterable[float],
    label: str,
    title: str | None = None,
    fmt: str | None = None,
    tick_labels: Iterable[str] | None = None,
    extend: str | None = None,
    backdrop: tuple[float, float, float, float] | None = None,
    cax_bounds: list[float] | tuple[float, float, float, float] | None = None,
    tick_position: str = "left",
    labelpad: float = 2.0,
    backdrop_edgecolor: str = "none",
    backdrop_linewidth: float = 0.0,
) -> None:
    backdrop = backdrop or COLORBAR_BACKDROP
    cax_bounds = cax_bounds or COLORBAR_AX
    ax.add_patch(
        Rectangle(
            backdrop[:2],
            backdrop[2],
            backdrop[3],
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=backdrop_edgecolor,
            linewidth=backdrop_linewidth,
            alpha=1.0,
            zorder=34,
        )
    )
    cax = ax.inset_axes(cax_bounds)
    cax.set_zorder(50)
    cax.set_facecolor("white")
    colorbar_kwargs = {"cax": cax, "ticks": list(ticks), "format": fmt}
    if extend is not None:
        colorbar_kwargs["extend"] = extend
    cbar = fig.colorbar(mappable, **colorbar_kwargs)
    cbar.outline.set_linewidth(0.7)
    tick_position = tick_position if tick_position in {"left", "right"} else "left"
    cbar.ax.yaxis.set_ticks_position(tick_position)
    cbar.ax.yaxis.set_label_position(tick_position)
    cbar.ax.tick_params(
        labelsize=6.8,
        length=2.2,
        pad=1.0,
        labelleft=tick_position == "left",
        labelright=tick_position == "right",
    )
    if tick_labels is not None:
        cbar.ax.set_yticklabels(list(tick_labels))
    cbar.set_label(label, fontsize=7.0, labelpad=labelpad)
    if title:
        cbar.ax.set_title(title, fontsize=7.2, fontweight="bold", pad=3.0)


def add_fourpanel_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    ticks: Iterable[float],
    label: str,
    **kwargs,
) -> None:
    add_internal_colorbar(
        fig,
        ax,
        mappable,
        ticks=ticks,
        label=label,
        backdrop=FOURPANEL_COLORBAR_BACKDROP,
        cax_bounds=FOURPANEL_COLORBAR_AX,
        **kwargs,
    )


def add_source_stamp(
    ax: plt.Axes,
    run,
    source_label: str = "CMC",
    fontsize: float = FOURPANEL_SOURCE_FONTSIZE,
    x: float = 0.986,
    y: float = 0.978,
) -> None:
    ax.text(
        x,
        y,
        f"Data:{source_label} | Init:{run.init_time:%Y%m%d%H}",
        transform=ax.transAxes,
        fontsize=fontsize,
        color="black",
        ha="right",
        va="top",
        zorder=45,
        bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def add_header(
    ax: plt.Axes,
    header: str,
    fontsize: float,
    y: float,
) -> None:
    ax.text(
        0.5,
        y,
        header,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        zorder=45,
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def add_footer(
    ax: plt.Axes,
    footer: str,
    fontsize: float,
    stroke_width: float,
) -> None:
    ax.text(
        0.5,
        0.014,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        zorder=45,
        path_effects=[path_effects.withStroke(linewidth=stroke_width, foreground="white", alpha=0.96)],
    )


def add_text_bands(
    ax: plt.Axes,
    header: str,
    footer: str,
    source: str,
    *,
    header_fontsize: float,
    footer_fontsize: float,
    source_fontsize: float,
    header_height: float,
    footer_height: float,
) -> None:
    """Draw consistent full-width title and footer bands inside a plot panel."""
    for y, height in ((1.0 - header_height, header_height), (0.0, footer_height)):
        ax.add_patch(
            Rectangle(
                (0.0, y),
                1.0,
                height,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="black",
                linewidth=TEXT_BAND_EDGE_WIDTH,
                zorder=76,
            )
        )

    header_y = 1.0 - header_height / 2.0
    ax.text(
        0.006,
        header_y,
        header,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=header_fontsize,
        fontweight="bold",
        color="black",
        zorder=77,
    )
    ax.text(
        0.994,
        header_y,
        source,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=source_fontsize,
        color="black",
        zorder=78,
    )
    ax.text(
        0.5,
        footer_height / 2.0,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=footer_fontsize,
        fontweight="normal",
        color="black",
        zorder=77,
    )


def add_single_panel_text(
    ax: plt.Axes,
    header: str,
    footer: str,
    run,
    source_label: str = "CMC",
    *,
    header_y: float = 0.992,
    source_x: float = 0.986,
    source_y: float = 0.966,
) -> None:
    del header_y, source_x, source_y
    add_text_bands(
        ax,
        header,
        footer,
        f"Data:{source_label} | Init:{run.init_time:%Y%m%d%H}",
        header_fontsize=SINGLE_HEADER_FONTSIZE,
        footer_fontsize=SINGLE_FOOTER_FONTSIZE,
        source_fontsize=SINGLE_SOURCE_FONTSIZE,
        header_height=SINGLE_HEADER_BAND_HEIGHT,
        footer_height=SINGLE_FOOTER_BAND_HEIGHT,
    )


def add_fourpanel_text(ax: plt.Axes, header: str, footer: str, run, source_label: str = "CMC") -> None:
    add_text_bands(
        ax,
        header,
        footer,
        f"Data:{source_label} | Init:{run.init_time:%Y%m%d%H}",
        header_fontsize=FOURPANEL_HEADER_FONTSIZE,
        footer_fontsize=FOURPANEL_FOOTER_FONTSIZE,
        source_fontsize=FOURPANEL_SOURCE_FONTSIZE,
        header_height=FOURPANEL_HEADER_BAND_HEIGHT,
        footer_height=FOURPANEL_FOOTER_BAND_HEIGHT,
    )
