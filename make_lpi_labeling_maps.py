#!/usr/bin/env python3
"""Build blinded observed-lightning case maps for BCH meteorologist labels."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from rasterio.windows import Window
from shapely import contains_xy
from shapely.geometry import GeometryCollection, LineString, MultiLineString, box, shape

import lightning_ml_archive as archive


UTC = dt.timezone.utc
MAP_EXTENT = (-139.2, -114.0, 48.2, 60.1)
CELL_AREA_KM2 = 6.25
POINTS_PER_SYMBOL = 1
POINT_LIMIT = 75_000
LABEL_VERSION = "bch_lpi_labels_v1"
DEFAULT_OUTPUT = Path("output/lpi_met_labeling/initial_20260812")
PERIOD_HOURS = tuple(range(3, 25, 3))
LINE_COLOR = "#153f8f"
BOUNDARY_COLOR = "#343434"


def parse_obs_end(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.name[:13], "%Y%m%dT%H%M").replace(tzinfo=UTC)


def observation_paths(root: Path) -> dict[dt.datetime, Path]:
    path = root / "observations/eccc_lightning_3h/schema_v1"
    return {parse_obs_end(item): item for item in path.glob("*/*/*.tif")}


def complete_days(paths: dict[dt.datetime, Path]) -> list[dt.datetime]:
    starts: list[dt.datetime] = []
    if not paths:
        return starts
    first = min(paths)
    last = max(paths)
    cursor = dt.datetime.combine(first.date(), dt.time(12), tzinfo=UTC) - dt.timedelta(days=1)
    while cursor + dt.timedelta(days=1) <= last:
        expected = [cursor + dt.timedelta(hours=hour) for hour in PERIOD_HOURS]
        if all(item in paths for item in expected):
            starts.append(cursor)
        cursor += dt.timedelta(days=1)
    return starts


def case_id(start: dt.datetime) -> str:
    digest = hashlib.sha256(f"{LABEL_VERSION}:{start.isoformat()}".encode()).hexdigest()[:8].upper()
    return f"L{digest}"


def crop_window(source: rasterio.io.DatasetReader, extent: tuple[float, float, float, float]) -> Window:
    west, east, south, north = extent
    return source.window(west, south, east, north).round_offsets().round_lengths()


def read_density(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        window = crop_window(source, MAP_EXTENT)
        data = source.read(1, window=window, boundless=True, fill_value=source.nodata or -999.0).astype(np.float32)
        transform = source.window_transform(window)
        nodata = source.nodata
    data = np.where(np.isfinite(data) & (data != nodata) & (data > 0.0), data, 0.0)
    rows, cols = np.indices(data.shape)
    lon, lat = rasterio.transform.xy(transform, rows, cols, offset="center")
    return (
        data,
        np.asarray(lon, dtype=np.float32).reshape(data.shape),
        np.asarray(lat, dtype=np.float32).reshape(data.shape),
    )


def synthetic_strike_points(
    density: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    seed: int,
    points_per_symbol: int = POINTS_PER_SYMBOL,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Place deterministic display points inside cells from gridded flash density."""

    expected_flashes = np.maximum(density, 0.0).astype(np.float64) * CELL_AREA_KM2
    expected_symbols = expected_flashes / max(1, points_per_symbol)
    base = np.floor(expected_symbols).astype(np.int32)
    random = np.random.default_rng(seed)
    counts = base + (random.random(base.shape) < (expected_symbols - base))
    rows, cols = np.nonzero(counts)
    repetitions = counts[rows, cols]
    total = int(repetitions.sum())
    if total == 0:
        return np.empty(0), np.empty(0), 0
    repeated_rows = np.repeat(rows, repetitions)
    repeated_cols = np.repeat(cols, repetitions)
    if total > POINT_LIMIT:
        selection = random.choice(total, POINT_LIMIT, replace=False)
        repeated_rows = repeated_rows[selection]
        repeated_cols = repeated_cols[selection]
    dx = float(np.nanmedian(np.abs(np.diff(lon, axis=1))))
    dy = float(np.nanmedian(np.abs(np.diff(lat, axis=0))))
    point_lon = lon[repeated_rows, repeated_cols] + random.uniform(-0.5 * dx, 0.5 * dx, len(repeated_rows))
    point_lat = lat[repeated_rows, repeated_cols] + random.uniform(-0.5 * dy, 0.5 * dy, len(repeated_rows))
    return point_lon, point_lat, total * max(1, points_per_symbol)


def geometry_lines(geometry: object) -> list[np.ndarray]:
    if isinstance(geometry, LineString):
        return [np.asarray(geometry.coords)]
    if isinstance(geometry, MultiLineString):
        return [np.asarray(line.coords) for line in geometry.geoms]
    if isinstance(geometry, GeometryCollection):
        return [coords for item in geometry.geoms for coords in geometry_lines(item)]
    boundary = getattr(geometry, "boundary", None)
    return geometry_lines(boundary) if boundary is not None else []


def load_map_geometries() -> tuple[list[np.ndarray], list[np.ndarray], object]:
    extent_shape = box(MAP_EXTENT[0], MAP_EXTENT[2], MAP_EXTENT[1], MAP_EXTENT[3])
    collection = json.loads(Path("data/bc_transmission_lines.geojson").read_text())
    transmission: list[np.ndarray] = []
    for feature in collection["features"]:
        if not feature.get("geometry"):
            continue
        clipped = shape(feature["geometry"]).intersection(extent_shape)
        transmission.extend(geometry_lines(clipped))

    from make_experimental_danger_class import load_bc_geometry

    bc = load_bc_geometry().intersection(extent_shape)
    return transmission, geometry_lines(bc.boundary), bc.buffer(0.4).intersection(extent_shape)


def setup_axis(ax: plt.Axes, transmission: list[np.ndarray], boundary: list[np.ndarray]) -> None:
    ax.add_collection(
        LineCollection(transmission, colors=LINE_COLOR, linewidths=0.32, alpha=0.72, zorder=3)
    )
    ax.add_collection(LineCollection(boundary, colors=BOUNDARY_COLOR, linewidths=0.75, zorder=4))
    ax.set_xlim(MAP_EXTENT[:2])
    ax.set_ylim(MAP_EXTENT[2:])
    ax.set_aspect(1.0 / math.cos(math.radians(54.0)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8f8f6")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)


def density_cmap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    colors = ["#ffffff00", "#d7edf6", "#95c9e2", "#f5d55c", "#ef8a34", "#c8322f"]
    levels = [0.0, 0.001, 0.02, 0.10, 0.50, 2.0, 100.0]
    cmap = mcolors.ListedColormap(colors)
    return cmap, mcolors.BoundaryNorm(levels, cmap.N)


def draw_observations(
    ax: plt.Axes,
    density: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    point_lon: np.ndarray,
    point_lat: np.ndarray,
) -> None:
    cmap, norm = density_cmap()
    ax.pcolormesh(lon, lat, density, cmap=cmap, norm=norm, shading="nearest", rasterized=True, zorder=1)
    if len(point_lon):
        ax.scatter(point_lon, point_lat, s=2.2, color="#1b1b1b", alpha=0.64, linewidths=0, zorder=2)


def make_case_maps(
    output: Path,
    start: dt.datetime,
    paths: dict[dt.datetime, Path],
    transmission: list[np.ndarray],
    boundary: list[np.ndarray],
    display_geometry: object,
) -> tuple[Path, Path, int]:
    identifier = case_id(start)
    ends = [start + dt.timedelta(hours=hour) for hour in PERIOD_HOURS]
    blocks = [read_density(paths[end]) for end in ends]
    lon = blocks[0][1]
    lat = blocks[0][2]
    display_mask = contains_xy(display_geometry, lon, lat)
    densities = [np.where(display_mask, item[0], 0.0) for item in blocks]
    daily_density = np.sum(densities, axis=0)
    seed_base = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)

    daily_points = synthetic_strike_points(daily_density, lon, lat, seed_base)
    daily_path = output / "cases" / f"{identifier}_24h.png"
    panel_path = output / "cases" / f"{identifier}_3h.png"
    if daily_path.exists() and panel_path.exists():
        return daily_path, panel_path, daily_points[2]
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 8.0), constrained_layout=True)
    setup_axis(ax, transmission, boundary)
    draw_observations(ax, daily_density, lon, lat, daily_points[0], daily_points[1])
    ax.set_title(f"Case {identifier} | 24-hour observed lightning", fontsize=16, fontweight="bold", pad=8)
    fig.text(
        0.5,
        0.005,
        "BCH transmission lines (blue). Dots are simulated placements within observed 2.5 km density cells; they are not strike coordinates.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.savefig(daily_path, dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.0), constrained_layout=True)
    for index, (ax, density) in enumerate(zip(axes.flat, densities), start=1):
        setup_axis(ax, transmission, boundary)
        point_data = synthetic_strike_points(density, lon, lat, seed_base + index)
        draw_observations(ax, density, lon, lat, point_data[0], point_data[1])
        ax.set_title(f"Period {index}", fontsize=11, fontweight="bold", pad=3)
    handles = [
        Line2D([0], [0], color=LINE_COLOR, linewidth=1.4, label="BCH transmission"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b1b1b", markersize=4, label="Synthetic flash placement"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Case {identifier} | Eight consecutive 3-hour periods", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.002,
        "Dots are simulated within observed 2.5 km density cells and are not strike coordinates. Period times remain blinded.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.savefig(panel_path, dpi=160, facecolor="white")
    plt.close(fig)
    return daily_path, panel_path, daily_points[2]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_package(args: argparse.Namespace) -> None:
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    paths = observation_paths(args.archive_root.expanduser())
    days = complete_days(paths)
    if args.limit:
        days = days[-args.limit :]
    if not days:
        raise RuntimeError("No complete 12Z-to-12Z observed-lightning days are available.")
    transmission, boundary, display_geometry = load_map_geometries()
    key_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index, start in enumerate(days, start=1):
        identifier = case_id(start)
        daily_path, panel_path, flash_count = make_case_maps(
            output, start, paths, transmission, boundary, display_geometry
        )
        key_rows.append(
            {
                "case_id": identifier,
                "period_start_utc": start.isoformat().replace("+00:00", "Z"),
                "period_end_utc": (start + dt.timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                "daily_map": str(daily_path),
                "three_hour_map": str(panel_path),
                "synthetic_daily_flash_count": flash_count,
            }
        )
        for period_index in range(0, 9):
            label_rows.append(
                {
                    "case_id": identifier,
                    "met_id": "",
                    "period_index": period_index,
                    "period_hours": 24 if period_index == 0 else 3,
                    "geometry_id": "overall",
                    "line_segment_id": "",
                    "rating": "",
                    "confidence": "",
                    "notes": "",
                    "label_version": LABEL_VERSION,
                    "created_at_utc": "",
                }
            )
        print(f"cases {index}/{len(days)}: {identifier}", flush=True)
    write_csv(output / "case_key_PRIVATE.csv", key_rows)
    write_csv(output / "labels_template.csv", label_rows)
    (output / "README.txt").write_text(
        "Give raters the cases directory and labels_template.csv. Keep case_key_PRIVATE.csv separate until adjudication.\n"
        "period_index 0 is the 24-hour rating; indices 1-8 correspond to the blinded three-hour panels.\n"
        "Allowed ratings: none, low, moderate, high, uncertain.\n"
        "Synthetic dots are deterministic display placements from ECCC gridded density, not observed strike coordinates.\n"
    )
    print(f"Wrote {len(days)} blinded cases to {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=archive.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Generate only the latest N complete days")
    return parser


if __name__ == "__main__":
    build_package(build_parser().parse_args())
