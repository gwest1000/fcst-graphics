#!/usr/bin/env python3
"""Create an experimental BC fire danger class graphic from CWFIS FWI+BUI grids.

This is intentionally not wired into the website publisher. Schedule 1 regions
are reconstructed from maintained BC Geographic Warehouse Natural Resource
District polygons corresponding to the former forest regions shown on the legal
map. Freehand lon/lat polygons remain only as an offline fallback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests
from cartopy.io import shapereader
from matplotlib.path import Path as MplPath
from PIL import Image
from shapely import contains_xy, prepare
from shapely.geometry import shape
from shapely.ops import unary_union

import make_hrdps_west_convective as hrdps
import plot_style
from make_hrdps_west_lightning import (
    DATA_CRS,
    FWI_CACHE_DIR,
    FWI_CACHE_RESOLUTION_M,
    FWI_FROM_3978,
    FWI_MAX_DIMENSION,
    FWI_NATIVE_BBOX_3978,
    FWI_TO_3978,
    FWI_WCS_URL,
    PLOT_CRS,
    add_transmission_lines,
    geotiff_projected_lon_lat,
    load_transmission_lines,
)


DANGER_EXTENT = hrdps.MODEL_CONFIGS["west"].extent
DANGER_REGION_CACHE = Path("data/bc_danger_regions/natural_resource_districts.geojson")
NR_DISTRICT_WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/WHSE_ADMIN_BOUNDARIES.ADM_NR_DISTRICTS_SPG/ows"
NR_DISTRICT_LAYER = "pub:WHSE_ADMIN_BOUNDARIES.ADM_NR_DISTRICTS_SPG"
CWFIS_COVERAGES = {
    "ffmc": "public:ffmc",
    "dmc": "public:dmc",
    "dc": "public:dc",
    "fwi": "public:fwi",
    "bui": "public:bui",
}
CWFIS_VALUE_LIMITS = {
    "ffmc": (0.0, 101.0),
    "dmc": (0.0, 1000.0),
    "dc": (0.0, 2000.0),
    "fwi": (0.0, 500.0),
    "bui": (0.0, 1000.0),
}


@dataclass(frozen=True)
class CwfisGrid:
    date: dt.date
    name: str
    lat: np.ndarray
    lon: np.ndarray
    data: np.ndarray


REGION2_POLYGON = np.array(
    [
        (-126.9, 51.35),
        (-126.1, 52.05),
        (-124.8, 52.45),
        (-123.2, 52.95),
        (-121.7, 53.05),
        (-120.5, 52.78),
        (-119.3, 52.55),
        (-118.75, 51.98),
        (-119.45, 51.45),
        (-120.75, 51.32),
        (-122.15, 51.23),
        (-123.35, 51.18),
        (-124.65, 51.28),
        (-126.0, 51.18),
        (-126.9, 51.35),
    ],
    dtype=np.float64,
)

REGION3_POLYGON = np.array(
    [
        (-123.95, 49.00),
        (-123.45, 49.85),
        (-122.90, 50.55),
        (-122.05, 51.05),
        (-120.85, 51.35),
        (-119.50, 51.48),
        (-118.55, 51.15),
        (-117.60, 51.25),
        (-116.60, 51.80),
        (-115.20, 51.55),
        (-113.20, 50.85),
        (-113.20, 49.00),
        (-123.95, 49.00),
    ],
    dtype=np.float64,
)

# Schedule 1 was drawn using former forest-district boundaries. These current
# Natural Resource Districts are their closest maintained BCGW successors and
# reproduce the coloured Schedule 1 areas far better than a freehand outline.
REGION_DISTRICT_NAMES = {
    2: frozenset(
        {
            "100 Mile House Natural Resource District",
            "Cariboo-Chilcotin Natural Resource District",
            "Quesnel Natural Resource District",
        }
    ),
    3: frozenset(
        {
            "Cascades Natural Resource District",
            "Okanagan Shuswap Natural Resource District",
            "Rocky Mountain Natural Resource District",
            "Selkirk Natural Resource District",
            "Thompson Rivers Natural Resource District",
        }
    ),
}


REGION_MATRICES: dict[int, dict[str, object]] = {
    1: {
        "fwi_edges": [1, 8, 17, 31],
        "bui_edges": [20, 43, 70, 119],
        "classes": np.array(
            [
                [1, 2, 2, 3, 3],
                [2, 2, 3, 3, 4],
                [2, 3, 3, 4, 4],
                [2, 3, 4, 4, 5],
                [3, 3, 4, 5, 5],
            ],
            dtype=np.uint8,
        ),
    },
    2: {
        "fwi_edges": [5, 17, 27, 38],
        "bui_edges": [49, 86, 119, 159],
        "classes": np.array(
            [
                [1, 2, 2, 3, 3],
                [2, 2, 3, 3, 4],
                [2, 3, 3, 4, 4],
                [2, 3, 4, 4, 5],
                [3, 3, 4, 5, 5],
            ],
            dtype=np.uint8,
        ),
    },
    3: {
        "fwi_edges": [5, 17, 28, 47],
        "bui_edges": [51, 91, 141, 201],
        "classes": np.array(
            [
                [1, 2, 2, 3, 3],
                [2, 2, 3, 3, 4],
                [2, 3, 3, 4, 5],
                [2, 3, 4, 4, 5],
                [3, 3, 4, 4, 5],
            ],
            dtype=np.uint8,
        ),
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def cwfis_bbox_3978(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, east, south, north = extent
    lons = np.linspace(west, east, 9)
    lats = np.linspace(south, north, 9)
    lon2d, lat2d = np.meshgrid(lons, lats)
    x, y = FWI_TO_3978.transform(lon2d, lat2d)
    pad = 35000.0
    xmin = max(float(np.nanmin(x)) - pad, FWI_NATIVE_BBOX_3978[0])
    xmax = min(float(np.nanmax(x)) + pad, FWI_NATIVE_BBOX_3978[2])
    ymin = max(float(np.nanmin(y)) - pad, FWI_NATIVE_BBOX_3978[1])
    ymax = min(float(np.nanmax(y)) + pad, FWI_NATIVE_BBOX_3978[3])
    if xmin >= xmax or ymin >= ymax:
        raise RuntimeError("CWFIS grid does not overlap the plot extent.")
    return xmin, ymin, xmax, ymax


def cwfis_dimensions(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    width = max(2, int(math.ceil((xmax - xmin) / FWI_CACHE_RESOLUTION_M)))
    height = max(2, int(math.ceil((ymax - ymin) / FWI_CACHE_RESOLUTION_M)))
    largest = max(width, height)
    if largest > FWI_MAX_DIMENSION:
        scale = largest / FWI_MAX_DIMENSION
        width = max(2, int(round(width / scale)))
        height = max(2, int(round(height / scale)))
    return width, height


def cwfis_cache_path(cache_dir: Path, name: str, valid_date: dt.date, extent: tuple[float, float, float, float]) -> Path:
    west, east, south, north = extent
    domain = f"{west:.1f}_{east:.1f}_{south:.1f}_{north:.1f}".replace("-", "m").replace(".", "p")
    return cache_dir / name / f"{valid_date:%Y%m%d}" / f"cwfis_{name}_{valid_date:%Y%m%d}_{domain}.tif"


def fetch_cwfis_geotiff(
    name: str,
    coverage: str,
    valid_date: dt.date,
    cache_dir: Path,
    extent: tuple[float, float, float, float],
) -> Path:
    dest = cwfis_cache_path(cache_dir, name, valid_date, extent)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    bbox = cwfis_bbox_3978(extent)
    width, height = cwfis_dimensions(bbox)
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": coverage,
        "BBOX": ",".join(f"{value:.1f}" for value in bbox),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "CRS": "EPSG:3978",
        "FORMAT": "geotiff",
        "time": valid_date.isoformat(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
    try:
        response = requests.get(FWI_WCS_URL, params=params, timeout=(20, 180))
        response.raise_for_status()
        tmp_path.write_bytes(response.content)
        with Image.open(tmp_path) as image:
            image.verify()
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def read_cwfis_grid(
    name: str,
    valid_date: dt.date,
    cache_dir: Path,
    extent: tuple[float, float, float, float],
) -> CwfisGrid:
    path = fetch_cwfis_geotiff(name, CWFIS_COVERAGES[name], valid_date, cache_dir, extent)
    with Image.open(path) as image:
        lon, lat = geotiff_projected_lon_lat(image)
        data = np.asarray(image, dtype=np.float32)
        nodata_tag = image.tag_v2.get(42113)
    if nodata_tag is not None:
        data = np.where(data == float(nodata_tag), np.nan, data)
    lower, upper = CWFIS_VALUE_LIMITS[name]
    data = np.where(np.isfinite(data) & (data >= lower) & (data <= upper), data, np.nan)
    return CwfisGrid(valid_date, name, lat, lon, data.astype(np.float32))


def polygon_mask(lon: np.ndarray, lat: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    path = MplPath(polygon)
    points = np.column_stack((lon.ravel(), lat.ravel()))
    return path.contains_points(points).reshape(lon.shape)


def fetch_nr_districts(cache_path: Path = DANGER_REGION_CACHE) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": NR_DISTRICT_LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    try:
        response = requests.get(NR_DISTRICT_WFS_URL, params=params, timeout=(20, 180))
        response.raise_for_status()
        payload = response.json()
        if not payload.get("features"):
            raise RuntimeError("BCGW Natural Resource District response contains no features.")
        tmp_path.write_text(response.text)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return cache_path


def load_danger_region_geometries(cache_path: Path = DANGER_REGION_CACHE) -> dict[int, object]:
    path = fetch_nr_districts(cache_path)
    payload = json.loads(path.read_text())
    by_name = {
        feature["properties"]["DISTRICT_NAME"]: shape(feature["geometry"])
        for feature in payload["features"]
        if feature.get("geometry") and feature.get("properties", {}).get("DISTRICT_NAME")
    }
    output = {}
    for region, names in REGION_DISTRICT_NAMES.items():
        missing = sorted(names - by_name.keys())
        if missing:
            raise RuntimeError(f"BCGW danger-region district cache is missing: {', '.join(missing)}")
        output[region] = unary_union([by_name[name] for name in sorted(names)])
    if output[2].intersects(output[3]):
        overlap = output[2].intersection(output[3])
        if overlap.area > 1.0e-9:
            raise RuntimeError("BCGW district unions for danger regions 2 and 3 overlap.")
    return output


def danger_regions(lon: np.ndarray, lat: np.ndarray, cache_path: Path = DANGER_REGION_CACHE) -> np.ndarray:
    regions = np.ones(lon.shape, dtype=np.uint8)
    try:
        geometries = load_danger_region_geometries(cache_path)
        for region in (2, 3):
            prepare(geometries[region])
            regions[contains_xy(geometries[region], lon, lat)] = region
    except Exception as exc:
        log(f"Using freehand Schedule 1 danger-region fallback: {exc}")
        regions[polygon_mask(lon, lat, REGION2_POLYGON)] = 2
        regions[polygon_mask(lon, lat, REGION3_POLYGON)] = 3
    return regions


def approximate_danger_regions(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for callers created before the BCGW boundary audit."""

    return danger_regions(lon, lat)


def classify_region(fwi: np.ndarray, bui: np.ndarray, region: int) -> np.ndarray:
    spec = REGION_MATRICES[region]
    fwi_col = np.digitize(fwi, np.asarray(spec["fwi_edges"], dtype=np.float32), right=False)
    bui_row = np.digitize(bui, np.asarray(spec["bui_edges"], dtype=np.float32), right=False)
    classes = spec["classes"]
    return classes[bui_row, fwi_col]


def classify_danger(fwi: np.ndarray, bui: np.ndarray, regions: np.ndarray) -> np.ndarray:
    out = np.full(fwi.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(fwi) & np.isfinite(bui)
    for region in (1, 2, 3):
        mask = valid & (regions == region)
        if np.any(mask):
            out[mask] = classify_region(fwi[mask], bui[mask], region).astype(np.float32)
    return out


def load_bc_geometry():
    path = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_1_states_provinces",
    )
    for record in shapereader.Reader(path).records():
        attrs = record.attributes
        if attrs.get("iso_3166_2") == "CA-BC" or (
            attrs.get("admin") == "Canada" and attrs.get("name") == "British Columbia"
        ):
            geometry = record.geometry
            prepare(geometry)
            return geometry
    raise RuntimeError("Could not find the British Columbia province polygon.")


def danger_cmap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    colors = ["#2774c6", "#4caf50", "#ffe45c", "#f39c12", "#d7191c"]
    cmap = mcolors.ListedColormap(colors, name="bc_danger_class")
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    return cmap, norm


def draw_region_boundaries(
    ax: plt.Axes,
    regions: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    in_bc: np.ndarray,
) -> None:
    masked_regions = np.where(in_bc, regions.astype(np.float32), np.nan)
    ax.contour(
        lon,
        lat,
        masked_regions,
        levels=[1.5, 2.5],
        colors="#333333",
        linewidths=1.1,
        linestyles="--",
        transform=DATA_CRS,
        transform_first=True,
        zorder=24,
    )


def plot_danger_class(
    out_path: Path,
    valid_date: dt.date,
    fwi_grid: CwfisGrid,
    bui_grid: CwfisGrid,
    watersheds: list,
    transmission_lines: list,
    extent: tuple[float, float, float, float],
) -> None:
    if fwi_grid.data.shape != bui_grid.data.shape:
        raise RuntimeError("FWI and BUI grids have different shapes.")

    bc_geometry = load_bc_geometry()
    in_bc = contains_xy(bc_geometry, fwi_grid.lon, fwi_grid.lat)
    regions = danger_regions(fwi_grid.lon, fwi_grid.lat)
    danger = classify_danger(fwi_grid.data, bui_grid.data, regions)
    danger = np.where(in_bc, danger, np.nan)

    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=PLOT_CRS)
    ax.set_extent(extent, crs=DATA_CRS)
    ax.set_facecolor("#dbeaf0")
    hrdps.add_map_features(ax)

    cmap, norm = danger_cmap()
    shaded = ax.contourf(
        fwi_grid.lon,
        fwi_grid.lat,
        danger,
        levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        cmap=cmap,
        norm=norm,
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    hrdps.add_hydro_features(ax)
    add_transmission_lines(ax, transmission_lines)
    hrdps.add_watersheds(ax, watersheds)
    draw_region_boundaries(ax, regions, fwi_grid.lon, fwi_grid.lat, in_bc)

    fwi_ct = ax.contour(
        fwi_grid.lon,
        fwi_grid.lat,
        fwi_grid.data,
        levels=[10, 20, 30, 45],
        colors="#171717",
        linewidths=[0.8, 1.0, 1.2, 1.5],
        transform=DATA_CRS,
        transform_first=True,
        zorder=25,
    )
    ax.clabel(fwi_ct, inline=True, fontsize=6.0, fmt="FWI %d")
    bui_ct = ax.contour(
        bui_grid.lon,
        bui_grid.lat,
        bui_grid.data,
        levels=[40, 80, 120, 160, 200],
        colors="#6b3f1d",
        linewidths=0.85,
        linestyles=":",
        transform=DATA_CRS,
        transform_first=True,
        zorder=25,
    )
    ax.clabel(bui_ct, inline=True, fontsize=5.8, fmt="BUI %d")
    hrdps.add_city_labels(ax, fontsize=7.0, marker_size=2.0, path_width=2.3, zorder=30)

    plot_style.add_internal_colorbar(
        fig,
        ax,
        shaded,
        ticks=[1, 2, 3, 4, 5],
        label="Experimental DGR",
        fmt="%d",
        tick_labels=["Very low", "Low", "Moderate", "High", "Extreme"],
        extend=None,
        backdrop=(0.930, 0.088, 0.060, 0.790),
        cax_bounds=[0.958, 0.112, 0.020, 0.730],
    )
    ax.text(
        0.5,
        0.992,
        f"Experimental BC Danger Class  |  CWFIS daily valid {valid_date:%d%b%Y}".upper(),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        zorder=45,
    )
    ax.text(
        0.5,
        0.014,
        "Shaded: BC Schedule 2 FWI+BUI danger class; Schedule 1 regions reconstructed from BCGW district boundaries; black FWI, brown dotted BUI; grey: BC transmission",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="black",
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
        zorder=45,
    )
    ax.text(
        0.986,
        0.966,
        "Data: CWFIS | region mask: BC Geographic Warehouse district unions",
        transform=ax.transAxes,
        fontsize=7.3,
        color="black",
        ha="right",
        va="top",
        bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        zorder=45,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_date(text: str | None) -> dt.date:
    if text:
        return dt.datetime.strptime(text, "%Y-%m-%d").date()
    return dt.datetime.now(plot_style.LOCAL_TZ).date()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="CWFIS date YYYY-MM-DD. Defaults to today's local date.")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/experimental_danger_class"))
    parser.add_argument("--cache-dir", type=Path, default=FWI_CACHE_DIR)
    parser.add_argument("--watershed-cache", type=Path, default=hrdps.WATERSHED_CACHE)
    parser.add_argument("--refresh-watersheds", action="store_true")
    parser.add_argument("--no-watersheds", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    hrdps.set_model("west")
    valid_date = parse_date(args.date)
    log(f"Reading CWFIS FWI/BUI for {valid_date:%Y-%m-%d}.")
    fwi = read_cwfis_grid("fwi", valid_date, args.cache_dir, DANGER_EXTENT)
    bui = read_cwfis_grid("bui", valid_date, args.cache_dir, DANGER_EXTENT)
    watersheds = [] if args.no_watersheds else hrdps.load_watersheds(args.watershed_cache, refresh=args.refresh_watersheds)
    transmission_lines = load_transmission_lines()
    out_path = args.output_dir / f"experimental_bc_danger_class_{valid_date:%Y%m%d}.png"
    plot_danger_class(out_path, valid_date, fwi, bui, watersheds, transmission_lines, DANGER_EXTENT)
    log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
