#!/usr/bin/env python3
"""Render selected frames with the operational HRDPS regional two-panel layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import make_fire_weather_regional_options as regional_options
import make_hrdps_fire_weather_twopanel as production
import make_hrdps_west_convective as hrdps
import make_hrdps_west_lightning as lightning


DEFAULT_STAMP = "20260715T12Z"
DEFAULT_FHOUR = 12
DEFAULT_OUTPUT_DIR = Path("plots/test_fire_weather_regional/twopanel_20260715T12Z_f012")
REGIONAL_EXTENTS = production.REGIONAL_EXTENTS


def render_region(
    region_key: str,
    out_path: Path,
    run: hrdps.RunInfo,
    fhour: int,
    lat: np.ndarray,
    lon: np.ndarray,
    fields: lightning.LightningFields,
    peak_danger: np.ndarray,
    transmission_lines,
    shade_stride: int,
    contour_stride: int,
) -> Path:
    extent = production.REGIONAL_EXTENTS[region_key]
    yslice, xslice = hrdps.subset_slices(lat, lon, extent)
    return production.plot_regional_twopanel(
        out_path,
        run,
        fhour,
        lat[yslice, xslice],
        lon[yslice, xslice],
        lightning.subset_lightning_fields(fields, yslice, xslice),
        peak_danger[yslice, xslice],
        transmission_lines,
        shade_stride,
        contour_stride,
        region_key,
        production.REGIONAL_LABELS[region_key],
        extent,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--fhour", type=int, default=DEFAULT_FHOUR)
    parser.add_argument("--regions", default="sw,se,ne")
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
    lat, lon, fields, peak_danger = regional_options.load_fields(run, args.fhour, args.data_dir)
    transmission_lines = lightning.load_transmission_lines()
    shade_stride = hrdps.grid_stride(5.0)
    contour_stride = hrdps.grid_stride(12.0)
    regions = tuple(item.strip() for item in args.regions.split(",") if item.strip())
    unknown = set(regions) - set(production.REGIONAL_EXTENTS)
    if unknown:
        raise ValueError(f"Unsupported regional area(s): {', '.join(sorted(unknown))}")
    for region_key in regions:
        render_region(
            region_key,
            args.output_dir / f"fire_weather_twopanel_{region_key}_{run.stamp}_f{args.fhour:03d}.png",
            run,
            args.fhour,
            lat,
            lon,
            fields,
            peak_danger,
            transmission_lines,
            shade_stride,
            contour_stride,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
