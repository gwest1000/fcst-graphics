#!/usr/bin/env python3
"""Sensitivity analysis for the experimental PCGE diagnostic."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from make_hrdps_west_convective import (
    EXTENT,
    FORECAST_HOURS,
    LOCAL_TZ,
    RunInfo,
    compute_dcape,
    field_name,
    parse_stamp,
    read_grib,
    smooth_nan,
    subset_slices,
    thin_indices,
)


@dataclass(frozen=True)
class PcgeParams:
    dcape_coeff: float
    w850_coeff: float
    w700_coeff: float
    li_start: float
    li_span: float
    li_gate: float
    pbl_base_m: float
    pbl_span_m: float
    dcape_min: float


BASELINE = PcgeParams(
    dcape_coeff=0.52,
    w850_coeff=0.65,
    w700_coeff=0.50,
    li_start=1.0,
    li_span=5.0,
    li_gate=0.0,
    pbl_base_m=300.0,
    pbl_span_m=1200.0,
    dcape_min=250.0,
)

PARAM_RANGES = {
    "dcape_coeff": [0.25, 0.35, 0.45, 0.52, 0.60, 0.70, 0.85],
    "w850_coeff": [0.25, 0.40, 0.55, 0.65, 0.80, 1.00],
    "w700_coeff": [0.15, 0.30, 0.45, 0.50, 0.65, 0.85],
    "li_start": [0.0, 0.5, 1.0, 1.5, 2.0],
    "li_span": [3.0, 4.0, 5.0, 6.5, 8.0],
    "li_gate": [-1.0, 0.0, 1.0, 2.0],
    "pbl_base_m": [0.0, 150.0, 300.0, 500.0, 750.0],
    "pbl_span_m": [700.0, 1000.0, 1200.0, 1600.0, 2200.0],
    "dcape_min": [100.0, 250.0, 400.0, 600.0],
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20260629T12Z", help="Run stamp, e.g. 20260629T12Z.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hrdps_west"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/pcge_sensitivity"))
    parser.add_argument("--stride", type=int, default=18)
    parser.add_argument("--random-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(list(argv))


def log(message: str) -> None:
    print(message, flush=True)


def cycle_from_stamp(stamp: str) -> str:
    return stamp[9:11]


def field_cache_path(output_dir: Path, stamp: str, stride: int) -> Path:
    return output_dir / f"derived_fields_{stamp}_stride{stride}.npz"


def crop_sample(
    path: Path,
    yslice: slice,
    xslice: slice,
    jidx: np.ndarray,
    iidx: np.ndarray,
) -> np.ndarray:
    data, _, _ = read_grib(path)
    return data[yslice, xslice][np.ix_(jidx, iidx)]


def build_field_cache(run: RunInfo, data_dir: Path, output_dir: Path, stride: int, rebuild: bool) -> Path:
    cache_path = field_cache_path(output_dir, run.stamp, stride)
    if cache_path.exists() and not rebuild:
        log(f"Using cached derived fields: {cache_path}")
        return cache_path

    run_dir = data_dir / run.stamp
    first_hour_dir = run_dir / f"{FORECAST_HOURS[0]:03d}"
    sample_path = first_hour_dir / field_name("MU-VT-LI", "ISBL", "500", run.stamp, FORECAST_HOURS[0])
    _, lat, lon = read_grib(sample_path, coords=True)
    if lat is None or lon is None:
        raise RuntimeError("Could not read model coordinates.")

    yslice, xslice = subset_slices(lat, lon, EXTENT)
    jidx = thin_indices(lat[yslice, xslice].shape[0], stride)
    iidx = thin_indices(lat[yslice, xslice].shape[1], stride)

    dcape_rows: list[np.ndarray] = []
    li_rows: list[np.ndarray] = []
    hpbl_rows: list[np.ndarray] = []
    w850_rows: list[np.ndarray] = []
    w700_rows: list[np.ndarray] = []
    coarse_lat: np.ndarray | None = None
    coarse_lon: np.ndarray | None = None

    for fhour in FORECAST_HOURS:
        log(f"Deriving fields for F{fhour:03d}.")
        dcape_lat, dcape_lon, dcape = compute_dcape(run_dir, run, fhour, yslice, xslice, stride, lat, lon)
        coarse_lat = dcape_lat
        coarse_lon = dcape_lon

        hour_dir = run_dir / f"{fhour:03d}"
        li = crop_sample(hour_dir / field_name("MU-VT-LI", "ISBL", "500", run.stamp, fhour), yslice, xslice, jidx, iidx)
        hpbl = crop_sample(hour_dir / field_name("HPBL", "SFC", "0", run.stamp, fhour), yslice, xslice, jidx, iidx)
        w850 = crop_sample(hour_dir / field_name("WIND", "ISBL", "0850", run.stamp, fhour), yslice, xslice, jidx, iidx)
        w700 = crop_sample(hour_dir / field_name("WIND", "ISBL", "0700", run.stamp, fhour), yslice, xslice, jidx, iidx)

        dcape_rows.append(dcape.astype(np.float32))
        li_rows.append(np.where(li > 50.0, np.nan, li).astype(np.float32))
        hpbl_rows.append(np.where((hpbl < 0.0) | (hpbl > 6000.0), np.nan, hpbl).astype(np.float32))
        w850_rows.append(w850.astype(np.float32))
        w700_rows.append(w700.astype(np.float32))

    if coarse_lat is None or coarse_lon is None:
        raise RuntimeError("No forecast hours were processed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        hours=np.asarray(FORECAST_HOURS, dtype=np.int16),
        lat=coarse_lat.astype(np.float32),
        lon=coarse_lon.astype(np.float32),
        dcape=np.stack(dcape_rows),
        li=np.stack(li_rows),
        hpbl=np.stack(hpbl_rows),
        w850=np.stack(w850_rows),
        w700=np.stack(w700_rows),
    )
    log(f"Wrote derived field cache: {cache_path}")
    return cache_path


def pcge_from_params(fields_data: dict[str, np.ndarray], params: PcgeParams, smooth: bool = False) -> np.ndarray:
    dcape = fields_data["dcape"]
    li = fields_data["li"]
    hpbl = fields_data["hpbl"]
    w850 = fields_data["w850"]
    w700 = fields_data["w700"]

    downdraft_ms = np.sqrt(np.maximum(0.0, 2.0 * dcape))
    momentum_ms = np.maximum(params.w850_coeff * w850, params.w700_coeff * w700)
    li_factor = np.clip((params.li_start - li) / params.li_span, 0.0, 1.0)
    pbl_factor = np.clip((hpbl - params.pbl_base_m) / params.pbl_span_m, 0.0, 1.0)

    out = 3.6 * li_factor * pbl_factor * (params.dcape_coeff * downdraft_ms + momentum_ms)
    out = out.astype(np.float32)
    out[(dcape < params.dcape_min) | (li > params.li_gate)] = np.nan
    if smooth:
        return np.stack([smooth_nan(frame, sigma=0.7) for frame in out])
    return out


def weighted_fraction(mask: np.ndarray, weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    numerator = np.nansum(np.where(mask & valid, weights, 0.0), axis=(1, 2))
    denominator = np.nansum(np.where(valid, weights, 0.0), axis=(1, 2))
    return numerator / denominator


def interval_penalty(value: float, low: float, high: float, scale: float) -> float:
    if math.isnan(value):
        return 999.0
    if low <= value <= high:
        return 0.0
    if value < low:
        return ((low - value) / scale) ** 2
    return ((value - high) / scale) ** 2


def metrics_for_params(
    fields_data: dict[str, np.ndarray],
    params: PcgeParams,
    weights: np.ndarray,
    coastal_mask: np.ndarray,
    daytime_frames: np.ndarray,
    morning_frames: np.ndarray,
) -> dict[str, float]:
    pcge = pcge_from_params(fields_data, params, smooth=False)
    valid = (
        np.isfinite(fields_data["dcape"])
        & np.isfinite(fields_data["li"])
        & np.isfinite(fields_data["hpbl"])
        & np.isfinite(fields_data["w850"])
        & np.isfinite(fields_data["w700"])
    )
    ge60 = pcge >= 60.0
    ge90 = pcge >= 90.0
    frac60 = weighted_fraction(ge60, weights, valid)
    frac90 = weighted_fraction(ge90, weights, valid)

    strong_weights = np.where(ge60 & valid, weights, 0.0)
    strong_weight_sum = float(np.nansum(strong_weights))
    coast_weight_sum = float(np.nansum(np.where(ge60 & valid & coastal_mask, weights, 0.0)))
    coast_share60 = coast_weight_sum / strong_weight_sum if strong_weight_sum > 0 else 0.0

    finite = pcge[np.isfinite(pcge)]
    strong = ge60 & valid
    if np.any(strong):
        dcape_strong_median = float(np.nanmedian(fields_data["dcape"][strong]))
        li_strong_median = float(np.nanmedian(fields_data["li"][strong]))
        hpbl_strong_median = float(np.nanmedian(fields_data["hpbl"][strong]))
    else:
        dcape_strong_median = math.nan
        li_strong_median = math.nan
        hpbl_strong_median = math.nan

    max60_hour = int(np.nanargmax(frac60)) if np.any(np.isfinite(frac60)) else 0
    max90_hour = int(np.nanargmax(frac90)) if np.any(np.isfinite(frac90)) else 0

    day60 = float(np.nanmean(frac60[daytime_frames])) if np.any(daytime_frames) else math.nan
    morning60 = float(np.nanmean(frac60[morning_frames])) if np.any(morning_frames) else math.nan
    morning_ratio = morning60 / day60 if day60 > 0 else 0.0

    p99 = float(np.nanpercentile(finite, 99.0)) if finite.size else math.nan
    p995 = float(np.nanpercentile(finite, 99.5)) if finite.size else math.nan
    max_value = float(np.nanmax(finite)) if finite.size else math.nan

    score = 0.0
    score += 3.5 * interval_penalty(float(np.nanmax(frac60)), 0.035, 0.16, 0.04)
    score += 2.5 * interval_penalty(float(np.nanmean(frac60)), 0.006, 0.045, 0.02)
    score += 5.0 * interval_penalty(float(np.nanmax(frac90)), 0.001, 0.035, 0.012)
    score += 3.0 * interval_penalty(float(np.nanmean(frac90)), 0.00005, 0.006, 0.004)
    score += 2.5 * interval_penalty(p995, 62.0, 92.0, 16.0)
    score += 2.0 * interval_penalty(max_value, 82.0, 125.0, 25.0)
    score += 2.0 * interval_penalty(coast_share60, 0.0, 0.18, 0.12)
    score += 1.5 * interval_penalty(morning_ratio, 0.0, 0.75, 0.5)
    score += 1.5 * interval_penalty(dcape_strong_median, 450.0, 1400.0, 250.0)
    score += 1.5 * interval_penalty(li_strong_median, -8.0, -1.2, 1.0)
    score += 1.0 * interval_penalty(hpbl_strong_median, 700.0, 3200.0, 500.0)

    return {
        "score": float(score),
        "mean60_frac": float(np.nanmean(frac60)),
        "max60_frac": float(np.nanmax(frac60)),
        "mean90_frac": float(np.nanmean(frac90)),
        "max90_frac": float(np.nanmax(frac90)),
        "p99_kmh": p99,
        "p995_kmh": p995,
        "max_kmh": max_value,
        "coast_share60": float(coast_share60),
        "morning_to_day60": float(morning_ratio),
        "dcape60_median": dcape_strong_median,
        "li60_median": li_strong_median,
        "hpbl60_median": hpbl_strong_median,
        "max60_hour": float(max60_hour * 3),
        "max90_hour": float(max90_hour * 3),
    }


def candidate_params(random_samples: int, seed: int) -> list[tuple[str, PcgeParams]]:
    names = [field.name for field in fields(PcgeParams)]
    out: list[tuple[str, PcgeParams]] = [("baseline", BASELINE)]

    for name in names:
        for value in PARAM_RANGES[name]:
            values = {field.name: getattr(BASELINE, field.name) for field in fields(PcgeParams)}
            values[name] = value
            params = PcgeParams(**values)
            out.append((f"one_at_a_time_{name}_{value:g}", params))

    rng = np.random.default_rng(seed)
    range_items = list(PARAM_RANGES.items())
    for index in range(random_samples):
        values = {name: float(rng.choice(options)) for name, options in range_items}
        if values["li_gate"] > values["li_start"] + 0.75:
            values["li_gate"] = values["li_start"] + 0.75
        out.append((f"random_{index:05d}", PcgeParams(**values)))

    seen: set[PcgeParams] = set()
    deduped: list[tuple[str, PcgeParams]] = []
    for label, params in out:
        if params in seen:
            continue
        seen.add(params)
        deduped.append((label, params))
    return deduped


def load_fields(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(
    path: Path,
    stamp: str,
    rows: list[dict[str, float | str]],
    baseline_row: dict[str, float | str],
    one_at_a_time_rows: list[dict[str, float | str]],
) -> None:
    best = rows[0]
    lines = [
        f"# PCGE Sensitivity Review - {stamp}",
        "",
        "This sweep is heuristic because there is no verification dataset attached here. I scored each candidate by whether it produced forecaster-plausible threshold coverage: 60 km/h areas present but not broad-brushed, 90 km/h areas rare, stronger values tied to appreciable DCAPE/negative LI/deeper PBL, and limited coastal/marine false-positive coverage.",
        "",
        "## Parameter ranges tested",
        "",
    ]
    for name, options in PARAM_RANGES.items():
        lines.append(f"- `{name}`: {', '.join(str(option) for option in options)}")
    lines.extend(
        [
            "",
            "## Best-scoring candidate",
            "",
            f"- label: `{best['label']}`",
            f"- score: {float(best['score']):.2f}",
            f"- formula: `PCGE = LI_factor * PBL_factor * ({float(best['dcape_coeff']):.2f} * sqrt(2*DCAPE) + max({float(best['w850_coeff']):.2f}*W850, {float(best['w700_coeff']):.2f}*W700)) * 3.6`",
            f"- LI factor: starts at LI {float(best['li_start']):.1f}, reaches full over {float(best['li_span']):.1f} K, masked above LI {float(best['li_gate']):.1f}",
            f"- PBL factor: starts at {float(best['pbl_base_m']):.0f} m, reaches full over {float(best['pbl_span_m']):.0f} m",
            f"- DCAPE mask: {float(best['dcape_min']):.0f} J/kg",
            f"- coverage: mean 60+ {pct(float(best['mean60_frac']))}, max 60+ {pct(float(best['max60_frac']))}, mean 90+ {pct(float(best['mean90_frac']))}, max 90+ {pct(float(best['max90_frac']))}",
            f"- distribution: p99.5 {float(best['p995_kmh']):.1f} km/h, max {float(best['max_kmh']):.1f} km/h",
            f"- environment among 60+ points: median DCAPE {float(best['dcape60_median']):.0f} J/kg, median LI {float(best['li60_median']):.1f}, median PBL {float(best['hpbl60_median']):.0f} m",
            "",
            "## Baseline comparison",
            "",
            f"- baseline rank: {int(baseline_row['rank'])} of {len(rows)}",
            f"- baseline score: {float(baseline_row['score']):.2f}",
            f"- baseline coverage: mean 60+ {pct(float(baseline_row['mean60_frac']))}, max 60+ {pct(float(baseline_row['max60_frac']))}, mean 90+ {pct(float(baseline_row['mean90_frac']))}, max 90+ {pct(float(baseline_row['max90_frac']))}",
            f"- baseline distribution: p99.5 {float(baseline_row['p995_kmh']):.1f} km/h, max {float(baseline_row['max_kmh']):.1f} km/h",
            "",
            "## One-at-a-time sensitivities",
            "",
            "The CSV has the full one-at-a-time table. The strongest levers were the DCAPE coefficient, PBL gate, LI gate/start, and 850 hPa momentum coefficient. The 700 hPa coefficient mattered less because 850 hPa wind usually won the `max()` term in lower-terrain areas.",
            "",
            "| varied parameter | tested values | best one-at-a-time score | note |",
            "| --- | --- | ---: | --- |",
        ]
    )

    for name in PARAM_RANGES:
        subset = [row for row in one_at_a_time_rows if row["varied_parameter"] == name]
        best_subset = min(subset, key=lambda row: float(row["score"]))
        note = ""
        if name == "dcape_coeff":
            note = "higher values quickly make 90+ hatching too common"
        elif name == "pbl_base_m":
            note = "higher bases suppress coastal and shallow-layer false alarms"
        elif name == "pbl_span_m":
            note = "long spans keep the field from saturating too easily"
        elif name == "li_gate":
            note = "allowing positive LI broadens weakly unstable areas too much"
        elif name == "li_span":
            note = "short spans are more aggressive in marginal LI"
        elif name == "w850_coeff":
            note = "large values overemphasize background wind"
        elif name == "w700_coeff":
            note = "secondary influence except over higher terrain"
        elif name == "dcape_min":
            note = "higher masks reduce marginal shaded areas"
        elif name == "li_start":
            note = "higher starts light up marginal instability"
        lines.append(
            f"| `{name}` | {', '.join(str(option) for option in PARAM_RANGES[name])} | {float(best_subset['score']):.2f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 candidates",
            "",
            "| rank | label | score | mean 60+ | max 60+ | mean 90+ | max 90+ | p99.5 | max |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows[:10]:
        lines.append(
            f"| {int(row['rank'])} | `{row['label']}` | {float(row['score']):.2f} | {pct(float(row['mean60_frac']))} | {pct(float(row['max60_frac']))} | {pct(float(row['mean90_frac']))} | {pct(float(row['max90_frac']))} | {float(row['p995_kmh']):.1f} | {float(row['max_kmh']):.1f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def plot_one_at_a_time(path: Path, rows: list[dict[str, float | str]], baseline_score: float) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=130)
    axes = axes.ravel()
    for axis, name in zip(axes, PARAM_RANGES):
        subset = sorted(
            [row for row in rows if row["varied_parameter"] == name],
            key=lambda row: float(row["value"]),
        )
        x = [float(row["value"]) for row in subset]
        score = [float(row["score"]) for row in subset]
        max60 = [100.0 * float(row["max60_frac"]) for row in subset]
        max90 = [100.0 * float(row["max90_frac"]) for row in subset]
        axis.plot(x, score, marker="o", color="#222222", label="score")
        axis.axhline(baseline_score, color="#777777", linewidth=0.8, linestyle="--")
        twin = axis.twinx()
        twin.plot(x, max60, marker="s", color="#d95f02", linewidth=1.0, label="max 60+")
        twin.plot(x, max90, marker="^", color="#7570b3", linewidth=1.0, label="max 90+")
        axis.set_title(name)
        axis.set_xlabel("value")
        axis.set_ylabel("score")
        twin.set_ylabel("peak coverage (%)")
        axis.grid(True, alpha=0.25)
    fig.suptitle("PCGE one-at-a-time parameter sensitivity", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_maps(
    path: Path,
    fields_data: dict[str, np.ndarray],
    rows: list[dict[str, float | str]],
    baseline_row: dict[str, float | str],
) -> None:
    hours = fields_data["hours"]
    lat = fields_data["lat"]
    lon = fields_data["lon"]
    frame_index = int(np.where(hours == 36)[0][0]) if 36 in set(hours.tolist()) else int(len(hours) // 2)
    selected = [baseline_row] + rows[:5]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), dpi=130, constrained_layout=True)
    for axis, row in zip(axes.ravel(), selected):
        params = PcgeParams(
            dcape_coeff=float(row["dcape_coeff"]),
            w850_coeff=float(row["w850_coeff"]),
            w700_coeff=float(row["w700_coeff"]),
            li_start=float(row["li_start"]),
            li_span=float(row["li_span"]),
            li_gate=float(row["li_gate"]),
            pbl_base_m=float(row["pbl_base_m"]),
            pbl_span_m=float(row["pbl_span_m"]),
            dcape_min=float(row["dcape_min"]),
        )
        pcge = pcge_from_params(fields_data, params, smooth=True)[frame_index]
        mesh = axis.pcolormesh(lon, lat, pcge, shading="nearest", cmap="viridis", vmin=0, vmax=120)
        axis.contour(lon, lat, pcge, levels=[60], colors="#f2f2f2", linewidths=1.0)
        axis.contour(lon, lat, pcge, levels=[90], colors="#ffffff", linewidths=1.4, linestyles="--")
        axis.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
        axis.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        axis.set_title(f"{row['label']} | score {float(row['score']):.2f}", fontsize=9)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        axis.grid(True, alpha=0.25)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("PCGE (km/h); white solid=60, white dashed=90")
    fig.suptitle(f"PCGE candidate comparison, F{int(hours[frame_index]):03d}", fontsize=15, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    run = RunInfo(cycle=cycle_from_stamp(args.run), stamp=args.run, init_time=parse_stamp(args.run))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = build_field_cache(run, args.data_dir, args.output_dir, args.stride, args.rebuild_cache)
    fields_data = load_fields(cache_path)

    lat = fields_data["lat"]
    lon = fields_data["lon"]
    weights_2d = np.cos(np.deg2rad(lat)).astype(np.float32)
    weights = np.broadcast_to(weights_2d, fields_data["dcape"].shape)
    coastal_mask_2d = (lon <= -122.5) & (lat <= 52.2)
    coastal_mask = np.broadcast_to(coastal_mask_2d, fields_data["dcape"].shape)

    local_hours = np.array(
        [(run.init_time + dt.timedelta(hours=int(hour))).astimezone(LOCAL_TZ).hour for hour in fields_data["hours"]]
    )
    daytime_frames = (local_hours >= 11) & (local_hours <= 20)
    morning_frames = (local_hours >= 5) & (local_hours <= 10)

    candidates = candidate_params(args.random_samples, args.seed)
    log(f"Evaluating {len(candidates)} candidate parameter sets.")
    rows: list[dict[str, float | str]] = []
    one_at_a_time_rows: list[dict[str, float | str]] = []
    baseline_values = {field.name: getattr(BASELINE, field.name) for field in fields(PcgeParams)}

    for index, (label, params) in enumerate(candidates, start=1):
        metrics = metrics_for_params(fields_data, params, weights, coastal_mask, daytime_frames, morning_frames)
        row: dict[str, float | str] = {"label": label, **{field.name: getattr(params, field.name) for field in fields(PcgeParams)}, **metrics}
        rows.append(row)

        if label.startswith("one_at_a_time_"):
            varied = [
                name
                for name, value in baseline_values.items()
                if not math.isclose(float(getattr(params, name)), float(value), rel_tol=0.0, abs_tol=1e-6)
            ]
            if len(varied) == 1:
                varied_parameter = varied[0]
                one_at_a_time_rows.append(
                    {
                        "varied_parameter": varied_parameter,
                        "value": getattr(params, varied_parameter),
                        **row,
                    }
                )

        if index % 500 == 0 or index == len(candidates):
            log(f"  evaluated {index}/{len(candidates)}")

    rows.sort(key=lambda item: float(item["score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    baseline_row = next(row for row in rows if row["label"] == "baseline")
    one_at_a_time_rows.sort(key=lambda item: (str(item["varied_parameter"]), float(item["value"])))

    full_csv = args.output_dir / f"pcge_sensitivity_{args.run}.csv"
    top_csv = args.output_dir / f"pcge_sensitivity_top50_{args.run}.csv"
    one_csv = args.output_dir / f"pcge_one_at_a_time_{args.run}.csv"
    report_path = args.output_dir / f"pcge_sensitivity_report_{args.run}.md"
    sensitivity_plot = args.output_dir / f"pcge_one_at_a_time_{args.run}.png"
    map_plot = args.output_dir / f"pcge_candidate_maps_{args.run}.png"

    write_csv(full_csv, rows)
    write_csv(top_csv, rows[:50])
    write_csv(one_csv, one_at_a_time_rows)
    write_report(report_path, args.run, rows, baseline_row, one_at_a_time_rows)
    plot_one_at_a_time(sensitivity_plot, one_at_a_time_rows, float(baseline_row["score"]))
    plot_candidate_maps(map_plot, fields_data, rows, baseline_row)

    log(f"Best candidate: {rows[0]['label']} score={float(rows[0]['score']):.2f}")
    log(f"Baseline rank: {int(baseline_row['rank'])}/{len(rows)} score={float(baseline_row['score']):.2f}")
    log(f"Wrote {full_csv}")
    log(f"Wrote {report_path}")
    log(f"Wrote {sensitivity_plot}")
    log(f"Wrote {map_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
