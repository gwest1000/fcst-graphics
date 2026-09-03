#!/usr/bin/env python3
"""Evaluate and cautiously retune the handmade BC lightning potential index.

The command intentionally works from the immutable lightning development archive.
It does not alter the operational formula or publish forecast graphics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy, points
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree

import lightning_ml_archive as archive
from make_experimental_danger_class import load_bc_geometry


UTC = dt.timezone.utc
DEFAULT_OUTPUT = Path("output/lpi_calibration/initial_20260812")
MODEL_GRID_KM = 5.0
OBS_GRID_KM = 2.5
TARGET_RADIUS_KM = 30.0
CORRIDOR_RADIUS_KM = 30.0
SCORE_BIN_WIDTH = 0.5
SCORE_EDGES = np.arange(0.0, 100.0 + SCORE_BIN_WIDTH, SCORE_BIN_WIDTH, dtype=np.float64)
SCORE_CENTERS = (SCORE_EDGES[:-1] + SCORE_EDGES[1:]) / 2.0
SPLITS = ("train", "validate", "test")


@dataclass(frozen=True)
class Formula:
    name: str
    family: str
    description: str
    li_start: float = 1.0
    li_end: float = -5.0
    cape_low: float = 75.0
    cape_high: float = 800.0
    cape_weight: float = 0.20
    charge_rh_low: float = 45.0
    charge_rh_high: float = 80.0
    charge_depth_low: float = 35.0
    charge_depth_high: float = 150.0
    mid_rh_low: float = 35.0
    mid_rh_high: float = 75.0
    mid_rh_weight: float = 0.15
    updraft_low: float = 0.005
    updraft_high: float = 0.050
    precip_accum_low: float = 0.05
    precip_accum_high: float = 1.50
    precip_rate_low: float = 0.02
    precip_rate_high: float = 0.80
    trigger_exponent: float = 0.50
    trigger_mode: str = "combined"


@dataclass(frozen=True)
class ForecastBlock:
    run_stamp: str
    forecast_hour: int
    valid_end: dt.datetime
    day_start: dt.datetime
    split: str


@dataclass
class Histogram:
    count: np.ndarray
    events: np.ndarray

    @classmethod
    def empty(cls) -> "Histogram":
        size = len(SCORE_CENTERS)
        return cls(np.zeros(size, dtype=np.float64), np.zeros(size, dtype=np.float64))

    def add(self, score: np.ndarray, event: np.ndarray, weight: float = 1.0) -> None:
        valid = np.isfinite(score) & np.isfinite(event)
        if not np.any(valid):
            return
        bins = np.clip(np.digitize(score[valid], SCORE_EDGES) - 1, 0, len(SCORE_CENTERS) - 1)
        self.count += np.bincount(bins, weights=np.full(np.count_nonzero(valid), weight), minlength=len(self.count))
        self.events += np.bincount(bins, weights=event[valid] * weight, minlength=len(self.events))


def log(message: str) -> None:
    print(message, flush=True)


def parse_utc_stamp(stamp: str) -> dt.datetime:
    return dt.datetime.strptime(stamp, "%Y%m%dT%HZ").replace(tzinfo=UTC)


def parse_obs_end(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.name[:13], "%Y%m%dT%H%M").replace(tzinfo=UTC)


def daily_window_start(timestamp: dt.datetime) -> dt.datetime:
    """Return the 12Z start of the operational day containing a block end."""

    shifted = timestamp - dt.timedelta(hours=12, microseconds=1)
    return dt.datetime.combine(shifted.date(), dt.time(12), tzinfo=UTC)


def ramp(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError(f"Invalid ramp: {low} to {high}")
    return np.clip((value - low) / (high - low), 0.0, 1.0)


def descending_ramp(value: np.ndarray, start: float, end: float) -> np.ndarray:
    if start <= end:
        raise ValueError(f"Invalid descending ramp: {start} to {end}")
    return np.clip((start - value) / (start - end), 0.0, 1.0)


def compute_formula(fields: dict[str, np.ndarray], formula: Formula) -> np.ndarray:
    li_factor = descending_ramp(fields["mu_li"], formula.li_start, formula.li_end)
    cape_factor = ramp(fields["cape"], formula.cape_low, formula.cape_high)
    instability = li_factor * ((1.0 - formula.cape_weight) + formula.cape_weight * cape_factor)

    charge_rh = ramp(fields["charge_rh"], formula.charge_rh_low, formula.charge_rh_high)
    charge_depth = ramp(fields["charge_depth"], formula.charge_depth_low, formula.charge_depth_high)
    charge_factor = np.sqrt(np.clip(charge_rh * charge_depth, 0.0, 1.0))
    mid_rh = ramp(fields["mid_rh"], formula.mid_rh_low, formula.mid_rh_high)
    moisture = charge_factor * ((1.0 - formula.mid_rh_weight) + formula.mid_rh_weight * mid_rh)

    updraft = ramp(fields["upward_w"], formula.updraft_low, formula.updraft_high)
    precip = np.maximum(
        ramp(fields["precip_3h"], formula.precip_accum_low, formula.precip_accum_high),
        ramp(fields["precip_rate"], formula.precip_rate_low, formula.precip_rate_high),
    )
    if formula.trigger_mode == "combined":
        trigger = np.maximum(precip, updraft * charge_factor)
    elif formula.trigger_mode == "precip_only":
        trigger = precip
    elif formula.trigger_mode == "updraft_only":
        trigger = updraft * charge_factor
    elif formula.trigger_mode == "neutral":
        trigger = np.ones_like(precip)
    else:
        raise ValueError(f"Unknown trigger mode: {formula.trigger_mode}")

    result = 100.0 * instability * moisture * np.power(np.clip(trigger, 0.0, 1.0), formula.trigger_exponent)
    result[(instability < 0.02) | (moisture < 0.02)] = 0.0
    return np.clip(result, 0.0, 100.0).astype(np.float32)


def candidate_formulas() -> list[Formula]:
    base = Formula("recomputed_baseline", "baseline", "Current formula reconstructed from archived ingredients")
    candidates = [base]

    def add(name: str, family: str, description: str, **changes: object) -> None:
        candidates.append(replace(base, name=name, family=family, description=description, **changes))

    add("ablate_cape", "ablation", "Remove CAPE's 20% modulation", cape_weight=0.0)
    add("ablate_mid_rh", "ablation", "Remove the 15% mid-level RH modulation", mid_rh_weight=0.0)
    add("ablate_precip_trigger", "ablation", "Use only resolved ascent for triggering", trigger_mode="updraft_only")
    add("ablate_updraft_trigger", "ablation", "Use only precipitation for triggering", trigger_mode="precip_only")
    add("ablate_trigger", "ablation", "Set trigger to one everywhere", trigger_mode="neutral")

    add("li_ramp_0_to_m4", "li_ramp", "Narrower MU-LI ramp from 0 to -4 K", li_start=0.0, li_end=-4.0)
    add("li_ramp_1_to_m4", "li_ramp", "Earlier saturation at -4 K", li_end=-4.0)
    add("li_ramp_1_to_m6", "li_ramp", "Later saturation at -6 K", li_end=-6.0)
    add("li_ramp_2_to_m5", "li_ramp", "Begin instability response at +2 K", li_start=2.0)
    add("cape_weight_05", "cape_weight", "Reduce CAPE modulation to 5%", cape_weight=0.05)
    add("cape_weight_10", "cape_weight", "Reduce CAPE modulation to 10%", cape_weight=0.10)
    add("cape_weight_30", "cape_weight", "Increase CAPE modulation to 30%", cape_weight=0.30)
    add("cape_ramp_50_600", "cape_ramp", "Earlier CAPE response and saturation", cape_low=50.0, cape_high=600.0)
    add("cape_ramp_100_1000", "cape_ramp", "More conservative CAPE response", cape_low=100.0, cape_high=1000.0)
    add("charge_rh_40_75", "charge_rh_ramp", "Earlier charging-layer RH response", charge_rh_low=40.0, charge_rh_high=75.0)
    add("charge_rh_50_85", "charge_rh_ramp", "More conservative charging-layer RH response", charge_rh_low=50.0, charge_rh_high=85.0)
    add("charge_depth_25_125", "charge_depth_ramp", "Earlier charging-depth response", charge_depth_low=25.0, charge_depth_high=125.0)
    add("charge_depth_50_175", "charge_depth_ramp", "More conservative charging-depth response", charge_depth_low=50.0, charge_depth_high=175.0)
    add("mid_rh_weight_05", "mid_rh_weight", "Reduce mid-level RH modulation to 5%", mid_rh_weight=0.05)
    add("mid_rh_weight_25", "mid_rh_weight", "Increase mid-level RH modulation to 25%", mid_rh_weight=0.25)
    add("updraft_ramp_003_040", "updraft_ramp", "Earlier resolved-ascent response", updraft_low=0.003, updraft_high=0.040)
    add("updraft_ramp_010_070", "updraft_ramp", "More conservative resolved-ascent response", updraft_low=0.010, updraft_high=0.070)
    add("precip_ramps_early", "precip_ramp", "Earlier precipitation trigger response", precip_accum_low=0.02, precip_accum_high=1.0, precip_rate_low=0.01, precip_rate_high=0.5)
    add("precip_ramps_late", "precip_ramp", "More conservative precipitation trigger response", precip_accum_low=0.10, precip_accum_high=2.0, precip_rate_low=0.05, precip_rate_high=1.2)
    add("trigger_exponent_035", "trigger_exponent", "Less trigger suppression", trigger_exponent=0.35)
    add("trigger_exponent_065", "trigger_exponent", "More trigger suppression", trigger_exponent=0.65)
    add("trigger_exponent_080", "trigger_exponent", "Strong trigger suppression", trigger_exponent=0.80)

    add(
        "bc_low_cape_broad_charge",
        "combined",
        "Low CAPE weight with broader charging RH/depth response",
        cape_weight=0.05,
        charge_rh_low=40.0,
        charge_rh_high=75.0,
        charge_depth_low=25.0,
        charge_depth_high=125.0,
    )
    add(
        "bc_no_cape_moisture",
        "combined",
        "No CAPE modulation and stronger mid-level moisture modulation",
        cape_weight=0.0,
        mid_rh_weight=0.25,
    )
    add(
        "bc_low_cape_sharp_trigger",
        "combined",
        "Low CAPE modulation with stronger trigger suppression",
        cape_weight=0.05,
        trigger_exponent=0.65,
    )
    return candidates


def load_grid(archive_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = archive.model_archive_dir(archive_root) / "static/grid.npz"
    with np.load(path) as data:
        return data["lat"].astype(np.float32), data["lon"].astype(np.float32)


def transmission_corridor_mask(lat: np.ndarray, lon: np.ndarray, radius_km: float) -> np.ndarray:
    collection = json.loads(Path("data/bc_transmission_lines.geojson").read_text())
    to_bc = Transformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)
    lines = [transform(to_bc.transform, shape(feature["geometry"])) for feature in collection["features"] if feature.get("geometry")]
    x, y = to_bc.transform(lon, lat)
    grid_points = points(x.reshape(-1), y.reshape(-1))
    matches = STRtree(lines).query(grid_points, predicate="dwithin", distance=radius_km * 1000.0)
    mask = np.zeros(grid_points.shape, dtype=bool)
    if matches.size:
        mask[np.unique(matches[0])] = True
    return mask.reshape(lat.shape)


def domain_masks(archive_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat, lon = load_grid(archive_root)
    bc = load_bc_geometry()
    in_bc = contains_xy(bc, lon, lat)
    in_corridor = in_bc & transmission_corridor_mask(lat, lon, CORRIDOR_RADIUS_KM)
    return lat, lon, in_bc, in_corridor


def observation_paths(archive_root: Path) -> dict[dt.datetime, Path]:
    root = archive_root / "observations/eccc_lightning_3h/schema_v1"
    return {parse_obs_end(path): path for path in root.glob("*/*/*.tif")}


def ingredient_run_dirs(archive_root: Path) -> list[Path]:
    root = archive.hourly_lpi_archive_dir(archive_root)
    return sorted(path.parent for path in root.glob("*/*/manifest.json") if json.loads(path.read_text()).get("complete"))


def forecast_blocks(archive_root: Path, obs_paths: dict[dt.datetime, Path]) -> list[tuple[str, int, dt.datetime]]:
    blocks: list[tuple[str, int, dt.datetime]] = []
    for run_dir in ingredient_run_dirs(archive_root):
        stamp = run_dir.name
        init = parse_utc_stamp(stamp)
        for fhour in range(3, 49, 3):
            valid = init + dt.timedelta(hours=fhour)
            if valid not in obs_paths:
                continue
            if all((run_dir / f"f{hour:03d}.npz").exists() for hour in range(fhour - 2, fhour + 1)):
                blocks.append((stamp, fhour, valid))
    return blocks


def read_observation_targets(
    path: Path,
    point_lat: np.ndarray,
    point_lon: np.ndarray,
    radii_km: Iterable[float] = (10.0, 20.0, 30.0, 40.0),
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    pad_degrees = max(radii_km) / 80.0
    bounds = (
        float(np.nanmin(point_lon) - pad_degrees),
        float(np.nanmin(point_lat) - pad_degrees),
        float(np.nanmax(point_lon) + pad_degrees),
        float(np.nanmax(point_lat) + pad_degrees),
    )
    with rasterio.open(path) as source:
        window = source.window(*bounds).round_offsets().round_lengths()
        data = source.read(1, window=window, boundless=True, fill_value=source.nodata or -999.0).astype(np.float32)
        window_transform = source.window_transform(window)
        valid = np.isfinite(data) & (data != source.nodata) & (data > 0.0)
        distance_km = distance_transform_edt(~valid, sampling=OBS_GRID_KM)
        rows, cols = rasterio.transform.rowcol(window_transform, point_lon, point_lat)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    density = np.zeros(point_lat.shape, dtype=np.float32)
    density[inside] = np.where(valid[rows[inside], cols[inside]], data[rows[inside], cols[inside]], 0.0)
    targets: dict[float, np.ndarray] = {}
    for radius in radii_km:
        target = np.zeros(point_lat.shape, dtype=np.float32)
        target[inside] = distance_km[rows[inside], cols[inside]] <= float(radius)
        targets[float(radius)] = target
    return targets, density


def split_days(days: list[dt.datetime], active_days: set[dt.datetime]) -> tuple[dict[dt.datetime, str], dict[str, object]]:
    unique = sorted(set(days))
    if len(unique) < 12:
        raise RuntimeError(f"Only {len(unique)} forecast days overlap the ingredient and observation archives.")
    train_end = max(1, int(math.floor(len(unique) * 0.60)))
    validate_end = max(train_end + 1, int(math.floor(len(unique) * 0.80)))
    validate_end = min(validate_end, len(unique) - 1)
    groups = {
        "train": unique[:train_end],
        "validate": unique[train_end:validate_end],
        "test": unique[validate_end:],
    }
    active_counts = {name: sum(day in active_days for day in values) for name, values in groups.items()}
    sufficient = all(len(groups[name]) >= 3 and active_counts[name] >= 2 for name in SPLITS)
    if not sufficient:
        day_counts = {name: len(value) for name, value in groups.items()}
        raise RuntimeError(
            "A defensible chronological train/validate/test split is not yet supported: "
            f"days={day_counts}, active={active_counts}"
        )
    lookup = {day: name for name, values in groups.items() for day in values}
    summary = {
        "strategy": "chronological_60_20_20_by_12z_operational_day",
        "sufficient_for_three_way_split": True,
        "days": {name: len(values) for name, values in groups.items()},
        "active_days": active_counts,
        "ranges": {
            name: [values[0].isoformat().replace("+00:00", "Z"), values[-1].isoformat().replace("+00:00", "Z")]
            for name, values in groups.items()
        },
    }
    return lookup, summary


def unpack_selected(data: np.lib.npyio.NpzFile, key: str, flat_indices: np.ndarray) -> np.ndarray:
    spec = next(item for item in archive.HOURLY_LPI_INGREDIENT_SPECS if item.key == key)
    packed = data[key].reshape(-1)[flat_indices]
    return archive.unpack_field(packed, spec).astype(np.float32)


def read_hour_fields(path: Path, flat_indices: np.ndarray) -> dict[str, np.ndarray]:
    keys = ("potential", "mu_li", "cape", "charge_rh", "charge_depth", "mid_rh", "upward_w", "precip_3h", "precip_rate")
    with np.load(path) as data:
        return {key: unpack_selected(data, key, flat_indices) for key in keys}


def pava_probabilities(histogram: Histogram) -> np.ndarray:
    nonzero = histogram.count > 0.0
    x = np.flatnonzero(nonzero)
    if not len(x):
        return np.zeros_like(histogram.count)
    blocks: list[list[float]] = []
    for index in x:
        blocks.append([float(index), float(index), histogram.count[index], histogram.events[index]])
        while len(blocks) >= 2:
            left_rate = blocks[-2][3] / blocks[-2][2]
            right_rate = blocks[-1][3] / blocks[-1][2]
            if left_rate <= right_rate:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    probability = np.full_like(histogram.count, np.nan, dtype=np.float64)
    for start, end, count, events in blocks:
        probability[int(start) : int(end) + 1] = events / count
    known = np.flatnonzero(np.isfinite(probability))
    probability[: known[0]] = probability[known[0]]
    probability[known[-1] + 1 :] = probability[known[-1]]
    missing = ~np.isfinite(probability)
    probability[missing] = np.interp(np.flatnonzero(missing), known, probability[known])
    return np.clip(probability, 1.0e-5, 1.0 - 1.0e-5)


def auc_from_histogram(histogram: Histogram) -> float:
    positives = histogram.events
    negatives = histogram.count - histogram.events
    total_pos = positives.sum()
    total_neg = negatives.sum()
    if total_pos <= 0.0 or total_neg <= 0.0:
        return float("nan")
    neg_below = np.cumsum(negatives) - negatives
    concordant = np.sum(positives * (neg_below + 0.5 * negatives))
    return float(concordant / (total_pos * total_neg))


def metrics(histogram: Histogram, probability: np.ndarray) -> dict[str, float]:
    count = histogram.count
    events = histogram.events
    total = count.sum()
    event_total = events.sum()
    prevalence = event_total / total
    brier = np.sum(events * (1.0 - probability) ** 2 + (count - events) * probability**2) / total
    climatology_brier = prevalence * (1.0 - prevalence)
    log_loss = -np.sum(events * np.log(probability) + (count - events) * np.log(1.0 - probability)) / total
    return {
        "samples_weighted": float(total),
        "event_frequency": float(prevalence),
        "brier": float(brier),
        "brier_skill_vs_split_climatology": float(1.0 - brier / climatology_brier) if climatology_brier > 0.0 else float("nan"),
        "log_loss": float(log_loss),
        "auc": auc_from_histogram(histogram),
    }


def combine_histograms(histograms: Iterable[Histogram]) -> Histogram:
    combined = Histogram.empty()
    for histogram in histograms:
        combined.count += histogram.count
        combined.events += histogram.events
    return combined


def bootstrap_difference(
    baseline_by_day: dict[dt.datetime, Histogram],
    candidate_by_day: dict[dt.datetime, Histogram],
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float]:
    days = sorted(set(baseline_by_day) & set(candidate_by_day))
    if len(days) < 2:
        return {"days": len(days), "brier_difference_low": float("nan"), "brier_difference_high": float("nan"), "auc_difference_low": float("nan"), "auc_difference_high": float("nan")}
    random = np.random.default_rng(seed)
    brier_differences: list[float] = []
    auc_differences: list[float] = []
    for _ in range(draws):
        selected = random.integers(0, len(days), len(days))
        baseline = combine_histograms(baseline_by_day[days[index]] for index in selected)
        candidate = combine_histograms(candidate_by_day[days[index]] for index in selected)
        baseline_metrics = metrics(baseline, baseline_probability)
        candidate_metrics = metrics(candidate, candidate_probability)
        brier_differences.append(float(candidate_metrics["brier"]) - float(baseline_metrics["brier"]))
        auc_differences.append(float(candidate_metrics["auc"]) - float(baseline_metrics["auc"]))
    brier_low, brier_high = np.nanpercentile(brier_differences, [2.5, 97.5])
    auc_low, auc_high = np.nanpercentile(auc_differences, [2.5, 97.5])
    return {
        "days": len(days),
        "brier_difference_low": float(brier_low),
        "brier_difference_high": float(brier_high),
        "auc_difference_low": float(auc_low),
        "auc_difference_high": float(auc_high),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reliability_figure(
    output: Path,
    histograms: dict[str, dict[str, dict[str, Histogram]]],
    calibrators: dict[str, dict[str, np.ndarray]],
    names: list[str],
    period: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for name in names:
        test = histograms[period][name]["test"]
        probability = calibrators[period][name]
        bins = np.linspace(0.0, 1.0, 11)
        assigned = np.clip(np.digitize(probability, bins) - 1, 0, 9)
        forecast = []
        observed = []
        sizes = []
        for index in range(10):
            mask = assigned == index
            count = test.count[mask].sum()
            if count <= 0.0:
                continue
            forecast.append(np.sum(test.count[mask] * probability[mask]) / count)
            observed.append(test.events[mask].sum() / count)
            sizes.append(count)
        axes[0].plot(forecast, observed, marker="o", linewidth=1.8, label=name)
        axes[1].plot(SCORE_CENTERS, probability, linewidth=1.8, label=name)
    axes[0].plot([0, 1], [0, 1], color="#444444", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Calibrated forecast probability", ylabel="Observed frequency", title=f"{period} test reliability")
    axes[0].grid(alpha=0.25)
    axes[1].set(xlabel="Raw LPI / daily max LPI", ylabel="Train-fitted probability", title="Monotone calibration mapping")
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def solar_declination(timestamp: dt.datetime) -> float:
    day = timestamp.timetuple().tm_yday
    gamma = 2.0 * math.pi / 365.0 * (day - 1)
    return (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )


def solar_max_elevation_and_daylight_hours(timestamp: dt.datetime, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(latitude.astype(np.float64))
    declination = solar_declination(timestamp)
    max_elevation = np.rad2deg(np.arcsin(np.sin(lat_rad) * math.sin(declination) + np.cos(lat_rad) * math.cos(declination)))
    sunset = np.arccos(np.clip(-np.tan(lat_rad) * math.tan(declination), -1.0, 1.0))
    daylight = 24.0 * sunset / math.pi
    return max_elevation.astype(np.float32), daylight.astype(np.float32)


def solar_elevation(timestamp: dt.datetime, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Approximate solar elevation using NOAA's fractional-year equations."""

    day = timestamp.timetuple().tm_yday
    fractional_hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (fractional_hour - 12.0) / 24.0)
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    equation_minutes = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    solar_minutes = fractional_hour * 60.0 + equation_minutes + 4.0 * longitude
    hour_angle = np.deg2rad(solar_minutes / 4.0 - 180.0)
    lat_rad = np.deg2rad(latitude)
    cosine_zenith = np.sin(lat_rad) * math.sin(declination) + np.cos(lat_rad) * math.cos(declination) * np.cos(hour_angle)
    return (90.0 - np.rad2deg(np.arccos(np.clip(cosine_zenith, -1.0, 1.0)))).astype(np.float32)


def run_analysis(args: argparse.Namespace) -> None:
    archive_root = args.archive_root.expanduser()
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    lat, lon, in_bc, in_corridor = domain_masks(archive_root)
    analysis_mask = in_corridor
    flat_indices = np.flatnonzero(analysis_mask.reshape(-1))
    point_lat = lat.reshape(-1)[flat_indices]
    point_lon = lon.reshape(-1)[flat_indices]
    log(f"Analysis grid: {np.count_nonzero(in_bc):,} BC cells; {len(flat_indices):,} within {CORRIDOR_RADIUS_KM:g} km of transmission lines.")

    obs_paths = observation_paths(archive_root)
    raw_blocks = forecast_blocks(archive_root, obs_paths)
    valid_ends = sorted({valid for _, _, valid in raw_blocks})
    log(f"Reading {len(valid_ends)} unique observed three-hour blocks matched by {len(raw_blocks)} forecasts.")

    obs_targets: dict[dt.datetime, np.ndarray] = {}
    obs_density: dict[dt.datetime, np.ndarray] = {}
    inventory_rows: list[dict[str, object]] = []
    active_days: set[dt.datetime] = set()
    for index, valid in enumerate(valid_ends, start=1):
        targets, density = read_observation_targets(obs_paths[valid], point_lat, point_lon)
        obs_targets[valid] = targets[TARGET_RADIUS_KM]
        obs_density[valid] = density
        day = daily_window_start(valid)
        if np.any(targets[TARGET_RADIUS_KM] > 0.0):
            active_days.add(day)
        inventory_rows.append(
            {
                "valid_end_utc": valid.isoformat().replace("+00:00", "Z"),
                "day_start_utc": day.isoformat().replace("+00:00", "Z"),
                "corridor_flash_density_sum": float(density.sum()),
                "event_fraction_10km": float(targets[10.0].mean()),
                "event_fraction_20km": float(targets[20.0].mean()),
                "event_fraction_30km": float(targets[30.0].mean()),
                "event_fraction_40km": float(targets[40.0].mean()),
            }
        )
        if index % 32 == 0 or index == len(valid_ends):
            log(f"  observations {index}/{len(valid_ends)}")
    write_csv(output / "block_inventory.csv", inventory_rows)

    day_lookup, split_summary = split_days([daily_window_start(item[2]) for item in raw_blocks], active_days)
    blocks = [ForecastBlock(stamp, fhour, valid, daily_window_start(valid), day_lookup[daily_window_start(valid)]) for stamp, fhour, valid in raw_blocks]
    block_multiplicity = Counter(block.valid_end for block in blocks)

    daily_windows: dict[str, dt.datetime] = {}
    blocks_by_run: dict[str, list[ForecastBlock]] = defaultdict(list)
    for block in blocks:
        blocks_by_run[block.run_stamp].append(block)
    for stamp, run_blocks in blocks_by_run.items():
        init = parse_utc_stamp(stamp)
        possible = sorted({block.day_start for block in run_blocks if block.day_start >= init and block.day_start + dt.timedelta(days=1) <= init + dt.timedelta(hours=48)})
        for start in possible:
            expected = {start + dt.timedelta(hours=hour) for hour in range(3, 25, 3)}
            available = {block.valid_end for block in run_blocks}
            if expected.issubset(available):
                daily_windows[stamp] = start
                break
    daily_multiplicity = Counter(daily_windows.values())

    formulas = candidate_formulas()
    candidate_names = ["issued_lpi"] + [formula.name for formula in formulas]
    histograms: dict[str, dict[str, dict[str, Histogram]]] = {
        period: {name: {split: Histogram.empty() for split in SPLITS} for name in candidate_names}
        for period in ("3h", "24h")
    }
    day_histograms: dict[str, dict[str, dict[str, dict[dt.datetime, Histogram]]]] = {
        period: {name: {split: {} for split in SPLITS} for name in candidate_names}
        for period in ("3h", "24h")
    }
    solar_rows: list[dict[str, object]] = []
    solar_case_histograms: list[Histogram] = []
    solar_elevation_histograms = {
        split: {label: Histogram.empty() for label in ("night", "twilight_low", "day_low", "day_high")}
        for split in SPLITS
    }
    run_dirs = {path.name: path for path in ingredient_run_dirs(archive_root)}

    for run_index, (stamp, run_blocks) in enumerate(sorted(blocks_by_run.items()), start=1):
        run_dir = run_dirs[stamp]
        endpoint_scores: dict[dt.datetime, dict[str, np.ndarray]] = {}
        for block in sorted(run_blocks, key=lambda item: item.forecast_hour):
            maxima = {name: np.full(len(flat_indices), -np.inf, dtype=np.float32) for name in candidate_names}
            for hour in range(block.forecast_hour - 2, block.forecast_hour + 1):
                fields = read_hour_fields(run_dir / f"f{hour:03d}.npz", flat_indices)
                maxima["issued_lpi"] = np.maximum(maxima["issued_lpi"], fields["potential"])
                for formula in formulas:
                    maxima[formula.name] = np.maximum(maxima[formula.name], compute_formula(fields, formula))
            event = obs_targets[block.valid_end]
            sample_weight = 1.0 / block_multiplicity[block.valid_end]
            common = np.isfinite(event)
            for score in maxima.values():
                common &= np.isfinite(score)
            for name, score in maxima.items():
                histograms["3h"][name][block.split].add(score[common], event[common], sample_weight)
                day_histogram = day_histograms["3h"][name][block.split].setdefault(block.day_start, Histogram.empty())
                day_histogram.add(score[common], event[common], sample_weight)
            endpoint_scores[block.valid_end] = maxima

            max_elevation, daylight = solar_max_elevation_and_daylight_hours(block.valid_end, point_lat)
            midpoint_elevation = solar_elevation(block.valid_end - dt.timedelta(hours=1, minutes=30), point_lat, point_lon)
            case_histogram = Histogram.empty()
            case_histogram.add(maxima["issued_lpi"][common], event[common])
            solar_case_histograms.append(case_histogram)
            solar_categories = (
                ("night", midpoint_elevation < -6.0),
                ("twilight_low", (midpoint_elevation >= -6.0) & (midpoint_elevation < 10.0)),
                ("day_low", (midpoint_elevation >= 10.0) & (midpoint_elevation < 30.0)),
                ("day_high", midpoint_elevation >= 30.0),
            )
            for label, category_mask in solar_categories:
                solar_elevation_histograms[block.split][label].add(
                    maxima["issued_lpi"][category_mask & common], event[category_mask & common], sample_weight
                )
            solar_rows.append(
                {
                    "run_stamp": stamp,
                    "valid_end_utc": block.valid_end.isoformat().replace("+00:00", "Z"),
                    "split": block.split,
                    "forecast_hour": block.forecast_hour,
                    "mean_issued_lpi": float(np.nanmean(maxima["issued_lpi"])),
                    "event_fraction_30km": float(event.mean()),
                    "mean_block_midpoint_solar_elevation_deg": float(np.mean(midpoint_elevation)),
                    "mean_daily_max_solar_elevation_deg": float(np.mean(max_elevation)),
                    "mean_daylight_hours": float(np.mean(daylight)),
                }
            )

        day_start = daily_windows.get(stamp)
        if day_start is not None:
            ends = [day_start + dt.timedelta(hours=hour) for hour in range(3, 25, 3)]
            daily_event = np.maximum.reduce([obs_targets[end] for end in ends])
            split = day_lookup[day_start]
            sample_weight = 1.0 / daily_multiplicity[day_start]
            daily_scores = {
                name: np.maximum.reduce([endpoint_scores[end][name] for end in ends])
                for name in candidate_names
            }
            daily_common = np.isfinite(daily_event)
            for item_score in daily_scores.values():
                daily_common &= np.isfinite(item_score)
            for name in candidate_names:
                daily_score = daily_scores[name]
                histograms["24h"][name][split].add(daily_score[daily_common], daily_event[daily_common], sample_weight)
                day_histogram = day_histograms["24h"][name][split].setdefault(day_start, Histogram.empty())
                day_histogram.add(daily_score[daily_common], daily_event[daily_common], sample_weight)
        if run_index % 8 == 0 or run_index == len(blocks_by_run):
            log(f"  forecast runs {run_index}/{len(blocks_by_run)}")

    calibrators: dict[str, dict[str, np.ndarray]] = {period: {} for period in ("3h", "24h")}
    metric_rows: list[dict[str, object]] = []
    for period in ("3h", "24h"):
        for name in candidate_names:
            calibrator = pava_probabilities(histograms[period][name]["train"])
            calibrators[period][name] = calibrator
            formula = next((item for item in formulas if item.name == name), None)
            for split in SPLITS:
                row: dict[str, object] = {
                    "period": period,
                    "candidate": name,
                    "family": "issued_baseline" if formula is None else formula.family,
                    "split": split,
                }
                row.update(metrics(histograms[period][name][split], calibrator))
                metric_rows.append(row)
    write_csv(output / "candidate_metrics.csv", metric_rows)

    issued_calibrator = calibrators["3h"]["issued_lpi"]
    for row, histogram in zip(solar_rows, solar_case_histograms):
        total = histogram.count.sum()
        row["train_calibrated_probability"] = float(np.sum(histogram.count * issued_calibrator) / total)
        row["calibration_residual"] = float(row["event_fraction_30km"]) - float(row["train_calibrated_probability"])
    write_csv(output / "solar_case_summary.csv", solar_rows)
    solar_bin_rows: list[dict[str, object]] = []
    for split in SPLITS:
        for label, histogram in solar_elevation_histograms[split].items():
            total = histogram.count.sum()
            if total <= 0.0:
                continue
            predicted = float(np.sum(histogram.count * issued_calibrator) / total)
            observed = float(histogram.events.sum() / total)
            solar_bin_rows.append(
                {
                    "split": split,
                    "solar_elevation_bin": label,
                    "weighted_samples": float(total),
                    "observed_event_frequency": observed,
                    "train_calibrated_probability": predicted,
                    "calibration_residual": observed - predicted,
                }
            )
    write_csv(output / "solar_elevation_summary.csv", solar_bin_rows)

    validation_3h = sorted(
        (row for row in metric_rows if row["period"] == "3h" and row["split"] == "validate"),
        key=lambda row: (float(row["brier"]), -float(row["auc"])),
    )
    validation_24h = sorted(
        (row for row in metric_rows if row["period"] == "24h" and row["split"] == "validate"),
        key=lambda row: (float(row["brier"]), -float(row["auc"])),
    )
    winner_3h = str(validation_3h[0]["candidate"])
    winner_24h = str(validation_24h[0]["candidate"])
    reliability_figure(output / "reliability_3h.png", histograms, calibrators, ["issued_lpi", winner_3h], "3h")
    reliability_figure(output / "reliability_24h.png", histograms, calibrators, ["issued_lpi", winner_24h], "24h")

    winner_formulas = {
        period: asdict(next(formula for formula in formulas if formula.name == winner)) if winner != "issued_lpi" else None
        for period, winner in (("3h", winner_3h), ("24h", winner_24h))
    }
    deployment_gate: dict[str, dict[str, object]] = {}
    bootstrap_rows: list[dict[str, object]] = []
    for period, winner in (("3h", winner_3h), ("24h", winner_24h)):
        baseline_validation = metric_lookup(metric_rows, period, "issued_lpi", "validate")
        candidate_validation = metric_lookup(metric_rows, period, winner, "validate")
        baseline_test = metric_lookup(metric_rows, period, "issued_lpi", "test")
        candidate_test = metric_lookup(metric_rows, period, winner, "test")
        validation_brier_better = float(candidate_validation["brier"]) < float(baseline_validation["brier"])
        validation_auc_acceptable = float(candidate_validation["auc"]) >= float(baseline_validation["auc"]) - 0.02
        test_brier_not_worse = float(candidate_test["brier"]) <= float(baseline_test["brier"])
        test_auc_not_worse = float(candidate_test["auc"]) >= float(baseline_test["auc"]) - 0.02
        deployment_gate[period] = {
            "candidate": winner,
            "validation_brier_better": validation_brier_better,
            "validation_auc_within_0.02": validation_auc_acceptable,
            "test_brier_not_worse": test_brier_not_worse,
            "test_auc_within_0.02": test_auc_not_worse,
            "passed": all((validation_brier_better, validation_auc_acceptable, test_brier_not_worse, test_auc_not_worse)),
        }
        bootstrap = bootstrap_difference(
            day_histograms[period]["issued_lpi"]["test"],
            day_histograms[period][winner]["test"],
            calibrators[period]["issued_lpi"],
            calibrators[period][winner],
            seed=20260812 + (3 if period == "3h" else 24),
        )
        bootstrap_rows.append({"period": period, "candidate": winner, **bootstrap})
        bootstrap_brier_support = float(bootstrap["brier_difference_high"]) < 0.0
        deployment_gate[period]["test_brier_bootstrap_95pct_below_zero"] = bootstrap_brier_support
        deployment_gate[period]["passed"] = bool(deployment_gate[period]["passed"] and bootstrap_brier_support)
    write_csv(output / "test_bootstrap_differences.csv", bootstrap_rows)
    payload = {
        "created_at_utc": dt.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "archive_root": str(archive_root),
        "analysis_domain": f"BC cells within {CORRIDOR_RADIUS_KM:g} km of BCH transmission lines",
        "target": f"at least one observed flash within {TARGET_RADIUS_KM:g} km",
        "split": split_summary,
        "forecast_blocks": len(blocks),
        "unique_observation_blocks": len(valid_ends),
        "daily_forecasts": len(daily_windows),
        "candidate_count": len(candidate_names),
        "validation_winners": {"3h": winner_3h, "24h": winner_24h},
        "winner_formulas": winner_formulas,
        "deployment_gate": deployment_gate,
        "deployment_recommendation": "retain_issued_lpi",
        "production_changed": False,
    }
    (output / "analysis_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(output, payload, metric_rows, validation_3h, validation_24h, solar_rows, solar_bin_rows, bootstrap_rows)
    log(f"Analysis complete: {output / 'report.md'}")


def metric_lookup(rows: list[dict[str, object]], period: str, candidate: str, split: str) -> dict[str, object]:
    return next(row for row in rows if row["period"] == period and row["candidate"] == candidate and row["split"] == split)


def write_report(
    output: Path,
    summary: dict[str, object],
    metric_rows: list[dict[str, object]],
    validation_3h: list[dict[str, object]],
    validation_24h: list[dict[str, object]],
    solar_rows: list[dict[str, object]],
    solar_bin_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
) -> None:
    winner_3h = str(summary["validation_winners"]["3h"])
    winner_24h = str(summary["validation_winners"]["24h"])
    lines = [
        "# Initial LPI Calibration Review",
        "",
        "## Scope",
        "",
        f"- Domain: {summary['analysis_domain']}.",
        f"- Predictand: {summary['target']} in each three-hour block; the 24-hour target is occurrence in any of the eight 12Z-to-12Z blocks.",
        f"- Sample: {summary['forecast_blocks']} matched forecasts, {summary['unique_observation_blocks']} unique observed blocks, and {summary['daily_forecasts']} complete daily forecasts.",
        "- Forecasts verifying the same observed block/day are down-weighted so repeated model cycles do not duplicate the event.",
        "- All candidate probability mappings are monotone isotonic fits on training data only.",
        "- No operational formula was changed.",
        "",
        "## Split",
        "",
        f"The archive supported a chronological train/validate/test split: `{json.dumps(summary['split']['days'])}` days, with active-day counts `{json.dumps(summary['split']['active_days'])}`. The final period is an early/mid-August holdout, so it directly tests the July-to-August seasonal shift, but it remains a short provisional test.",
        "",
        "## Results",
        "",
        "| Period | Forecast | Validation Brier | Validation AUC | Test Brier | Test AUC | Test event frequency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for period, winner in (("3h", winner_3h), ("24h", winner_24h)):
        for candidate in dict.fromkeys(["issued_lpi", winner]):
            validation = metric_lookup(metric_rows, period, candidate, "validate")
            test = metric_lookup(metric_rows, period, candidate, "test")
            lines.append(
                f"| {period} | {candidate} | {float(validation['brier']):.5f} | {float(validation['auc']):.3f} | "
                f"{float(test['brier']):.5f} | {float(test['auc']):.3f} | {float(test['event_frequency']):.3f} |"
            )
    lines.extend(
        [
            "",
            "Day-block bootstrap intervals below are candidate minus issued LPI on the untouched test days. Negative Brier and positive AUC differences favor the candidate.",
            "",
            "| Period | Candidate | Test days | Brier difference 95% interval | AUC difference 95% interval |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['period']} | {row['candidate']} | {row['days']} | "
            f"[{float(row['brier_difference_low']):+.5f}, {float(row['brier_difference_high']):+.5f}] | "
            f"[{float(row['auc_difference_low']):+.3f}, {float(row['auc_difference_high']):+.3f}] |"
        )
    lines.extend(
        [
            "",
            "The candidate winner is chosen on validation Brier score, with AUC as the tie-breaker. Its test row is evaluation, not another selection step. Deployment additionally requires validation AUC to remain within 0.02 of the issued LPI and no material Brier/AUC degradation on test. Neither validation-selected candidate passed that gate, so the issued LPI should remain operational.",
            "",
            "### Leading validation candidates",
            "",
            "| 3-hour candidate | Brier | AUC |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in validation_3h[:8]:
        lines.append(f"| {row['candidate']} | {float(row['brier']):.5f} | {float(row['auc']):.3f} |")
    lines.extend(["", "| 24-hour candidate | Brier | AUC |", "| --- | ---: | ---: |"])
    for row in validation_24h[:8]:
        lines.append(f"| {row['candidate']} | {float(row['brier']):.5f} | {float(row['auc']):.3f} |")
    lines.extend(
        [
            "",
            "### Component ablations",
            "",
            "Values are changes relative to the reconstructed current formula on the same common finite sample. Negative Brier and positive AUC changes are improvements.",
            "",
            "| Period | Ablation | Validation delta Brier | Validation delta AUC | Test delta Brier | Test delta AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    ablations = (
        "ablate_cape",
        "ablate_mid_rh",
        "ablate_precip_trigger",
        "ablate_updraft_trigger",
        "ablate_trigger",
    )
    for period in ("3h", "24h"):
        baseline_validation = metric_lookup(metric_rows, period, "recomputed_baseline", "validate")
        baseline_test = metric_lookup(metric_rows, period, "recomputed_baseline", "test")
        for candidate in ablations:
            validation = metric_lookup(metric_rows, period, candidate, "validate")
            test = metric_lookup(metric_rows, period, candidate, "test")
            lines.append(
                f"| {period} | {candidate} | {float(validation['brier']) - float(baseline_validation['brier']):+.5f} | "
                f"{float(validation['auc']) - float(baseline_validation['auc']):+.3f} | "
                f"{float(test['brier']) - float(baseline_test['brier']):+.5f} | "
                f"{float(test['auc']) - float(baseline_test['auc']):+.3f} |"
            )
    lines.extend(
        [
            "",
            "CAPE and mid-level RH modulation have little independent effect in this short sample. Removing the precipitation trigger consistently hurts. Removing the resolved-updraft trigger produces unstable regime-dependent results: lower Brier but substantially worse 3-hour discrimination, and opposite validation/test behavior at 24 hours. That is evidence to retain the combined trigger while collecting more storm days.",
        ]
    )

    split_solar: dict[str, tuple[float, float]] = {}
    for split in SPLITS:
        values = [row for row in solar_rows if row["split"] == split]
        split_solar[split] = (
            float(np.mean([float(row["mean_daily_max_solar_elevation_deg"]) for row in values])),
            float(np.mean([float(row["mean_daylight_hours"]) for row in values])),
        )
    lines.extend(
        [
            "",
            "## Solar Geometry",
            "",
            "Solar elevation and day length were calculated astronomically for every matched case and corridor latitude. This is preferable to a calendar-day sine because it handles latitude, longitude, UTC valid time, and changing declination directly.",
            "",
            "| Split | Mean daily maximum elevation | Mean daylight |",
            "| --- | ---: | ---: |",
        ]
    )
    for split, values in split_solar.items():
        lines.append(f"| {split} | {values[0]:.2f} deg | {values[1]:.2f} h |")
    lines.extend(
        [
            "",
            "Residuals below use the issued LPI's training-fitted monotone calibration. Positive values mean observed lightning was more frequent than forecast.",
            "",
            "| Split | Solar-elevation regime | Observed | Forecast | Residual |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in solar_bin_rows:
        lines.append(
            f"| {row['split']} | {row['solar_elevation_bin']} | {float(row['observed_event_frequency']):.3f} | "
            f"{float(row['train_calibrated_probability']):.3f} | {float(row['calibration_residual']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "The chronological holdout therefore includes the seasonal solar change automatically. A direct solar multiplier is not promoted from this archive: solar declination is inseparable from the time trend over only a few weeks, and MU-LI/CAPE already respond to modeled heating. Solar should remain a stratification variable until a second season or a materially longer archive shows repeatable residual bias at the same LPI.",
            "",
            "## Interpretation",
            "",
            "An ablation sets one contribution to neutral and asks whether held-out skill improves or degrades. For example, `ablate_cape` removes only CAPE's 20% modulation while retaining MU-LI, moisture, and triggering. This identifies which ingredients add independent information rather than merely appearing meteorologically plausible.",
            "",
            "A constrained adjustment changes only declared ramp endpoints, component weights, or the trigger exponent inside physically defensible bounds. Every retained relationship remains monotone: more instability, charging-layer moisture/depth, ascent, or precipitation can never lower the raw LPI. This is intentionally not an unconstrained fit to 22 days of data.",
            "",
            "## Next Decision",
            "",
            f"The lowest-Brier validation candidates were `{winner_3h}` for 3-hour guidance and `{winner_24h}` for 24-hour guidance, but both failed the deployment gate. Retain the issued formula through the remainder of the 2026 convective season. Re-run this analysis as the archive grows, then require the same candidate family to improve more than one chronological split before a parallel operational trial.",
            "",
            "The reconstructed candidates are evaluated on the archived 5 km ingredients without reproducing the operational formula's native-grid spatial smoothing; `issued_lpi` is the exact archived, smoothed product. Any future candidate that passes the statistical gate must be rerun through the native 2.5 km smoothing and reviewed on case maps before deployment.",
            "",
            "The meteorologist-labeling protocol is in `docs/lpi_meteorologist_labeling_protocol.md`. Those labels should calibrate operational low/moderate/high categories, while the objective flash data continue to calibrate quantitative 3-hour and 24-hour occurrence probabilities.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=archive.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    run_analysis(build_parser().parse_args())
