#!/usr/bin/env python3
"""Evaluate the spatial scale and raw-score calibration of the issued BC LPI."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, label

import analyze_lpi_calibration as calibration
import lightning_ml_archive as archive


UTC = dt.timezone.utc
SMOOTHING_KM = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
TARGET_RADII_KM = (10.0, 20.0, 30.0)
PERIODS = ("3h", "24h")
RAW_SCORE_LEVELS = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0)
DEFAULT_OUTPUT = Path("output/lpi_calibration/smoothing_20260812")


def log(message: str) -> None:
    print(message, flush=True)


def smooth_nan(data: np.ndarray, smoothing_km: float) -> np.ndarray:
    if smoothing_km <= 0.0:
        return data.astype(np.float32, copy=True)
    sigma = smoothing_km / calibration.MODEL_GRID_KM
    valid = np.isfinite(data)
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    values = gaussian_filter(np.where(valid, data, 0.0).astype(np.float32), sigma=sigma, mode="nearest")
    result = np.full(data.shape, np.nan, dtype=np.float32)
    np.divide(values, weights, out=result, where=weights > 1.0e-4)
    return result


def read_full_potential(path: Path) -> np.ndarray:
    spec = next(item for item in archive.HOURLY_LPI_INGREDIENT_SPECS if item.key == "potential")
    with np.load(path) as data:
        return archive.unpack_field(data["potential"], spec).astype(np.float32)


def common_finite_scores(scores: dict[float, np.ndarray], event: np.ndarray) -> np.ndarray:
    common = np.isfinite(event)
    for score in scores.values():
        common &= np.isfinite(score)
    return common


def spatial_detail(score: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    masked = np.where(mask, score, np.nan)
    finite = mask & np.isfinite(score)
    dx_valid = finite[:, 1:] & finite[:, :-1]
    dy_valid = finite[1:, :] & finite[:-1, :]
    dx = np.abs(np.diff(masked, axis=1))[dx_valid]
    dy = np.abs(np.diff(masked, axis=0))[dy_valid]
    threshold = mask & np.isfinite(score) & (score >= 20.0)
    _, components = label(threshold, structure=np.ones((3, 3), dtype=np.int8))
    return {
        "mean_abs_neighbor_difference": float((dx.sum() + dy.sum()) / (len(dx) + len(dy))),
        "area_fraction_lpi_ge_20": float(np.count_nonzero(threshold) / np.count_nonzero(mask)),
        "components_lpi_ge_20": float(components),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoothing_name(value: float) -> str:
    return "issued" if value == 0.0 else f"gaussian_{value:g}km"


def score_level_probability(calibrator: np.ndarray, level: float) -> float:
    index = int(np.clip(np.digitize([level], calibration.SCORE_EDGES)[0] - 1, 0, len(calibrator) - 1))
    return float(calibrator[index])


def bootstrap_difference(
    baseline: dict[dt.datetime, calibration.Histogram],
    candidate: dict[dt.datetime, calibration.Histogram],
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float]:
    return calibration.bootstrap_difference(
        baseline,
        candidate,
        baseline_probability,
        candidate_probability,
        seed=seed,
        draws=draws,
    )


def reliability_figure(
    output: Path,
    histograms: dict[str, dict[float, dict[float, dict[str, calibration.Histogram]]]],
    calibrators: dict[str, dict[float, dict[float, np.ndarray]]],
    selected: dict[str, dict[float, float]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, period in enumerate(PERIODS):
        for column, radius in enumerate(TARGET_RADII_KM):
            ax = axes[row, column]
            values = dict.fromkeys((0.0, selected[period][radius]))
            for smoothing in values:
                histogram = histograms[period][radius][smoothing]["test"]
                probability = calibrators[period][radius][smoothing]
                bins = np.linspace(0.0, 1.0, 11)
                assigned = np.clip(np.digitize(probability, bins) - 1, 0, 9)
                forecasts: list[float] = []
                observed: list[float] = []
                for index in range(10):
                    mask = assigned == index
                    count = histogram.count[mask].sum()
                    if count <= 0.0:
                        continue
                    forecasts.append(float(np.sum(histogram.count[mask] * probability[mask]) / count))
                    observed.append(float(histogram.events[mask].sum() / count))
                ax.plot(forecasts, observed, marker="o", label=smoothing_name(smoothing))
            ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=0.8)
            ax.set_title(f"{period}, event within {radius:g} km")
            ax.set_xlabel("Calibrated probability")
            ax.set_ylabel("Observed frequency")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def smoothing_case_figure(
    output: Path,
    score_grids: dict[float, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    in_bc: np.ndarray,
    title: str,
) -> None:
    scales = (0.0, 5.0, 10.0, 15.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    levels = [5, 10, 20, 30, 45, 60, 75, 90, 100]
    colors = ["#f1f1f1", "#ded7ec", "#c2afe2", "#9b78ce", "#b447a6", "#d71e72", "#f05289", "#ff9fc2"]
    last = None
    for ax, smoothing in zip(axes.flat, scales):
        field = np.where(in_bc, score_grids[smoothing], np.nan)
        last = ax.contourf(lon, lat, field, levels=levels, colors=colors, extend="max")
        ax.contour(lon, lat, in_bc.astype(float), levels=[0.5], colors="#303030", linewidths=0.6)
        ax.set_xlim(-139.2, -114.0)
        ax.set_ylim(48.2, 60.1)
        ax.set_aspect(1.65)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(smoothing_name(smoothing), fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if last is not None:
        fig.colorbar(last, ax=axes, shrink=0.78, label="Issued LPI")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = args.archive_root.expanduser()
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    lat, lon, in_bc, in_corridor = calibration.domain_masks(root)
    flat_indices = np.flatnonzero(in_corridor.reshape(-1))
    point_lat = lat.reshape(-1)[flat_indices]
    point_lon = lon.reshape(-1)[flat_indices]
    obs_paths = calibration.observation_paths(root)
    raw_blocks = calibration.forecast_blocks(root, obs_paths)
    valid_ends = sorted({valid for _, _, valid in raw_blocks})

    obs_targets: dict[dt.datetime, dict[float, np.ndarray]] = {}
    active_days: set[dt.datetime] = set()
    for index, valid in enumerate(valid_ends, start=1):
        targets, _ = calibration.read_observation_targets(obs_paths[valid], point_lat, point_lon, TARGET_RADII_KM)
        obs_targets[valid] = targets
        if np.any(targets[30.0] > 0.0):
            active_days.add(calibration.daily_window_start(valid))
        if index % 40 == 0 or index == len(valid_ends):
            log(f"observations {index}/{len(valid_ends)}")

    day_lookup, split_summary = calibration.split_days(
        [calibration.daily_window_start(item[2]) for item in raw_blocks], active_days
    )
    blocks = [
        calibration.ForecastBlock(stamp, fhour, valid, calibration.daily_window_start(valid), day_lookup[calibration.daily_window_start(valid)])
        for stamp, fhour, valid in raw_blocks
    ]
    block_multiplicity = Counter(block.valid_end for block in blocks)
    blocks_by_run: dict[str, list[calibration.ForecastBlock]] = defaultdict(list)
    for block in blocks:
        blocks_by_run[block.run_stamp].append(block)

    daily_windows: dict[str, dt.datetime] = {}
    for stamp, run_blocks in blocks_by_run.items():
        init = calibration.parse_utc_stamp(stamp)
        possible = sorted({block.day_start for block in run_blocks if block.day_start >= init and block.day_start + dt.timedelta(days=1) <= init + dt.timedelta(hours=48)})
        available = {block.valid_end for block in run_blocks}
        for start in possible:
            if {start + dt.timedelta(hours=hour) for hour in range(3, 25, 3)}.issubset(available):
                daily_windows[stamp] = start
                break
    daily_multiplicity = Counter(daily_windows.values())

    histograms = {
        period: {
            radius: {smoothing: {split: calibration.Histogram.empty() for split in calibration.SPLITS} for smoothing in SMOOTHING_KM}
            for radius in TARGET_RADII_KM
        }
        for period in PERIODS
    }
    day_histograms = {
        period: {
            radius: {smoothing: {split: {} for split in calibration.SPLITS} for smoothing in SMOOTHING_KM}
            for radius in TARGET_RADII_KM
        }
        for period in PERIODS
    }
    detail_accumulator = {smoothing: defaultdict(float) for smoothing in SMOOTHING_KM}
    detail_count = 0
    best_case_strength = -1.0
    best_case: tuple[str, dt.datetime, dict[float, np.ndarray]] | None = None
    run_dirs = {path.name: path for path in calibration.ingredient_run_dirs(root)}

    for run_index, (stamp, run_blocks) in enumerate(sorted(blocks_by_run.items()), start=1):
        run_dir = run_dirs[stamp]
        required_hours = sorted({hour for block in run_blocks for hour in range(block.forecast_hour - 2, block.forecast_hour + 1)})
        hourly = {hour: read_full_potential(run_dir / f"f{hour:03d}.npz") for hour in required_hours}
        endpoint_scores: dict[dt.datetime, dict[float, np.ndarray]] = {}
        for block in sorted(run_blocks, key=lambda item: item.forecast_hour):
            raw = np.fmax.reduce([hourly[hour] for hour in range(block.forecast_hour - 2, block.forecast_hour + 1)])
            grids = {smoothing: smooth_nan(raw, smoothing) for smoothing in SMOOTHING_KM}
            scores = {smoothing: grid.reshape(-1)[flat_indices] for smoothing, grid in grids.items()}
            endpoint_scores[block.valid_end] = scores
            weight = 1.0 / block_multiplicity[block.valid_end]
            for radius in TARGET_RADII_KM:
                event = obs_targets[block.valid_end][radius]
                common = common_finite_scores(scores, event)
                for smoothing, score in scores.items():
                    histograms["3h"][radius][smoothing][block.split].add(score[common], event[common], weight)
                    day_histogram = day_histograms["3h"][radius][smoothing][block.split].setdefault(block.day_start, calibration.Histogram.empty())
                    day_histogram.add(score[common], event[common], weight)
            if block.split == "test":
                detail_count += 1
                for smoothing, grid in grids.items():
                    for key, value in spatial_detail(grid, in_bc).items():
                        detail_accumulator[smoothing][key] += value
                strength = float(np.nanpercentile(raw[in_corridor], 99.5))
                if strength > best_case_strength:
                    best_case_strength = strength
                    best_case = (stamp, block.valid_end, {key: value.copy() for key, value in grids.items()})

        start = daily_windows.get(stamp)
        if start is not None:
            ends = [start + dt.timedelta(hours=hour) for hour in range(3, 25, 3)]
            daily_scores = {
                smoothing: np.maximum.reduce([endpoint_scores[end][smoothing] for end in ends])
                for smoothing in SMOOTHING_KM
            }
            weight = 1.0 / daily_multiplicity[start]
            split = day_lookup[start]
            for radius in TARGET_RADII_KM:
                event = np.maximum.reduce([obs_targets[end][radius] for end in ends])
                common = common_finite_scores(daily_scores, event)
                for smoothing, score in daily_scores.items():
                    histograms["24h"][radius][smoothing][split].add(score[common], event[common], weight)
                    day_histogram = day_histograms["24h"][radius][smoothing][split].setdefault(start, calibration.Histogram.empty())
                    day_histogram.add(score[common], event[common], weight)
        if run_index % 8 == 0 or run_index == len(blocks_by_run):
            log(f"runs {run_index}/{len(blocks_by_run)}")

    calibrators = {
        period: {
            radius: {smoothing: calibration.pava_probabilities(histograms[period][radius][smoothing]["train"]) for smoothing in SMOOTHING_KM}
            for radius in TARGET_RADII_KM
        }
        for period in PERIODS
    }
    metric_rows: list[dict[str, object]] = []
    for period in PERIODS:
        for radius in TARGET_RADII_KM:
            for smoothing in SMOOTHING_KM:
                for split in calibration.SPLITS:
                    row: dict[str, object] = {
                        "period": period,
                        "target_radius_km": radius,
                        "smoothing_km": smoothing,
                        "candidate": smoothing_name(smoothing),
                        "split": split,
                    }
                    row.update(calibration.metrics(histograms[period][radius][smoothing][split], calibrators[period][radius][smoothing]))
                    metric_rows.append(row)
    write_csv(output / "smoothing_metrics.csv", metric_rows)

    selected: dict[str, dict[float, float]] = {period: {} for period in PERIODS}
    bootstrap_rows: list[dict[str, object]] = []
    for period in PERIODS:
        for radius in TARGET_RADII_KM:
            baseline_validation = next(row for row in metric_rows if row["period"] == period and row["target_radius_km"] == radius and row["smoothing_km"] == 0.0 and row["split"] == "validate")
            candidates = [
                row
                for row in metric_rows
                if row["period"] == period
                and row["target_radius_km"] == radius
                and row["split"] == "validate"
                and float(row["auc"]) >= float(baseline_validation["auc"]) - 0.02
            ]
            winner = min(candidates, key=lambda row: (float(row["brier"]), float(row["smoothing_km"])))
            smoothing = float(winner["smoothing_km"])
            selected[period][radius] = smoothing
            bootstrap = bootstrap_difference(
                day_histograms[period][radius][0.0]["test"],
                day_histograms[period][radius][smoothing]["test"],
                calibrators[period][radius][0.0],
                calibrators[period][radius][smoothing],
                seed=20260812 + int(radius) + int(smoothing) + (0 if period == "3h" else 1000),
            )
            bootstrap_rows.append({"period": period, "target_radius_km": radius, "selected_smoothing_km": smoothing, **bootstrap})
    write_csv(output / "smoothing_bootstrap.csv", bootstrap_rows)

    mapping_rows: list[dict[str, object]] = []
    for period in PERIODS:
        for radius in TARGET_RADII_KM:
            for smoothing in dict.fromkeys((0.0, selected[period][radius])):
                probability = calibrators[period][radius][smoothing]
                for level in RAW_SCORE_LEVELS:
                    mapping_rows.append(
                        {
                            "period": period,
                            "target_radius_km": radius,
                            "smoothing_km": smoothing,
                            "raw_lpi": level,
                            "train_fitted_probability": score_level_probability(probability, level),
                        }
                    )
    write_csv(output / "raw_score_probability_mapping.csv", mapping_rows)

    detail_rows = [
        {"smoothing_km": smoothing, **{key: value / detail_count for key, value in values.items()}}
        for smoothing, values in detail_accumulator.items()
    ]
    write_csv(output / "spatial_detail_metrics.csv", detail_rows)
    reliability_figure(output / "smoothing_reliability.png", histograms, calibrators, selected)
    if best_case is not None:
        stamp, valid, grids = best_case
        smoothing_case_figure(
            output / "active_case_smoothing.png",
            grids,
            lat,
            lon,
            in_bc,
            f"Strongest held-out corridor case | run {stamp} | valid {valid:%Y-%m-%d %HZ}",
        )

    summary = {
        "created_at_utc": dt.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "forecast_blocks": len(blocks),
        "unique_observation_blocks": len(valid_ends),
        "split": split_summary,
        "smoothing_scales_km": list(SMOOTHING_KM),
        "target_radii_km": list(TARGET_RADII_KM),
        "validation_selected_smoothing_km": selected,
        "production_changed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(output, summary, metric_rows, bootstrap_rows, mapping_rows, detail_rows)
    log(f"complete: {output / 'report.md'}")


def find_metric(
    rows: list[dict[str, object]], period: str, radius: float, smoothing: float, split: str
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["period"] == period
        and row["target_radius_km"] == radius
        and row["smoothing_km"] == smoothing
        and row["split"] == split
    )


def write_report(
    output: Path,
    summary: dict[str, object],
    metrics: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    mapping_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
) -> None:
    selected = summary["validation_selected_smoothing_km"]
    lines = [
        "# LPI Spatial Smoothing and Score Calibration",
        "",
        "This review starts from the exact archived issued LPI. Each candidate applies Gaussian smoothing after the operational three-hour maximum, so it tests the user-visible risk field without changing the meteorological ingredient formula.",
        "",
        "## Held-Out Results",
        "",
        "| Period | Target | Validation-selected smoothing | Validation Brier change | Test Brier change | Test AUC change | Test day-bootstrap Brier interval | Test day-bootstrap AUC interval |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for period in PERIODS:
        for radius in TARGET_RADII_KM:
            smoothing = float(selected[period][str(radius)] if str(radius) in selected[period] else selected[period][radius])
            baseline_validation = find_metric(metrics, period, radius, 0.0, "validate")
            candidate_validation = find_metric(metrics, period, radius, smoothing, "validate")
            baseline_test = find_metric(metrics, period, radius, 0.0, "test")
            candidate_test = find_metric(metrics, period, radius, smoothing, "test")
            bootstrap = next(row for row in bootstrap_rows if row["period"] == period and row["target_radius_km"] == radius)
            lines.append(
                f"| {period} | {radius:g} km | {smoothing:g} km | "
                f"{float(candidate_validation['brier']) - float(baseline_validation['brier']):+.5f} | "
                f"{float(candidate_test['brier']) - float(baseline_test['brier']):+.5f} | "
                f"{float(candidate_test['auc']) - float(baseline_test['auc']):+.3f} | "
                f"[{float(bootstrap['brier_difference_low']):+.5f}, {float(bootstrap['brier_difference_high']):+.5f}] | "
                f"[{float(bootstrap['auc_difference_low']):+.3f}, {float(bootstrap['auc_difference_high']):+.3f}] |"
            )

    lines.extend(
        [
            "",
            "## Raw Score Meaning",
            "",
            "The LPI is an index, not a probability. The table shows training-fitted observed occurrence probabilities for the issued unsmoothed index. These mappings are provisional because the archive spans only one part of one convective season.",
            "",
            "| Period | Target | LPI 10 | LPI 20 | LPI 40 | LPI 60 | LPI 80 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for period in PERIODS:
        for radius in TARGET_RADII_KM:
            values = {
                float(row["raw_lpi"]): float(row["train_fitted_probability"])
                for row in mapping_rows
                if row["period"] == period and row["target_radius_km"] == radius and row["smoothing_km"] == 0.0
            }
            lines.append(
                f"| {period} | {radius:g} km | {values[10.0]:.3f} | {values[20.0]:.3f} | {values[40.0]:.3f} | {values[60.0]:.3f} | {values[80.0]:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Spatial Detail",
            "",
            "A lower neighboring-cell difference and fewer disconnected LPI>=20 regions indicate a less granular field on held-out forecasts.",
            "",
            "| Smoothing | Mean neighboring-cell difference | LPI>=20 area fraction | LPI>=20 components |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in detail_rows:
        lines.append(
            f"| {float(row['smoothing_km']):g} km | {float(row['mean_abs_neighbor_difference']):.3f} | "
            f"{float(row['area_fraction_lpi_ge_20']):.3f} | {float(row['components_lpi_ge_20']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Smoothing is selected on validation Brier score subject to retaining AUC within 0.02 of the issued field. Test scores and day-block bootstrap intervals are then reported without reselection. A visually smoother field is not enough by itself; the selected scale should improve or at least preserve held-out corridor skill at both the 20 and 30 km targets.",
            "",
            "A monotone exponent applied only to the displayed 0-100 values cannot improve discrimination and is redundant once a monotone probability calibration is fitted. The defensible score adjustment is therefore a probability/category remapping learned from more data and the meteorologist labels, not an arbitrary cosmetic exponent.",
            "",
            "## Recommendation",
            "",
            "Apply 10 km Gaussian smoothing after the three-hour temporal maximum for the operational 20-30 km BCH corridor use case. The current single-panel LPI has no post-window smoothing, so this is a 0-to-10 km change. The two-panel LPI contours currently use 5 km post-window smoothing, so this is a 5-to-10 km change. Keep the ingredient formula unchanged for now; this recommendation changes spatial presentation and risk scale, not the underlying instability/moisture/trigger weights.",
            "",
            "Do not interpret the displayed 0-100 values as percentages. Keep the LPI name during the next data-collection period, then replace the numeric display with separately calibrated 3-hour and 24-hour probabilities or low/moderate/high categories after the BCH meteorologist labels are available. The same raw value has different observed meaning at the two periods, so one global exponent or lookup table would be incorrect.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive-root", type=Path, default=archive.DEFAULT_ARCHIVE_ROOT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
