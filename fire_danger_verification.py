#!/usr/bin/env python3
"""Archive station forecasts and render BC fire-danger verification summaries."""

from __future__ import annotations

import datetime as dt
import json
import os
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

import fire_danger_peak
import plot_style


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONCRETE_DATA_ROOT = Path(
    os.environ.get("FIRE_DANGER_BCWS_MIRROR_ROOT", PROJECT_ROOT / "data" / "fire_danger_bcws")
)
DEFAULT_ARCHIVE_ROOT = Path(
    os.environ.get(
        "FIRE_DANGER_VERIFICATION_ROOT",
        PROJECT_ROOT / "data" / "fire_danger_verification",
    )
)
CLASS_NAMES = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Extreme"}
MODEL_LABELS = {"continental": "HRDPS 2.5 km", "west": "HRDPS-West 1 km", "bcws": "BCWS Forecast"}
GRID_TO_BC_ALBERS = Transformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)


def _station_catalog_path(concrete_data_root: Path) -> Path:
    return concrete_data_root / "observations" / "bcws" / "datamart" / "stations" / "current_stations.csv"


def load_bcws_stations(concrete_data_root: Path = DEFAULT_CONCRETE_DATA_ROOT) -> pd.DataFrame:
    path = _station_catalog_path(concrete_data_root)
    frame = pd.read_csv(path, dtype={"STATION_CODE": "string"})
    frame["STATION_CODE"] = frame["STATION_CODE"].astype("string")
    frame["LATITUDE"] = pd.to_numeric(frame["LATITUDE"], errors="coerce")
    frame["LONGITUDE"] = pd.to_numeric(frame["LONGITUDE"], errors="coerce")
    return frame.dropna(subset=["STATION_CODE", "LATITUDE", "LONGITUDE"]).drop_duplicates("STATION_CODE")


def station_grid_mapping(
    model_key: str,
    lat: np.ndarray,
    lon: np.ndarray,
    cache_dir: Path,
    concrete_data_root: Path = DEFAULT_CONCRETE_DATA_ROOT,
) -> pd.DataFrame:
    path = cache_dir / "peak_daily" / model_key / "station_grid_mapping.csv"
    stations_path = _station_catalog_path(concrete_data_root)
    if path.exists() and path.stat().st_mtime_ns >= stations_path.stat().st_mtime_ns:
        mapping = pd.read_csv(path, dtype={"station_code": "string"})
        cached_shape = None
        if not mapping.empty and {"grid_shape_y", "grid_shape_x"}.issubset(mapping.columns):
            cached_shape = (
                int(mapping["grid_shape_y"].iloc[0]),
                int(mapping["grid_shape_x"].iloc[0]),
            )
        if cached_shape == lat.shape:
            return mapping

    stations = load_bcws_stations(concrete_data_root)
    grid_x, grid_y = GRID_TO_BC_ALBERS.transform(lon.ravel(), lat.ravel())
    valid = np.isfinite(grid_x) & np.isfinite(grid_y)
    flat_indices = np.flatnonzero(valid)
    tree = cKDTree(np.column_stack((grid_x[valid], grid_y[valid])))
    station_x, station_y = GRID_TO_BC_ALBERS.transform(
        stations["LONGITUDE"].to_numpy(),
        stations["LATITUDE"].to_numpy(),
    )
    distance_m, nearest = tree.query(np.column_stack((station_x, station_y)), k=1)
    selected = flat_indices[nearest]
    rows, cols = np.unravel_index(selected, lat.shape)
    mapping = pd.DataFrame(
        {
            "station_code": stations["STATION_CODE"].astype("string"),
            "station_name": stations["STATION_NAME"].astype("string"),
            "station_latitude": stations["LATITUDE"],
            "station_longitude": stations["LONGITUDE"],
            "grid_row": rows,
            "grid_col": cols,
            "grid_distance_km": distance_m / 1000.0,
            "grid_shape_y": lat.shape[0],
            "grid_shape_x": lat.shape[1],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(path, index=False)
    return mapping


def archive_station_peak_forecast(
    run_stamp: str,
    run_init_utc: dt.datetime,
    model_key: str,
    day: fire_danger_peak.PeakBurnDay,
    danger: np.ndarray,
    mapping: pd.DataFrame,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> Path:
    rows = mapping["grid_row"].to_numpy(dtype=np.int64)
    cols = mapping["grid_col"].to_numpy(dtype=np.int64)
    target_valid = dt.datetime.combine(day.fire_date, dt.time(20), tzinfo=dt.timezone.utc)
    output = mapping[
        ["station_code", "station_name", "station_latitude", "station_longitude", "grid_distance_km"]
    ].copy()
    output["model_key"] = model_key
    output["model_label"] = MODEL_LABELS.get(model_key, model_key)
    output["run_stamp"] = run_stamp
    output["run_init_utc"] = run_init_utc.isoformat()
    output["fire_date"] = day.fire_date.isoformat()
    output["target_valid_utc"] = target_valid.isoformat()
    output["lead_hours"] = (target_valid - run_init_utc).total_seconds() / 3600.0
    output["forecast_fwi"] = day.fwi[rows, cols]
    output["forecast_bui"] = day.bui[rows, cols]
    output["forecast_danger_class"] = danger[rows, cols]
    output["peak_local_hour"] = day.peak_local_hour[rows, cols]
    output = output.loc[
        np.isfinite(output["forecast_danger_class"])
        & output["grid_distance_km"].le(15.0)
        & output["lead_hours"].ge(0.0)
    ].copy()
    path = archive_root / "forecasts" / model_key / run_stamp / f"{day.fire_date:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return path


def load_observed_daily(
    dates: Iterable[dt.date],
    concrete_data_root: Path = DEFAULT_CONCRETE_DATA_ROOT,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    daily_root = concrete_data_root / "observations" / "bcws" / "datamart" / "daily"
    for day in sorted(set(dates)):
        path = daily_root / f"{day:%Y}" / f"{day:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"STATION_CODE": "string"}, low_memory=False)
        timestamp = frame["DATE_TIME"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
        frame = frame.loc[timestamp.str.endswith("12")].copy()
        frame["observed_danger_class"] = pd.to_numeric(frame.get("DANGER_RATING"), errors="coerce")
        frame["observed_fwi"] = pd.to_numeric(frame.get("FIRE_WEATHER_INDEX"), errors="coerce")
        frame["observed_bui"] = pd.to_numeric(frame.get("BUILDUP_INDEX"), errors="coerce")
        frame["fire_date"] = day.isoformat()
        frame = frame.rename(columns={"STATION_CODE": "station_code", "STATION_NAME": "station_name_observed"})
        frames.append(
            frame[
                ["station_code", "station_name_observed", "fire_date", "observed_danger_class", "observed_fwi", "observed_bui"]
            ].dropna(subset=["observed_danger_class"])
        )
    if not frames:
        return pd.DataFrame(
            columns=["station_code", "station_name_observed", "fire_date", "observed_danger_class", "observed_fwi", "observed_bui"]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(["station_code", "fire_date"], keep="last")


def load_gridded_forecasts(archive_root: Path = DEFAULT_ARCHIVE_ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((archive_root / "forecasts").glob("*/*/*.csv")):
        try:
            frame = pd.read_csv(path, dtype={"station_code": "string"})
        except (OSError, pd.errors.ParserError):
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_bcws_forecast_snapshots(
    concrete_data_root: Path = DEFAULT_CONCRETE_DATA_ROOT,
) -> pd.DataFrame:
    root = concrete_data_root / "observations" / "bcws" / "danger_summaries" / "snapshots"
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        snapshot_time = payload.get("snapshot_time_utc")
        response = payload.get("response", payload)
        for station in response.get("collection", []):
            for item in station.get("summaryData", []):
                if str(item.get("recordType", "")).upper() not in {"FORECAST", "ESTIMATE", "MANUAL"}:
                    continue
                timestamp = str(item.get("dailyTimestamp", ""))
                if len(timestamp) < 8:
                    continue
                rows.append(
                    {
                        "station_code": str(station.get("stationCode")),
                        "station_name": station.get("stationName"),
                        "model_key": "bcws",
                        "model_label": MODEL_LABELS["bcws"],
                        "run_stamp": snapshot_time,
                        "run_init_utc": snapshot_time,
                        "fire_date": f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
                        "forecast_danger_class": item.get("dangerClass"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(rows)
    frame["run_init_utc"] = pd.to_datetime(frame["run_init_utc"], utc=True, errors="coerce")
    target = pd.to_datetime(frame["fire_date"], utc=True, errors="coerce") + pd.Timedelta(hours=20)
    frame["lead_hours"] = (target - frame["run_init_utc"]).dt.total_seconds() / 3600.0
    frame["forecast_danger_class"] = pd.to_numeric(frame["forecast_danger_class"], errors="coerce")
    return frame.loc[frame["lead_hours"].ge(0.0)].drop_duplicates(
        ["station_code", "fire_date", "run_init_utc"], keep="last"
    )


def matched_verification_frame(
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    concrete_data_root: Path = DEFAULT_CONCRETE_DATA_ROOT,
) -> pd.DataFrame:
    gridded = load_gridded_forecasts(archive_root)
    bcws = load_bcws_forecast_snapshots(concrete_data_root)
    forecasts = pd.concat([frame for frame in (gridded, bcws) if not frame.empty], ignore_index=True) if not gridded.empty or not bcws.empty else pd.DataFrame()
    if forecasts.empty:
        return forecasts
    forecasts["station_code"] = forecasts["station_code"].astype("string")
    forecasts["forecast_danger_class"] = pd.to_numeric(forecasts["forecast_danger_class"], errors="coerce")
    dates = [dt.date.fromisoformat(value) for value in forecasts["fire_date"].dropna().unique()]
    observations = load_observed_daily(dates, concrete_data_root)
    matched = forecasts.merge(observations, on=["station_code", "fire_date"], how="inner")
    matched = matched.dropna(subset=["forecast_danger_class", "observed_danger_class", "lead_hours"])
    matched["error"] = matched["forecast_danger_class"] - matched["observed_danger_class"]
    matched["absolute_error"] = matched["error"].abs()
    matched["lead_group"] = pd.cut(
        matched["lead_hours"],
        bins=[-0.001, 18.0, 42.0, 72.0],
        labels=["Day 0", "Day 1", "Day 2"],
    ).astype("string")
    matched = matched.dropna(subset=["lead_group"])
    archive_root.mkdir(parents=True, exist_ok=True)
    matched.to_csv(archive_root / "matched_verification.csv", index=False)
    return matched


def _metric_row(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "n": len(frame),
        "exact": 100.0 * frame["absolute_error"].eq(0).mean() if len(frame) else np.nan,
        "within_one": 100.0 * frame["absolute_error"].le(1).mean() if len(frame) else np.nan,
        "mae": frame["absolute_error"].mean() if len(frame) else np.nan,
        "bias": frame["error"].mean() if len(frame) else np.nan,
    }


def _draw_metric_table(ax: plt.Axes, title: str, rows: list[tuple[str, dict[str, object]]]) -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=9)
    headers = ["Forecast", "N", "Exact", "Within 1", "MAE", "Bias"]
    data = []
    for label, metrics in rows:
        data.append(
            [
                label,
                f"{metrics['n']:,}",
                "--" if not np.isfinite(metrics["exact"]) else f"{metrics['exact']:.0f}%",
                "--" if not np.isfinite(metrics["within_one"]) else f"{metrics['within_one']:.0f}%",
                "--" if not np.isfinite(metrics["mae"]) else f"{metrics['mae']:.2f}",
                "--" if not np.isfinite(metrics["bias"]) else f"{metrics['bias']:+.2f}",
            ]
        )
    if not data:
        ax.text(0.01, 0.78, "No matched forecasts are available yet.", fontsize=12, color="#59656c", va="top")
        return
    table = ax.table(cellText=data, colLabels=headers, loc="upper left", cellLoc="right", colLoc="right", bbox=[0, 0.08, 1, 0.82])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d1d5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#074f70")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2:
            cell.set_facecolor("#eef3f5")
        if col == 0:
            cell.get_text().set_ha("left")


def render_dashboard(
    out_path: Path,
    matched: pd.DataFrame,
    as_of: dt.date | None = None,
) -> Path:
    as_of = as_of or dt.datetime.now(plot_style.LOCAL_TZ).date()
    previous_day = as_of - dt.timedelta(days=1)
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="#f3f6f7")
    grid = fig.add_gridspec(2, 2, left=0.045, right=0.975, bottom=0.075, top=0.86, hspace=0.27, wspace=0.18)
    fig.text(0.045, 0.945, "Experimental BC Fire Danger Verification", fontsize=22, fontweight="bold", color="#16262d")
    fig.text(
        0.045,
        0.902,
        f"Previous day: {previous_day:%d %b %Y}  |  Running window: latest 30 verified days  |  Updated {dt.datetime.now(plot_style.LOCAL_TZ):%d %b %Y %H:%M %Z}",
        fontsize=11.5,
        color="#526069",
    )

    previous = matched.loc[matched.get("fire_date", pd.Series(dtype=str)).eq(previous_day.isoformat())].copy() if not matched.empty else matched
    previous_rows: list[tuple[str, dict[str, object]]] = []
    if not previous.empty:
        latest = previous.sort_values("lead_hours").drop_duplicates(["model_key", "station_code"], keep="first")
        for model_key, frame in latest.groupby("model_key", sort=False):
            previous_rows.append((MODEL_LABELS.get(str(model_key), str(model_key)), _metric_row(frame)))
    _draw_metric_table(fig.add_subplot(grid[0, 0]), "Previous-Day Verification", previous_rows)

    cutoff = as_of - dt.timedelta(days=30)
    running = matched.loc[pd.to_datetime(matched.get("fire_date"), errors="coerce").dt.date.ge(cutoff)].copy() if not matched.empty else matched
    running_rows: list[tuple[str, dict[str, object]]] = []
    if not running.empty:
        for (model_key, lead_group), frame in running.groupby(["model_key", "lead_group"], sort=False):
            running_rows.append((f"{MODEL_LABELS.get(str(model_key), str(model_key))} {lead_group}", _metric_row(frame)))
    _draw_metric_table(fig.add_subplot(grid[0, 1]), "Running Verification", running_rows)

    ax_confusion = fig.add_subplot(grid[1, 0])
    ax_confusion.set_title("Previous-Day Confusion Matrix", loc="left", fontsize=14, fontweight="bold", pad=9)
    primary = previous.loc[previous.get("model_key", pd.Series(dtype=str)).eq("continental")].copy() if not previous.empty else previous
    if primary.empty:
        ax_confusion.set_axis_off()
        ax_confusion.text(0.01, 0.88, "Awaiting matched HRDPS 2.5 km and BCWS station classes.", fontsize=12, color="#59656c", va="top")
    else:
        primary = primary.sort_values("lead_hours").drop_duplicates("station_code", keep="first")
        matrix = np.zeros((5, 5), dtype=int)
        for observed, forecast in zip(primary["observed_danger_class"], primary["forecast_danger_class"]):
            matrix[int(observed) - 1, int(forecast) - 1] += 1
        image = ax_confusion.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for row in range(5):
            for col in range(5):
                ax_confusion.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=10, color="#152126")
        short_names = ["VLow", "Low", "Mod", "High", "Ext"]
        ax_confusion.set_xticks(range(5), short_names)
        ax_confusion.set_yticks(range(5), short_names)
        ax_confusion.set_xlabel("Forecast class")
        ax_confusion.set_ylabel("Observed class")
        fig.colorbar(image, ax=ax_confusion, fraction=0.046, pad=0.04, label="Stations")

    ax_coverage = fig.add_subplot(grid[1, 1])
    ax_coverage.set_title("Verification Coverage", loc="left", fontsize=14, fontweight="bold", pad=9)
    ax_coverage.set_axis_off()
    if matched.empty:
        lines = [
            "Forecast sampling is configured and waiting for the first complete 05:00-05:00 fire day.",
            "BCWS observations and forecast danger-summary snapshots will be matched automatically.",
        ]
    else:
        verified_dates = pd.to_datetime(matched["fire_date"], errors="coerce").dropna()
        lines = [
            f"Verified dates: {verified_dates.min():%d %b %Y} to {verified_dates.max():%d %b %Y}",
            f"Matched station forecasts: {len(matched):,}",
            f"Unique stations: {matched['station_code'].nunique():,}",
            "Exact and within-one-class scores treat classes as an ordered 1-5 scale.",
            "Observed target: BCWS noon-PST daily DANGER_RATING.",
        ]
    y = 0.88
    for line in lines:
        wrapped = textwrap.fill(line, width=72)
        ax_coverage.text(0.01, y, wrapped, fontsize=11.5, color="#26363d", va="top", linespacing=1.35)
        y -= 0.13 * (wrapped.count("\n") + 1)

    fig.text(
        0.045,
        0.025,
        "Experimental guidance: FWI2025 hourly evolution + NRCan peak-burn selection + BC Schedule 2 matrices. Not an official BCWS danger rating.",
        fontsize=10.5,
        color="#4d5b62",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
