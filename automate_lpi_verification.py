#!/usr/bin/env python3
"""Mirror ECCC lightning density and render LPI verification graphics."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from shapely.geometry.base import BaseGeometry

import make_hrdps_west_convective as hrdps
import make_hrdps_west_lightning as lightning
import lightning_ml_archive as ml_archive
import plot_style
from publish_hrdps_west import DEFAULT_PAGES_REPO, publish, write_manifest

LIGHTNING_OBS_URL = "https://dd.weather.gc.ca/today/lightning"
OBS_RE = re.compile(r"(?P<stamp>\d{8}T\d{4}Z)_MSC_Lightning_2\.5km\.tif$")
LPI_CACHE_RE = re.compile(r"_f(?P<fhour>\d{3})_lpi\.npz$")
LOCK_PATH = Path("logs/lpi_verification.lock")
PUBLISH_LOCK = Path("logs/hrdps_publish.lock")
STATUS_PATH = Path("logs/state/lpi_verification.status.json")
DEFAULT_OBS_DIR = Path("data/lightning_obs")
DEFAULT_WEST_CACHE_DIR = Path("plots/hrdps_west_lightning")
DEFAULT_CONTINENTAL_CACHE_DIR = Path("plots/hrdps_continental_lightning")
DEFAULT_WEST_OUTPUT_DIR = Path("plots/hrdps_west_lightning_verif")
DEFAULT_CONTINENTAL_OUTPUT_DIR = Path("plots/hrdps_continental_lightning_verif")
OBS_LOW_FLASH_KM2 = 0.05
OBS_MED_FLASH_KM2 = 0.50
OBS_HIGH_FLASH_KM2 = 2.00
OBS_CONTOUR_COLORS = ("#a97700", "#e06400", "#c41424")
OBS_CONTOUR_WIDTHS = (0.95, 1.30, 1.75)
DAILY_VERIFICATION_KEEP_DAYS = 60
LPI_TUNING_MIN_ARCHIVE_DAYS = 21
LPI_TUNING_READY_MARKER = Path("logs/state/lpi_tuning_ready.notified")


@dataclass(frozen=True)
class LpiCache:
    path: Path
    cache_version: int
    formula_version: str
    model_key: str
    model_label: str
    source_label: str
    run: hrdps.RunInfo
    fhour: int
    lat: np.ndarray
    lon: np.ndarray
    potential: np.ndarray


@dataclass(frozen=True)
class DailyLpiWindow:
    run: hrdps.RunInfo
    start: dt.datetime
    end: dt.datetime
    included_hours: tuple[int, ...]
    end_fhour: int


@dataclass(frozen=True)
class DailyLpiForecast:
    formula_version: str
    model_key: str
    model_label: str
    source_label: str
    run: hrdps.RunInfo
    start: dt.datetime
    end: dt.datetime
    end_fhour: int
    lat: np.ndarray
    lon: np.ndarray
    potential: np.ndarray


@dataclass(frozen=True)
class ObsGrid:
    lat: np.ndarray
    lon: np.ndarray
    flash_km2: np.ndarray


def log(message: str) -> None:
    print(message, flush=True)


def write_status(status: str, **metadata: object) -> None:
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        **metadata,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(STATUS_PATH)


def lpi_tuning_readiness(
    archive_root: Path,
    minimum_days: int = LPI_TUNING_MIN_ARCHIVE_DAYS,
) -> dict[str, object]:
    archive_dir = ml_archive.observation_archive_dir(archive_root)
    timestamps: list[dt.datetime] = []
    active_blocks = 0
    for path in sorted(archive_dir.glob("*/*/*.tif")):
        try:
            timestamp = dt.datetime.strptime(path.stem.split("_MSC_")[0], "%Y%m%dT%H%MZ").replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            continue
        timestamps.append(timestamp)
        sidecar = path.with_suffix(".json")
        try:
            active_blocks += int(json.loads(sidecar.read_text()).get("nonzero_cells", 0) > 0)
        except (OSError, json.JSONDecodeError):
            pass

    if not timestamps:
        return {
            "ready": False,
            "minimum_days": minimum_days,
            "archive_span_days": 0.0,
            "coverage_fraction": 0.0,
            "observation_blocks": 0,
            "active_observation_blocks": 0,
        }

    unique_timestamps = sorted(set(timestamps))
    span = unique_timestamps[-1] - unique_timestamps[0]
    expected_blocks = int(span.total_seconds() // (ml_archive.OBS_BLOCK_HOURS * 3600)) + 1
    coverage = len(unique_timestamps) / expected_blocks if expected_blocks else 0.0
    span_days = span.total_seconds() / 86400.0
    return {
        "ready": span_days >= minimum_days and coverage >= 0.95 and active_blocks >= 12,
        "minimum_days": minimum_days,
        "archive_span_days": round(span_days, 3),
        "coverage_fraction": round(coverage, 4),
        "observation_blocks": len(unique_timestamps),
        "active_observation_blocks": active_blocks,
        "first_block_utc": ml_archive.utc_iso(unique_timestamps[0]),
        "latest_block_utc": ml_archive.utc_iso(unique_timestamps[-1]),
    }


def notify_lpi_tuning_ready(readiness: dict[str, object]) -> bool:
    if not readiness.get("ready") or LPI_TUNING_READY_MARKER.exists():
        return False
    message = (
        "At least 21 days of lightning observations are archived. "
        "Continue the LPI tuning and regional fire-start forecast project."
    )
    result = subprocess.run(
        [
            "/usr/bin/osascript",
            "-e",
            f'display notification "{message}" with title "Forecast Graphics Pilot"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"Could not send LPI tuning readiness notification: {result.stderr.strip()}")
        return False
    LPI_TUNING_READY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    LPI_TUNING_READY_MARKER.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    log("Sent the one-time LPI tuning readiness notification.")
    return True


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-dir", type=Path, default=DEFAULT_OBS_DIR)
    parser.add_argument("--west-cache-dir", type=Path, default=DEFAULT_WEST_CACHE_DIR)
    parser.add_argument("--continental-cache-dir", type=Path, default=DEFAULT_CONTINENTAL_CACHE_DIR)
    parser.add_argument("--west-output-dir", type=Path, default=DEFAULT_WEST_OUTPUT_DIR)
    parser.add_argument("--continental-output-dir", type=Path, default=DEFAULT_CONTINENTAL_OUTPUT_DIR)
    parser.add_argument("--pages-repo", type=Path, default=DEFAULT_PAGES_REPO)
    parser.add_argument("--ml-archive-root", type=Path, default=ml_archive.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--keep-days", type=int, default=DAILY_VERIFICATION_KEEP_DAYS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", choices=["all", "west", "continental"], default="all")
    parser.add_argument("--stamps", default=None, help="Comma-separated run stamps to verify.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mirror-only", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--keep-raw-observations", action="store_true")
    return parser.parse_args(list(argv))


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"Another LPI verification job is already running: {path}")
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def publish_lock():
    PUBLISH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PUBLISH_LOCK.open("w") as handle:
        log(f"Waiting for publish lock: {PUBLISH_LOCK}")
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_obs_time(name: str) -> dt.datetime | None:
    match = OBS_RE.match(name)
    if not match:
        return None
    return dt.datetime.strptime(match.group("stamp"), "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc)


def obs_path_for_time(obs_dir: Path, timestamp: dt.datetime) -> Path:
    return obs_dir / f"{timestamp:%Y%m%d}" / f"{timestamp:%Y%m%dT%H%MZ}_MSC_Lightning_2.5km.tif"


def expected_obs_times(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Observation window times must be timezone aware.")
    start = start.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)
    end = end.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)
    minutes_to_next = 10 - (start.minute % 10)
    first = start + dt.timedelta(minutes=minutes_to_next)
    if first <= start:
        first += dt.timedelta(minutes=10)
    out: list[dt.datetime] = []
    timestamp = first
    while timestamp <= end:
        out.append(timestamp)
        timestamp += dt.timedelta(minutes=10)
    return out


def fetch_obs_listing() -> list[str]:
    response = requests.get(f"{LIGHTNING_OBS_URL}/", timeout=30)
    response.raise_for_status()
    return [name for name in hrdps.parse_links(response.text) if OBS_RE.match(name)]


def mirror_lightning_observations(obs_dir: Path, workers: int, archive_root: Path | None = None) -> int:
    names = fetch_obs_listing()
    jobs: list[tuple[str, Path]] = []
    for name in names:
        timestamp = parse_obs_time(name)
        if timestamp is None:
            continue
        if archive_root is not None and ml_archive.observation_block_archived(archive_root, timestamp):
            continue
        dest = obs_dir / f"{timestamp:%Y%m%d}" / name
        if not dest.exists() or dest.stat().st_size == 0:
            jobs.append((f"{LIGHTNING_OBS_URL}/{name}", dest))

    if not jobs:
        log("No new ECCC lightning observation files to mirror.")
        return 0

    log(f"Mirroring {len(jobs)} ECCC lightning observation file(s).")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(hrdps.download_one, url, dest) for url, dest in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                log(f"  observation files ready: {completed}/{len(futures)}")
    return len(jobs)


def prune_obs(obs_dir: Path, keep_days: int) -> None:
    if not obs_dir.exists():
        return
    cutoff_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)).date()
    for child in obs_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            child_date = dt.datetime.strptime(child.name, "%Y%m%d").date()
        except ValueError:
            continue
        if child_date < cutoff_date:
            log(f"Removing old lightning observation mirror: {child}")
            shutil.rmtree(child)


def prune_local_plots(output_dir: Path, keep_days: int) -> None:
    if not output_dir.exists():
        return
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            init_time = hrdps.parse_stamp(child.name)
        except ValueError:
            continue
        if init_time < cutoff:
            log(f"Removing old LPI verification plots: {child}")
            shutil.rmtree(child)


def npz_string(npz, key: str, fallback: str = "") -> str:
    if key not in npz:
        return fallback
    value = npz[key]
    return str(value.item() if hasattr(value, "item") else value)


def load_lpi_cache(path: Path) -> LpiCache:
    with np.load(path) as cached:
        cache_version = int(cached["version"][0]) if "version" in cached else 1
        formula_version = npz_string(cached, "formula_version", "legacy_v1")
        model_key = npz_string(cached, "model_key", "west")
        model_label = npz_string(cached, "model_label", "HRDPS-West 1 km")
        source_label = npz_string(cached, "source_label", "ECCC HRDPS")
        stamp = npz_string(cached, "run_stamp")
        init_iso = npz_string(cached, "init_iso")
        init_time = dt.datetime.fromisoformat(init_iso.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        fhour = int(cached["fhour"][0])
        run = hrdps.RunInfo(cycle=f"{init_time:%H}", stamp=stamp, init_time=init_time)
        return LpiCache(
            path=path,
            cache_version=cache_version,
            formula_version=formula_version,
            model_key=model_key,
            model_label=model_label,
            source_label=source_label,
            run=run,
            fhour=fhour,
            lat=cached["lat"].astype(np.float32),
            lon=cached["lon"].astype(np.float32),
            potential=cached["potential"].astype(np.float32),
        )


def find_lpi_cache_groups(args: argparse.Namespace) -> dict[tuple[str, str], dict[int, Path]]:
    roots: list[tuple[str, Path]] = []
    if args.model in {"all", "west"}:
        roots.append(("west", args.west_cache_dir))
    if args.model in {"all", "continental"}:
        roots.append(("continental", args.continental_cache_dir))

    stamps = {item.strip() for item in args.stamps.split(",")} if args.stamps else None
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.keep_days)
    groups: dict[tuple[str, str], dict[int, Path]] = {}
    for model_key, root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/lpi_cache/*_lpi.npz"):
            run_stamp = path.parent.parent.name
            if stamps and run_stamp not in stamps:
                continue
            try:
                if hrdps.parse_stamp(run_stamp) < cutoff:
                    continue
            except ValueError:
                continue
            match = LPI_CACHE_RE.search(path.name)
            if not match:
                continue
            fhour = int(match.group("fhour"))
            groups.setdefault((model_key, run_stamp), {})[fhour] = path
    return groups


def geotiff_lon_lat(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    tags = image.tag_v2
    tiepoint = tags.get(33922)
    scale = tags.get(33550)
    if not tiepoint or not scale:
        raise RuntimeError("Lightning GeoTIFF is missing tiepoint/pixel-scale metadata.")
    width, height = image.size
    x0 = float(tiepoint[3])
    y0 = float(tiepoint[4])
    dx = float(scale[0])
    dy = float(scale[1])
    lon = x0 + (np.arange(width, dtype=np.float32) + 0.5) * dx
    lat = y0 - (np.arange(height, dtype=np.float32) + 0.5) * dy
    return lon, lat


def obs_crop_slices(lon: np.ndarray, lat: np.ndarray, extent: tuple[float, float, float, float]) -> tuple[slice, slice]:
    west, east, south, north = extent
    xidx = np.where((lon >= west) & (lon <= east))[0]
    yidx = np.where((lat >= south) & (lat <= north))[0]
    if not len(xidx) or not len(yidx):
        raise RuntimeError("Lightning observations do not overlap plot extent.")
    return slice(max(int(yidx.min()) - 1, 0), min(int(yidx.max()) + 2, len(lat))), slice(
        max(int(xidx.min()) - 1, 0), min(int(xidx.max()) + 2, len(lon))
    )


def read_aggregated_obs_window(
    archive_root: Path,
    start: dt.datetime,
    end: dt.datetime,
    extent: tuple[float, float, float, float],
) -> ObsGrid | None:
    paths = ml_archive.expected_observation_aggregate_paths(archive_root, start, end)
    if not paths or any(not path.exists() or path.stat().st_size == 0 for path in paths):
        return None

    accum: np.ndarray | None = None
    out_lon: np.ndarray | None = None
    out_lat: np.ndarray | None = None
    yslice: slice | None = None
    xslice: slice | None = None
    for path in paths:
        with Image.open(path) as image:
            if yslice is None or xslice is None:
                lon, lat = geotiff_lon_lat(image)
                yslice, xslice = obs_crop_slices(lon, lat, extent)
                out_lon = lon[xslice]
                out_lat = lat[yslice]
                accum = np.zeros((len(out_lat), len(out_lon)), dtype=np.float32)
            nodata = float(image.tag_v2.get(42113, -999.0))
            data = np.asarray(image, dtype=np.float32)[yslice, xslice]
            data = np.where(np.isfinite(data) & (data != nodata) & (data > 0.0), data, 0.0)
            accum += data

    if accum is None or out_lon is None or out_lat is None:
        return None
    lon2d, lat2d = np.meshgrid(out_lon, out_lat)
    return ObsGrid(lat=lat2d.astype(np.float32), lon=lon2d.astype(np.float32), flash_km2=accum)


def read_obs_window(
    obs_dir: Path,
    start: dt.datetime,
    end: dt.datetime,
    extent: tuple[float, float, float, float],
    archive_root: Path | None = None,
) -> ObsGrid | None:
    if archive_root is not None:
        archived = read_aggregated_obs_window(archive_root, start, end, extent)
        if archived is not None:
            return archived
    timestamps = expected_obs_times(start, end)
    paths = [obs_path_for_time(obs_dir, timestamp) for timestamp in timestamps]
    missing = [path for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        return None

    accum: np.ndarray | None = None
    out_lon: np.ndarray | None = None
    out_lat: np.ndarray | None = None
    yslice: slice | None = None
    xslice: slice | None = None
    for path in paths:
        with Image.open(path) as image:
            if yslice is None or xslice is None:
                lon, lat = geotiff_lon_lat(image)
                yslice, xslice = obs_crop_slices(lon, lat, extent)
                out_lon = lon[xslice]
                out_lat = lat[yslice]
                accum = np.zeros((len(out_lat), len(out_lon)), dtype=np.float32)
            nodata = float(image.tag_v2.get(42113, -999.0))
            data = np.asarray(image, dtype=np.float32)[yslice, xslice]
            data = np.where(np.isfinite(data) & (data != nodata) & (data > 0.0), data, 0.0)
            accum += data * 10.0

    if accum is None or out_lon is None or out_lat is None:
        return None
    lon2d, lat2d = np.meshgrid(out_lon, out_lat)
    return ObsGrid(lat=lat2d.astype(np.float32), lon=lon2d.astype(np.float32), flash_km2=accum)


def output_dir_for_model(args: argparse.Namespace, model_key: str) -> Path:
    if model_key == "continental":
        return args.continental_output_dir
    if model_key == "west":
        return args.west_output_dir
    raise ValueError(f"Unsupported model for LPI verification: {model_key}")


def product_model_for_forecast_model(model_key: str) -> str:
    if model_key == "continental":
        return "continental_verif"
    if model_key == "west":
        return "west_verif"
    raise ValueError(f"Unsupported model for LPI verification: {model_key}")


def output_prefix_for_model(model_key: str) -> str:
    if model_key == "continental":
        return "hrdps_continental_lightning_verif"
    if model_key == "west":
        return "hrdps_west_lightning_verif"
    raise ValueError(f"Unsupported model for LPI verification: {model_key}")


def first_full_12z_window(run: hrdps.RunInfo) -> DailyLpiWindow | None:
    """Return the first complete 12Z-12Z verification window covered by the run.

    LPI frames represent the preceding 3-hour block, so the forecast blocks used
    for a 12Z-12Z verification are the frames valid at 15Z, 18Z, ..., 12Z.
    """
    start = run.init_time.astimezone(dt.timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    if start < run.init_time:
        start += dt.timedelta(days=1)
    end = start + dt.timedelta(days=1)

    included: list[int] = []
    valid = start + dt.timedelta(hours=3)
    while valid <= end:
        fhour = int(round((valid - run.init_time).total_seconds() / 3600.0))
        if fhour not in lightning.FORECAST_HOURS:
            return None
        included.append(fhour)
        valid += dt.timedelta(hours=3)

    if not included:
        return None
    return DailyLpiWindow(run=run, start=start, end=end, included_hours=tuple(included), end_fhour=included[-1])


def aggregate_daily_lpi(paths_by_hour: dict[int, Path], window: DailyLpiWindow) -> DailyLpiForecast | None:
    missing = [fhour for fhour in window.included_hours if fhour not in paths_by_hour]
    if missing:
        return None

    caches = [load_lpi_cache(paths_by_hour[fhour]) for fhour in window.included_hours]
    first = caches[0]
    stack = []
    for cache in caches:
        if cache.formula_version != first.formula_version:
            log(
                f"Skipping mixed LPI formulas in {window.run.stamp}: "
                f"{first.formula_version} and {cache.formula_version}."
            )
            return None
        if cache.lat.shape != first.lat.shape or cache.lon.shape != first.lon.shape:
            raise RuntimeError(f"LPI cache grid shape changed within {window.run.stamp}.")
        stack.append(cache.potential)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        potential = np.nanmax(np.stack(stack), axis=0)

    return DailyLpiForecast(
        formula_version=first.formula_version,
        model_key=first.model_key,
        model_label=first.model_label,
        source_label=first.source_label,
        run=window.run,
        start=window.start,
        end=window.end,
        end_fhour=window.end_fhour,
        lat=first.lat,
        lon=first.lon,
        potential=potential.astype(np.float32),
    )


def verification_output_path(args: argparse.Namespace, forecast: DailyLpiForecast) -> Path:
    return (
        output_dir_for_model(args, forecast.model_key)
        / forecast.run.stamp
        / f"{output_prefix_for_model(forecast.model_key)}_{forecast.run.stamp}_f{forecast.end_fhour:03d}.png"
    )


def prune_stale_verification_frames(output_dir: Path, stamp: str, prefix: str, keep_path: Path) -> None:
    run_dir = output_dir / stamp
    if not run_dir.exists():
        return
    for path in run_dir.glob(f"{prefix}_{stamp}_f*.png"):
        if path != keep_path:
            path.unlink()


def expected_verification_name(model_key: str, stamp: str) -> str | None:
    run = hrdps.RunInfo(cycle=stamp[-3:-1], stamp=stamp, init_time=hrdps.parse_stamp(stamp))
    window = first_full_12z_window(run)
    if window is None:
        return None
    return f"{output_prefix_for_model(model_key)}_{stamp}_f{window.end_fhour:03d}.png"


def prune_superseded_local_frames(args: argparse.Namespace) -> list[Path]:
    removed: list[Path] = []
    for model_key in ("west", "continental"):
        output_dir = output_dir_for_model(args, model_key)
        if not output_dir.exists():
            continue
        prefix = output_prefix_for_model(model_key)
        for run_dir in output_dir.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                expected_name = expected_verification_name(model_key, run_dir.name)
            except ValueError:
                continue
            if expected_name is None:
                continue
            for path in run_dir.glob(f"{prefix}_{run_dir.name}_f*.png"):
                if path.name != expected_name:
                    log(f"Removing superseded 3-hour verification frame: {path}")
                    path.unlink()
                    removed.append(path)
    return removed


def pages_verification_dir(pages_repo: Path, model_key: str, stamp: str) -> Path:
    run_dir = pages_repo / "hrdps-west" / "images" / stamp
    if model_key == "continental":
        return run_dir / "continental" / "lightning_verif"
    return run_dir / "lightning_verif"


def prune_superseded_pages_frames(args: argparse.Namespace) -> list[Path]:
    removed: list[Path] = []
    images_root = args.pages_repo / "hrdps-west" / "images"
    if not images_root.exists():
        return removed
    for run_dir in images_root.iterdir():
        if not run_dir.is_dir():
            continue
        for model_key in ("west", "continental"):
            try:
                expected_name = expected_verification_name(model_key, run_dir.name)
            except ValueError:
                continue
            if expected_name is None:
                continue
            archive_dir = pages_verification_dir(args.pages_repo, model_key, run_dir.name)
            prefix = output_prefix_for_model(model_key)
            for path in archive_dir.glob(f"{prefix}_{run_dir.name}_f*.png"):
                if path.name != expected_name:
                    log(f"Removing superseded published verification frame: {path}")
                    path.unlink()
                    removed.append(path)
    return removed


def add_observed_lightning_contours(ax: plt.Axes, obs: ObsGrid) -> object | None:
    thresholds = (OBS_LOW_FLASH_KM2, OBS_MED_FLASH_KM2, OBS_HIGH_FLASH_KM2)
    finite = obs.flash_km2[np.isfinite(obs.flash_km2)]
    if not finite.size:
        return None
    maximum = float(np.max(finite))
    available = [index for index, threshold in enumerate(thresholds) if maximum >= threshold]
    if not available:
        return None
    contours = ax.contour(
        obs.lon,
        obs.lat,
        obs.flash_km2,
        levels=[thresholds[index] for index in available],
        colors=[OBS_CONTOUR_COLORS[index] for index in available],
        linewidths=[OBS_CONTOUR_WIDTHS[index] for index in available],
        transform=lightning.DATA_CRS,
        zorder=30,
    )
    for collection, width in zip(contours.collections, (OBS_CONTOUR_WIDTHS[index] for index in available)):
        collection.set_path_effects(
            [path_effects.Stroke(linewidth=width + 1.15, foreground="white"), path_effects.Normal()]
        )
    return contours


def daily_window_header(forecast: DailyLpiForecast) -> str:
    start_label = forecast.start.strftime("%d%b%Y %HZ").upper()
    end_label = forecast.end.strftime("%d%b%Y %HZ").upper()
    return f"HRDPS  |  LPI verification {start_label}-{end_label}"


def render_verification(
    forecast: DailyLpiForecast,
    obs: ObsGrid,
    out_path: Path,
    watersheds: list[BaseGeometry],
    transmission_lines: list[BaseGeometry],
) -> None:
    hrdps.set_model(forecast.model_key)
    lightning.set_model(forecast.model_key)
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    ax = fig.add_axes(plot_style.SINGLE_PANEL_AX_POS, projection=lightning.PLOT_CRS)
    hrdps.add_base_features(ax)

    cmap, norm, levels = lightning.lightning_cmap()
    shaded = ax.contourf(
        forecast.lon,
        forecast.lat,
        forecast.potential,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=lightning.DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    hrdps.add_hydro_features(ax)
    lightning.add_transmission_lines(ax, transmission_lines)
    add_observed_lightning_contours(ax, obs)
    hrdps.add_watersheds(ax, watersheds)
    hrdps.add_city_labels(ax, fontsize=7.1, marker_size=2.2, path_width=2.35, zorder=40)

    plot_style.add_internal_colorbar(fig, ax, shaded, ticks=levels, label="BC-LPI", fmt="%g")
    footer = (
        "max LPI(shaded); observed 12Z-12Z lightning density contours: "
        f"gold={OBS_LOW_FLASH_KM2:g}, orange={OBS_MED_FLASH_KM2:g}, red={OBS_HIGH_FLASH_KM2:g} flash km$^{{-2}}$; grey: BC transmission"
    )
    plot_style.add_single_panel_text(
        ax,
        daily_window_header(forecast),
        footer,
        forecast.run,
        source_label=f"{forecast.source_label} + ECCC lightning density",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_ready_verifications(args: argparse.Namespace) -> list[Path]:
    generated: list[Path] = []
    watershed_cache: dict[str, list[BaseGeometry]] = {}
    transmission_cache: dict[str, list[BaseGeometry]] = {}
    for (model_key, stamp), paths_by_hour in sorted(find_lpi_cache_groups(args).items()):
        if model_key not in {"west", "continental"}:
            continue
        run = hrdps.RunInfo(cycle=stamp[-3:-1], stamp=stamp, init_time=hrdps.parse_stamp(stamp))
        window = first_full_12z_window(run)
        if window is None:
            continue

        forecast = aggregate_daily_lpi(paths_by_hour, window)
        if forecast is None:
            continue

        out_path = verification_output_path(args, forecast)
        if out_path.exists() and not args.force:
            continue

        hrdps.set_model(forecast.model_key)
        obs = read_obs_window(
            args.obs_dir,
            forecast.start,
            forecast.end,
            hrdps.model_config().extent,
            archive_root=args.ml_archive_root,
        )
        if obs is None:
            continue

        if forecast.model_key not in watershed_cache:
            watershed_cache[forecast.model_key] = hrdps.load_watersheds(hrdps.WATERSHED_CACHE)
        if forecast.model_key not in transmission_cache:
            transmission_cache[forecast.model_key] = lightning.load_transmission_lines()
        log(
            f"Rendering 12Z-12Z LPI verification {forecast.model_label} {forecast.run.stamp} "
            f"F{forecast.end_fhour:03d} using {','.join(f'F{hour:03d}' for hour in window.included_hours)}."
        )
        prune_stale_verification_frames(
            output_dir_for_model(args, forecast.model_key),
            forecast.run.stamp,
            output_prefix_for_model(forecast.model_key),
            out_path,
        )
        render_verification(
            forecast,
            obs,
            out_path,
            watershed_cache[forecast.model_key],
            transmission_cache[forecast.model_key],
        )
        log(f"  wrote {out_path}")
        generated.append(out_path)
    return generated


def verification_queue_summary(args: argparse.Namespace) -> dict[str, object]:
    candidate_windows = 0
    completed_windows = 0
    waiting_for_cache = 0
    waiting_for_observations = 0
    ready_to_render = 0
    for (model_key, stamp), paths_by_hour in find_lpi_cache_groups(args).items():
        run = hrdps.RunInfo(cycle=stamp[-3:-1], stamp=stamp, init_time=hrdps.parse_stamp(stamp))
        window = first_full_12z_window(run)
        if window is None:
            continue
        candidate_windows += 1
        missing_cache = set(window.included_hours) - set(paths_by_hour)
        if missing_cache:
            waiting_for_cache += 1
            continue
        expected_name = expected_verification_name(model_key, stamp)
        output_path = output_dir_for_model(args, model_key) / stamp / str(expected_name)
        if output_path.exists():
            completed_windows += 1
            continue
        raw_obs_complete = all(
            obs_path_for_time(args.obs_dir, timestamp).exists()
            for timestamp in expected_obs_times(window.start, window.end)
        )
        aggregate_obs_complete = all(
            path.exists()
            for path in ml_archive.expected_observation_aggregate_paths(
                args.ml_archive_root,
                window.start,
                window.end,
            )
        )
        obs_complete = raw_obs_complete or aggregate_obs_complete
        if obs_complete:
            ready_to_render += 1
        else:
            waiting_for_observations += 1

    latest_obs: dt.datetime | None = None
    if args.obs_dir.exists():
        for path in args.obs_dir.glob("*/*_MSC_Lightning_2.5km.tif"):
            timestamp = parse_obs_time(path.name)
            if timestamp is not None and (latest_obs is None or timestamp > latest_obs):
                latest_obs = timestamp
    aggregate_dir = ml_archive.observation_archive_dir(args.ml_archive_root)
    if aggregate_dir.exists():
        for path in aggregate_dir.glob("*/*/*_MSC_LightningDensity_3h_2.5km.tif"):
            try:
                timestamp = dt.datetime.strptime(path.name[:14], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            if latest_obs is None or timestamp > latest_obs:
                latest_obs = timestamp
    return {
        "candidate_windows": candidate_windows,
        "completed_windows": completed_windows,
        "waiting_for_cache": waiting_for_cache,
        "waiting_for_observations": waiting_for_observations,
        "ready_to_render": ready_to_render,
        "latest_observation_utc": latest_obs.isoformat().replace("+00:00", "Z") if latest_obs else None,
    }


def publish_generated(args: argparse.Namespace, generated: list[Path]) -> None:
    if args.no_publish:
        return

    targets: set[tuple[str, str]] = set()
    for path in generated:
        stamp = path.parent.name
        if path.name.startswith("hrdps_continental_lightning_verif_"):
            targets.add(("continental", stamp))
        elif path.name.startswith("hrdps_west_lightning_verif_"):
            targets.add(("west", stamp))

    with publish_lock():
        for model_key, stamp in sorted(targets):
            publish(
                stamp=stamp,
                plots_dir=None,
                pages_repo=args.pages_repo,
                keep_days=args.keep_days,
                lightning_verif_plots_dir=output_dir_for_model(args, model_key),
                model=product_model_for_forecast_model(model_key),
                partial=True,
            )
        removed = prune_superseded_pages_frames(args)
        if removed:
            write_manifest(args.pages_repo / "hrdps-west", args.keep_days)
        if not targets and not removed:
            return
        from automate_hrdps_west import commit_and_push_pages

        stamps = ",".join(sorted(stamp for _, stamp in targets)) or "cleanup"
        commit_and_push_pages(args.pages_repo, stamps, "LPI verification")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    with file_lock(LOCK_PATH) as acquired:
        if not acquired:
            return 0
        started = dt.datetime.now(dt.timezone.utc)
        log(f"LPI verification check started {started:%Y-%m-%dT%H:%M:%SZ}.")
        write_status("running", started_at_utc=started.isoformat().replace("+00:00", "Z"))
        try:
            mirrored = mirror_lightning_observations(args.obs_dir, args.workers, args.ml_archive_root)
            aggregated: list[Path] = []
            archive_error: str | None = None
            try:
                ml_archive.verify_archive_writable(args.ml_archive_root)
                aggregated = ml_archive.aggregate_available_observations(
                    args.obs_dir,
                    args.ml_archive_root,
                    delete_sources=not args.keep_raw_observations,
                )
                ml_archive.write_archive_status(args.ml_archive_root)
            except Exception as exc:
                archive_error = str(exc)
                log(f"Lightning observation archive unavailable; retaining ten-minute sources: {exc}")
            if archive_error is None:
                prune_obs(args.obs_dir, args.keep_days + 1)
            else:
                log("Skipping raw lightning pruning because the aggregate archive is unavailable.")
            try:
                tuning_readiness = lpi_tuning_readiness(args.ml_archive_root)
                tuning_notification_sent = notify_lpi_tuning_ready(tuning_readiness)
            except OSError as exc:
                tuning_readiness = {"ready": False, "error": str(exc)}
                tuning_notification_sent = False
            prune_local_plots(args.west_output_dir, args.keep_days)
            prune_local_plots(args.continental_output_dir, args.keep_days)
            removed_local = prune_superseded_local_frames(args)
            generated: list[Path] = []
            if not args.mirror_only:
                generated = render_ready_verifications(args)
                publish_generated(args, generated)
            summary = verification_queue_summary(args)
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            log(
                "LPI verification check complete: "
                f"mirrored={mirrored}, aggregated={len(aggregated)}, generated={len(generated)}, "
                f"waiting_for_observations={summary['waiting_for_observations']}, elapsed={elapsed:.1f}s."
            )
            write_status(
                "success",
                started_at_utc=started.isoformat().replace("+00:00", "Z"),
                elapsed_seconds=round(elapsed, 2),
                mirrored_files=mirrored,
                aggregated_blocks=len(aggregated),
                observation_archive_error=archive_error,
                lpi_tuning=tuning_readiness,
                lpi_tuning_notification_sent=tuning_notification_sent,
                generated_files=[str(path) for path in generated],
                removed_superseded_local_frames=len(removed_local),
                **summary,
            )
        except Exception as exc:
            write_status(
                "failed",
                started_at_utc=started.isoformat().replace("+00:00", "Z"),
                error=str(exc),
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
