#!/usr/bin/env python3
"""Operational update job for HRDPS graphics."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

import make_hrdps_west_convective as convective
import make_hrdps_evolved_danger_class as danger
import fire_danger_peak
import make_hrdps_west_fourpanel as fourpanel
import make_hrdps_west_lightning as lightning
import lightning_ml_archive as ml_archive
from publish_hrdps_west import (
    DEFAULT_PAGES_REPO,
    PRODUCTS_BY_MODEL,
    archive_dir_for_product,
    expected_images,
    image_name_for_hour,
    publish,
)

FOURPANEL_WATERSHED_CACHE = convective.WATERSHED_CACHE
PUBLISH_LOCK = Path("logs/hrdps_publish.lock")
JOB_STATE_ROOT = Path("logs/state")
FIRE_WEATHER_REGION_KEYS_BY_MODEL = {
    "west": ("sw", "se", "ne"),
    "continental": ("bc",),
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(convective.MODEL_CONFIGS), default="west")
    parser.add_argument("--cycle", choices=["latest", "00", "06", "12", "18"], required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Convective output directory.")
    parser.add_argument("--fourpanel-output-dir", type=Path, default=None)
    parser.add_argument("--lightning-output-dir", type=Path, default=None)
    parser.add_argument("--danger-output-dir", type=Path, default=None)
    parser.add_argument("--pages-repo", type=Path, default=DEFAULT_PAGES_REPO)
    parser.add_argument("--ml-archive-root", type=Path, default=ml_archive.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--fourpanel-shade-stride", type=int, default=None)
    parser.add_argument("--fourpanel-contour-stride", type=int, default=None)
    parser.add_argument("--fourpanel-barb-stride", type=int, default=None)
    parser.add_argument("--lightning-shade-stride", type=int, default=None)
    parser.add_argument("--lightning-contour-stride", type=int, default=None)
    parser.add_argument("--lightning-dcape-stride", type=int, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--keep-days", type=int, default=7)
    parser.add_argument("--wait-minutes", type=int, default=600)
    parser.add_argument("--poll-minutes", type=int, default=5)
    parser.add_argument(
        "--publish-cooldown-minutes",
        type=int,
        default=12,
        help="Minimum time between partial GitHub Pages publishes; full-run publishes are immediate.",
    )
    parser.add_argument("--max-runtime-minutes", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--complete-only", action="store_true", help="Wait for a full run before rendering/publishing.")
    parser.add_argument(
        "--legacy-pages-publish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish to the legacy FCSTPP GitHub Page during the R2 migration.",
    )
    return parser.parse_args(list(argv))


def plot_set_complete(output_dir: Path, stamp: str, product_key: str) -> bool:
    plot_dir = output_dir / stamp
    return all((plot_dir / name).exists() for name in expected_images(stamp, product_key))


def published_run_exists(pages_repo: Path, stamp: str, model: str) -> bool:
    run_dir = pages_repo / "hrdps-west" / "images" / stamp
    product_keys = PRODUCTS_BY_MODEL[model]
    return all(
        (archive_dir_for_product(run_dir, product_key) / name).exists()
        for product_key in product_keys
        for name in expected_images(stamp, product_key)
    )


def model_has_product_suffix(model: str, suffix: str) -> bool:
    return any(product_key.endswith(suffix) for product_key in PRODUCTS_BY_MODEL[model])


def run_has_all_fields(run: convective.RunInfo) -> bool:
    complete = convective.run_is_complete(run) and lightning.run_is_complete(run)
    if model_has_product_suffix(convective.model_config().key, "fourpanel"):
        complete = complete and fourpanel.run_is_complete(run)
    return complete


def expected_stamp_for_cycle(cycle: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    cycle_hour = int(cycle)
    candidate = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if candidate > now:
        candidate -= dt.timedelta(days=1)
    return f"{candidate:%Y%m%dT%HZ}"


def current_cycle_run(cycle: str) -> convective.RunInfo | None:
    try:
        stamp = convective.run_stamp_from_listing(cycle)
    except Exception as exc:
        convective.log(f"{convective.model_config().label} {cycle}Z listing is not available yet: {exc}")
        return None
    if not stamp:
        return None
    expected_stamp = expected_stamp_for_cycle(cycle)
    if stamp != expected_stamp:
        convective.log(f"Ignoring stale {convective.model_config().label} {cycle}Z listing {stamp}; waiting for {expected_stamp}.")
        return None
    return convective.RunInfo(cycle=cycle, stamp=stamp, init_time=convective.parse_stamp(stamp))


def hour_has_all_fields(run: convective.RunInfo, fhour: int) -> bool:
    try:
        html = convective.fetch_text(f"{convective.model_config().base_url}/{run.cycle}/{fhour:03d}/")
    except Exception as exc:
        convective.log(f"F{fhour:03d} is not listed yet: {exc}")
        return False
    links = set(convective.parse_links(html))
    needed = set(convective.required_names(run.stamp, fhour, include_static=(fhour == convective.TERRAIN_FHOUR)))
    if model_has_product_suffix(convective.model_config().key, "fourpanel"):
        needed.update(fourpanel.required_names(run.stamp, fhour))
    needed.update(lightning.required_names(run.stamp, fhour))
    missing = needed - links
    if missing:
        convective.log(f"F{fhour:03d} not ready: missing {len(missing)} files.")
        return False
    return True


def ready_hours(run: convective.RunInfo) -> tuple[int, ...]:
    ready = [fhour for fhour in convective.FORECAST_HOURS if hour_has_all_fields(run, fhour)]
    # Plotting uses F000 for grid geometry and terrain, so hold later hours until F000 is ready.
    if convective.TERRAIN_FHOUR not in ready:
        return ()

    listing_cache: dict[int, set[str] | None] = {}

    def lightning_hour_ready(fhour: int, required: set[str]) -> bool:
        if fhour not in listing_cache:
            try:
                html = convective.fetch_text(f"{convective.model_config().base_url}/{run.cycle}/{fhour:03d}/")
                listing_cache[fhour] = set(convective.parse_links(html))
            except Exception as exc:
                convective.log(f"F{fhour:03d} lightning-window input is not listed yet: {exc}")
                listing_cache[fhour] = None
        links = listing_cache[fhour]
        if links is None:
            return False
        missing = required - links
        if missing:
            convective.log(f"F{fhour:03d} lightning-window input not ready: missing {len(missing)} files.")
            return False
        return True

    output: list[int] = []
    for fhour in ready:
        requirements = lightning.required_names_by_hour(run.stamp, (fhour,))
        requirements.pop(convective.TERRAIN_FHOUR, None)
        requirements.pop(fhour, None)
        if all(
            lightning_hour_ready(hour, requirements[hour])
            for hour in sorted(requirements)
        ):
            output.append(fhour)
    return tuple(output)


def danger_hour_ready(run: convective.RunInfo, fhour: int) -> bool:
    """Check the hourly surface forcing needed since the previous plotted frame."""

    first = max(0, fhour - 2)
    for hourly in range(first, fhour + 1):
        try:
            html = convective.fetch_text(f"{convective.model_config().base_url}/{run.cycle}/{hourly:03d}/")
        except Exception:
            return False
        links = set(convective.parse_links(html))
        if not set(danger.required_names(run.stamp, hourly)).issubset(links):
            return False
    return True


def download_hours(run: convective.RunInfo, data_dir: Path, hours: Iterable[int], workers: int) -> None:
    hours = tuple(sorted(set(int(hour) for hour in hours)))
    if not hours:
        return
    job_map: dict[Path, str] = {}
    run_dir = data_dir / run.stamp
    plot_hours = set(hours)
    lightning_requirements = lightning.required_names_by_hour(run.stamp, hours)
    prerequisite_hours = plot_hours | {convective.TERRAIN_FHOUR} | set(lightning_requirements)
    for fhour in sorted(prerequisite_hours):
        names = set(lightning_requirements.get(fhour, ()))
        if fhour in plot_hours or fhour == convective.TERRAIN_FHOUR:
            names.update(
                convective.required_names(
                    run.stamp,
                    fhour,
                    include_static=(fhour == convective.TERRAIN_FHOUR),
                )
            )
            if model_has_product_suffix(convective.model_config().key, "fourpanel"):
                names.update(fourpanel.required_names(run.stamp, fhour))
            if ml_archive.should_archive_model_run(convective.model_config().key, run.cycle):
                names.update(ml_archive.required_model_names(run.stamp, fhour))
        for name in names:
            dest = run_dir / f"{fhour:03d}" / name
            job_map[dest] = f"{convective.model_config().base_url}/{run.cycle}/{fhour:03d}/{name}"
    if model_has_product_suffix(convective.model_config().key, "fwi2025_danger"):
        for fhour in danger.prerequisite_hours(hours):
            for name in danger.required_names(run.stamp, fhour):
                dest = run_dir / f"{fhour:03d}" / name
                job_map[dest] = f"{convective.model_config().base_url}/{run.cycle}/{fhour:03d}/{name}"

    convective.log(f"Downloading/reusing {len(job_map)} GRIB2 files for hours {','.join(f'{hour:03d}' for hour in hours)}.")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convective.download_one, url, dest) for dest, url in job_map.items()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                convective.log(f"  files ready: {completed}/{len(futures)}")


def existing_plot_hours(output_dir: Path, stamp: str, product_key: str) -> set[int]:
    plot_dir = output_dir / stamp
    return {
        fhour
        for fhour in convective.FORECAST_HOURS
        if (plot_dir / image_name_for_hour(stamp, product_key, fhour)).exists()
    }


def existing_product_family_hours(output_dir: Path, stamp: str, product_keys: Iterable[str]) -> set[int]:
    hour_sets = [existing_plot_hours(output_dir, stamp, product_key) for product_key in product_keys]
    if not hour_sets:
        return set()
    return set.intersection(*hour_sets)


def full_product_hours_published(pages_repo: Path, stamp: str, product_key: str) -> set[int]:
    run_dir = pages_repo / "hrdps-west" / "images" / stamp
    archive_dir = archive_dir_for_product(run_dir, product_key)
    return {
        fhour
        for fhour in convective.FORECAST_HOURS
        if (archive_dir / image_name_for_hour(stamp, product_key, fhour)).exists()
    }


def full_product_family_hours_published(pages_repo: Path, stamp: str, product_keys: Iterable[str]) -> set[int]:
    hour_sets = [full_product_hours_published(pages_repo, stamp, product_key) for product_key in product_keys]
    if not hour_sets:
        return set()
    return set.intersection(*hour_sets)


def lightning_product_keys(model: str) -> tuple[str, ...]:
    return tuple(key for key in PRODUCTS_BY_MODEL[model] if "lightning" in key and "verif" not in key)


def fire_weather_region_keys(model: str) -> tuple[str, ...]:
    return FIRE_WEATHER_REGION_KEYS_BY_MODEL.get(model, ("bc",))


def latest_complete_run() -> convective.RunInfo:
    candidates: list[convective.RunInfo] = []
    for cycle in convective.AVAILABLE_CYCLES:
        try:
            stamp = convective.run_stamp_from_listing(cycle)
        except Exception as exc:
            convective.log(f"Could not inspect cycle {cycle}Z: {exc}")
            continue
        if stamp:
            candidates.append(convective.RunInfo(cycle=cycle, stamp=stamp, init_time=convective.parse_stamp(stamp)))

    for run in sorted(candidates, key=lambda item: item.init_time, reverse=True):
        convective.log(f"Checking completeness for {convective.model_config().label} {run.stamp}.")
        if run_has_all_fields(run):
            return run

    raise RuntimeError(f"No complete {convective.model_config().label} run was found for both product sets.")


def wait_for_run(cycle: str, wait_minutes: int, poll_minutes: int) -> convective.RunInfo:
    if cycle == "latest":
        return latest_complete_run()

    deadline = time.monotonic() + wait_minutes * 60
    last_error = ""
    while True:
        try:
            run = current_cycle_run(cycle)
            if run:
                convective.log(f"Checking completeness for {convective.model_config().label} {run.stamp}.")
                if run_has_all_fields(run):
                    return run
        except Exception as exc:
            last_error = str(exc)
            convective.log(f"Cycle {cycle}Z is not ready: {exc}")

        if time.monotonic() >= deadline:
            detail = f" Last error: {last_error}" if last_error else ""
            raise RuntimeError(f"Timed out waiting for complete {convective.model_config().label} {cycle}Z run.{detail}")

        convective.log(f"Waiting {poll_minutes} minutes before retrying {cycle}Z.")
        time.sleep(poll_minutes * 60)


def cleanup_model_data(
    data_dir: Path,
    keep_stamp: str,
    archive_root: Path = ml_archive.DEFAULT_ARCHIVE_ROOT,
) -> None:
    if not data_dir.exists():
        return
    for child in data_dir.iterdir():
        if child.is_dir() and child.name != keep_stamp:
            cycle = child.name[9:11] if len(child.name) >= 11 else ""
            if ml_archive.should_archive_model_run(convective.model_config().key, cycle) and not ml_archive.run_archive_complete(
                archive_root, child.name
            ):
                convective.log(f"Preserving unarchived HRDPS training run: {child}")
                continue
            convective.log(f"Removing old model data: {child}")
            shutil.rmtree(child)


def archive_downloaded_hours(
    args: argparse.Namespace,
    run: convective.RunInfo,
    data_dir: Path,
    hours: Iterable[int],
) -> bool:
    hours = tuple(sorted(set(int(hour) for hour in hours)))
    if not hours or not ml_archive.should_archive_model_run(args.model, run.cycle):
        return True
    preflight_error = getattr(args, "ml_archive_preflight_error", None)
    if preflight_error:
        convective.log(f"Skipping HRDPS training archive; startup health check failed: {preflight_error}")
        return False
    try:
        ml_archive.verify_archive_writable(args.ml_archive_root)
        ml_archive.archive_model_hours(args.ml_archive_root, run, data_dir, hours)
        ml_archive.write_archive_status(args.ml_archive_root)
        return True
    except Exception as exc:
        args.ml_archive_preflight_error = str(exc)
        convective.log(f"HRDPS training archive failed; raw run will be retained for retry: {exc}")
        return False


def archive_lpi_baselines(
    args: argparse.Namespace,
    run: convective.RunInfo,
    lightning_output_dir: Path,
    hours: Iterable[int],
) -> bool:
    hours = tuple(sorted(set(int(hour) for hour in hours)))
    if not hours or not ml_archive.should_archive_model_run(args.model, run.cycle):
        return True
    preflight_error = getattr(args, "ml_archive_preflight_error", None)
    if preflight_error:
        convective.log(f"Skipping handmade-LPI archive; startup health check failed: {preflight_error}")
        return False
    try:
        ml_archive.verify_archive_writable(args.ml_archive_root)
        ml_archive.archive_lpi_baseline_hours(args.ml_archive_root, run, lightning_output_dir, hours)
        ml_archive.write_archive_status(args.ml_archive_root)
        return True
    except Exception as exc:
        args.ml_archive_preflight_error = str(exc)
        convective.log(f"Handmade-LPI baseline archive failed: {exc}")
        return False


def cleanup_local_plots(output_dir: Path, keep_days: int) -> None:
    if not output_dir.exists():
        return
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            init_time = convective.parse_stamp(child.name)
        except ValueError:
            continue
        if init_time < cutoff:
            convective.log(f"Removing local plot archive older than retention: {child}")
            shutil.rmtree(child)


def pages_worktree_has_changes(pages_repo: Path) -> bool:
    status = subprocess.run(
        ["git", "-C", str(pages_repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(status.stdout.strip())


def pages_head_has_parents(pages_repo: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(pages_repo), "rev-list", "--parents", "-n", "1", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return len(result.stdout.strip().split()) > 1


def compact_pages_git_store(pages_repo: Path) -> None:
    subprocess.run(["git", "-C", str(pages_repo), "reflog", "expire", "--expire=now", "--all"], check=True)
    subprocess.run(["git", "-C", str(pages_repo), "gc", "--prune=now", "--quiet"], check=True)


def commit_and_push_pages(pages_repo: Path, stamp: str, model_label: str) -> None:
    has_changes = pages_worktree_has_changes(pages_repo)
    has_parent_history = pages_head_has_parents(pages_repo)
    if not has_changes and not has_parent_history:
        convective.log("No GitHub Pages changes to snapshot.")
        return

    message = f"Update {model_label} {stamp}"
    subprocess.run(["git", "-C", str(pages_repo), "fetch", "origin", "gh-pages"], check=True)
    subprocess.run(["git", "-C", str(pages_repo), "add", "-A"], check=True)
    tree = subprocess.run(
        ["git", "-C", str(pages_repo), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(pages_repo), "commit-tree", tree, "-m", message],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(pages_repo), "update-ref", "refs/heads/gh-pages", commit], check=True)
    subprocess.run(["git", "-C", str(pages_repo), "push", "--force-with-lease", "origin", "gh-pages:gh-pages"], check=True)
    compact_pages_git_store(pages_repo)
    convective.log(f"Pushed single-snapshot GitHub Pages commit {commit[:12]}.")


def status_path(model: str, cycle: str) -> Path:
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return JOB_STATE_ROOT / f"{model}_{cycle}.status.json"


def write_status(model: str, cycle: str, status: str, **metadata: object) -> None:
    payload = {
        "model": model,
        "cycle": cycle,
        "status": status,
        "pid": os.getpid(),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        **metadata,
    }
    path = status_path(model, cycle)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def job_lock(model: str, cycle: str):
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = JOB_STATE_ROOT / f"{model}_{cycle}.lock"
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            convective.log(f"Another HRDPS job is already running for {model} {cycle}; skipping this launch.")
            write_status(model, cycle, "skipped_existing_lock")
            yield False
            return
        handle.write(f"pid={os.getpid()} started_at_utc={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        handle.flush()
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def max_runtime(minutes: int):
    if minutes <= 0:
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"HRDPS job exceeded max runtime of {minutes} minutes.")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, minutes * 60)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def publish_lock():
    PUBLISH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PUBLISH_LOCK.open("w") as handle:
        convective.log(f"Waiting for publish lock: {PUBLISH_LOCK}")
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def render_convective_worker(
    model: str,
    run: convective.RunInfo,
    data_dir: Path,
    output_dir: Path,
    stride: int,
    hours: tuple[int, ...],
) -> int:
    convective.set_model(model)
    return len(convective.make_plots(run, data_dir, output_dir, stride, hours))


def render_fourpanel_worker(
    model: str,
    run: convective.RunInfo,
    data_dir: Path,
    output_dir: Path,
    shade_stride: int,
    contour_stride: int,
    barb_stride: int,
    hours: tuple[int, ...],
) -> int:
    convective.set_model(model)
    fourpanel.set_model(model)
    return len(
        fourpanel.make_plots(
            run,
            data_dir,
            output_dir,
            FOURPANEL_WATERSHED_CACHE,
            False,
            False,
            shade_stride,
            contour_stride,
            barb_stride,
            hours,
        )
    )


def render_lightning_worker(
    model: str,
    run: convective.RunInfo,
    data_dir: Path,
    output_dir: Path,
    shade_stride: int,
    contour_stride: int,
    dcape_stride: int,
    hours: tuple[int, ...],
) -> int:
    convective.set_model(model)
    lightning.set_model(model)
    return len(
        lightning.make_region_plots(
            run,
            data_dir,
            output_dir,
            shade_stride,
            contour_stride,
            dcape_stride,
            hours,
            region_keys=fire_weather_region_keys(model),
        )
    )


def render_danger_worker(
    model: str,
    run: convective.RunInfo,
    data_dir: Path,
    output_dir: Path,
    hours: tuple[int, ...],
) -> int:
    convective.set_model(model)
    return len(
        danger.make_plots(
            run,
            data_dir,
            output_dir,
            lightning.FWI_CACHE_DIR,
            hours,
            classification="schedule2",
            plot_stride=1,
            allow_bootstrap=True,
        )
    )


def render_products(
    args: argparse.Namespace,
    run: convective.RunInfo,
    data_dir: Path,
    output_dir: Path,
    fourpanel_output_dir: Path,
    lightning_output_dir: Path,
    danger_output_dir: Path,
    stride: int,
    shade_stride: int,
    contour_stride: int,
    barb_stride: int,
    lightning_shade_stride: int,
    lightning_contour_stride: int,
    lightning_dcape_stride: int,
    convective_hours: Iterable[int],
    fourpanel_hours: Iterable[int],
    lightning_hours: Iterable[int],
    danger_hours: Iterable[int],
) -> None:
    convective_hours = tuple(sorted(set(int(hour) for hour in convective_hours)))
    fourpanel_hours = tuple(sorted(set(int(hour) for hour in fourpanel_hours)))
    lightning_hours = tuple(sorted(set(int(hour) for hour in lightning_hours)))
    danger_hours = tuple(sorted(set(int(hour) for hour in danger_hours)))
    full_danger_render = set(convective.FORECAST_HOURS).issubset(danger_hours)
    if full_danger_render and lightning_hours:
        convective.log("Rendering the complete hourly danger sequence before fire-weather peak overlays.")
        render_danger_worker(args.model, run, data_dir, danger_output_dir, danger_hours)
        danger_hours = ()
    job_count = sum(bool(hours) for hours in (convective_hours, fourpanel_hours, lightning_hours, danger_hours))
    if job_count == 0:
        return
    if job_count == 1:
        if convective_hours:
            render_convective_worker(args.model, run, data_dir, output_dir, stride, convective_hours)
        elif fourpanel_hours:
            render_fourpanel_worker(
                args.model, run, data_dir, fourpanel_output_dir, shade_stride, contour_stride, barb_stride, fourpanel_hours
            )
        elif lightning_hours:
            render_lightning_worker(
                args.model,
                run,
                data_dir,
                lightning_output_dir,
                lightning_shade_stride,
                lightning_contour_stride,
                lightning_dcape_stride,
                lightning_hours,
            )
        else:
            render_danger_worker(args.model, run, data_dir, danger_output_dir, danger_hours)
        return
    convective.log(
        "Rendering products with up to two worker processes: "
        f"convective={len(convective_hours)}, four-panel={len(fourpanel_hours)}, "
        f"lightning={len(lightning_hours)}, FWI2025 danger={len(danger_hours)}."
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures: dict[str, concurrent.futures.Future[int]] = {}
        if convective_hours:
            futures["convective"] = executor.submit(
                render_convective_worker, args.model, run, data_dir, output_dir, stride, convective_hours
            )
        if fourpanel_hours:
            futures["four-panel"] = executor.submit(
                render_fourpanel_worker,
                args.model,
                run,
                data_dir,
                fourpanel_output_dir,
                shade_stride,
                contour_stride,
                barb_stride,
                fourpanel_hours,
            )
        if lightning_hours:
            futures["lightning"] = executor.submit(
                render_lightning_worker,
                args.model,
                run,
                data_dir,
                lightning_output_dir,
                lightning_shade_stride,
                lightning_contour_stride,
                lightning_dcape_stride,
                lightning_hours,
            )
        if danger_hours:
            futures["FWI2025 danger"] = executor.submit(
                render_danger_worker,
                args.model,
                run,
                data_dir,
                danger_output_dir,
                danger_hours,
            )
        for future in concurrent.futures.as_completed(futures.values()):
            name = next(key for key, value in futures.items() if value is future)
            count = future.result()
            convective.log(f"Finished {name} render worker with {count} frames.")


def wait_for_current_run(cycle: str, wait_minutes: int, poll_minutes: int) -> convective.RunInfo:
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        run = current_cycle_run(cycle)
        if run is not None:
            return run
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for current {convective.model_config().label} {cycle}Z listing "
                f"({expected_stamp_for_cycle(cycle)})."
            )
        convective.log(f"Waiting {poll_minutes} minutes for current {cycle}Z listing.")
        time.sleep(poll_minutes * 60)


def render_incremental_run(
    args: argparse.Namespace,
    config: convective.ModelConfig,
    data_dir: Path,
    output_dir: Path,
    fourpanel_output_dir: Path,
    lightning_output_dir: Path,
    danger_output_dir: Path,
    stride: int,
    shade_stride: int,
    contour_stride: int,
    barb_stride: int,
    lightning_shade_stride: int,
    lightning_contour_stride: int,
    lightning_dcape_stride: int,
    convective_key: str,
    fourpanel_key: str | None,
    danger_key: str | None,
) -> convective.RunInfo:
    if args.cycle == "latest":
        run = latest_complete_run()
    else:
        run = wait_for_current_run(args.cycle, args.wait_minutes, args.poll_minutes)

    convective.log(f"Using {config.label} run {run.stamp}.")
    deadline = time.monotonic() + args.wait_minutes * 60
    all_hours = set(convective.FORECAST_HOURS)
    lightning_keys = lightning_product_keys(args.model)
    # Experimental danger frames can begin after F000 when CWFIS bootstrap occurs, so they never gate core products.
    if args.legacy_pages_publish:
        last_published_hours: set[int] = full_product_hours_published(
            args.pages_repo, run.stamp, convective_key
        ) & full_product_family_hours_published(args.pages_repo, run.stamp, lightning_keys)
        if fourpanel_key is not None:
            last_published_hours &= full_product_hours_published(args.pages_repo, run.stamp, fourpanel_key)
    else:
        last_published_hours = set()
    last_publish_monotonic = 0.0

    while True:
        ready = set(ready_hours(run))
        conv_done = existing_plot_hours(output_dir, run.stamp, convective_key)
        four_done = existing_plot_hours(fourpanel_output_dir, run.stamp, fourpanel_key) if fourpanel_key is not None else set(ready)
        lightning_done = existing_product_family_hours(lightning_output_dir, run.stamp, lightning_keys)
        danger_done = existing_plot_hours(danger_output_dir, run.stamp, danger_key) if danger_key is not None else set(ready)
        core_done = conv_done & four_done & lightning_done
        needs_convective = sorted(ready - conv_done)
        needs_fourpanel = sorted(ready - four_done)
        needs_lightning = sorted(ready - lightning_done)
        danger_is_bootstrap_shortened = bool(danger_done) and all_hours.issubset(core_done)
        needs_danger = (
            sorted(hour for hour in ready - danger_done if danger_hour_ready(run, hour))
            if danger_key is not None and not danger_is_bootstrap_shortened
            else []
        )

        if ready:
            convective.log(
                f"Ready hours for {run.stamp}: {','.join(f'F{hour:03d}' for hour in sorted(ready))}; "
                f"new convective={len(needs_convective)}, four-panel={len(needs_fourpanel)}, "
                f"lightning={len(needs_lightning)}, FWI2025 danger={len(needs_danger)}."
            )

        if needs_convective or needs_fourpanel or needs_lightning or needs_danger or args.force:
            plot_hours = sorted(
                ready
                if args.force
                else set(needs_convective) | set(needs_fourpanel) | set(needs_lightning) | set(needs_danger)
            )
            download_hours(run, data_dir, plot_hours, args.workers)
            archive_downloaded_hours(args, run, data_dir, plot_hours)
            render_products(
                args,
                run,
                data_dir,
                output_dir,
                fourpanel_output_dir,
                lightning_output_dir,
                danger_output_dir,
                stride,
                shade_stride,
                contour_stride,
                barb_stride,
                lightning_shade_stride,
                lightning_contour_stride,
                lightning_dcape_stride,
                sorted(ready if args.force else needs_convective),
                sorted(ready if args.force else needs_fourpanel),
                sorted(ready if args.force else needs_lightning),
                sorted(ready if args.force and danger_key is not None else needs_danger),
            )
            archive_lpi_baselines(
                args,
                run,
                lightning_output_dir,
                sorted(ready if args.force else needs_lightning),
            )

        if danger_key is not None and all_hours.issubset(ready):
            peak_marker = fire_danger_peak.peak_run_marker_path(
                lightning.FWI_CACHE_DIR,
                args.model,
                run.stamp,
            )
            lightning_plot_paths = [
                lightning_output_dir / run.stamp / image_name_for_hour(run.stamp, key, hour)
                for key in lightning_keys
                for hour in convective.FORECAST_HOURS
            ]
            peak_overlay_stale = (
                not peak_marker.exists()
                or any(not path.exists() or path.stat().st_mtime_ns < peak_marker.stat().st_mtime_ns for path in lightning_plot_paths)
            )
            if peak_overlay_stale:
                convective.log("Finalizing peak-daily fire danger and refreshing all fire-weather overlays.")
                render_danger_worker(
                    args.model,
                    run,
                    data_dir,
                    danger_output_dir,
                    tuple(convective.FORECAST_HOURS),
                )
                render_lightning_worker(
                    args.model,
                    run,
                    data_dir,
                    lightning_output_dir,
                    lightning_shade_stride,
                    lightning_contour_stride,
                    lightning_dcape_stride,
                    tuple(convective.FORECAST_HOURS),
                )
                archive_lpi_baselines(args, run, lightning_output_dir, convective.FORECAST_HOURS)

        # Publish core products independently of the optional, potentially shorter danger sequence.
        publishable_hours = existing_plot_hours(
            output_dir, run.stamp, convective_key
        ) & existing_product_family_hours(lightning_output_dir, run.stamp, lightning_keys)
        if fourpanel_key is not None:
            publishable_hours &= existing_plot_hours(fourpanel_output_dir, run.stamp, fourpanel_key)
        if publishable_hours and (publishable_hours != last_published_hours or args.force):
            full_run_ready = all_hours.issubset(publishable_hours)
            cooldown_seconds = max(0, args.publish_cooldown_minutes) * 60
            elapsed_since_publish = time.monotonic() - last_publish_monotonic
            publish_now = args.force or full_run_ready or last_publish_monotonic == 0.0 or elapsed_since_publish >= cooldown_seconds
            if publish_now:
                if args.legacy_pages_publish:
                    try:
                        with publish_lock():
                            publish(
                                stamp=run.stamp,
                                plots_dir=output_dir,
                                pages_repo=args.pages_repo,
                                keep_days=args.keep_days,
                                fourpanel_plots_dir=fourpanel_output_dir,
                                lightning_plots_dir=lightning_output_dir,
                                danger_plots_dir=danger_output_dir,
                                model=args.model,
                                partial=True,
                            )
                            commit_and_push_pages(args.pages_repo, run.stamp, config.label)
                    except Exception as exc:
                        convective.log(
                            f"Legacy GitHub Pages publish failed for {run.stamp}; "
                            f"rendering will continue and the independent publisher can retry: {exc}"
                        )
                last_publish_monotonic = time.monotonic()
                last_published_hours = set(publishable_hours)
                write_status(
                    args.model,
                    args.cycle,
                    "running",
                    model_label=config.label,
                    stamp=run.stamp,
                    published_hours=sorted(last_published_hours),
                )
            else:
                remaining = max(0.0, cooldown_seconds - elapsed_since_publish) / 60.0
                convective.log(
                    f"Deferring partial publish for {run.stamp}; "
                    f"{len(publishable_hours)} hours are render-complete, cooldown has {remaining:.1f} minutes left."
                )

        if all_hours.issubset(last_published_hours):
            convective.log(f"All forecast hours are published for {run.stamp}.")
            cleanup_local_plots(output_dir, args.keep_days)
            cleanup_local_plots(fourpanel_output_dir, args.keep_days)
            cleanup_local_plots(lightning_output_dir, args.keep_days)
            cleanup_local_plots(danger_output_dir, args.keep_days)
            cleanup_model_data(data_dir, run.stamp, args.ml_archive_root)
            return run

        if time.monotonic() >= deadline:
            if last_published_hours:
                convective.log(
                    f"Timed out before full run, but published {len(last_published_hours)} hours for {run.stamp}."
                )
                return run
            raise RuntimeError(f"Timed out before any publishable hours were ready for {run.stamp}.")

        convective.log(f"Waiting {args.poll_minutes} minutes for more {run.stamp} forecast hours.")
        time.sleep(args.poll_minutes * 60)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = convective.set_model(args.model)
    fourpanel.set_model(args.model)
    data_dir = args.data_dir or Path(config.default_data_dir)
    output_dir = args.output_dir or Path(config.default_output_dir)
    fourpanel_output_dir = args.fourpanel_output_dir or Path(f"{config.default_output_dir}_fourpanel")
    lightning_output_dir = args.lightning_output_dir or Path(f"{config.default_output_dir}_lightning")
    danger_output_dir = args.danger_output_dir or Path("plots/experimental_fwi2025_danger")
    stride = args.stride or convective.grid_stride(18.0)
    shade_stride = args.fourpanel_shade_stride or convective.grid_stride(5.0)
    contour_stride = args.fourpanel_contour_stride or convective.grid_stride(12.0)
    barb_stride = args.fourpanel_barb_stride or convective.grid_stride(27.0)
    lightning_shade_stride = args.lightning_shade_stride or convective.grid_stride(5.0)
    lightning_contour_stride = args.lightning_contour_stride or convective.grid_stride(12.0)
    lightning_dcape_stride = args.lightning_dcape_stride or convective.grid_stride(18.0)
    product_keys = PRODUCTS_BY_MODEL[args.model]
    fourpanel_key = next((key for key in product_keys if key.endswith("fourpanel")), None)
    convective_key = next(key for key in product_keys if key.endswith("convective"))
    lightning_keys = lightning_product_keys(args.model)
    danger_key = next((key for key in product_keys if key.endswith("fwi2025_danger")), None)
    if args.cycle not in (*convective.AVAILABLE_CYCLES, "latest"):
        raise RuntimeError(f"Unsupported {config.label} cycle: {args.cycle}")

    args.ml_archive_preflight_error = None
    if args.model == "continental" and args.cycle in ml_archive.MODEL_ARCHIVE_CYCLES:
        try:
            ml_archive.verify_archive_writable(args.ml_archive_root)
        except Exception as exc:
            args.ml_archive_preflight_error = str(exc)
            convective.log(f"Lightning ML archive startup health check failed: {exc}")

    run: convective.RunInfo | None = None
    with job_lock(args.model, args.cycle) as acquired:
        if not acquired:
            return 0
        write_status(
            args.model,
            args.cycle,
            "running",
            model_label=config.label,
            lightning_ml_archive_error=args.ml_archive_preflight_error,
        )
        try:
            with max_runtime(args.max_runtime_minutes):
                if not args.complete_only:
                    run = render_incremental_run(
                        args,
                        config,
                        data_dir,
                        output_dir,
                        fourpanel_output_dir,
                        lightning_output_dir,
                        danger_output_dir,
                        stride,
                        shade_stride,
                        contour_stride,
                        barb_stride,
                        lightning_shade_stride,
                        lightning_contour_stride,
                        lightning_dcape_stride,
                        convective_key,
                        fourpanel_key,
                        danger_key,
                    )
                    write_status(
                        args.model,
                        args.cycle,
                        "success",
                        model_label=config.label,
                        stamp=run.stamp,
                        published_hours=sorted(
                            full_product_hours_published(args.pages_repo, run.stamp, convective_key)
                            & (
                                full_product_hours_published(args.pages_repo, run.stamp, fourpanel_key)
                                if fourpanel_key is not None
                                else set(convective.FORECAST_HOURS)
                            )
                            & full_product_family_hours_published(args.pages_repo, run.stamp, lightning_keys)
                        ),
                        lightning_ml_archive_error=args.ml_archive_preflight_error,
                    )
                    return 0

                run = wait_for_run(args.cycle, args.wait_minutes, args.poll_minutes)
                convective.log(f"Using {config.label} run {run.stamp}.")

                if (
                    args.legacy_pages_publish
                    and published_run_exists(args.pages_repo, run.stamp, args.model)
                    and not args.force
                ):
                    convective.log(f"{run.stamp} is already published; skipping render.")
                    cleanup_model_data(data_dir, run.stamp, args.ml_archive_root)
                    cleanup_local_plots(output_dir, args.keep_days)
                    cleanup_local_plots(fourpanel_output_dir, args.keep_days)
                    cleanup_local_plots(lightning_output_dir, args.keep_days)
                    write_status(
                        args.model,
                        args.cycle,
                        "success",
                        model_label=config.label,
                        stamp=run.stamp,
                        skipped=True,
                        lightning_ml_archive_error=args.ml_archive_preflight_error,
                    )
                    return 0

                convective_complete = plot_set_complete(output_dir, run.stamp, convective_key)
                fourpanel_complete = fourpanel_key is None or plot_set_complete(
                    fourpanel_output_dir, run.stamp, fourpanel_key
                )
                lightning_complete = all(plot_set_complete(lightning_output_dir, run.stamp, key) for key in lightning_keys)
                danger_complete = danger_key is None or plot_set_complete(danger_output_dir, run.stamp, danger_key)
                if args.force or not (convective_complete and fourpanel_complete and lightning_complete and danger_complete):
                    download_hours(run, data_dir, convective.FORECAST_HOURS, args.workers)
                    archive_downloaded_hours(args, run, data_dir, convective.FORECAST_HOURS)

                if convective_complete and not args.force:
                    convective.log(f"Using existing complete convective plot set for {run.stamp}.")

                if fourpanel_complete and not args.force:
                    convective.log(f"Using existing complete four-panel plot set for {run.stamp}.")

                if lightning_complete and not args.force:
                    convective.log(f"Using existing complete lightning plot set for {run.stamp}.")
                if danger_complete and danger_key is not None and not args.force:
                    convective.log(f"Using existing complete FWI2025 danger plot set for {run.stamp}.")
                render_products(
                    args,
                    run,
                    data_dir,
                    output_dir,
                    fourpanel_output_dir,
                    lightning_output_dir,
                    danger_output_dir,
                    stride,
                    shade_stride,
                    contour_stride,
                    barb_stride,
                    lightning_shade_stride,
                    lightning_contour_stride,
                    lightning_dcape_stride,
                    () if convective_complete and not args.force else convective.FORECAST_HOURS,
                    () if fourpanel_complete and not args.force else convective.FORECAST_HOURS,
                    () if lightning_complete and not args.force else convective.FORECAST_HOURS,
                    () if danger_complete and not args.force else convective.FORECAST_HOURS,
                )
                archive_lpi_baselines(
                    args,
                    run,
                    lightning_output_dir,
                    () if lightning_complete and not args.force else convective.FORECAST_HOURS,
                )

                if args.legacy_pages_publish:
                    try:
                        with publish_lock():
                            if published_run_exists(args.pages_repo, run.stamp, args.model) and not args.force:
                                convective.log(
                                    f"{run.stamp} was published by another job while this job rendered; skipping publish."
                                )
                            else:
                                publish(
                                    stamp=run.stamp,
                                    plots_dir=output_dir,
                                    pages_repo=args.pages_repo,
                                    keep_days=args.keep_days,
                                    fourpanel_plots_dir=fourpanel_output_dir,
                                    lightning_plots_dir=lightning_output_dir,
                                    danger_plots_dir=danger_output_dir,
                                    model=args.model,
                                )
                                commit_and_push_pages(args.pages_repo, run.stamp, config.label)
                    except Exception as exc:
                        convective.log(
                            f"Legacy GitHub Pages publish failed for {run.stamp}; "
                            f"the completed plots remain available for retry: {exc}"
                        )
                cleanup_model_data(data_dir, run.stamp, args.ml_archive_root)
                cleanup_local_plots(output_dir, args.keep_days)
                cleanup_local_plots(fourpanel_output_dir, args.keep_days)
                cleanup_local_plots(lightning_output_dir, args.keep_days)
                cleanup_local_plots(danger_output_dir, args.keep_days)
                write_status(
                    args.model,
                    args.cycle,
                    "success",
                    model_label=config.label,
                    stamp=run.stamp,
                    lightning_ml_archive_error=args.ml_archive_preflight_error,
                )
                return 0
        except Exception as exc:
            write_status(
                args.model,
                args.cycle,
                "failed",
                model_label=config.label,
                stamp=run.stamp if run else None,
                error=str(exc),
                lightning_ml_archive_error=args.ml_archive_preflight_error,
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
