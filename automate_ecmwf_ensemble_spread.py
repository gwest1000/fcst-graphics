#!/usr/bin/env python3
"""Operational update job for ECMWF ENS 500 hPa mean/spread graphics."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import signal
import shutil
import time
from pathlib import Path
from typing import Iterable

import ecmwf_ensemble_stats_data as stats_data
import make_ecmwf_ensemble_spread as spread_plot
from make_hrdps_west_convective import RunInfo, parse_stamp


JOB_STATE_ROOT = Path("logs/state")
DEFAULT_LOCAL_PLOT_KEEP_DAYS = 7


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ecmwf_ensemble"], default="ecmwf_ensemble")
    parser.add_argument("--cycle", type=int, default=0, choices=[0, 12])
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--hours", default=None)
    parser.add_argument("--archive-root", type=Path, default=stats_data.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=spread_plot.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wait-minutes", type=int, default=180)
    parser.add_argument("--poll-minutes", type=int, default=10)
    parser.add_argument("--max-runtime-minutes", type=int, default=300)
    parser.add_argument(
        "--local-plot-keep-days",
        type=int,
        default=DEFAULT_LOCAL_PLOT_KEEP_DAYS,
        help="Retain local rendered run directories for this many days.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-legacy-pages-publish", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(list(argv))


def status_path(cycle: int) -> Path:
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return JOB_STATE_ROOT / f"ecmwf_ensemble_{cycle:02d}.status.json"


def write_status(cycle: int, status: str, **metadata: object) -> None:
    payload = {
        "model": "ecmwf_ensemble",
        "cycle": f"{cycle:02d}",
        "status": status,
        "pid": os.getpid(),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        **metadata,
    }
    target = status_path(cycle)
    staged = target.with_suffix(target.suffix + ".tmp")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    staged.replace(target)


@contextlib.contextmanager
def job_lock(cycle: int):
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = JOB_STATE_ROOT / f"ecmwf_ensemble_{cycle:02d}.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"Another ECMWF ENS {cycle:02d}Z job is already running; skipping.")
            yield False
            return
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
        raise TimeoutError(f"ECMWF ENS graphics job exceeded {minutes} minutes.")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, minutes * 60)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def plot_set_complete(output_dir: Path, stamp: str, hours: Iterable[int]) -> bool:
    run_dir = output_dir / stamp
    return all(
        (run_dir / spread_plot.image_name(stamp, int(hour))).exists()
        for hour in hours
    )


def prune_local_plots(
    output_dir: Path,
    keep_days: int,
    *,
    now: dt.datetime | None = None,
) -> list[Path]:
    if keep_days < 0 or not output_dir.exists():
        return []
    reference = now or dt.datetime.now(dt.timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=dt.timezone.utc)
    cutoff = reference.astimezone(dt.timezone.utc) - dt.timedelta(days=keep_days)
    removed: list[Path] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            init_time = parse_stamp(child.name)
        except ValueError:
            continue
        if init_time < cutoff:
            shutil.rmtree(child)
            removed.append(child)
            log(f"Removed expired local ECMWF ENS plot run {child}.")
    return removed


def ensure_with_wait(
    run: RunInfo,
    hours: tuple[int, ...],
    archive_root: Path,
    force: bool,
    wait_minutes: int,
    poll_minutes: int,
) -> stats_data.ArchivePaths:
    deadline = time.monotonic() + max(0, wait_minutes) * 60
    while True:
        try:
            return stats_data.ensure_archives(
                f"{run.init_time:%Y%m%d}",
                run.cycle,
                hours,
                archive_root=archive_root,
                force=force,
            )
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise
            delay = max(1, poll_minutes) * 60
            log(f"ECMWF ENS statistics are not ready ({exc}); retrying in {delay // 60} minutes.")
            time.sleep(delay)
            force = False


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    stamp = args.stamp or spread_plot.latest_cycle_stamp(args.cycle)
    init_time = parse_stamp(stamp)
    run = RunInfo(cycle=f"{init_time.hour:02d}", stamp=stamp, init_time=init_time)
    hours = (
        spread_plot.FORECAST_HOURS
        if args.hours is None
        else tuple(int(value) for value in args.hours.split(",") if value.strip())
    )
    with job_lock(args.cycle) as acquired:
        if not acquired:
            return 0
        write_status(args.cycle, "running", stamp=stamp, hours=list(hours))
        try:
            with max_runtime(args.max_runtime_minutes):
                if args.force or not plot_set_complete(args.output_dir, stamp, hours):
                    ensure_with_wait(
                        run,
                        hours,
                        args.archive_root,
                        args.force_download,
                        args.wait_minutes,
                        args.poll_minutes,
                    )
                    spread_plot.make_plots(run, args.archive_root, args.output_dir, hours)
                else:
                    log(f"Using existing complete ECMWF ENS plot set for {stamp}.")
            removed = prune_local_plots(args.output_dir, args.local_plot_keep_days)
            write_status(
                args.cycle,
                "success",
                stamp=stamp,
                hours=list(hours),
                local_plot_runs_pruned=len(removed),
            )
            return 0
        except Exception as exc:
            write_status(args.cycle, "failed", stamp=stamp, error=str(exc))
            raise


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
