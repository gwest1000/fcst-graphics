#!/usr/bin/env python3
"""Operational update job for ECMWF/GEFS control-member four-panel graphics."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import signal
from pathlib import Path
from typing import Iterable

import make_ensemble_control_fourpanel as fourpanel
from publish_hrdps_west import DEFAULT_PAGES_REPO, publish

PUBLISH_LOCK = Path("logs/hrdps_publish.lock")
JOB_STATE_ROOT = Path("logs/state")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(fourpanel.MODEL_CONFIGS), required=True)
    parser.add_argument("--cycle", type=int, default=0, choices=[0, 6, 12, 18])
    parser.add_argument("--stamp", default=None, help="Run stamp, e.g. 20260630T00Z. Defaults to latest cycle.")
    parser.add_argument("--data-root", type=Path, default=fourpanel.CONCRETE_DATA_ROOT)
    parser.add_argument("--concrete-repo", type=Path, default=fourpanel.CONCRETE_REPO)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pages-repo", type=Path, default=DEFAULT_PAGES_REPO)
    parser.add_argument("--keep-days", type=int, default=7)
    parser.add_argument("--hours", default=None, help="Comma-separated forecast hours. Defaults to 0..48 every 3h.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-runtime-minutes", type=int, default=360)
    parser.add_argument("--no-watersheds", action="store_true")
    parser.add_argument(
        "--legacy-pages-publish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish to the legacy FCSTPP GitHub Page during the R2 migration.",
    )
    return parser.parse_args(list(argv))


def log(message: str) -> None:
    print(message, flush=True)


def status_path(model: str, cycle: int) -> Path:
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return JOB_STATE_ROOT / f"{model}_{cycle:02d}.status.json"


def write_status(model: str, cycle: int, status: str, **metadata: object) -> None:
    payload = {
        "model": model,
        "cycle": f"{cycle:02d}",
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
def job_lock(model: str, cycle: int):
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = JOB_STATE_ROOT / f"{model}_{cycle:02d}.lock"
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"Another {model} {cycle:02d}Z graphics job is already running; skipping.")
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
def publish_lock():
    PUBLISH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PUBLISH_LOCK.open("w") as handle:
        log(f"Waiting for publish lock: {PUBLISH_LOCK}")
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def max_runtime(minutes: int):
    if minutes <= 0:
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Graphics job exceeded max runtime of {minutes} minutes.")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, minutes * 60)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def image_name(config: fourpanel.ModelConfig, stamp: str, fhour: int) -> str:
    return f"{config.output_prefix}_fourpanel_{stamp}_f{fhour:03d}.png"


def plot_set_complete(output_dir: Path, config: fourpanel.ModelConfig, stamp: str, hours: Iterable[int]) -> bool:
    plot_dir = output_dir / stamp
    return all((plot_dir / image_name(config, stamp, int(hour))).exists() for hour in hours)


def commit_and_push_pages(pages_repo: Path, stamp: str, model_label: str) -> None:
    from automate_hrdps_west import commit_and_push_pages as commit_snapshot

    commit_snapshot(pages_repo, stamp, model_label)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = fourpanel.MODEL_CONFIGS[args.model]
    stamp = args.stamp or fourpanel.latest_cycle_stamp(args.cycle)
    run = fourpanel.RunInfo(cycle=f"{fourpanel.cycle_hour_from_stamp(stamp):02d}", stamp=stamp, init_time=fourpanel.parse_stamp(stamp))
    hours = fourpanel.model_hours(args.model) if args.hours is None else tuple(int(item) for item in args.hours.split(",") if item.strip())
    output_dir = args.output_dir or Path(config.default_output_dir)

    with job_lock(args.model, args.cycle) as acquired:
        if not acquired:
            return 0
        write_status(args.model, args.cycle, "running", model_label=config.label, stamp=stamp)
        try:
            with max_runtime(args.max_runtime_minutes):
                if args.force or not plot_set_complete(output_dir, config, stamp, hours):
                    if args.force_download or not fourpanel.required_files_present(args.model, args.data_root, run, hours):
                        fourpanel.ensure_downloads(
                            args.model,
                            run,
                            hours,
                            args.concrete_repo,
                            args.data_root,
                            force=args.force_download,
                        )
                    fourpanel.make_plots(
                        args.model,
                        run,
                        args.data_root,
                        output_dir,
                        fourpanel.WATERSHED_CACHE,
                        False,
                        args.no_watersheds,
                        hours=hours,
                    )
                else:
                    log(f"Using existing complete {config.label} four-panel plot set for {stamp}.")

                if args.legacy_pages_publish:
                    try:
                        with publish_lock():
                            publish(
                                stamp=stamp,
                                plots_dir=None,
                                pages_repo=args.pages_repo,
                                keep_days=args.keep_days,
                                fourpanel_plots_dir=output_dir,
                                model=args.model,
                            )
                            commit_and_push_pages(args.pages_repo, stamp, config.label)
                    except Exception as exc:
                        log(
                            f"Legacy GitHub Pages publish failed for {stamp}; "
                            f"the completed plots remain available for retry: {exc}"
                        )
                write_status(args.model, args.cycle, "success", model_label=config.label, stamp=stamp, hours=list(hours))
                return 0
        except Exception as exc:
            write_status(args.model, args.cycle, "failed", model_label=config.label, stamp=stamp, error=str(exc))
            raise


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
