#!/usr/bin/env python3
"""Refresh and publish the static BC fire-danger verification dashboard."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import fire_danger_verification as verification
import plot_style
from publish_hrdps_west import DEFAULT_PAGES_REPO, PRODUCTS, minimum_manifest_hours, publish


DEFAULT_OUTPUT_DIR = Path("plots/fire_danger_verification")
LOCK_PATH = Path("logs/fire_danger_verification.lock")
STATE_PATH = Path("logs/state/fire_danger_verification.status.json")
RUN_STAMP_RE = re.compile(r"^\d{8}T(?:00|06|12|18)Z$")
DEFAULT_FORECAST_DIRS = (Path("plots/experimental_fwi2025_danger"),)
DEFAULT_FORECAST_PRODUCT = "continental_fwi2025_danger"


def _latest_mtime_ns(paths: Iterable[Path]) -> int:
    latest = 0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime_ns)
        except OSError:
            continue
    return latest


def source_signature(stamp: str, archive_root: Path, concrete_data_root: Path) -> dict[str, object]:
    local_date = dt.datetime.now(plot_style.LOCAL_TZ).date().isoformat()
    return {
        "stamp": stamp,
        "local_date": local_date,
        "forecast_mtime_ns": _latest_mtime_ns((archive_root / "forecasts").glob("*/*/*.csv")),
        "danger_summary_mtime_ns": _latest_mtime_ns(
            (concrete_data_root / "observations/bcws/danger_summaries/snapshots").rglob("*.json")
        ),
        "daily_observation_mtime_ns": _latest_mtime_ns(
            (concrete_data_root / "observations/bcws/datamart/daily").glob("*/*.csv")
        ),
    }


def load_state(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(path: Path, signature: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def latest_run_stamp(pages_repo: Path) -> str:
    manifest_path = pages_repo / "hrdps-west" / "runs.json"
    payload = json.loads(manifest_path.read_text())
    for run in payload.get("runs", []):
        products = run.get("products", {})
        if "continental_fwi2025_danger" in products or "continental_lightning" in products:
            return str(run["stamp"])
    if payload.get("runs"):
        return str(payload["runs"][0]["stamp"])
    raise RuntimeError("No published forecast run is available for the verification dashboard.")


def latest_local_run_stamp(
    forecast_dirs: Iterable[Path] = DEFAULT_FORECAST_DIRS,
    product_key: str = DEFAULT_FORECAST_PRODUCT,
) -> str:
    product = PRODUCTS[product_key]
    required_frames = minimum_manifest_hours(product_key)
    stamps = {
        run_dir.name
        for root in forecast_dirs
        if root.exists()
        for run_dir in root.iterdir()
        if run_dir.is_dir()
        and RUN_STAMP_RE.match(run_dir.name)
        and len(list(run_dir.glob(f"{product.prefix}_{run_dir.name}_f*.png"))) >= required_frames
    }
    if not stamps:
        raise RuntimeError("No publishable local continental forecast run is available for verification.")
    return max(stamps)


def commit_and_push(pages_repo: Path, stamp: str) -> None:
    subprocess.run(["git", "add", "hrdps-west"], cwd=pages_repo, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=pages_repo).returncode != 0
    if not changed:
        return
    subprocess.run(
        ["git", "commit", "-m", f"Update fire danger verification {stamp}"],
        cwd=pages_repo,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=pages_repo, check=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-repo", type=Path, default=DEFAULT_PAGES_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive-root", type=Path, default=verification.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--concrete-data-root", type=Path, default=verification.DEFAULT_CONCRETE_DATA_ROOT)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--stamp", help="Published run stamp to carry the static dashboard asset.")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        stamp = args.stamp or (
            latest_local_run_stamp() if args.no_publish else latest_run_stamp(args.pages_repo)
        )
        out_path = args.output_dir / stamp / f"fire_danger_verification_{stamp}_f000.png"
        signature = source_signature(stamp, args.archive_root, args.concrete_data_root)
        if out_path.exists() and load_state(args.state_path) == signature:
            return 0
        matched = verification.matched_verification_frame(args.archive_root, args.concrete_data_root)
        verification.render_dashboard(out_path, matched)
        if not args.no_publish:
            publish(
                stamp=stamp,
                plots_dir=None,
                pages_repo=args.pages_repo,
                keep_days=7,
                lightning_verif_plots_dir=args.output_dir,
                model="danger_verif",
                partial=True,
            )
            if not args.no_push:
                commit_and_push(args.pages_repo, stamp)
        save_state(args.state_path, signature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
