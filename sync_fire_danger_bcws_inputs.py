#!/usr/bin/env python3
"""Mirror the recent BCWS inputs needed by the fire-danger verifier."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path
from typing import Iterable

import fire_danger_verification as verification


DEFAULT_SOURCE_ROOT = Path("/Volumes/Greg1_2tb/concrete_fcst_data")


def _copy_if_changed(source: Path, target: Path) -> bool:
    if target.exists() and target.stat().st_size == source.stat().st_size and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _dated_paths(root: Path, suffix: str, cutoff: dt.date) -> list[Path]:
    selected: list[Path] = []
    for path in root.rglob(f"*{suffix}"):
        try:
            if suffix == ".csv":
                file_date = dt.date.fromisoformat(path.stem)
            else:
                file_date = dt.date(int(path.parts[-4]), int(path.parts[-3]), int(path.parts[-2]))
        except (ValueError, IndexError):
            continue
        if file_date >= cutoff:
            selected.append(path)
    return selected


def sync_inputs(
    source_root: Path,
    target_root: Path,
    *,
    as_of: dt.date | None = None,
    keep_days: int = 45,
) -> tuple[int, int]:
    as_of = as_of or dt.date.today()
    cutoff = as_of - dt.timedelta(days=max(1, keep_days))
    copied = 0
    considered = 0

    station_source = source_root / "observations/bcws/datamart/stations/current_stations.csv"
    if station_source.exists():
        considered += 1
        copied += int(
            _copy_if_changed(
                station_source,
                target_root / "observations/bcws/datamart/stations/current_stations.csv",
            )
        )

    daily_source = source_root / "observations/bcws/datamart/daily"
    for source in _dated_paths(daily_source, ".csv", cutoff):
        considered += 1
        relative = source.relative_to(daily_source)
        copied += int(_copy_if_changed(source, target_root / "observations/bcws/datamart/daily" / relative))

    snapshot_source = source_root / "observations/bcws/danger_summaries/snapshots"
    for source in _dated_paths(snapshot_source, ".json", cutoff):
        considered += 1
        relative = source.relative_to(snapshot_source)
        copied += int(
            _copy_if_changed(
                source,
                target_root / "observations/bcws/danger_summaries/snapshots" / relative,
            )
        )
    return copied, considered


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=verification.DEFAULT_CONCRETE_DATA_ROOT)
    parser.add_argument("--keep-days", type=int, default=45)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    copied, considered = sync_inputs(args.source_root, args.target_root, keep_days=args.keep_days)
    print(f"BCWS fire-danger mirror: copied {copied} of {considered} current input files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
