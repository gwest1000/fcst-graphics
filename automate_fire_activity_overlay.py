#!/usr/bin/env python3
"""Hourly retrieval and R2 publication for the live fire-activity layer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import fire_activity
import fire_activity_overlay


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=fire_activity_overlay.OUTPUT_DIR)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--force-upload", action="store_true")
    return parser.parse_args(list(argv))


def log(message: str) -> None:
    print(message, flush=True)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    with fire_activity_overlay.overlay_lock() as acquired:
        if not acquired:
            log("Fire-activity overlay refresh is already running; skipping.")
            return 0
        activity = fire_activity.load_fire_activity(logger=log)
        if args.render_only:
            if activity is None:
                log("No usable fire-activity observations are available.")
                return 0
            paths = fire_activity_overlay.render_overlays(activity, args.output_dir)
            log(f"Rendered {len(paths)} fire-activity overlay(s) from {len(activity.observations)} observations.")
            return 0
        result = fire_activity_overlay.publish_overlays(
            activity,
            output_dir=args.output_dir,
            force_upload=args.force_upload,
        )
        log(
            "Fire-activity overlay: "
            f"observations={result['observations']}, uploaded={result['uploaded']}, "
            f"unchanged={result['unchanged']}, manifest={result['manifest']}"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
