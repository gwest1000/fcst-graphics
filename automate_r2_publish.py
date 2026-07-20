#!/usr/bin/env python3
"""Independent, retryable R2 publication worker for one forecast model."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import time
from pathlib import Path
from typing import Iterable

from r2_publish import MODEL_PRODUCTS, R2ConfigurationError, publish_model

LOCK_ROOT = Path("logs/state")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_PRODUCTS), required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--sync-retained", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(list(argv))


@contextlib.contextmanager
def model_lock(model: str):
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOCK_ROOT / f"r2_{model}.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"R2 publisher for {model} is already running; skipping.", flush=True)
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_once(args: argparse.Namespace) -> bool:
    try:
        publish_model(args.model, stamp=args.stamp, sync_retained=args.sync_retained or not args.stamp)
        return True
    except R2ConfigurationError as exc:
        print(f"R2 publisher is not configured: {exc}", flush=True)
        return False
    except Exception as exc:
        print(f"R2 publication failed for {args.model}: {exc}", flush=True)
        return False


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    with model_lock(args.model) as acquired:
        if not acquired:
            return 0
        if args.once:
            return 0 if run_once(args) else 1
        while True:
            run_once(args)
            time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

