#!/usr/bin/env python3
"""Download static surface geopotential for an ECMWF IFS control cycle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from ecmwf.opendata import Client


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Forecast date in YYYYMMDD format.")
    parser.add_argument("--cycle", required=True, type=int, choices=[0, 6, 12, 18])
    parser.add_argument("--target", required=True, type=Path)
    return parser.parse_args(list(argv))


def download(date: str, cycle: int, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    errors: list[str] = []
    try:
        for source in ("google", "aws"):
            try:
                Client(source=source, model="ifs").retrieve(
                    date=date,
                    time=cycle,
                    stream="oper",
                    type="fc",
                    step=0,
                    levtype="sfc",
                    param=["z"],
                    target=str(temporary),
                )
                temporary.replace(target)
                return
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                errors.append(f"{source}: {exc}")
    finally:
        temporary.unlink(missing_ok=True)
    raise RuntimeError("ECMWF terrain download failed; " + "; ".join(errors))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    download(args.date, args.cycle, args.target)
    print(f"Wrote {args.target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
