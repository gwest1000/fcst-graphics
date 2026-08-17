#!/usr/bin/env python3
"""Access compact ECMWF ENS fields owned by the concrete forecast archive."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eccodes import codes_get, codes_grib_new_from_file, codes_release

DEFAULT_CONCRETE_REPO = Path(
    os.environ.get("CONCRETE_FCST_REPO_ROOT", "/Users/greg/projects/concrete_fcst")
).expanduser()
DEFAULT_CONCRETE_PYTHON = Path(
    os.environ.get("CONCRETE_FCST_PYTHON", str(DEFAULT_CONCRETE_REPO / ".venv/bin/python"))
).expanduser()
PRODUCT_TYPES = ("em", "es")


def concrete_archive_root(repo_root: Path = DEFAULT_CONCRETE_REPO) -> Path:
    override = os.environ.get("CONCRETE_FCST_DATA_ROOT", "").strip()
    if override:
        data_root = Path(os.path.expandvars(override)).expanduser()
    else:
        config_path = repo_root / "configs/project.toml"
        with config_path.open("rb") as handle:
            configured = tomllib.load(handle)["paths"]["data_root"]
        data_root = Path(os.path.expandvars(str(configured))).expanduser()
        if not data_root.is_absolute():
            data_root = repo_root / data_root
    return data_root.resolve(strict=False) / "raw/ecmwf/realtime"


DEFAULT_ARCHIVE_ROOT = concrete_archive_root()


@dataclass(frozen=True)
class ArchivePaths:
    mean: Path
    spread: Path

    def for_type(self, data_type: str) -> Path:
        if data_type == "em":
            return self.mean
        if data_type == "es":
            return self.spread
        raise ValueError(f"Unsupported ECMWF ensemble statistic: {data_type}")


def archive_paths(
    date_label: str,
    cycle: str,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> ArchivePaths:
    cycle_root = archive_root / date_label / cycle
    return ArchivePaths(
        mean=cycle_root / "gh500_ensemble_mean.grib2",
        spread=cycle_root / "gh500_ensemble_spread.grib2",
    )


def grib_steps(path: Path, expected_type: str) -> set[int]:
    if not path.exists():
        return set()
    steps: set[int] = set()
    try:
        with path.open("rb") as handle:
            while True:
                gid = codes_grib_new_from_file(handle)
                if gid is None:
                    break
                try:
                    if (
                        str(codes_get(gid, "shortName")) == "gh"
                        and str(codes_get(gid, "typeOfLevel")) == "isobaricInhPa"
                        and int(codes_get(gid, "level")) == 500
                        and str(codes_get(gid, "dataType")) == expected_type
                    ):
                        steps.add(int(codes_get(gid, "step")))
                finally:
                    codes_release(gid)
    except (OSError, RuntimeError):
        return set()
    return steps


def archive_has_hours(path: Path, data_type: str, hours: Iterable[int]) -> bool:
    requested = {int(hour) for hour in hours}
    return requested.issubset(grib_steps(path, data_type))


def _download_with_concrete_archive(
    date_label: str,
    cycle: str,
    hours: tuple[int, ...],
    *,
    force: bool,
) -> None:
    command = [
        str(DEFAULT_CONCRETE_PYTHON),
        "-m",
        "concrete_fcst.ingest.ecmwf_ensemble_stats",
        "--repo-root",
        str(DEFAULT_CONCRETE_REPO),
        "--date",
        date_label,
        "--cycle",
        str(int(cycle)),
        "--hours",
        *(str(hour) for hour in hours),
    ]
    if force:
        command.append("--force")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Concrete ECMWF ensemble-statistics archive failed: {detail}")


def ensure_archives(
    date_label: str,
    cycle: str,
    hours: Iterable[int],
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    force: bool = False,
) -> ArchivePaths:
    requested = tuple(sorted(set(int(hour) for hour in hours)))
    if not requested:
        raise ValueError("At least one ECMWF ENS forecast hour is required.")
    if archive_root.resolve(strict=False) != DEFAULT_ARCHIVE_ROOT.resolve(strict=False):
        raise ValueError(
            "ECMWF ensemble-statistics downloads are owned by concrete_fcst; "
            f"use its archive root at {DEFAULT_ARCHIVE_ROOT}."
        )
    paths = archive_paths(date_label, cycle, archive_root)
    if force or any(
        not archive_has_hours(paths.for_type(data_type), data_type, requested)
        for data_type in PRODUCT_TYPES
    ):
        _download_with_concrete_archive(
            date_label,
            cycle,
            requested,
            force=force,
        )
    for data_type in PRODUCT_TYPES:
        if not archive_has_hours(paths.for_type(data_type), data_type, requested):
            raise RuntimeError(
                f"Concrete archive is incomplete for ECMWF {data_type} hours {requested[0]}-{requested[-1]}."
            )
    return paths
