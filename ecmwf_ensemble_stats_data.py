#!/usr/bin/env python3
"""Download compact ECMWF ENS 500 hPa mean and spread fields."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eccodes import codes_get, codes_grib_new_from_file, codes_release

import project_paths


DEFAULT_ARCHIVE_ROOT = project_paths.data_path("ecmwf_ensemble_stats")
PRODUCT_TYPES = ("em", "es")


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


def _download_statistic(
    target: Path,
    data_type: str,
    date_label: str,
    cycle: str,
    hours: tuple[int, ...],
) -> None:
    try:
        from ecmwf.opendata import Client
    except ImportError as exc:
        raise RuntimeError("ecmwf-opendata is required for ECMWF ENS downloads.") from exc

    request = {
        "date": date_label,
        "time": int(cycle),
        "stream": "enfo",
        "type": data_type,
        "step": list(hours),
        "levtype": "pl",
        "levelist": [500],
        "param": ["gh"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    with tempfile.TemporaryDirectory(prefix=f"ecmwf-gh500-{data_type}-") as directory:
        temporary = Path(directory) / target.name
        # Google Cloud does not accept the multi-range request used by ECMWF's
        # aggregated ensemble-statistics files. ECMWF and AWS do.
        for source in ("ecmwf", "aws"):
            temporary.unlink(missing_ok=True)
            try:
                Client(source=source, model="ifs", resol="0p25").retrieve(
                    request=request,
                    target=str(temporary),
                )
                if not archive_has_hours(temporary, data_type, hours):
                    raise RuntimeError(
                        f"Downloaded {data_type} GRIB does not contain every requested forecast hour."
                    )
                staged = target.with_suffix(target.suffix + ".tmp")
                staged.unlink(missing_ok=True)
                shutil.copy2(temporary, staged)
                staged.replace(target)
                return
            except Exception as exc:
                last_error = exc
    assert last_error is not None
    raise RuntimeError(
        f"ECMWF ENS 500 hPa {data_type} download failed: {last_error}"
    ) from last_error


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
    paths = archive_paths(date_label, cycle, archive_root)
    for data_type in PRODUCT_TYPES:
        target = paths.for_type(data_type)
        if force or not archive_has_hours(target, data_type, requested):
            _download_statistic(target, data_type, date_label, cycle, requested)
    return paths
