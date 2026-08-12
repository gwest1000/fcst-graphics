#!/usr/bin/env python3
"""Download and retain BC-only ECMWF control convective diagnostics."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release


ARCHIVE_VERSION = 2
PARAMETERS = ("mucape", "sp")
# Retain a small halo around the plotted domain for stable edge interpolation.
REGIONAL_EXTENT = (-141.0, -106.5, 43.5, 60.75)
ARCHIVE_NAME = "convective_control_bc.npz"


@dataclass(frozen=True)
class RegionalConvectiveArchive:
    steps: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    mucape: np.ndarray
    surface_pressure_pa: np.ndarray

    def field(self, short_name: str, fhour: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matches = np.flatnonzero(self.steps == int(fhour))
        if matches.size != 1:
            raise KeyError(f"F{fhour:03d} is not present in the ECMWF convective archive.")
        if short_name == "mucape":
            data = self.mucape[int(matches[0])]
        elif short_name == "sp":
            data = self.surface_pressure_pa[int(matches[0])]
        else:
            raise KeyError(short_name)
        return data, self.lat, self.lon


def archive_path(data_root: Path, date_label: str, cycle: str) -> Path:
    return data_root / "raw" / "ecmwf" / "realtime" / date_label / cycle / ARCHIVE_NAME


def _regional_slices(
    lat: np.ndarray,
    lon: np.ndarray,
    extent: tuple[float, float, float, float] = REGIONAL_EXTENT,
) -> tuple[slice, slice]:
    west, east, south, north = extent
    mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lon >= west)
        & (lon <= east)
        & (lat >= south)
        & (lat <= north)
    )
    rows, columns = np.where(mask)
    if rows.size == 0 or columns.size == 0:
        raise RuntimeError(f"ECMWF grid does not intersect regional extent {extent}.")
    return slice(int(rows.min()), int(rows.max()) + 1), slice(int(columns.min()), int(columns.max()) + 1)


def extract_regional_archive(
    grib_paths: Path | Iterable[Path],
    target: Path,
    requested_steps: Iterable[int],
) -> Path:
    requested = tuple(sorted(set(int(step) for step in requested_steps)))
    fields: dict[str, dict[int, np.ndarray]] = {name: {} for name in PARAMETERS}
    regional_lat: np.ndarray | None = None
    regional_lon: np.ndarray | None = None

    paths = (grib_paths,) if isinstance(grib_paths, Path) else tuple(grib_paths)
    for grib_path in paths:
        with grib_path.open("rb") as handle:
            while True:
                gid = codes_grib_new_from_file(handle)
                if gid is None:
                    break
                try:
                    short_name = str(codes_get(gid, "shortName"))
                    step = int(codes_get(gid, "step"))
                    if short_name not in fields or step not in requested:
                        continue
                    nx = int(codes_get(gid, "Ni"))
                    ny = int(codes_get(gid, "Nj"))
                    values = codes_get_array(gid, "values").reshape(ny, nx).astype(np.float32)
                    lat = codes_get_array(gid, "latitudes").reshape(ny, nx).astype(np.float32)
                    lon = codes_get_array(gid, "longitudes").reshape(ny, nx).astype(np.float32)
                    lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
                    yslice, xslice = _regional_slices(lat, lon)
                    if regional_lat is None:
                        regional_lat = lat[yslice, xslice]
                        regional_lon = lon[yslice, xslice]
                    fields[short_name][step] = values[yslice, xslice]
                finally:
                    codes_release(gid)

    missing = [
        f"{name}:F{step:03d}"
        for name in PARAMETERS
        for step in requested
        if step not in fields[name]
    ]
    if missing:
        raise RuntimeError("Missing ECMWF convective fields: " + ", ".join(missing[:12]))
    if regional_lat is None or regional_lon is None:
        raise RuntimeError("No ECMWF convective fields were extracted.")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                version=np.asarray([ARCHIVE_VERSION], dtype=np.int16),
                steps=np.asarray(requested, dtype=np.int16),
                lat=regional_lat,
                lon=regional_lon,
                mucape=np.stack([fields["mucape"][step] for step in requested]),
                surface_pressure_pa=np.stack([fields["sp"][step] for step in requested]),
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_archive(path: Path) -> RegionalConvectiveArchive:
    with np.load(path) as data:
        if int(data["version"][0]) != ARCHIVE_VERSION:
            raise ValueError(f"Unsupported ECMWF convective archive version: {path}")
        return RegionalConvectiveArchive(
            steps=data["steps"].astype(np.int16),
            lat=data["lat"].astype(np.float32),
            lon=data["lon"].astype(np.float32),
            mucape=data["mucape"].astype(np.float32),
            surface_pressure_pa=data["surface_pressure_pa"].astype(np.float32),
        )


def archive_has_hours(path: Path, hours: Iterable[int]) -> bool:
    if not path.exists():
        return False
    try:
        archive = load_archive(path)
    except (OSError, KeyError, ValueError):
        return False
    return set(int(value) for value in hours).issubset(int(value) for value in archive.steps)


def ensure_archive(
    data_root: Path,
    date_label: str,
    cycle: str,
    hours: Iterable[int],
    force: bool = False,
) -> Path:
    requested = tuple(sorted(set(int(hour) for hour in hours)))
    target = archive_path(data_root, date_label, cycle)
    if not force and archive_has_hours(target, requested):
        return target
    if target.exists() and not force:
        try:
            requested = tuple(sorted(set(requested) | set(int(value) for value in load_archive(target).steps)))
        except (OSError, KeyError, ValueError):
            pass

    try:
        from ecmwf.opendata import Client
    except ImportError as exc:
        raise RuntimeError("ecmwf-opendata is required for ECMWF convective downloads.") from exc

    request = {
        "date": date_label,
        "time": int(cycle),
        "stream": "oper",
        "type": "fc",
        "step": list(requested),
        "levtype": "sfc",
    }
    last_error: Exception | None = None
    with tempfile.TemporaryDirectory(prefix="ecmwf-convective-") as directory:
        for source in ("google", "aws", "azure"):
            global_gribs = [Path(directory) / f"{parameter}_global.grib2" for parameter in PARAMETERS]
            try:
                client = Client(source=source, model="ifs")
                for parameter, global_grib in zip(PARAMETERS, global_gribs):
                    client.retrieve({**request, "param": parameter}, str(global_grib))
                return extract_regional_archive(global_gribs, target, requested)
            except Exception as exc:
                last_error = exc
                for global_grib in global_gribs:
                    global_grib.unlink(missing_ok=True)
    assert last_error is not None
    raise RuntimeError(f"ECMWF convective download failed: {last_error}") from last_error
