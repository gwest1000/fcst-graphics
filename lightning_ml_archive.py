#!/usr/bin/env python3
"""Build compact HRDPS and observed-lightning archives for LPI development."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, TiffImagePlugin

import make_hrdps_west_convective as hrdps

ARCHIVE_SCHEMA_VERSION = 1
MODEL_ARCHIVE_CYCLES = frozenset({"00", "12"})
MODEL_FORECAST_HOURS = tuple(range(0, 49, 3))
MODEL_RESOLUTION_KM = 5.0
OBS_BLOCK_HOURS = 3
OBS_SOURCE_MINUTES = 10
OBS_SOURCES_PER_BLOCK = OBS_BLOCK_HOURS * 60 // OBS_SOURCE_MINUTES
OBS_RE = re.compile(r"(?P<stamp>\d{8}T\d{4}Z)_MSC_Lightning_2\.5km\.tif$")
DEFAULT_ARCHIVE_ROOT = Path(
    os.environ.get(
        "LIGHTNING_ML_ARCHIVE_ROOT",
        "/Volumes/Greg1_2tb/concrete_fcst_data/derived/lightning_ml",
    )
)
DEFAULT_OBS_DIR = Path("data/lightning_obs")
DEFAULT_MODEL_DATA_DIR = Path("data/hrdps_continental")
DEFAULT_LPI_CACHE_DIR = Path("plots/hrdps_continental_lightning")
FILL_VALUE = np.int16(-32768)
PROFILE_LEVELS_HPA = (
    1000,
    985,
    970,
    950,
    925,
    900,
    875,
    850,
    800,
    750,
    700,
    650,
    600,
    550,
    500,
    450,
    400,
    350,
    300,
    250,
)


@dataclass(frozen=True)
class FieldSpec:
    key: str
    variable: str
    level_tag: str
    units: str
    offset: float
    scale: float
    optional_at_f000: bool = False


def _profile_specs() -> list[FieldSpec]:
    specs: list[FieldSpec] = []
    for level in PROFILE_LEVELS_HPA:
        specs.append(FieldSpec(f"tmp_{level}", "TMP", f"ISBL_{level:04d}", "K", 273.15, 0.05))
        specs.append(FieldSpec(f"spfh_{level}", "SPFH", f"ISBL_{level:04d}", "kg kg-1", 0.0, 1.0e-5))
    return specs


MODEL_FIELD_SPECS = tuple(
    _profile_specs()
    + [
        FieldSpec(f"ugrd_{level}", "UGRD", f"ISBL_{level:04d}", "m s-1", 0.0, 0.1)
        for level in (850, 700, 500, 250)
    ]
    + [
        FieldSpec(f"vgrd_{level}", "VGRD", f"ISBL_{level:04d}", "m s-1", 0.0, 0.1)
        for level in (850, 700, 500, 250)
    ]
    + [
        FieldSpec(f"vvel_{level}", "VVEL", f"ISBL_{level:04d}", "Pa s-1", 0.0, 0.01)
        for level in (1000, 850, 700, 500, 250)
    ]
    + [
        FieldSpec("cape", "CAPE", "Sfc", "J kg-1", 0.0, 1.0),
        FieldSpec("mu_li", "MU-VT-LI", "ISBL_0500", "K", 0.0, 0.05),
        FieldSpec("lifted_index", "LFTX", "ISBL_0500", "K", 0.0, 0.05),
        FieldSpec("surface_pressure", "PRES", "Sfc", "Pa", 100000.0, 10.0),
        FieldSpec("mslp", "PRMSL", "MSL", "Pa", 100000.0, 10.0),
        FieldSpec("tmp_2m", "TMP", "AGL-2m", "K", 273.15, 0.05),
        FieldSpec("dpt_2m", "DPT", "AGL-2m", "K", 273.15, 0.05),
        FieldSpec("ugrd_10m", "UGRD", "AGL-10m", "m s-1", 0.0, 0.1),
        FieldSpec("vgrd_10m", "VGRD", "AGL-10m", "m s-1", 0.0, 0.1),
        FieldSpec("pbl_height", "HPBL", "Sfc", "m", 0.0, 1.0),
        FieldSpec("storm_relative_helicity", "HLCY", "Sfc", "m2 s-2", 0.0, 0.1),
        FieldSpec("precip_rate", "PRATE", "Sfc", "kg m-2 s-1", 0.0, 2.0e-6, True),
        FieldSpec("precip_accum", "APCP", "Sfc", "kg m-2", 0.0, 0.05, True),
        FieldSpec("column_cloud_water", "CWAT", "EATM", "kg m-2", 0.0, 0.01),
        FieldSpec("total_cloud_cover", "TCDC", "Sfc", "%", 0.0, 0.01),
        FieldSpec("height_500", "HGT", "ISBL_0500", "gpm", 5500.0, 0.5),
        FieldSpec("absolute_vorticity_500", "ABSV", "ISBL_0500", "s-1", 0.0, 1.0e-6),
    ]
)


def utc_iso(timestamp: dt.datetime) -> str:
    return timestamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(message, flush=True)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_archive_root(root: Path) -> Path:
    root = root.expanduser()
    if root.is_absolute() and len(root.parts) >= 3 and root.parts[1] == "Volumes":
        mount = Path("/") / root.parts[1] / root.parts[2]
        if not mount.is_mount():
            raise RuntimeError(f"Lightning ML archive volume is not mounted: {mount}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def verify_archive_writable(root: Path) -> Path:
    """Fail fast when a scheduled process cannot write the archive volume."""
    root = root.expanduser()
    probe = root / f".lightning_ml_write_probe.{os.getpid()}"
    try:
        root = ensure_archive_root(root)
        probe = root / probe.name
        with probe.open("xb") as handle:
            handle.write(b"ok\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError(
            f"Lightning ML archive is not writable by this process: {root}. "
            "On macOS, grant Removable Volumes or Full Disk Access to the exact "
            f"Python runtime used by launchd. Original error: {exc}"
        ) from exc
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
    return root


def observation_archive_dir(root: Path) -> Path:
    return root / "observations" / "eccc_lightning_3h" / f"schema_v{ARCHIVE_SCHEMA_VERSION}"


def model_archive_dir(root: Path) -> Path:
    return root / "model" / "hrdps_continental_5km" / f"schema_v{ARCHIVE_SCHEMA_VERSION}"


def baseline_archive_dir(root: Path) -> Path:
    return root / "baseline" / "hrdps_continental_lpi_5km" / f"schema_v{ARCHIVE_SCHEMA_VERSION}"


def parse_obs_time(path_or_name: str | Path) -> dt.datetime | None:
    name = Path(path_or_name).name
    match = OBS_RE.match(name)
    if not match:
        return None
    return dt.datetime.strptime(match.group("stamp"), "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc)


def observation_block_end(timestamp: dt.datetime) -> dt.datetime:
    timestamp = timestamp.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)
    if timestamp.minute == 0 and timestamp.hour % OBS_BLOCK_HOURS == 0:
        return timestamp
    block_hour = timestamp.hour - timestamp.hour % OBS_BLOCK_HOURS
    return timestamp.replace(hour=block_hour, minute=0) + dt.timedelta(hours=OBS_BLOCK_HOURS)


def observation_block_times(end: dt.datetime) -> tuple[dt.datetime, ...]:
    end = end.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)
    start = end - dt.timedelta(hours=OBS_BLOCK_HOURS)
    return tuple(start + dt.timedelta(minutes=OBS_SOURCE_MINUTES * index) for index in range(1, OBS_SOURCES_PER_BLOCK + 1))


def observation_source_path(obs_dir: Path, timestamp: dt.datetime) -> Path:
    return obs_dir / f"{timestamp:%Y%m%d}" / f"{timestamp:%Y%m%dT%H%MZ}_MSC_Lightning_2.5km.tif"


def observation_aggregate_path(root: Path, end: dt.datetime) -> Path:
    archive_dir = observation_archive_dir(root)
    return (
        archive_dir
        / f"{end:%Y}"
        / f"{end:%Y%m%d}"
        / f"{end:%Y%m%dT%H%MZ}_MSC_LightningDensity_3h_2.5km.tif"
    )


def expected_observation_aggregate_paths(root: Path, start: dt.datetime, end: dt.datetime) -> tuple[Path, ...]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Observation window times must be timezone aware.")
    timestamp = start.astimezone(dt.timezone.utc) + dt.timedelta(hours=OBS_BLOCK_HOURS)
    end = end.astimezone(dt.timezone.utc)
    paths: list[Path] = []
    while timestamp <= end:
        paths.append(observation_aggregate_path(root, timestamp))
        timestamp += dt.timedelta(hours=OBS_BLOCK_HOURS)
    return tuple(paths)


def observation_block_archived(root: Path, timestamp: dt.datetime) -> bool:
    path = observation_aggregate_path(root, observation_block_end(timestamp))
    return path.exists() and path.with_suffix(".json").exists()


def _copy_geotiff_tags(image: Image.Image) -> TiffImagePlugin.ImageFileDirectory_v2:
    info = TiffImagePlugin.ImageFileDirectory_v2()
    for tag in (33550, 33922, 34735, 34736, 34737):
        value = image.tag_v2.get(tag)
        if value is not None:
            info[tag] = value
    info[42113] = "-999"
    return info


def aggregate_observation_block(
    obs_dir: Path,
    root: Path,
    end: dt.datetime,
    delete_sources: bool = True,
) -> Path | None:
    root = ensure_archive_root(root)
    timestamps = observation_block_times(end)
    sources = tuple(observation_source_path(obs_dir, timestamp) for timestamp in timestamps)
    if not all(path.exists() and path.stat().st_size > 0 for path in sources):
        return None

    out_path = observation_aggregate_path(root, end)
    sidecar_path = out_path.with_suffix(".json")
    if out_path.exists() and sidecar_path.exists():
        if delete_sources:
            for source in sources:
                source.unlink(missing_ok=True)
        return out_path

    total: np.ndarray | None = None
    valid_count: np.ndarray | None = None
    tiff_info: TiffImagePlugin.ImageFileDirectory_v2 | None = None
    image_size: tuple[int, int] | None = None
    for source in sources:
        with Image.open(source) as image:
            if image_size is None:
                image_size = image.size
                tiff_info = _copy_geotiff_tags(image)
            elif image.size != image_size:
                raise RuntimeError(f"Lightning grid changed within block ending {utc_iso(end)}.")
            nodata = float(image.tag_v2.get(42113, -999.0))
            data = np.asarray(image, dtype=np.float32)
        valid = np.isfinite(data) & (data != nodata)
        if total is None:
            total = np.zeros(data.shape, dtype=np.float32)
            valid_count = np.zeros(data.shape, dtype=np.uint8)
        total += np.where(valid & (data > 0.0), data * OBS_SOURCE_MINUTES, 0.0)
        valid_count += valid.astype(np.uint8)

    assert total is not None and valid_count is not None and tiff_info is not None
    aggregate = np.where(valid_count == OBS_SOURCES_PER_BLOCK, total, -999.0).astype(np.float32)
    tiff_info[42112] = (
        "<GDALMetadata>"
        '<Item name="DESCRIPTION" sample="0">Three-hour total lightning flash density from 18 ECCC ten-minute grids</Item>'
        '<Item name="UNITS" sample="0">Flash/km**2</Item>'
        f'<Item name="WINDOW_START_DATETIME" sample="0">{utc_iso(end - dt.timedelta(hours=3))}</Item>'
        f'<Item name="WINDOW_END_DATETIME" sample="0">{utc_iso(end)}</Item>'
        "</GDALMetadata>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    try:
        Image.fromarray(aggregate).save(
            tmp_path,
            format="TIFF",
            compression="tiff_adobe_deflate",
            tiffinfo=tiff_info,
        )
        with Image.open(tmp_path) as check:
            if check.size != image_size or check.mode != "F":
                raise RuntimeError(f"Failed verification of aggregate lightning file {tmp_path}.")
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    finite = aggregate != -999.0
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "window_start_utc": utc_iso(end - dt.timedelta(hours=OBS_BLOCK_HOURS)),
        "window_end_utc": utc_iso(end),
        "source_interval_minutes": OBS_SOURCE_MINUTES,
        "source_count": len(sources),
        "source_files": [path.name for path in sources],
        "units": "flash km-2 per 3h",
        "shape": list(aggregate.shape),
        "valid_cells": int(np.count_nonzero(finite)),
        "nonzero_cells": int(np.count_nonzero(finite & (aggregate > 0.0))),
        "max_flash_km2": float(np.max(aggregate[finite])) if np.any(finite) else None,
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
        "created_at_utc": utc_iso(dt.datetime.now(dt.timezone.utc)),
    }
    write_json_atomic(sidecar_path, payload)

    if delete_sources:
        for source in sources:
            source.unlink(missing_ok=True)
        for directory in sorted({path.parent for path in sources}):
            try:
                directory.rmdir()
            except OSError:
                pass
    return out_path


def aggregate_available_observations(obs_dir: Path, root: Path, delete_sources: bool = True) -> list[Path]:
    ends: set[dt.datetime] = set()
    for path in obs_dir.glob("*/*_MSC_Lightning_2.5km.tif"):
        timestamp = parse_obs_time(path)
        if timestamp is not None:
            ends.add(observation_block_end(timestamp))
    outputs: list[Path] = []
    for end in sorted(ends):
        path = aggregate_observation_block(obs_dir, root, end, delete_sources=delete_sources)
        if path is not None:
            outputs.append(path)
    if outputs:
        log(f"Archived {len(outputs)} complete three-hour lightning block(s).")
    return outputs


def should_archive_model_run(model_key: str, cycle: str) -> bool:
    return model_key == "continental" and cycle in MODEL_ARCHIVE_CYCLES


def model_source_filename(spec: FieldSpec, stamp: str, fhour: int) -> str:
    return f"{stamp}_MSC_HRDPS_{spec.variable}_{spec.level_tag}_RLatLon0.0225_PT{fhour:03d}H.grib2"


def model_field_specs(fhour: int) -> tuple[FieldSpec, ...]:
    return tuple(spec for spec in MODEL_FIELD_SPECS if not (fhour == 0 and spec.optional_at_f000))


def required_model_names(stamp: str, fhour: int) -> tuple[str, ...]:
    return tuple(model_source_filename(spec, stamp, fhour) for spec in model_field_specs(fhour))


def _pack_field(data: np.ndarray, spec: FieldSpec) -> tuple[np.ndarray, int]:
    output = np.full(data.shape, FILL_VALUE, dtype=np.int16)
    valid = np.isfinite(data)
    if not np.any(valid):
        return output, 0
    scaled = np.rint((data[valid] - spec.offset) / spec.scale)
    clipped = int(np.count_nonzero((scaled < -32767.0) | (scaled > 32767.0)))
    output[valid] = np.clip(scaled, -32767.0, 32767.0).astype(np.int16)
    return output, clipped


def unpack_field(data: np.ndarray, spec: FieldSpec) -> np.ndarray:
    return np.where(data == FILL_VALUE, np.nan, data.astype(np.float32) * spec.scale + spec.offset)


def _grid_hash(lat: np.ndarray, lon: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(lat, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(lon, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _write_model_schema(root: Path) -> None:
    path = model_archive_dir(root) / "schema.json"
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "model": "ECCC HRDPS continental 2.5 km",
        "archived_grid_resolution_km": MODEL_RESOLUTION_KM,
        "eligible_cycles_utc": sorted(MODEL_ARCHIVE_CYCLES),
        "forecast_hours": list(MODEL_FORECAST_HOURS),
        "fill_value": int(FILL_VALUE),
        "packing_formula": "physical_value = packed_int16 * scale + offset",
        "fields": [asdict(spec) for spec in MODEL_FIELD_SPECS],
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError(f"Archive schema mismatch at {path}; increment the schema version before changing fields.")
        return
    write_json_atomic(path, payload)


def _ensure_grid_archive(root: Path, lat: np.ndarray, lon: np.ndarray) -> str:
    static_dir = model_archive_dir(root) / "static"
    path = static_dir / "grid.npz"
    grid_hash = _grid_hash(lat, lon)
    if path.exists():
        with np.load(path) as archived:
            archived_hash = str(archived["grid_hash"].item())
        if archived_hash != grid_hash:
            raise RuntimeError(f"HRDPS archive grid changed: {archived_hash} != {grid_hash}.")
        return grid_hash

    static_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.int16),
                grid_hash=np.asarray(grid_hash),
                lat=lat.astype(np.float32, copy=False),
                lon=lon.astype(np.float32, copy=False),
            )
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return grid_hash


def _ensure_terrain_archive(
    root: Path,
    run_dir: Path,
    stamp: str,
    yslice: slice,
    xslice: slice,
    stride: int,
    grid_hash: str,
) -> None:
    path = model_archive_dir(root) / "static" / "terrain.npz"
    if path.exists():
        return
    source = run_dir / "003" / f"{stamp}_MSC_HRDPS_HGT_Sfc_RLatLon0.0225_PT003H.grib2"
    if not source.exists():
        return
    terrain, _, _ = hrdps.read_grib(source)
    sample = terrain[yslice, xslice][::stride, ::stride]
    packed = np.full(sample.shape, FILL_VALUE, dtype=np.int16)
    valid = np.isfinite(sample)
    packed[valid] = np.clip(np.rint(sample[valid]), -32767.0, 32767.0).astype(np.int16)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.int16),
                grid_hash=np.asarray(grid_hash),
                elevation_m=packed,
                fill_value=np.asarray([FILL_VALUE], dtype=np.int16),
            )
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def model_run_archive_dir(root: Path, stamp: str) -> Path:
    return model_archive_dir(root) / stamp[:4] / stamp


def model_hour_archive_path(root: Path, stamp: str, fhour: int) -> Path:
    return model_run_archive_dir(root, stamp) / f"f{fhour:03d}.npz"


def baseline_hour_archive_path(root: Path, stamp: str, fhour: int) -> Path:
    return baseline_archive_dir(root) / stamp[:4] / stamp / f"f{fhour:03d}.npz"


def archive_lpi_baseline_hour(root: Path, run: hrdps.RunInfo, cache_path: Path, fhour: int) -> Path:
    if not should_archive_model_run("continental", run.cycle):
        raise ValueError(f"Run {run.stamp} is not an eligible LPI baseline archive cycle.")
    root = ensure_archive_root(root)
    out_path = baseline_hour_archive_path(root, run.stamp, fhour)
    if out_path.exists():
        return out_path
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    with np.load(cache_path) as cache:
        potential = cache["potential"].astype(np.float32)
        formula_version = str(cache["formula_version"].item()) if "formula_version" in cache else "unknown"
        cache_grid_hash = _grid_hash(cache["lat"].astype(np.float32), cache["lon"].astype(np.float32))
    packed = np.full(potential.shape, 255, dtype=np.uint8)
    valid = np.isfinite(potential)
    packed[valid] = np.clip(np.rint(potential[valid] * 2.0), 0.0, 200.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.int16),
                run_stamp=np.asarray(run.stamp),
                forecast_hour=np.asarray([fhour], dtype=np.int16),
                valid_unix=np.asarray([int((run.init_time + dt.timedelta(hours=fhour)).timestamp())], dtype=np.int64),
                formula_version=np.asarray(formula_version),
                cache_grid_hash=np.asarray(cache_grid_hash),
                potential_x2=packed,
                fill_value=np.asarray([255], dtype=np.uint8),
                scale=np.asarray([0.5], dtype=np.float32),
            )
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    write_json_atomic(
        out_path.with_suffix(".json"),
        {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "run_stamp": run.stamp,
            "forecast_hour": fhour,
            "valid_time_utc": utc_iso(run.init_time + dt.timedelta(hours=fhour)),
            "formula_version": formula_version,
            "packing_formula": "potential_percent = potential_x2 * 0.5; 255 is missing",
            "cache_grid_hash": cache_grid_hash,
            "bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "created_at_utc": utc_iso(dt.datetime.now(dt.timezone.utc)),
        },
    )
    return out_path


def archive_lpi_baseline_hours(
    root: Path,
    run: hrdps.RunInfo,
    lightning_output_dir: Path,
    hours: Iterable[int],
) -> list[Path]:
    outputs: list[Path] = []
    cache_dir = lightning_output_dir / run.stamp / "lpi_cache"
    for fhour in sorted(set(int(hour) for hour in hours)):
        cache_path = cache_dir / f"hrdps_continental_lightning_{run.stamp}_f{fhour:03d}_lpi.npz"
        if not cache_path.exists():
            continue
        outputs.append(archive_lpi_baseline_hour(root, run, cache_path, fhour))
    if outputs:
        log(f"Archived {len(outputs)} handmade-LPI baseline hour(s) for {run.stamp}.")
    return outputs


def _update_run_manifest(root: Path, run: hrdps.RunInfo, grid_hash: str) -> Path:
    run_archive_dir = model_run_archive_dir(root, run.stamp)
    archived_hours = [hour for hour in MODEL_FORECAST_HOURS if model_hour_archive_path(root, run.stamp, hour).exists()]
    total_bytes = sum(model_hour_archive_path(root, run.stamp, hour).stat().st_size for hour in archived_hours)
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "model_key": "continental",
        "model_label": "ECCC HRDPS continental 2.5 km",
        "archive_resolution_km": MODEL_RESOLUTION_KM,
        "run_stamp": run.stamp,
        "cycle_utc": run.cycle,
        "init_time_utc": utc_iso(run.init_time),
        "expected_hours": list(MODEL_FORECAST_HOURS),
        "archived_hours": archived_hours,
        "complete": archived_hours == list(MODEL_FORECAST_HOURS),
        "grid_hash": grid_hash,
        "archive_bytes": total_bytes,
        "updated_at_utc": utc_iso(dt.datetime.now(dt.timezone.utc)),
    }
    path = run_archive_dir / "manifest.json"
    write_json_atomic(path, payload)
    return path


def archive_model_hour(root: Path, run: hrdps.RunInfo, data_dir: Path, fhour: int) -> Path:
    if not should_archive_model_run("continental", run.cycle):
        raise ValueError(f"Run {run.stamp} is not an eligible twice-daily continental archive cycle.")
    if fhour not in MODEL_FORECAST_HOURS:
        raise ValueError(f"Unsupported HRDPS archive forecast hour: {fhour}")
    root = ensure_archive_root(root)
    _write_model_schema(root)
    out_path = model_hour_archive_path(root, run.stamp, fhour)
    sidecar_path = out_path.with_suffix(".json")
    if out_path.exists() and sidecar_path.exists():
        return out_path

    source_dir = data_dir / run.stamp / f"{fhour:03d}"
    specs = model_field_specs(fhour)
    missing = [model_source_filename(spec, run.stamp, fhour) for spec in specs if not (source_dir / model_source_filename(spec, run.stamp, fhour)).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot archive {run.stamp} F{fhour:03d}; missing {len(missing)} field(s): {missing[:4]}")

    coordinate_source = source_dir / model_source_filename(
        next(spec for spec in specs if spec.key == "surface_pressure"), run.stamp, fhour
    )
    _, full_lat, full_lon = hrdps.read_grib(coordinate_source, coords=True)
    assert full_lat is not None and full_lon is not None
    yslice, xslice = hrdps.subset_slices(full_lat, full_lon, hrdps.MODEL_CONFIGS["continental"].extent)
    stride = max(1, int(round(MODEL_RESOLUTION_KM / hrdps.MODEL_CONFIGS["continental"].resolution_km)))
    lat = full_lat[yslice, xslice][::stride, ::stride]
    lon = full_lon[yslice, xslice][::stride, ::stride]
    grid_hash = _ensure_grid_archive(root, lat, lon)
    _ensure_terrain_archive(root, data_dir / run.stamp, run.stamp, yslice, xslice, stride, grid_hash)

    arrays: dict[str, np.ndarray] = {}
    clipped_counts: dict[str, int] = {}
    source_files: dict[str, str] = {}
    for spec in specs:
        name = model_source_filename(spec, run.stamp, fhour)
        data, _, _ = hrdps.read_grib(source_dir / name)
        sample = data[yslice, xslice][::stride, ::stride]
        if sample.shape != lat.shape:
            raise RuntimeError(f"Field {name} has shape {sample.shape}; expected {lat.shape}.")
        arrays[spec.key], clipped_counts[spec.key] = _pack_field(sample, spec)
        source_files[spec.key] = name

    valid_time = run.init_time + dt.timedelta(hours=int(fhour))
    arrays.update(
        schema_version=np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.int16),
        run_stamp=np.asarray(run.stamp),
        cycle_utc=np.asarray([int(run.cycle)], dtype=np.int8),
        forecast_hour=np.asarray([int(fhour)], dtype=np.int16),
        init_unix=np.asarray([int(run.init_time.timestamp())], dtype=np.int64),
        valid_unix=np.asarray([int(valid_time.timestamp())], dtype=np.int64),
        grid_hash=np.asarray(grid_hash),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(tmp_path) as check:
            if int(check["forecast_hour"][0]) != fhour or str(check["grid_hash"].item()) != grid_hash:
                raise RuntimeError(f"Failed verification of HRDPS archive {tmp_path}.")
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    write_json_atomic(
        sidecar_path,
        {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "run_stamp": run.stamp,
            "forecast_hour": fhour,
            "valid_time_utc": utc_iso(valid_time),
            "grid_hash": grid_hash,
            "shape": list(lat.shape),
            "field_count": len(specs),
            "source_files": source_files,
            "clipped_counts": clipped_counts,
            "bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "created_at_utc": utc_iso(dt.datetime.now(dt.timezone.utc)),
        },
    )
    _update_run_manifest(root, run, grid_hash)
    return out_path


def archive_model_hours(root: Path, run: hrdps.RunInfo, data_dir: Path, hours: Iterable[int]) -> list[Path]:
    if not should_archive_model_run("continental", run.cycle):
        return []
    outputs = []
    for hour in sorted(set(hours)):
        path = archive_model_hour(root, run, data_dir, int(hour))
        outputs.append(path)
        log(f"  archived {run.stamp} F{int(hour):03d}: {path.stat().st_size / (1024 * 1024):.1f} MiB")
    if outputs:
        log(f"Archived {len(outputs)} HRDPS feature hour(s) for {run.stamp}.")
    return outputs


def run_archive_complete(root: Path, stamp: str) -> bool:
    manifest_path = model_run_archive_dir(root, stamp) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not manifest.get("complete"):
        return False
    return all(model_hour_archive_path(root, stamp, hour).exists() for hour in MODEL_FORECAST_HOURS)


def archive_status(root: Path) -> dict[str, object]:
    obs_dir = observation_archive_dir(root)
    model_dir = model_archive_dir(root)
    obs_paths = sorted(obs_dir.glob("*/*/*.tif")) if obs_dir.exists() else []
    manifests = sorted(model_dir.glob("*/*/manifest.json")) if model_dir.exists() else []
    model_hours = sorted(model_dir.glob("*/*/f*.npz")) if model_dir.exists() else []
    baseline_dir = baseline_archive_dir(root)
    baseline_hours = sorted(baseline_dir.glob("*/*/f*.npz")) if baseline_dir.exists() else []
    complete_runs = 0
    latest_run = None
    for path in manifests:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        complete_runs += bool(payload.get("complete"))
        stamp = payload.get("run_stamp")
        if isinstance(stamp, str) and (latest_run is None or stamp > latest_run):
            latest_run = stamp
    usage = shutil.disk_usage(root) if root.exists() else None
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "archive_root": str(root),
        "observation_blocks": len(obs_paths),
        "latest_observation_block": obs_paths[-1].stem.split("_MSC_")[0] if obs_paths else None,
        "model_runs": len(manifests),
        "complete_model_runs": complete_runs,
        "model_hours": len(model_hours),
        "baseline_lpi_hours": len(baseline_hours),
        "latest_model_run": latest_run,
        "archive_bytes": sum(path.stat().st_size for path in (*obs_paths, *model_hours, *baseline_hours)),
        "filesystem_free_bytes": usage.free if usage else None,
        "updated_at_utc": utc_iso(dt.datetime.now(dt.timezone.utc)),
    }


def write_archive_status(root: Path) -> Path:
    root = ensure_archive_root(root)
    path = root / "status.json"
    write_json_atomic(path, archive_status(root))
    return path


def parse_hours(text: str | None) -> tuple[int, ...]:
    if not text:
        return MODEL_FORECAST_HOURS
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    obs_parser = subparsers.add_parser("aggregate-observations")
    obs_parser.add_argument("--obs-dir", type=Path, default=DEFAULT_OBS_DIR)
    obs_parser.add_argument("--keep-sources", action="store_true")

    model_parser = subparsers.add_parser("archive-model")
    model_parser.add_argument("--run", required=True)
    model_parser.add_argument("--data-dir", type=Path, default=DEFAULT_MODEL_DATA_DIR)
    model_parser.add_argument("--hours", default=None)

    baseline_parser = subparsers.add_parser("archive-baseline")
    baseline_parser.add_argument("--run", required=True)
    baseline_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_LPI_CACHE_DIR)
    baseline_parser.add_argument("--hours", default=None)

    subparsers.add_parser("status")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if args.command == "aggregate-observations":
        verify_archive_writable(args.archive_root)
        aggregate_available_observations(args.obs_dir, args.archive_root, delete_sources=not args.keep_sources)
        write_archive_status(args.archive_root)
        return 0
    if args.command == "archive-model":
        verify_archive_writable(args.archive_root)
        init_time = hrdps.parse_stamp(args.run)
        run = hrdps.RunInfo(cycle=f"{init_time:%H}", stamp=args.run, init_time=init_time)
        hrdps.set_model("continental")
        archive_model_hours(args.archive_root, run, args.data_dir, parse_hours(args.hours))
        write_archive_status(args.archive_root)
        return 0
    if args.command == "archive-baseline":
        verify_archive_writable(args.archive_root)
        init_time = hrdps.parse_stamp(args.run)
        run = hrdps.RunInfo(cycle=f"{init_time:%H}", stamp=args.run, init_time=init_time)
        hrdps.set_model("continental")
        archive_lpi_baseline_hours(args.archive_root, run, args.cache_dir, parse_hours(args.hours))
        write_archive_status(args.archive_root)
        return 0
    if args.command == "status":
        print(json.dumps(archive_status(args.archive_root), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
