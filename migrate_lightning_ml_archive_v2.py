#!/usr/bin/env python3
"""Migrate large LPI development archives to the bounded schema-v2 policy."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

import lightning_ml_archive as archive


UTC = dt.timezone.utc
REPORT_NAME = "migration_schema_v1_to_v2.json"


def run_info(stamp: str) -> archive.hrdps.RunInfo:
    init = archive.hrdps.parse_stamp(stamp)
    return archive.hrdps.RunInfo(cycle=f"{init:%H}", stamp=stamp, init_time=init)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_npz(
    source: Path,
    destination: Path,
    domain_mask: np.ndarray,
    schema_version: int,
) -> None:
    with np.load(source) as packed:
        arrays = {key: packed[key] for key in packed.files}
    for key, value in list(arrays.items()):
        if value.shape == domain_mask.shape:
            masked = value.copy()
            masked[~domain_mask] = archive.FILL_VALUE
            arrays[key] = masked
    arrays["schema_version"] = np.asarray([schema_version], dtype=np.int16)
    atomic_npz(destination, arrays)

    source_sidecar = source.with_suffix(".json")
    payload = json.loads(source_sidecar.read_text()) if source_sidecar.exists() else {}
    payload.update(
        schema_version=schema_version,
        archive_domain="British Columbia plus buffer",
        archive_domain_buffer_km=archive.ARCHIVE_DOMAIN_BUFFER_KM,
        bytes=destination.stat().st_size,
        sha256=archive.sha256_file(destination),
        migrated_from=str(source),
        migrated_at_utc=archive.utc_iso(dt.datetime.now(UTC)),
    )
    archive.write_json_atomic(destination.with_suffix(".json"), payload)


def verify_npz(source: Path, destination: Path, domain_mask: np.ndarray, schema_version: int) -> None:
    sidecar = destination.with_suffix(".json")
    if not destination.is_file() or not sidecar.is_file():
        raise RuntimeError(f"Incomplete migrated pair for {destination}.")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("sha256") != archive.sha256_file(destination):
        raise RuntimeError(f"Checksum mismatch for {destination}.")
    with np.load(source) as old, np.load(destination) as new:
        if set(old.files) != set(new.files):
            raise RuntimeError(f"Field set changed while migrating {source}.")
        if int(new["schema_version"][0]) != schema_version:
            raise RuntimeError(f"Wrong schema version in {destination}.")
        for key in old.files:
            if key == "schema_version":
                continue
            old_value = old[key]
            new_value = new[key]
            if old_value.shape == domain_mask.shape:
                if not np.array_equal(old_value[domain_mask], new_value[domain_mask]):
                    raise RuntimeError(f"Retained values changed in {destination}: {key}.")
                if not np.all(new_value[~domain_mask] == archive.FILL_VALUE):
                    raise RuntimeError(f"Unmasked values remain outside the archive domain: {destination}: {key}.")
            elif not np.array_equal(old_value, new_value):
                raise RuntimeError(f"Metadata array changed in {destination}: {key}.")


def migrate_tree(
    source_root: Path,
    destination_root: Path,
    hours_for_cycle,
    domain_mask: np.ndarray,
    schema_version: int,
) -> tuple[list[tuple[Path, Path]], int]:
    migrated: list[tuple[Path, Path]] = []
    skipped_bytes = 0
    for source in sorted(source_root.glob("*/*/f*.npz")):
        run = run_info(source.parent.name)
        fhour = int(source.stem[1:])
        if fhour not in hours_for_cycle(run.cycle):
            skipped_bytes += source.stat().st_size
            continue
        destination = destination_root / source.relative_to(source_root)
        migrate_npz(source, destination, domain_mask, schema_version)
        migrated.append((source, destination))
    return migrated, skipped_bytes


def copy_masked_terrain(source_root: Path, destination_root: Path, domain_mask: np.ndarray) -> None:
    source = source_root / "static/terrain.npz"
    if not source.exists():
        return
    with np.load(source) as old:
        arrays = {key: old[key] for key in old.files}
    terrain = arrays["elevation_m"].copy()
    terrain[~domain_mask] = archive.FILL_VALUE
    arrays["elevation_m"] = terrain
    arrays["schema_version"] = np.asarray([archive.MODEL_ARCHIVE_SCHEMA_VERSION], dtype=np.int16)
    atomic_npz(destination_root / "static/terrain.npz", arrays)


def update_manifests(root: Path, pairs: Iterable[tuple[Path, Path]], *, hourly: bool, grid_hash: str) -> None:
    stamps = sorted({destination.parent.name for _, destination in pairs})
    for stamp in stamps:
        run = run_info(stamp)
        if hourly:
            archive._update_hourly_lpi_manifest(root, run, grid_hash)
        else:
            archive._update_run_manifest(root, run, grid_hash)


def source_complete_implies_destination_complete(
    source_root: Path,
    destination_root: Path,
) -> None:
    for source_manifest in source_root.glob("*/*/manifest.json"):
        source = json.loads(source_manifest.read_text())
        if not source.get("complete"):
            continue
        destination_manifest = destination_root / source_manifest.relative_to(source_root)
        if not destination_manifest.exists() or not json.loads(destination_manifest.read_text()).get("complete"):
            raise RuntimeError(f"Complete source run did not migrate completely: {source_manifest.parent.name}.")


def retire_v1_trees(root: Path, old_model: Path, old_hourly: Path) -> None:
    """Atomically withdraw both v1 trees before deleting either one."""

    root = root.resolve()
    for path in (old_model, old_hourly):
        if root not in path.resolve().parents or path.name != "schema_v1":
            raise RuntimeError(f"Refusing to retire unexpected archive path: {path}")

    retired_model = old_model.with_name(f".schema_v1.retiring.{os.getpid()}")
    retired_hourly = old_hourly.with_name(f".schema_v1.retiring.{os.getpid()}")
    if retired_model.exists() or retired_hourly.exists():
        raise RuntimeError("A previous schema-v1 retirement directory already exists.")

    old_model.rename(retired_model)
    try:
        old_hourly.rename(retired_hourly)
    except BaseException:
        retired_model.rename(old_model)
        raise

    try:
        shutil.rmtree(retired_model)
        shutil.rmtree(retired_hourly)
    except BaseException as exc:
        raise RuntimeError(
            "Both schema-v1 trees were retired from active use, but removing their files failed. "
            f"Inspect {retired_model} and {retired_hourly}."
        ) from exc


def migrate(root: Path, *, delete_v1: bool) -> dict[str, object]:
    root = archive.verify_archive_writable(root)
    old_model = archive.model_archive_dir(root, schema_version=1)
    old_hourly = archive.hourly_lpi_archive_dir(root, schema_version=1)
    if not old_model.is_dir() or not old_hourly.is_dir():
        raise RuntimeError("Both schema-v1 model and hourly archives must exist before migration.")

    with np.load(old_model / "static/grid.npz") as old_grid:
        lat = old_grid["lat"].astype(np.float32)
        lon = old_grid["lon"].astype(np.float32)
    archive._write_model_schema(root)
    archive._write_hourly_lpi_schema(root)
    grid_hash, domain_mask = archive._ensure_grid_archive(root, lat, lon)
    copy_masked_terrain(old_model, archive.model_archive_dir(root), domain_mask)

    model_pairs, model_skipped = migrate_tree(
        old_model,
        archive.model_archive_dir(root),
        archive.model_forecast_hours,
        domain_mask,
        archive.MODEL_ARCHIVE_SCHEMA_VERSION,
    )
    hourly_pairs, hourly_skipped = migrate_tree(
        old_hourly,
        archive.hourly_lpi_archive_dir(root),
        archive.hourly_lpi_forecast_hours,
        domain_mask,
        archive.HOURLY_LPI_SCHEMA_VERSION,
    )
    update_manifests(root, model_pairs, hourly=False, grid_hash=grid_hash)
    update_manifests(root, hourly_pairs, hourly=True, grid_hash=grid_hash)

    for index, (source, destination) in enumerate(model_pairs + hourly_pairs, start=1):
        verify_npz(
            source,
            destination,
            domain_mask,
            archive.MODEL_ARCHIVE_SCHEMA_VERSION if index <= len(model_pairs) else archive.HOURLY_LPI_SCHEMA_VERSION,
        )
        if index % 500 == 0:
            archive.log(f"  verified {index}/{len(model_pairs) + len(hourly_pairs)} migrated files")

    source_complete_implies_destination_complete(old_model, archive.model_archive_dir(root))
    source_complete_implies_destination_complete(old_hourly, archive.hourly_lpi_archive_dir(root))
    source_bytes = sum(path.stat().st_size for path in old_model.rglob("*") if path.is_file()) + sum(
        path.stat().st_size for path in old_hourly.rglob("*") if path.is_file()
    )
    destination_bytes = sum(
        path.stat().st_size for path in archive.model_archive_dir(root).rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in archive.hourly_lpi_archive_dir(root).rglob("*") if path.is_file())

    result: dict[str, object] = {
        "status": "verified",
        "updated_at_utc": archive.utc_iso(dt.datetime.now(UTC)),
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "bytes_saved": source_bytes - destination_bytes,
        "model_files": len(model_pairs),
        "hourly_files": len(hourly_pairs),
        "model_excluded_bytes": model_skipped,
        "hourly_excluded_bytes": hourly_skipped,
        "domain_cells": int(domain_mask.size),
        "retained_domain_cells": int(np.count_nonzero(domain_mask)),
        "source_deleted": False,
    }
    archive.write_json_atomic(root / REPORT_NAME, result)

    if delete_v1:
        result["status"] = "retiring_v1"
        archive.write_json_atomic(root / REPORT_NAME, result)
        retire_v1_trees(root, old_model, old_hourly)
        result["source_deleted"] = True
        result["status"] = "complete"
        archive.write_json_atomic(root / REPORT_NAME, result)
    archive.write_archive_status(root)
    return result


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=archive.DEFAULT_ARCHIVE_ROOT)
    parser.add_argument(
        "--delete-v1",
        action="store_true",
        help="Delete the large model/hourly schema-v1 trees only after complete validation.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    lock_path = args.archive_root / ".migration_v2.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = migrate(args.archive_root, delete_v1=args.delete_v1)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
