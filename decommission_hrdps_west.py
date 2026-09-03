#!/usr/bin/env python3
"""Audit or remove the retired HRDPS-West public objects from forecast R2."""

from __future__ import annotations

import argparse
import json
import shutil
from typing import Iterable

from botocore.exceptions import ClientError

from r2_publish import R2Config, boto3_client, delete_object_keys
import project_paths


PREFIXES = ("models/west/",)
EXACT_KEYS = (
    "manifests/west.json",
    "live/fire_activity/lightning_sw.png",
    "live/fire_activity/lightning_se.png",
    "live/fire_activity/lightning_ne.png",
)
LOCAL_PATHS = (
    project_paths.data_path("hrdps_west"),
    project_paths.data_path("cwfis_fwi", "fwi2025_state", "west"),
    project_paths.data_path("cwfis_fwi", "peak_daily", "west"),
    project_paths.data_path("cwfis_fwi", "regridded_anchor", "west"),
    project_paths.data_path("cwfis_fwi", "regridded_anchor_v2", "west"),
    project_paths.data_path("cwfis_fwi", "regridded_anchor_v3", "west"),
    project_paths.data_path("fire_danger_verification", "forecasts", "west"),
)


def listed_keys(client, bucket: str, prefix: str) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        request: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        output.extend(
            (str(item["Key"]), int(item.get("Size", 0)))
            for item in response.get("Contents", ())
        )
        if not response.get("IsTruncated"):
            return output
        token = str(response["NextContinuationToken"])


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(list(argv))
    config = R2Config.from_environment()
    client = boto3_client(config)
    objects = [item for prefix in PREFIXES for item in listed_keys(client, config.bucket, prefix)]
    exact_existing: list[tuple[str, int]] = []
    for key in EXACT_KEYS:
        try:
            response = client.head_object(Bucket=config.bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                continue
            raise
        exact_existing.append((key, int(response.get("ContentLength", 0))))
    candidates = [*objects, *exact_existing]
    deleted = delete_object_keys(client, config, (key for key, _ in candidates)) if args.apply else 0
    local_existing = [path for path in LOCAL_PATHS if path.exists()]
    local_bytes = sum(
        item.stat().st_size
        for path in local_existing
        for item in path.rglob("*")
        if item.is_file()
    )
    if args.apply:
        for path in local_existing:
            shutil.rmtree(path)
    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "objects": len(candidates),
        "bytes": sum(size for _, size in candidates),
        "deleted": deleted,
        "localPaths": [str(path) for path in local_existing],
        "localBytes": local_bytes,
        "prefixes": PREFIXES,
        "exactKeys": EXACT_KEYS,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
