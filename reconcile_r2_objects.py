#!/usr/bin/env python3
"""Reconcile forecast R2 objects against model-init retention policy."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from make_hrdps_west_convective import parse_stamp
from r2_publish import (
    MODEL_PRODUCTS,
    RETIRED_PRODUCTS,
    R2Config,
    boto3_client,
    delete_object_keys,
    retention_class,
    run_is_retained,
)


LOCK_PATH = Path("logs/state/r2_reconcile.lock")
STATUS_PATH = Path("logs/state/r2_reconcile.status.json")
DECOMMISSIONED_MODELS = frozenset({"west"})


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int


def list_remote_objects(client, bucket: str) -> list[RemoteObject]:
    objects: list[RemoteObject] = []
    token: str | None = None
    while True:
        request: dict[str, object] = {"Bucket": bucket, "Prefix": "models/"}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        objects.extend(
            RemoteObject(str(item["Key"]), int(item.get("Size", 0)))
            for item in response.get("Contents", ())
        )
        if not response.get("IsTruncated"):
            return objects
        token = str(response["NextContinuationToken"])


def classify_object(key: str, now: dt.datetime) -> str:
    parts = key.split("/")
    if len(parts) < 6 or parts[0] != "models":
        return "unrecognized"
    _, model, storage_class, product, stamp, *_ = parts
    if model in DECOMMISSIONED_MODELS:
        return "decommissioned"
    if model not in MODEL_PRODUCTS:
        return "unrecognized"
    if product in RETIRED_PRODUCTS.get(model, ()):
        return "retired"
    if product not in MODEL_PRODUCTS[model]:
        return "unrecognized"
    if storage_class != retention_class(product):
        return "misplaced"
    try:
        init = parse_stamp(stamp)
    except ValueError:
        return "unrecognized"
    if not run_is_retained(product, init, now):
        return "expired"
    return "retained"


def reconcile(
    client,
    config: R2Config,
    *,
    apply: bool,
    now: dt.datetime | None = None,
    status_path: Path = STATUS_PATH,
) -> dict[str, object]:
    now = now or dt.datetime.now(dt.timezone.utc)
    objects = list_remote_objects(client, config.bucket)
    grouped: dict[str, list[RemoteObject]] = {}
    for item in objects:
        grouped.setdefault(classify_object(item.key, now), []).append(item)
    delete_candidates = [
        item
        for reason in ("expired", "retired", "misplaced", "decommissioned")
        for item in grouped.get(reason, ())
    ]
    deleted = delete_object_keys(client, config, (item.key for item in delete_candidates)) if apply else 0
    result: dict[str, object] = {
        "status": "success",
        "mode": "apply" if apply else "dry-run",
        "updatedAt": now.isoformat().replace("+00:00", "Z"),
        "bucket": config.bucket,
        "objectsScanned": len(objects),
        "bytesScanned": sum(item.size for item in objects),
        "deleteCandidates": len(delete_candidates),
        "deleteCandidateBytes": sum(item.size for item in delete_candidates),
        "deleted": deleted,
        "counts": {reason: len(items) for reason, items in sorted(grouped.items())},
        "unrecognizedSample": [item.key for item in grouped.get("unrecognized", ())[:10]],
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(status_path)
    return result


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete objects selected by policy.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("R2 reconciliation is already running; skipping.", flush=True)
            return 0
        config = R2Config.from_environment()
        result = reconcile(boto3_client(config), config, apply=args.apply)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
