#!/usr/bin/env python3
"""Report exact forecast-graphics R2 storage by model, class, and product."""

from __future__ import annotations

from collections import defaultdict
import json

from r2_publish import R2Config, boto3_client


def add(group: dict[str, dict[str, int]], key: str, size: int) -> None:
    row = group.setdefault(key, {"objects": 0, "bytes": 0})
    row["objects"] += 1
    row["bytes"] += size


def main() -> int:
    config = R2Config.from_environment()
    client = boto3_client(config)
    groups: dict[str, dict[str, dict[str, int]]] = {
        name: defaultdict(dict) for name in ("topPrefixes", "models", "storageClasses", "products")
    }
    objects = 0
    total_bytes = 0
    token: str | None = None
    while True:
        request: dict[str, object] = {"Bucket": config.bucket}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        for item in response.get("Contents", ()):
            key = str(item["Key"])
            size = int(item.get("Size", 0))
            parts = key.split("/")
            objects += 1
            total_bytes += size
            add(groups["topPrefixes"], parts[0], size)
            if len(parts) >= 5 and parts[0] == "models":
                add(groups["models"], parts[1], size)
                add(groups["storageClasses"], parts[2], size)
                add(groups["products"], f"{parts[1]}/{parts[3]}", size)
        if not response.get("IsTruncated"):
            break
        token = str(response["NextContinuationToken"])

    payload = {
        "bucket": config.bucket,
        "objects": objects,
        "bytes": total_bytes,
        **{
            name: dict(sorted(values.items(), key=lambda item: item[1]["bytes"], reverse=True))
            for name, values in groups.items()
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
