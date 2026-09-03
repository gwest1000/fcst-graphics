#!/usr/bin/env python3
"""Apply CORS and retention policies to the forecast-graphics R2 bucket."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

from r2_publish import (
    FORECAST_KEEP_DAYS,
    MODEL_PRODUCTS,
    R2Config,
    VERIFICATION_KEEP_DAYS,
    boto3_client,
)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-origin",
        default=os.environ.get("FCST_SITE_ORIGIN", "https://gwest1000.github.io"),
        help="Browser origin allowed to fetch model manifests.",
    )
    return parser.parse_args(list(argv))


def lifecycle_rules() -> list[dict[str, object]]:
    rules = [
        {
            "ID": "expire-all-model-data",
            "Status": "Enabled",
            "Filter": {"Prefix": "models/"},
            "Expiration": {"Days": VERIFICATION_KEEP_DAYS + 1},
        }
    ]
    for model in MODEL_PRODUCTS:
        rules.append(
            {
                "ID": f"expire-{model.replace('_', '-')}-forecasts",
                "Status": "Enabled",
                "Filter": {"Prefix": f"models/{model}/forecast/"},
                "Expiration": {"Days": FORECAST_KEEP_DAYS + 1},
            }
        )
    return rules


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = R2Config.from_environment()
    client = boto3_client(config)
    client.put_bucket_cors(
        Bucket=config.bucket,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedMethods": ["GET", "HEAD"],
                    "AllowedOrigins": [args.site_origin],
                    "AllowedHeaders": ["*"],
                    "ExposeHeaders": [
                        "ETag",
                        "Content-Length",
                        "Content-Type",
                        "Last-Modified",
                    ],
                    "MaxAgeSeconds": 86400,
                }
            ]
        },
    )
    rules = lifecycle_rules()
    client.put_bucket_lifecycle_configuration(
        Bucket=config.bucket,
        LifecycleConfiguration={"Rules": rules},
    )
    print(
        f"Configured CORS for {args.site_origin} and {len(rules)} lifecycle rules on {config.bucket}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
