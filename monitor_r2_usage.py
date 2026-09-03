#!/usr/bin/env python3
"""Monitor R2 free-tier storage and operations and alert before overage."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import telegram_notify

GRAPHQL_URL = "https://api.cloudflare.com/client/v4/graphql"
KEYCHAIN_ACCOUNT = "fcst-graphics"
KEYCHAIN_API_TOKEN_SERVICE = "fcstGraphics-cloudflare-api-token"
FREE_STORAGE_GB_MONTH = 10.0
FREE_CLASS_A = 1_000_000
FREE_CLASS_B = 10_000_000
STORAGE_WARNING_FRACTION = 0.85
STORAGE_CRITICAL_FRACTION = 0.98
DEFAULT_BILLING_DAY = 20
STATE_PATH = Path("logs/state/r2_usage_monitor.json")
LATEST_PATH = Path("logs/r2_usage_latest.json")
HISTORY_PATH = Path("logs/r2_usage_history.jsonl")
HISTORY_MAX_BYTES = 5_000_000

CLASS_A_ACTIONS = {
    "listbuckets",
    "putbucket",
    "listobjects",
    "listobjectsv2",
    "putobject",
    "copyobject",
    "completemultipartupload",
    "createmultipartupload",
    "lifecyclestoragetiertransition",
    "listmultipartuploads",
    "uploadpart",
    "uploadpartcopy",
    "listparts",
    "putbucketencryption",
    "putbucketcors",
    "putbucketlifecycleconfiguration",
}
CLASS_B_ACTIONS = {
    "headbucket",
    "headobject",
    "getobject",
    "usagesummary",
    "getbucketencryption",
    "getbucketlocation",
    "getbucketcors",
    "getbucketlifecycleconfiguration",
}
FREE_ACTIONS = {"deleteobject", "deleteobjects", "deletebucket", "abortmultipartupload"}


@dataclass(frozen=True)
class BillingPeriod:
    start: dt.datetime
    end: dt.datetime


def keychain_password(service: str, account: str = KEYCHAIN_ACCOUNT) -> str:
    security = shutil.which("security")
    if security is None:
        return ""
    result = subprocess.run(
        [security, "find-generic-password", "-a", account, "-s", service, "-w"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--billing-day",
        type=int,
        default=int(os.environ.get("FCST_R2_BILLING_DAY", DEFAULT_BILLING_DAY)),
    )
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--latest-path", type=Path, default=LATEST_PATH)
    parser.add_argument("--history-path", type=Path, default=HISTORY_PATH)
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument(
        "--always-notify",
        action="store_true",
        help="Send a heartbeat notification even when usage is below warning thresholds.",
    )
    return parser.parse_args(list(argv))


def month_shift(year: int, month: int, delta: int) -> tuple[int, int]:
    value = year * 12 + month - 1 + delta
    return value // 12, value % 12 + 1


def month_day(year: int, month: int, day: int) -> dt.datetime:
    bounded_day = min(day, calendar.monthrange(year, month)[1])
    return dt.datetime(year, month, bounded_day, tzinfo=dt.timezone.utc)


def billing_period(now: dt.datetime, billing_day: int) -> BillingPeriod:
    if not 1 <= billing_day <= 31:
        raise ValueError("billing day must be between 1 and 31")
    now = now.astimezone(dt.timezone.utc)
    this_anchor = month_day(now.year, now.month, billing_day)
    if now >= this_anchor:
        start = this_anchor
        end_year, end_month = month_shift(now.year, now.month, 1)
        end = month_day(end_year, end_month, billing_day)
    else:
        start_year, start_month = month_shift(now.year, now.month, -1)
        start = month_day(start_year, start_month, billing_day)
        end = this_anchor
    return BillingPeriod(start=start, end=end)


def graphql_usage(
    account_id: str,
    token: str,
    period: BillingPeriod,
) -> dict[str, object]:
    query = """
    query R2Usage($accountTag: string!, $startDate: Time!, $endDate: Time!) {
      viewer {
        accounts(filter: {accountTag: $accountTag}) {
          r2OperationsAdaptiveGroups(
            limit: 10000
            filter: {datetime_geq: $startDate, datetime_leq: $endDate}
          ) {
            sum { requests }
            dimensions { actionType bucketName }
          }
          r2StorageAdaptiveGroups(
            limit: 10000
            filter: {datetime_geq: $startDate, datetime_leq: $endDate}
            orderBy: [datetime_DESC]
          ) {
            max { objectCount uploadCount payloadSize metadataSize }
            dimensions { bucketName datetime }
          }
        }
      }
    }
    """
    payload = json.dumps(
        {
            "query": query,
            "variables": {
                "accountTag": account_id,
                "startDate": period.start.isoformat().replace("+00:00", "Z"),
                "endDate": period.end.isoformat().replace("+00:00", "Z"),
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(f"Cloudflare GraphQL error: {result['errors']}")
    accounts = result.get("data", {}).get("viewer", {}).get("accounts", [])
    if not accounts:
        raise RuntimeError("Cloudflare GraphQL returned no account metrics")
    account = accounts[0]
    return {
        "operations": list(account.get("r2OperationsAdaptiveGroups", [])),
        "storage": list(account.get("r2StorageAdaptiveGroups", [])),
    }


def classify_operations(groups: Iterable[Mapping[str, object]]) -> dict[str, object]:
    totals = {"class_a": 0, "class_b": 0, "free": 0, "unknown": 0}
    unknown_types: dict[str, int] = {}
    buckets: dict[str, dict[str, object]] = {}
    for group in groups:
        dimensions = group.get("dimensions", {})
        sums = group.get("sum", {})
        action = str(dimensions.get("actionType", "unknown"))
        normalized = "".join(character for character in action.lower() if character.isalnum())
        requests = int(sums.get("requests", 0) or 0)
        bucket = str(dimensions.get("bucketName", "account-level") or "account-level")
        bucket_totals = buckets.setdefault(
            bucket,
            {"class_a": 0, "class_b": 0, "free": 0, "unknown": 0, "actions": {}},
        )
        actions = bucket_totals["actions"]
        if isinstance(actions, dict):
            actions[action] = actions.get(action, 0) + requests
        if normalized in CLASS_A_ACTIONS:
            totals["class_a"] += requests
            bucket_totals["class_a"] = int(bucket_totals["class_a"]) + requests
        elif normalized in CLASS_B_ACTIONS:
            totals["class_b"] += requests
            bucket_totals["class_b"] = int(bucket_totals["class_b"]) + requests
        elif normalized in FREE_ACTIONS:
            totals["free"] += requests
            bucket_totals["free"] = int(bucket_totals["free"]) + requests
        else:
            totals["unknown"] += requests
            bucket_totals["unknown"] = int(bucket_totals["unknown"]) + requests
            unknown_types[action] = unknown_types.get(action, 0) + requests
    totals["unknown_types"] = unknown_types
    totals["buckets"] = buckets
    return totals


def latest_storage(groups: Iterable[Mapping[str, object]]) -> dict[str, object]:
    latest_by_bucket: dict[str, Mapping[str, object]] = {}
    for group in groups:
        dimensions = group.get("dimensions", {})
        bucket_name = str(dimensions.get("bucketName", "unknown"))
        if bucket_name not in latest_by_bucket:
            latest_by_bucket[bucket_name] = group
    if not latest_by_bucket:
        return {
            "payload_bytes": 0,
            "metadata_bytes": 0,
            "object_count": 0,
            "pending_uploads": 0,
            "observed_at": None,
            "buckets": {},
        }
    buckets: dict[str, dict[str, object]] = {}
    for bucket_name, group in latest_by_bucket.items():
        maximum = group.get("max", {})
        dimensions = group.get("dimensions", {})
        buckets[bucket_name] = {
            "payload_bytes": int(maximum.get("payloadSize", 0) or 0),
            "metadata_bytes": int(maximum.get("metadataSize", 0) or 0),
            "object_count": int(maximum.get("objectCount", 0) or 0),
            "pending_uploads": int(maximum.get("uploadCount", 0) or 0),
            "observed_at": dimensions.get("datetime"),
        }
    observed_times = [
        str(bucket["observed_at"])
        for bucket in buckets.values()
        if bucket["observed_at"]
    ]
    return {
        "payload_bytes": sum(int(bucket["payload_bytes"]) for bucket in buckets.values()),
        "metadata_bytes": sum(int(bucket["metadata_bytes"]) for bucket in buckets.values()),
        "object_count": sum(int(bucket["object_count"]) for bucket in buckets.values()),
        "pending_uploads": sum(int(bucket["pending_uploads"]) for bucket in buckets.values()),
        "observed_at": max(observed_times) if observed_times else None,
        "buckets": buckets,
    }


def daily_storage_peaks(groups: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Return account-wide daily peak bytes by summing buckets at each sample time."""
    samples: dict[str, int] = {}
    for group in groups:
        maximum = group.get("max", {})
        dimensions = group.get("dimensions", {})
        observed_at = dimensions.get("datetime")
        if not observed_at:
            continue
        sample_bytes = int(maximum.get("payloadSize", 0) or 0) + int(
            maximum.get("metadataSize", 0) or 0
        )
        timestamp = str(observed_at)
        samples[timestamp] = samples.get(timestamp, 0) + sample_bytes
    peaks: dict[str, int] = {}
    for timestamp, sample_bytes in samples.items():
        day = timestamp[:10]
        peaks[day] = max(peaks.get(day, 0), sample_bytes)
    return dict(sorted(peaks.items()))


def projected_storage_gb_month(
    peaks: Mapping[str, int],
    period: BillingPeriod,
    now: dt.datetime,
    current_bytes: int,
) -> tuple[float, float]:
    """Return observed-to-date and projected full-cycle average daily peak GB."""
    now = now.astimezone(dt.timezone.utc)
    total_days = max(1, (period.end.date() - period.start.date()).days)
    today = min(now.date(), period.end.date() - dt.timedelta(days=1))
    elapsed_days = max(1, (today - period.start.date()).days + 1)
    fallback = max(0, current_bytes)
    observed_bytes = 0
    for offset in range(elapsed_days):
        day = period.start.date() + dt.timedelta(days=offset)
        observed_bytes += int(peaks.get(day.isoformat(), fallback))
    observed_gb = observed_bytes / elapsed_days / 1_000_000_000.0
    future_days = max(0, total_days - elapsed_days)
    projected_bytes = observed_bytes + future_days * fallback
    projected_gb = projected_bytes / total_days / 1_000_000_000.0
    return observed_gb, projected_gb


def projected_count(observed: int, period: BillingPeriod, now: dt.datetime) -> float:
    # A one-day floor avoids treating a one-time upload burst at cycle start as
    # though it will repeat every hour for the rest of the month.
    elapsed = max(86400.0, (now - period.start).total_seconds())
    duration = (period.end - period.start).total_seconds()
    return float(observed) / max(elapsed / duration, 1.0 / 31.0)


def assess_usage(
    storage_gb: float,
    class_a: int,
    class_b: int,
    period: BillingPeriod,
    now: dt.datetime,
    projected_storage_gb: float | None = None,
) -> dict[str, object]:
    projected_a = projected_count(class_a, period, now)
    projected_b = projected_count(class_b, period, now)
    fractions = {
        "storage": storage_gb / FREE_STORAGE_GB_MONTH,
        "storage_projected": (
            projected_storage_gb if projected_storage_gb is not None else storage_gb
        )
        / FREE_STORAGE_GB_MONTH,
        "class_a_current": class_a / FREE_CLASS_A,
        "class_a_projected": projected_a / FREE_CLASS_A,
        "class_b_current": class_b / FREE_CLASS_B,
        "class_b_projected": projected_b / FREE_CLASS_B,
    }
    operations_highest = max(
        fractions["class_a_current"],
        fractions["class_a_projected"],
        fractions["class_b_current"],
        fractions["class_b_projected"],
    )
    storage_highest = max(fractions["storage"], fractions["storage_projected"])
    if storage_highest >= STORAGE_CRITICAL_FRACTION or operations_highest >= 0.90:
        level = "critical"
    elif storage_highest >= STORAGE_WARNING_FRACTION or operations_highest >= 0.70:
        level = "warning"
    else:
        level = "ok"
    return {
        "level": level,
        "fractions": fractions,
        "projected_class_a": round(projected_a),
        "projected_class_b": round(projected_b),
    }


def notify(message: str) -> bool:
    if telegram_notify.configured():
        telegram_notify.send_message("R2 Free-Tier Monitor", message)
        return True
    escaped = message.replace("\\", "\\\\").replace('"', '\\"')
    result = subprocess.run(
        [
            "/usr/bin/osascript",
            "-e",
            f'display notification "{escaped}" with title "R2 Free-Tier Monitor"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def read_state(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_history(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= HISTORY_MAX_BYTES:
        rotated = path.with_suffix(path.suffix + ".1")
        rotated.unlink(missing_ok=True)
        path.replace(rotated)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def should_notify(level: str, state: Mapping[str, object], now: dt.datetime) -> bool:
    if level == "ok":
        return False
    if state.get("level") != level:
        return True
    try:
        previous = dt.datetime.fromisoformat(str(state["last_notified_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return True
    return now - previous >= dt.timedelta(hours=24)


def bucket_summary(storage: Mapping[str, object]) -> str:
    buckets = storage.get("buckets", {})
    if not isinstance(buckets, Mapping):
        return ""
    parts = []
    for name, values in sorted(buckets.items()):
        if not isinstance(values, Mapping):
            continue
        total = int(values.get("payload_bytes", 0) or 0) + int(
            values.get("metadata_bytes", 0) or 0
        )
        parts.append(f"{name} {total / 1_000_000_000:.2f} GB")
    return "; ".join(parts)


def operation_bucket_summary(operations: Mapping[str, object]) -> str:
    buckets = operations.get("buckets", {})
    if not isinstance(buckets, Mapping):
        return ""
    parts = []
    for name, values in sorted(buckets.items()):
        if not isinstance(values, Mapping):
            continue
        class_a = int(values.get("class_a", 0) or 0)
        class_b = int(values.get("class_b", 0) or 0)
        if class_a or class_b:
            parts.append(f"{name} A {class_a:,}, B {class_b:,}")
    return "; ".join(parts)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    period = billing_period(now, args.billing_day)
    state = read_state(args.state_path)
    try:
        account_id = os.environ.get("FCST_R2_ACCOUNT_ID", "").strip()
        bucket = os.environ.get("FCST_R2_BUCKET", "").strip()
        if not account_id or not bucket:
            raise RuntimeError("FCST_R2_ACCOUNT_ID and FCST_R2_BUCKET are required")
        api_token = os.environ.get("FCST_CLOUDFLARE_API_TOKEN", "").strip() or keychain_password(
            KEYCHAIN_API_TOKEN_SERVICE
        )
        if not api_token:
            raise RuntimeError("Cloudflare analytics token is not configured")
        usage = graphql_usage(account_id, api_token, period)
        storage = latest_storage(usage["storage"])
        operations = classify_operations(usage["operations"])
        conservative_a = int(operations["class_a"]) + int(operations["unknown"])
        storage_bytes = int(storage["payload_bytes"]) + int(storage["metadata_bytes"])
        storage_gb = storage_bytes / 1_000_000_000.0
        storage_peaks = daily_storage_peaks(usage["storage"])
        observed_storage_gb, projected_storage_gb = projected_storage_gb_month(
            storage_peaks,
            period,
            now,
            storage_bytes,
        )
        assessment = assess_usage(
            storage_gb,
            conservative_a,
            int(operations["class_b"]),
            period,
            now,
            projected_storage_gb,
        )
        payload = {
            "checked_at": now.isoformat().replace("+00:00", "Z"),
            "billing_period_start": period.start.isoformat().replace("+00:00", "Z"),
            "billing_period_end": period.end.isoformat().replace("+00:00", "Z"),
            "bucket": bucket,
            "storage_bytes": storage_bytes,
            "storage_payload_bytes": storage["payload_bytes"],
            "storage_metadata_bytes": storage["metadata_bytes"],
            "storage_gb": round(storage_gb, 4),
            "observed_storage_gb_month": round(observed_storage_gb, 4),
            "projected_storage_gb_month": round(projected_storage_gb, 4),
            "daily_storage_peak_bytes": storage_peaks,
            "object_count": storage["object_count"],
            "pending_uploads": storage["pending_uploads"],
            "storage_observed_at": storage["observed_at"],
            "storage_buckets": storage["buckets"],
            "operations": operations,
            "conservative_class_a": conservative_a,
            "class_b": int(operations["class_b"]),
            **assessment,
        }
        write_json(args.latest_path, payload)
        fractions = assessment["fractions"]
        summary_prefix = "R2 weekly report" if args.always_notify else f"R2 {assessment['level']}"
        summary = (
            f"{summary_prefix}: storage {storage_gb:.2f}/10 GB; "
            f"projected billing average {projected_storage_gb:.2f}/10 GB-month; "
            f"projected Class A {fractions['class_a_projected']:.1%}; "
            f"projected Class B {fractions['class_b_projected']:.1%}."
        )
        by_bucket = bucket_summary(storage)
        if by_bucket:
            summary += f" Buckets: {by_bucket}."
        operation_buckets = operation_bucket_summary(operations)
        if operation_buckets and assessment["level"] != "ok":
            summary += f" Cycle operations: {operation_buckets}."
        print(summary, flush=True)
        notified = False
        if not args.no_notify and (
            args.always_notify or should_notify(str(assessment["level"]), state, now)
        ):
            notified = notify(summary)
        new_state = {
            "level": assessment["level"],
            "consecutive_failures": 0,
            "last_checked_at": payload["checked_at"],
            "last_notified_at": (
                payload["checked_at"] if notified else state.get("last_notified_at")
            ),
        }
        write_json(args.state_path, new_state)
        append_history(args.history_path, payload)
        return 0
    except Exception as exc:
        failures = int(state.get("consecutive_failures", 0)) + 1
        message = (
            f"R2 usage check failed ({failures} consecutive): {exc}. "
            "Forecast graphics are not known to be affected, but the free-tier "
            "cost guardrail cannot currently measure storage or request usage."
        )
        print(message, flush=True)
        notified = (
            (args.always_notify or failures == 3)
            and not args.no_notify
            and notify(message)
        )
        write_json(
            args.state_path,
            {
                **state,
                "consecutive_failures": failures,
                "last_checked_at": now.isoformat().replace("+00:00", "Z"),
                "last_notified_at": (
                    now.isoformat().replace("+00:00", "Z")
                    if notified
                    else state.get("last_notified_at")
                ),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
