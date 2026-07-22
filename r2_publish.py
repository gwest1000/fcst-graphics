#!/usr/bin/env python3
"""Publish forecast frames and model manifests to Cloudflare R2."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from publish_hrdps_west import (
    PRODUCTS,
    VERIFICATION_KEEP_DAYS,
    VERIFICATION_PRODUCT_KEYS,
    default_plot_dir_for_product,
    image_name_for_hour,
    minimum_manifest_hours,
)
from make_hrdps_west_convective import parse_stamp

FORECAST_KEEP_DAYS = 7
IMAGE_CACHE_CONTROL = "public, max-age=31536000, immutable"
MANIFEST_CACHE_CONTROL = "no-cache, max-age=0, must-revalidate"
PUBLISH_FORMAT_VERSION = "pngquant-70-90-speed1-v1"
STATE_ROOT = Path("logs/state")
PNG_END_MARKER = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
KEYCHAIN_ACCOUNT = "fcst-graphics"
KEYCHAIN_ACCESS_KEY_SERVICE = "fcstGraphics-r2-access-key-id"
KEYCHAIN_SECRET_KEY_SERVICE = "fcstGraphics-r2-secret-access-key"

MODEL_PRODUCTS: dict[str, tuple[str, ...]] = {
    "continental": (
        "continental_fourpanel",
        "continental_lightning_twopanel",
        "continental_fwi2025_danger",
        "continental_lightning_verif",
        "fire_danger_verif",
    ),
    "west": (
        "lightning_sw",
        "lightning_se",
        "lightning_ne",
        "fwi2025_danger",
        "lightning_verif",
    ),
    "gefs_control": ("gefs_control_fourpanel",),
    "ecmwf_control": ("ecmwf_control_fourpanel",),
}

DEFAULT_PRODUCTS = {
    "continental": "continental_fourpanel",
    "west": "lightning_sw",
    "gefs_control": "gefs_control_fourpanel",
    "ecmwf_control": "ecmwf_control_fourpanel",
}

RETIRED_PRODUCTS: dict[str, tuple[str, ...]] = {
    "continental": ("continental_convective", "continental_lightning"),
    "west": ("convective",),
}
RETIRED_PRODUCTS_VERSION = "2026-07-21-v1"


class R2ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_base_url: str

    @classmethod
    def from_environment(cls) -> "R2Config":
        access_key_id = os.environ.get("FCST_R2_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.environ.get("FCST_R2_SECRET_ACCESS_KEY", "").strip()
        if not access_key_id:
            access_key_id = keychain_password(KEYCHAIN_ACCESS_KEY_SERVICE)
        if not secret_access_key:
            secret_access_key = keychain_password(KEYCHAIN_SECRET_KEY_SERVICE)
        values = {
            "account_id": os.environ.get("FCST_R2_ACCOUNT_ID", "").strip(),
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "bucket": os.environ.get("FCST_R2_BUCKET", "").strip(),
            "public_base_url": os.environ.get("FCST_R2_PUBLIC_BASE_URL", "").strip().rstrip("/"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise R2ConfigurationError("Missing R2 configuration: " + ", ".join(missing))
        return cls(**values)

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


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


@dataclass(frozen=True)
class LocalFrame:
    model: str
    product_key: str
    stamp: str
    hour: int
    path: Path
    object_key: str


class PublishState:
    def __init__(self, path: Path, storage_scope: str = ""):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                object_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                product_key TEXT NOT NULL,
                stamp TEXT NOT NULL,
                forecast_hour INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                format_version TEXT NOT NULL DEFAULT '',
                uploaded_at TEXT NOT NULL
            )
            """
        )
        columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(artifacts)").fetchall()
        }
        if "format_version" not in columns:
            self.connection.execute(
                "ALTER TABLE artifacts ADD COLUMN format_version TEXT NOT NULL DEFAULT ''"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS artifacts_model_stamp ON artifacts(model, stamp)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        previous_scope = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'storage_scope'"
        ).fetchone()
        if previous_scope is not None and str(previous_scope[0]) != storage_scope:
            self.connection.execute("DELETE FROM artifacts")
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES('storage_scope', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (storage_scope,),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def metadata(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.connection.commit()

    def unchanged(self, frame: LocalFrame) -> bool:
        row = self.connection.execute(
            "SELECT size_bytes, mtime_ns, format_version FROM artifacts WHERE object_key = ?",
            (frame.object_key,),
        ).fetchone()
        if row is None:
            return False
        stat = frame.path.stat()
        return (
            int(row["size_bytes"]) == stat.st_size
            and int(row["mtime_ns"]) == stat.st_mtime_ns
            and str(row["format_version"]) == PUBLISH_FORMAT_VERSION
        )

    def record(self, frame: LocalFrame, sha256: str) -> None:
        stat = frame.path.stat()
        now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        self.connection.execute(
            """
            INSERT INTO artifacts (
                object_key, model, product_key, stamp, forecast_hour, source_path,
                size_bytes, mtime_ns, sha256, format_version, uploaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(object_key) DO UPDATE SET
                source_path = excluded.source_path,
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                sha256 = excluded.sha256,
                format_version = excluded.format_version,
                uploaded_at = excluded.uploaded_at
            """,
            (
                frame.object_key,
                frame.model,
                frame.product_key,
                frame.stamp,
                frame.hour,
                str(frame.path),
                stat.st_size,
                stat.st_mtime_ns,
                sha256,
                PUBLISH_FORMAT_VERSION,
                now,
            ),
        )
        self.connection.commit()

    def retained_rows(self, model: str, now: dt.datetime | None = None) -> list[sqlite3.Row]:
        now = now or dt.datetime.now(dt.timezone.utc)
        rows = self.connection.execute(
            "SELECT * FROM artifacts WHERE model = ? ORDER BY stamp DESC, product_key, forecast_hour",
            (model,),
        ).fetchall()
        retained = []
        for row in rows:
            init = parse_stamp(str(row["stamp"]))
            keep_days = retention_days(str(row["product_key"]))
            if init >= now - dt.timedelta(days=keep_days):
                retained.append(row)
        return retained

    def prune_expired(self, model: str, now: dt.datetime | None = None) -> int:
        retained_keys = {str(row["object_key"]) for row in self.retained_rows(model, now)}
        rows = self.connection.execute(
            "SELECT object_key FROM artifacts WHERE model = ?", (model,)
        ).fetchall()
        expired = [str(row["object_key"]) for row in rows if str(row["object_key"]) not in retained_keys]
        if expired:
            self.connection.executemany("DELETE FROM artifacts WHERE object_key = ?", ((key,) for key in expired))
            self.connection.commit()
        return len(expired)

    def prune_inactive_products(self, model: str, active_products: Iterable[str]) -> int:
        active = tuple(active_products)
        rows = self.connection.execute(
            "SELECT object_key, product_key FROM artifacts WHERE model = ?", (model,)
        ).fetchall()
        stale = [
            str(row["object_key"])
            for row in rows
            if str(row["product_key"]) not in active
        ]
        if stale:
            self.connection.executemany(
                "DELETE FROM artifacts WHERE object_key = ?", ((key,) for key in stale)
            )
            self.connection.commit()
        return len(stale)


def retention_days(product_key: str) -> int:
    return VERIFICATION_KEEP_DAYS if product_key in VERIFICATION_PRODUCT_KEYS else FORECAST_KEEP_DAYS


def retention_class(product_key: str) -> str:
    return "verification" if product_key in VERIFICATION_PRODUCT_KEYS else "forecast"


def object_key_for(model: str, product_key: str, stamp: str, filename: str) -> str:
    return f"models/{model}/{retention_class(product_key)}/{product_key}/{stamp}/{filename}"


def source_root(product_key: str) -> Path:
    product = PRODUCTS[product_key]
    return default_plot_dir_for_product(product_key, product.model_key)


def candidate_stamps(model: str, sync_retained: bool, requested_stamp: str | None) -> list[str]:
    if requested_stamp:
        return [requested_stamp]
    if not sync_retained:
        return []
    now = dt.datetime.now(dt.timezone.utc)
    stamps = set()
    for product_key in MODEL_PRODUCTS[model]:
        root = source_root(product_key)
        if not root.exists():
            continue
        keep_days = retention_days(product_key)
        cutoff = now - dt.timedelta(days=keep_days)
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                init = parse_stamp(child.name)
            except ValueError:
                continue
            if init >= cutoff:
                stamps.add(child.name)
    return sorted(stamps)


def discover_frames(
    model: str,
    stamps: Iterable[str],
    enforce_retention: bool = False,
    now: dt.datetime | None = None,
) -> list[LocalFrame]:
    frames: list[LocalFrame] = []
    now = now or dt.datetime.now(dt.timezone.utc)
    for stamp in sorted(set(stamps)):
        init = parse_stamp(stamp)
        for product_key in MODEL_PRODUCTS[model]:
            if enforce_retention and init < now - dt.timedelta(days=retention_days(product_key)):
                continue
            product = PRODUCTS[product_key]
            plot_dir = source_root(product_key) / stamp
            for hour in product.hours:
                filename = image_name_for_hour(stamp, product_key, hour)
                path = plot_dir / filename
                if path.exists():
                    frames.append(
                        LocalFrame(
                            model=model,
                            product_key=product_key,
                            stamp=stamp,
                            hour=hour,
                            path=path,
                            object_key=object_key_for(model, product_key, stamp, filename),
                        )
                    )
    return frames


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete_png(path: Path) -> bool:
    try:
        if path.stat().st_size < len(PNG_END_MARKER):
            return False
        with path.open("rb") as handle:
            handle.seek(-len(PNG_END_MARKER), os.SEEK_END)
            return handle.read() == PNG_END_MARKER
    except OSError:
        return False


@contextmanager
def optimized_png(source: Path):
    pngquant = shutil.which("pngquant")
    if pngquant is None:
        yield source
        return
    with tempfile.TemporaryDirectory(prefix="fcst-r2-png-") as tmp:
        target = Path(tmp) / source.name
        result = subprocess.run(
            [
                pngquant,
                "--force",
                "--strip",
                "--quality=70-90",
                "--speed=1",
                "--output",
                str(target),
                str(source),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and target.exists() and target.stat().st_size < source.stat().st_size:
            yield target
        else:
            yield source


def retry(operation, description: str, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt >= attempts:
                raise
            delay = min(60.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            print(f"{description} failed; retrying in {delay:.1f}s ({attempt}/{attempts}).", flush=True)
            time.sleep(delay)


def boto3_client(config: R2Config):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is required for R2 publication") from exc
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 1}, connect_timeout=15, read_timeout=60),
    )


def upload_frame(client, config: R2Config, frame: LocalFrame, body_path: Path, sha256: str) -> None:
    def put():
        with body_path.open("rb") as body:
            client.put_object(
                Bucket=config.bucket,
                Key=frame.object_key,
                Body=body,
                ContentType="image/png",
                CacheControl=IMAGE_CACHE_CONTROL,
                Metadata={"sha256": sha256},
            )

    retry(put, f"Upload {frame.object_key}")


def purge_retired_objects(client, config: R2Config, model: str) -> int:
    deleted = 0
    for product_key in RETIRED_PRODUCTS.get(model, ()):
        prefix = f"models/{model}/forecast/{product_key}/"
        continuation_token: str | None = None
        while True:
            request: dict[str, object] = {"Bucket": config.bucket, "Prefix": prefix}
            if continuation_token:
                request["ContinuationToken"] = continuation_token
            response = retry(
                lambda request=request: client.list_objects_v2(**request),
                f"List retired prefix {prefix}",
            )
            keys = [str(item["Key"]) for item in response.get("Contents", ())]
            for offset in range(0, len(keys), 1000):
                batch = keys[offset : offset + 1000]
                retry(
                    lambda batch=batch: client.delete_objects(
                        Bucket=config.bucket,
                        Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
                    ),
                    f"Delete retired prefix {prefix}",
                )
                deleted += len(batch)
            if not response.get("IsTruncated"):
                break
            continuation_token = str(response["NextContinuationToken"])
    return deleted


def optimize_and_upload(client, config: R2Config, frame: LocalFrame) -> tuple[LocalFrame, str]:
    with optimized_png(frame.path) as upload_path:
        sha256 = sha256_file(upload_path)
        upload_frame(client, config, frame, upload_path, sha256)
    return frame, sha256


def asset_version(stamp: str, product_key: str, rows: Iterable[sqlite3.Row]) -> str:
    digest = hashlib.blake2s(digest_size=8)
    for row in sorted(rows, key=lambda item: int(item["forecast_hour"])):
        digest.update(str(row["forecast_hour"]).encode("ascii"))
        digest.update(str(row["sha256"]).encode("ascii"))
    return f"{stamp}-{product_key}-{digest.hexdigest()}"


def build_manifest(
    model: str,
    rows: Iterable[sqlite3.Row],
    public_base_url: str,
    generated: dt.datetime | None = None,
) -> dict[str, object]:
    generated = generated or dt.datetime.now(dt.timezone.utc)
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        product_key = str(row["product_key"])
        if product_key not in MODEL_PRODUCTS[model] or product_key not in PRODUCTS:
            continue
        grouped.setdefault((str(row["stamp"]), product_key), []).append(row)

    runs: dict[str, dict[str, object]] = {}
    for (stamp, product_key), product_rows in grouped.items():
        product = PRODUCTS[product_key]
        product_rows.sort(key=lambda item: int(item["forecast_hour"]))
        if len(product_rows) < minimum_manifest_hours(product_key):
            continue
        init = parse_stamp(stamp)
        base_key = object_key_for(model, product_key, stamp, "").rstrip("/") + "/"
        product_record = {
            "key": product.key,
            "label": product.label,
            "category": product.category,
            "plotType": product.plot_type,
            "area": product.area,
            "model": product.model,
            "modelKey": product.model_key,
            "description": product.description,
            "hours": [int(row["forecast_hour"]) for row in product_rows],
            "imageBase": f"{public_base_url}/{base_key}",
            "filePrefix": f"{product.prefix}_{stamp}",
            "assetVersion": asset_version(stamp, product_key, product_rows),
            "validStart": init.isoformat().replace("+00:00", "Z"),
        }
        run = runs.setdefault(
            stamp,
            {
                "stamp": stamp,
                "init": init.isoformat().replace("+00:00", "Z"),
                "label": f"{init:%Y-%m-%d %HZ}",
                "products": {},
            },
        )
        run["products"][product_key] = product_record

    ordered_runs = sorted(runs.values(), key=lambda item: str(item["init"]), reverse=True)
    for run in ordered_runs:
        products: Mapping[str, dict[str, object]] = run["products"]
        default = products.get(DEFAULT_PRODUCTS[model]) or next(iter(products.values()))
        run["assetVersion"] = default["assetVersion"]
        run["imageBase"] = default["imageBase"]

    return {
        "schemaVersion": 1,
        "model": model,
        "generated": generated.isoformat().replace("+00:00", "Z"),
        "retentionDays": FORECAST_KEEP_DAYS,
        "verificationRetentionDays": VERIFICATION_KEEP_DAYS,
        "defaultProduct": DEFAULT_PRODUCTS[model],
        "runs": ordered_runs,
    }


def upload_manifest(client, config: R2Config, model: str, manifest: Mapping[str, object]) -> str:
    key = f"manifests/{model}.json"
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    retry(
        lambda: client.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=payload,
            ContentType="application/json; charset=utf-8",
            CacheControl=MANIFEST_CACHE_CONTROL,
        ),
        f"Upload {key}",
    )
    return f"{config.public_base_url}/{key}"


def publish_model(
    model: str,
    stamp: str | None = None,
    sync_retained: bool = False,
    config: R2Config | None = None,
    state_path: Path | None = None,
    client=None,
) -> dict[str, object]:
    if model not in MODEL_PRODUCTS:
        raise ValueError(f"Unsupported R2 model group: {model}")
    config = config or R2Config.from_environment()
    client = client or boto3_client(config)
    state = PublishState(
        state_path or STATE_ROOT / f"r2_{model}.sqlite3",
        storage_scope=f"{config.account_id}/{config.bucket}",
    )
    try:
        retirement_key = f"retired_products:{model}"
        retired_deleted = 0
        if state.metadata(retirement_key) != RETIRED_PRODUCTS_VERSION:
            retired_deleted = purge_retired_objects(client, config, model)
            state.set_metadata(retirement_key, RETIRED_PRODUCTS_VERSION)
        inactive_pruned = state.prune_inactive_products(model, MODEL_PRODUCTS[model])
        stamps = candidate_stamps(model, sync_retained, stamp)
        frames = discover_frames(
            model,
            stamps,
            enforce_retention=sync_retained and stamp is None,
        )
        uploaded = 0
        skipped = 0
        incomplete = 0
        pending: list[LocalFrame] = []
        for frame in frames:
            if state.unchanged(frame):
                skipped += 1
                continue
            if not is_complete_png(frame.path):
                incomplete += 1
                continue
            pending.append(frame)

        workers = max(1, min(8, int(os.environ.get("FCST_R2_UPLOAD_WORKERS", "2"))))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(optimize_and_upload, client, config, frame): frame for frame in pending
            }
            for future in as_completed(futures):
                frame, sha256 = future.result()
                state.record(frame, sha256)
                uploaded += 1
        pruned = state.prune_expired(model)
        manifest = build_manifest(model, state.retained_rows(model), config.public_base_url)
        manifest_url = upload_manifest(client, config, model, manifest)
        print(
            f"R2 {model}: uploaded={uploaded}, unchanged={skipped}, incomplete={incomplete}, "
            f"state_pruned={pruned}, inactive_pruned={inactive_pruned}, retired_deleted={retired_deleted}, "
            f"runs={len(manifest['runs'])}, manifest={manifest_url}",
            flush=True,
        )
        return {
            "model": model,
            "uploaded": uploaded,
            "unchanged": skipped,
            "incomplete": incomplete,
            "state_pruned": pruned,
            "inactive_pruned": inactive_pruned,
            "retired_deleted": retired_deleted,
            "manifest_url": manifest_url,
            "runs": len(manifest["runs"]),
        }
    finally:
        state.close()
