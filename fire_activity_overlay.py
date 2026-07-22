#!/usr/bin/env python3
"""Render and publish lightweight live fire-activity overlays for fire-weather maps."""

from __future__ import annotations

import datetime as dt
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import fire_activity
import make_hrdps_fire_weather_twopanel as firewx
import make_hrdps_west_lightning as lightning
import plot_style
from r2_publish import (
    MANIFEST_CACHE_CONTROL,
    R2Config,
    boto3_client,
    optimized_png,
    retry,
    sha256_file,
)


OUTPUT_DIR = Path("plots/fire_activity_overlay")
STATE_PATH = Path("logs/state/fire_activity_overlay.json")
LOCK_PATH = Path("logs/state/fire_activity_overlay.lock")
MANIFEST_KEY = "manifests/fire_activity.json"
OVERLAY_KEY_PREFIX = "live/fire_activity"
OVERLAY_CACHE_CONTROL = "no-cache, max-age=0, must-revalidate"


@dataclass(frozen=True)
class OverlaySpec:
    product_key: str
    model: str
    region_key: str


OVERLAY_SPECS = (
    OverlaySpec("continental_lightning_twopanel", "continental", "bc"),
    OverlaySpec("lightning_sw", "west", "sw"),
    OverlaySpec("lightning_se", "west", "se"),
    OverlaySpec("lightning_ne", "west", "ne"),
)
MINIMUM_BASE_RUN = {
    "continental_lightning_twopanel": "20260722T12Z",
    "lightning_sw": "20260721T12Z",
    "lightning_se": "20260721T12Z",
    "lightning_ne": "20260721T12Z",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def overlay_object_key(product_key: str) -> str:
    return f"{OVERLAY_KEY_PREFIX}/{product_key}.png"


def _transparent_axes(spec: OverlaySpec):
    if spec.region_key == "bc":
        projection = firewx.PLOT_CRS
    else:
        projection = lightning.PLOT_CRS
    fig = plt.figure(
        figsize=plot_style.PLOT_FIGSIZE,
        dpi=plot_style.PLOT_DPI,
        facecolor=(0.0, 0.0, 0.0, 0.0),
    )
    ax = fig.add_axes(firewx.EDGE_BAND_PANEL_POSITIONS[1], projection=projection)
    if spec.region_key == "bc":
        ax.set_xlim(*firewx.PROJECTED_X_LIMITS)
        ax.set_ylim(*firewx.PROJECTED_Y_LIMITS)
    else:
        ax.set_extent(firewx.REGIONAL_EXTENTS[spec.region_key], crs=lightning.DATA_CRS)
    ax.patch.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def render_overlay(
    spec: OverlaySpec,
    activity: fire_activity.FireActivity,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Render one transparent overlay with the exact operational map geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.product_key}.png"
    temporary_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.png")
    fig, ax = _transparent_axes(spec)
    try:
        firewx.add_fire_activity(ax, activity)
        fig.savefig(
            temporary_path,
            dpi=plot_style.PLOT_DPI,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
    finally:
        plt.close(fig)
    temporary_path.replace(output_path)
    return output_path


def render_overlays(
    activity: fire_activity.FireActivity,
    output_dir: Path = OUTPUT_DIR,
    specs: Iterable[OverlaySpec] = OVERLAY_SPECS,
) -> dict[str, Path]:
    return {spec.product_key: render_overlay(spec, activity, output_dir) for spec in specs}


def read_state(path: Path = STATE_PATH) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_state(payload: Mapping[str, object], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def build_manifest(
    config: R2Config,
    activity: fire_activity.FireActivity | None,
    versions: Mapping[str, str],
    generated: dt.datetime | None = None,
) -> dict[str, object]:
    generated = generated or utc_now()
    products = {
        product_key: {
            "image": f"{config.public_base_url}/{overlay_object_key(product_key)}",
            "assetVersion": version,
            "minimumRunStamp": MINIMUM_BASE_RUN[product_key],
        }
        for product_key, version in versions.items()
    }
    return {
        "schemaVersion": 1,
        "generated": generated.isoformat().replace("+00:00", "Z"),
        "available": activity is not None,
        "source": activity.source if activity is not None else None,
        "observationTime": (
            activity.retrieved_at.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            if activity is not None
            else None
        ),
        "cached": bool(activity.cached) if activity is not None else False,
        "stale": bool(activity.stale) if activity is not None else False,
        "observationCount": len(activity.observations) if activity is not None else 0,
        "products": products,
    }


def _put_overlay(client, config: R2Config, product_key: str, body_path: Path, sha256: str) -> None:
    object_key = overlay_object_key(product_key)

    def put() -> None:
        with body_path.open("rb") as body:
            client.put_object(
                Bucket=config.bucket,
                Key=object_key,
                Body=body,
                ContentType="image/png",
                CacheControl=OVERLAY_CACHE_CONTROL,
                Metadata={"sha256": sha256},
            )

    retry(put, f"Upload {object_key}")


def _put_manifest(client, config: R2Config, manifest: Mapping[str, object]) -> None:
    body = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    retry(
        lambda: client.put_object(
            Bucket=config.bucket,
            Key=MANIFEST_KEY,
            Body=body,
            ContentType="application/json; charset=utf-8",
            CacheControl=MANIFEST_CACHE_CONTROL,
        ),
        f"Upload {MANIFEST_KEY}",
    )


def publish_overlays(
    activity: fire_activity.FireActivity | None,
    *,
    output_dir: Path = OUTPUT_DIR,
    state_path: Path = STATE_PATH,
    config: R2Config | None = None,
    client=None,
    force_upload: bool = False,
    generated: dt.datetime | None = None,
) -> dict[str, object]:
    """Render current overlays and atomically advance the small live manifest."""
    config = config or R2Config.from_environment()
    client = client or boto3_client(config)
    previous_state = read_state(state_path)
    previous_versions = previous_state.get("versions")
    previous_versions = previous_versions if isinstance(previous_versions, dict) else {}
    versions: dict[str, str] = {}
    uploaded = 0
    unchanged = 0

    if activity is not None:
        paths = render_overlays(activity, output_dir)
        for product_key, path in paths.items():
            with optimized_png(path) as body_path:
                sha256 = sha256_file(body_path)
                versions[product_key] = sha256
                if force_upload or previous_versions.get(product_key) != sha256:
                    _put_overlay(client, config, product_key, body_path, sha256)
                    uploaded += 1
                else:
                    unchanged += 1

    manifest = build_manifest(config, activity, versions, generated)
    _put_manifest(client, config, manifest)
    write_state(
        {
            "updated": manifest["generated"],
            "versions": versions,
            "source": manifest["source"],
            "observationTime": manifest["observationTime"],
        },
        state_path,
    )
    return {
        "available": manifest["available"],
        "observations": manifest["observationCount"],
        "uploaded": uploaded,
        "unchanged": unchanged,
        "manifest": f"{config.public_base_url}/{MANIFEST_KEY}",
    }


@contextmanager
def overlay_lock(path: Path = LOCK_PATH):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
