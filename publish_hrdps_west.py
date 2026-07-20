#!/usr/bin/env python3
"""Publish forecast plot images into the fcstpp GitHub Pages checkout."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from make_ensemble_control_fourpanel import FORECAST_HOURS as ENSEMBLE_CONTROL_FORECAST_HOURS
from make_ensemble_control_fourpanel import GEFS_FORECAST_HOURS as GEFS_CONTROL_FORECAST_HOURS
from make_hrdps_west_convective import FORECAST_HOURS as CONVECTIVE_FORECAST_HOURS
from make_hrdps_west_convective import parse_stamp
from make_hrdps_west_fourpanel import FORECAST_HOURS as FOURPANEL_FORECAST_HOURS
from make_hrdps_west_lightning import FORECAST_HOURS as LIGHTNING_FORECAST_HOURS
from make_hrdps_evolved_danger_class import FORECAST_HOURS as FWI2025_DANGER_FORECAST_HOURS

DEFAULT_PAGES_REPO = Path("/Users/greg/projects/fcstpp-reports-pages")
STAMP_RE = re.compile(r"^\d{8}T\d{2}Z$")
CONTACT_SHEET_RE = re.compile(r".*_contact_sheet\.png$")
TOP_LEVEL_IMAGE_RE = re.compile(r"^hrdps_west_(?:convective|fourpanel|lightning)_(\d{8}T\d{2}Z)_f\d{3}\.png$")
VERIFICATION_PRODUCT_KEYS = frozenset({"lightning_verif", "continental_lightning_verif", "fire_danger_verif"})
VERIFICATION_KEEP_DAYS = 60
MIN_MANIFEST_FRAME_FRACTION = 0.60
PNGQUANT_QUALITY = "70-90"
FIRE_WEATHER_DESCRIPTION = (
    "Fire-weather ingredients: three-hour maximum all-cause gust colored vectors (regular HRDPS gust, ECCC-style "
    "downslope adjustment, and triggered PCGE), experimental peak-daily BC fire-danger contours, 10 m RH hatching "
    "(brown at 21-30%, dark brown at 20% or less, pale blue at 61-80%, and blue above 80%), "
    "three-hour maximum LPI shading, and three-hour maximum dry-lightning asterisks."
)
FIRE_WEATHER_TWO_PANEL_DESCRIPTION = (
    "Two-panel fire weather: valid-time categorical 10 m RH and three-hour maximum colored all-cause gust vectors "
    "on the left; experimental peak-daily BC fire-danger categories, three-hour maximum LPI contours and "
    "dry-lightning asterisks, "
    "and 3-hour precipitation dots at 2.5 and 10 mm on the right."
)


@dataclass(frozen=True)
class ProductConfig:
    key: str
    prefix: str
    label: str
    category: str
    plot_type: str
    area: str
    model: str
    description: str
    hours: tuple[int, ...]
    archive_subdir: str
    model_key: str


PRODUCTS: dict[str, ProductConfig] = {
    "fourpanel": ProductConfig(
        key="fourpanel",
        prefix="hrdps_west_fourpanel",
        label="Convective 4-Panel",
        category="4-Panel",
        plot_type="Convective 4-Panel",
        area="BC",
        model="HRDPS-West 1 km",
        description="500 hPa vorticity/height/250 hPa wind, IPW/LI/CAPE, 850-700 hPa RH/850 hPa temperature/850-or-700 hPa wind, and 3-hour precipitation/MSLP/10 m wind.",
        hours=FOURPANEL_FORECAST_HOURS,
        archive_subdir="fourpanel",
        model_key="west",
    ),
    "convective": ProductConfig(
        key="convective",
        prefix="hrdps_west_convective",
        label="Convective Gust Potential",
        category="Surface",
        plot_type="Convective Gust Potential",
        area="BC",
        model="HRDPS-West 1 km",
        description="DCAPE shading, LI contours, storm-relative helicity contours, and PCGE hatching at 60 and 90 km/h.",
        hours=CONVECTIVE_FORECAST_HOURS,
        archive_subdir="",
        model_key="west",
    ),
    "lightning": ProductConfig(
        key="lightning",
        prefix="hrdps_west_lightning",
        label="Fire Weather",
        category="Surface",
        plot_type="Fire Weather",
        area="BC",
        model="HRDPS-West 1 km",
        description=FIRE_WEATHER_DESCRIPTION,
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="lightning",
        model_key="west",
    ),
    "fwi2025_danger": ProductConfig(
        key="fwi2025_danger",
        prefix="hrdps_west_fwi2025_danger",
        label="Experimental Hourly Fire Danger",
        category="Surface",
        plot_type="Experimental Hourly Fire Danger",
        area="BC",
        model="HRDPS-West 1 km",
        description="Hourly FWI2025 evolution anchored to CWFIS FFMC/DMC/DC, classified with the BC Schedule 2 FWI+BUI danger-region matrices and lightly smoothed over 2 km for display.",
        hours=FWI2025_DANGER_FORECAST_HOURS,
        archive_subdir="fwi2025_danger",
        model_key="west",
    ),
    "continental_fwi2025_danger": ProductConfig(
        key="continental_fwi2025_danger",
        prefix="hrdps_continental_fwi2025_danger",
        label="Experimental Hourly Fire Danger",
        category="Surface",
        plot_type="Experimental Hourly Fire Danger",
        area="BC",
        model="HRDPS 2.5 km",
        description="Hourly FWI2025 evolution anchored to CWFIS FFMC/DMC/DC, classified with the BC Schedule 2 FWI+BUI danger-region matrices and lightly smoothed over 2 km for display.",
        hours=FWI2025_DANGER_FORECAST_HOURS,
        archive_subdir="continental/fwi2025_danger",
        model_key="continental",
    ),
    "lightning_sw": ProductConfig(
        key="lightning_sw",
        prefix="hrdps_west_lightning_sw",
        label="Fire Weather",
        category="Surface",
        plot_type="Fire Weather",
        area="SW BC",
        model="HRDPS-West 1 km",
        description=FIRE_WEATHER_TWO_PANEL_DESCRIPTION,
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="lightning/sw",
        model_key="west",
    ),
    "lightning_se": ProductConfig(
        key="lightning_se",
        prefix="hrdps_west_lightning_se",
        label="Fire Weather",
        category="Surface",
        plot_type="Fire Weather",
        area="SE BC",
        model="HRDPS-West 1 km",
        description=FIRE_WEATHER_TWO_PANEL_DESCRIPTION,
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="lightning/se",
        model_key="west",
    ),
    "lightning_ne": ProductConfig(
        key="lightning_ne",
        prefix="hrdps_west_lightning_ne",
        label="Fire Weather",
        category="Surface",
        plot_type="Fire Weather",
        area="NE BC",
        model="HRDPS-West 1 km",
        description=FIRE_WEATHER_TWO_PANEL_DESCRIPTION,
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="lightning/ne",
        model_key="west",
    ),
    "lightning_verif": ProductConfig(
        key="lightning_verif",
        prefix="hrdps_west_lightning_verif",
        label="LPI Verification",
        category="Surface",
        plot_type="LPI Verification",
        area="BC",
        model="HRDPS-West 1 km",
        description="First 12Z-12Z LPI forecast-period maximum shading with observed 24-hour ECCC lightning-density categories overlaid.",
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="lightning_verif",
        model_key="west",
    ),
    "continental_fourpanel": ProductConfig(
        key="continental_fourpanel",
        prefix="hrdps_continental_fourpanel",
        label="Convective 4-Panel",
        category="4-Panel",
        plot_type="Convective 4-Panel",
        area="BC",
        model="HRDPS 2.5 km",
        description="500 hPa vorticity/height/250 hPa wind, IPW/LI/CAPE, 850-700 hPa RH/850 hPa temperature/850-or-700 hPa wind, and 3-hour precipitation/MSLP/10 m wind.",
        hours=FOURPANEL_FORECAST_HOURS,
        archive_subdir="continental/fourpanel",
        model_key="continental",
    ),
    "continental_convective": ProductConfig(
        key="continental_convective",
        prefix="hrdps_continental_convective",
        label="Convective Gust Potential",
        category="Surface",
        plot_type="Convective Gust Potential",
        area="BC",
        model="HRDPS 2.5 km",
        description="DCAPE shading, LI contours, storm-relative helicity contours, and PCGE hatching at 60 and 90 km/h.",
        hours=CONVECTIVE_FORECAST_HOURS,
        archive_subdir="continental",
        model_key="continental",
    ),
    "continental_lightning": ProductConfig(
        key="continental_lightning",
        prefix="hrdps_continental_lightning",
        label="Fire Weather 1-panel",
        category="Surface",
        plot_type="Fire Weather 1-panel",
        area="BC",
        model="HRDPS 2.5 km",
        description=FIRE_WEATHER_DESCRIPTION,
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="continental/lightning",
        model_key="continental",
    ),
    "continental_lightning_twopanel": ProductConfig(
        key="continental_lightning_twopanel",
        prefix="hrdps_continental_lightning_twopanel",
        label="Fire Weather",
        category="Surface",
        plot_type="Fire Weather",
        area="BC",
        model="HRDPS 2.5 km",
        description=FIRE_WEATHER_TWO_PANEL_DESCRIPTION,
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="continental/lightning_twopanel",
        model_key="continental",
    ),
    "continental_lightning_verif": ProductConfig(
        key="continental_lightning_verif",
        prefix="hrdps_continental_lightning_verif",
        label="LPI Verification",
        category="Surface",
        plot_type="LPI Verification",
        area="BC",
        model="HRDPS 2.5 km",
        description="First 12Z-12Z LPI forecast-period maximum shading with observed 24-hour ECCC lightning-density categories overlaid.",
        hours=LIGHTNING_FORECAST_HOURS,
        archive_subdir="continental/lightning_verif",
        model_key="continental",
    ),
    "fire_danger_verif": ProductConfig(
        key="fire_danger_verif",
        prefix="fire_danger_verification",
        label="Fire Danger Verification",
        category="Verification",
        plot_type="Fire Danger Verification",
        area="BC",
        model="HRDPS + BCWS",
        description="Previous-day and running station verification of experimental FWI2025 peak-daily BC danger classes, with BCWS forecast danger classes included as a benchmark when archived.",
        hours=(0,),
        archive_subdir="fire_danger_verif",
        model_key="verification",
    ),
    "ecmwf_control_fourpanel": ProductConfig(
        key="ecmwf_control_fourpanel",
        prefix="ecmwf_control_fourpanel",
        label="Synoptic 4-Panel",
        category="4-Panel",
        plot_type="Synoptic 4-Panel",
        area="BC",
        model="ECMWF IFS Control",
        description="500 hPa vorticity/height/wind, IPW/vapor transport/low-level vertical velocity where available, 850-700 hPa RH/850 hPa temperature/850-or-700 hPa wind, and 3-hour precipitation/MSLP/10 m wind.",
        hours=ENSEMBLE_CONTROL_FORECAST_HOURS,
        archive_subdir="ecmwf/control/fourpanel",
        model_key="ecmwf_control",
    ),
    "gefs_control_fourpanel": ProductConfig(
        key="gefs_control_fourpanel",
        prefix="gefs_control_fourpanel",
        label="Synoptic 4-Panel",
        category="4-Panel",
        plot_type="Synoptic 4-Panel",
        area="BC",
        model="GEFS Control",
        description="500 hPa vorticity/height/wind, IPW/vapor transport/low-level vertical velocity where available, 850-700 hPa RH/850 hPa temperature/850-or-700 hPa wind, and 3-hour precipitation/MSLP/10 m wind.",
        hours=GEFS_CONTROL_FORECAST_HOURS,
        archive_subdir="gefs/control/fourpanel",
        model_key="gefs_control",
    ),
}

PRODUCTS_BY_MODEL = {
    "west": ("convective", "lightning_sw", "lightning_se", "lightning_ne", "fwi2025_danger"),
    "continental": (
        "continental_fourpanel",
        "continental_convective",
        "continental_lightning",
        "continental_lightning_twopanel",
        "continental_fwi2025_danger",
    ),
    "west_verif": ("lightning_verif",),
    "continental_verif": ("continental_lightning_verif",),
    "danger_verif": ("fire_danger_verif",),
    "ecmwf_control": ("ecmwf_control_fourpanel",),
    "gefs_control": ("gefs_control_fourpanel",),
}


def manifest_product_keys() -> tuple[str, ...]:
    """Products that should appear on the public menu/manifest."""
    keys: list[str] = []
    for product_keys in PRODUCTS_BY_MODEL.values():
        for key in product_keys:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stamp", help="Run stamp to publish, e.g. 20260629T12Z.")
    parser.add_argument("--model", choices=sorted(PRODUCTS_BY_MODEL), default="west")
    parser.add_argument("--plots-dir", type=Path, default=None, help="Convective plot directory.")
    parser.add_argument("--fourpanel-plots-dir", type=Path, default=None)
    parser.add_argument("--lightning-plots-dir", type=Path, default=None)
    parser.add_argument("--danger-plots-dir", type=Path, default=None)
    parser.add_argument("--lightning-verif-plots-dir", type=Path, default=None)
    parser.add_argument("--pages-repo", type=Path, default=DEFAULT_PAGES_REPO)
    parser.add_argument("--keep-days", type=int, default=7)
    parser.add_argument("--partial", action="store_true", help="Publish available frames instead of requiring full product sets.")
    return parser.parse_args(list(argv))


def log(message: str) -> None:
    print(message, flush=True)


def optimize_png_for_pages(path: Path) -> None:
    pngquant = shutil.which("pngquant")
    if pngquant is None:
        return
    tmp_path = path.with_suffix(path.suffix + ".pngquant.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [
                pngquant,
                "--force",
                "--strip",
                f"--quality={PNGQUANT_QUALITY}",
                "--speed=1",
                "--output",
                str(tmp_path),
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0 or not tmp_path.exists():
            return
        if tmp_path.stat().st_size < path.stat().st_size:
            tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def expected_images(stamp: str, product_key: str = "convective") -> list[str]:
    product = PRODUCTS[product_key]
    return [f"{product.prefix}_{stamp}_f{fhour:03d}.png" for fhour in product.hours]


def image_name_for_hour(stamp: str, product_key: str, fhour: int) -> str:
    product = PRODUCTS[product_key]
    return f"{product.prefix}_{stamp}_f{fhour:03d}.png"


def available_image_names(plot_dir: Path, stamp: str, product_key: str) -> list[str]:
    product = PRODUCTS[product_key]
    return [image_name_for_hour(stamp, product_key, fhour) for fhour in product.hours if (plot_dir / image_name_for_hour(stamp, product_key, fhour)).exists()]


def default_plot_dir_for_product(product_key: str, model: str) -> Path:
    if product_key == "fire_danger_verif":
        return Path("plots/fire_danger_verification")
    if product_key.endswith("lightning_verif"):
        if product_key.startswith("continental_"):
            return Path("plots/hrdps_continental_lightning_verif")
        return Path("plots/hrdps_west_lightning_verif")
    if product_key.startswith("continental_lightning"):
        return Path("plots/hrdps_continental_lightning")
    if product_key.startswith("lightning"):
        return Path("plots/hrdps_west_lightning")
    if product_key.endswith("fwi2025_danger"):
        return Path("plots/experimental_fwi2025_danger")
    if product_key == "continental_convective":
        return Path("plots/hrdps_continental")
    if product_key == "continental_fourpanel":
        return Path("plots/hrdps_continental_fourpanel")
    if product_key == "convective":
        return Path("plots/hrdps_west")
    if product_key == "fourpanel":
        return Path("plots/hrdps_west_fourpanel")
    if product_key == "ecmwf_control_fourpanel":
        return Path("plots/ecmwf_control_fourpanel")
    if product_key == "gefs_control_fourpanel":
        return Path("plots/gefs_control_fourpanel")
    raise ValueError(f"Unsupported product for {model}: {product_key}")


def plot_dir_for_product(
    product_key: str,
    model: str,
    plots_dir: Path | None,
    fourpanel_plots_dir: Path | None,
    lightning_plots_dir: Path | None,
    danger_plots_dir: Path | None = None,
    lightning_verif_plots_dir: Path | None = None,
) -> Path:
    if product_key in VERIFICATION_PRODUCT_KEYS:
        return lightning_verif_plots_dir or default_plot_dir_for_product(product_key, model)
    if product_key.endswith("fourpanel") or product_key.endswith("_fourpanel"):
        return fourpanel_plots_dir or default_plot_dir_for_product(product_key, model)
    if "lightning" in product_key:
        return lightning_plots_dir or default_plot_dir_for_product(product_key, model)
    if product_key.endswith("fwi2025_danger"):
        return danger_plots_dir or default_plot_dir_for_product(product_key, model)
    return plots_dir or default_plot_dir_for_product(product_key, model)


def ensure_plot_set(plot_dir: Path, stamp: str, product_key: str, partial: bool = False) -> None:
    if partial:
        if not available_image_names(plot_dir, stamp, product_key):
            raise RuntimeError(f"{product_key} plot set for {stamp} has no available files.")
        return
    missing = [name for name in expected_images(stamp, product_key) if not (plot_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{product_key} plot set for {stamp} is incomplete; missing {len(missing)} files.")


def archive_dir_for_product(run_dir: Path, product_key: str) -> Path:
    subdir = PRODUCTS[product_key].archive_subdir
    return run_dir / subdir if subdir else run_dir


def copy_product_images(plot_dir: Path, run_dir: Path, stamp: str, product_key: str, partial: bool = False) -> Path:
    archive_dir = archive_dir_for_product(run_dir, product_key)
    archive_dir.mkdir(parents=True, exist_ok=True)
    names = available_image_names(plot_dir, stamp, product_key) if partial else expected_images(stamp, product_key)
    if product_key in VERIFICATION_PRODUCT_KEYS:
        product = PRODUCTS[product_key]
        for stale in archive_dir.glob(f"{product.prefix}_{stamp}_f*.png"):
            stale.unlink()
    for name in names:
        dest = archive_dir / name
        shutil.copy2(plot_dir / name, dest)
        optimize_png_for_pages(dest)
    return archive_dir


def cutoff_time(days: int) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)


def retention_days_for_product(product_key: str, default_keep_days: int) -> int:
    if product_key in VERIFICATION_PRODUCT_KEYS:
        return max(default_keep_days, VERIFICATION_KEEP_DAYS)
    return default_keep_days


def remove_empty_dirs(path: Path, stop_at: Path) -> None:
    stop_at = stop_at.resolve()
    current = path
    while current.exists() and current.is_dir() and current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def prune_product_images(run_dir: Path, stamp: str, product_key: str, cutoff: dt.datetime) -> None:
    if parse_stamp(stamp) >= cutoff:
        return
    product = PRODUCTS[product_key]
    archive_dir = archive_dir_for_product(run_dir, product_key)
    if not archive_dir.exists():
        return
    removed = False
    for path in archive_dir.glob(f"{product.prefix}_{stamp}_f*.png"):
        log(f"Removing archived {product_key} image older than retention: {path}")
        path.unlink()
        removed = True
    if removed and product.archive_subdir:
        remove_empty_dirs(archive_dir, run_dir)


def prune_archived_images(images_root: Path, keep_days: int) -> None:
    if not images_root.exists():
        return
    for child in images_root.iterdir():
        if child.is_dir() and STAMP_RE.match(child.name):
            for product_key in PRODUCTS:
                prune_product_images(
                    child,
                    child.name,
                    product_key,
                    cutoff_time(retention_days_for_product(product_key, keep_days)),
                )
            remove_empty_dirs(child, images_root)
        elif child.is_file():
            match = TOP_LEVEL_IMAGE_RE.match(child.name)
            if match and parse_stamp(match.group(1)) < cutoff_time(keep_days):
                log(f"Removing legacy top-level image older than retention: {child}")
                child.unlink()


def prune_contact_sheets(images_root: Path) -> None:
    if not images_root.exists():
        return
    for path in images_root.rglob("*_contact_sheet.png"):
        if path.is_file() and CONTACT_SHEET_RE.match(path.name):
            log(f"Removing deprecated overview image: {path}")
            path.unlink()


def minimum_manifest_hours(product_key: str) -> int:
    if product_key in VERIFICATION_PRODUCT_KEYS:
        return 1
    return max(1, math.ceil(len(PRODUCTS[product_key].hours) * MIN_MANIFEST_FRAME_FRACTION))


def product_asset_version(stamp: str, product_key: str, frame_paths: Iterable[Path]) -> str:
    """Return a stable cache key that changes when any published frame changes."""
    digest = hashlib.blake2s(digest_size=8)
    for path in sorted(frame_paths, key=lambda item: item.name):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return f"{stamp}-{product_key}-{digest.hexdigest()}"


def product_record(stamp: str, product_key: str, pages_dir: Path) -> dict[str, object] | None:
    init = parse_stamp(stamp)
    product = PRODUCTS[product_key]
    archive_dir = archive_dir_for_product(pages_dir, product_key)
    hours = [fhour for fhour in product.hours if (archive_dir / image_name_for_hour(stamp, product_key, fhour)).exists()]
    if len(hours) < minimum_manifest_hours(product_key):
        return None
    frame_paths = [archive_dir / image_name_for_hour(stamp, product_key, fhour) for fhour in hours]
    image_base = f"./images/{stamp}/"
    if product.archive_subdir:
        image_base = f"{image_base}{product.archive_subdir}/"
    return {
        "key": product.key,
        "label": product.label,
        "category": product.category,
        "plotType": product.plot_type,
        "area": product.area,
        "model": product.model,
        "modelKey": product.model_key,
        "description": product.description,
        "hours": hours,
        "imageBase": image_base,
        "filePrefix": f"{product.prefix}_{stamp}",
        "assetVersion": product_asset_version(stamp, product.key, frame_paths),
        "validStart": init.isoformat().replace("+00:00", "Z"),
    }


def run_record(stamp: str, pages_dir: Path) -> dict[str, object]:
    init = parse_stamp(stamp)
    products = {}
    for key in manifest_product_keys():
        record = product_record(stamp, key, pages_dir)
        if record is not None:
            products[key] = record
    if not products:
        raise RuntimeError(f"Archived run {stamp} has no complete product sets in {pages_dir}.")
    default = products.get("continental_convective") or products.get("convective") or next(iter(products.values()))
    return {
        "stamp": stamp,
        "init": init.isoformat().replace("+00:00", "Z"),
        "label": f"{init:%Y-%m-%d %HZ}",
        "assetVersion": str(default["assetVersion"]),
        "imageBase": str(default["imageBase"]),
        "products": products,
    }


def write_manifest(page_root: Path, keep_days: int) -> list[dict[str, str]]:
    images_root = page_root / "images"
    runs: list[dict[str, str]] = []
    for child in sorted(images_root.iterdir(), reverse=True):
        if not (child.is_dir() and STAMP_RE.match(child.name)):
            continue
        try:
            runs.append(run_record(child.name, child))
        except RuntimeError:
            continue
    runs.sort(key=lambda item: item["init"], reverse=True)
    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "retentionDays": keep_days,
        "verificationRetentionDays": VERIFICATION_KEEP_DAYS,
        "defaultProduct": "continental_fourpanel",
        "runs": runs,
    }
    (page_root / "runs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return runs


def publish(
    stamp: str,
    plots_dir: Path | None,
    pages_repo: Path,
    keep_days: int,
    fourpanel_plots_dir: Path | None = None,
    lightning_plots_dir: Path | None = None,
    danger_plots_dir: Path | None = None,
    lightning_verif_plots_dir: Path | None = None,
    model: str = "west",
    partial: bool = False,
) -> list[dict[str, object]]:
    if not STAMP_RE.match(stamp):
        raise ValueError(f"Invalid run stamp: {stamp}")

    if model not in PRODUCTS_BY_MODEL:
        raise ValueError(f"Unsupported model: {model}")
    page_root = pages_repo / "hrdps-west"
    images_root = page_root / "images"
    run_dir = images_root / stamp
    product_keys = PRODUCTS_BY_MODEL[model]
    if partial:
        copied_any = False
        for product_key in product_keys:
            source_dir = (
                plot_dir_for_product(product_key, model, plots_dir, fourpanel_plots_dir, lightning_plots_dir, danger_plots_dir, lightning_verif_plots_dir)
                / stamp
            )
            if available_image_names(source_dir, stamp, product_key):
                copy_product_images(source_dir, run_dir, stamp, product_key, partial=True)
                copied_any = True
        if not copied_any:
            raise RuntimeError(f"No publishable plot images exist for {model} {stamp}.")
    else:
        for product_key in product_keys:
            source_dir = (
                plot_dir_for_product(product_key, model, plots_dir, fourpanel_plots_dir, lightning_plots_dir, danger_plots_dir, lightning_verif_plots_dir)
                / stamp
            )
            ensure_plot_set(source_dir, stamp, product_key)
            copy_product_images(source_dir, run_dir, stamp, product_key)

    prune_contact_sheets(images_root)
    prune_archived_images(images_root, keep_days)
    runs = write_manifest(page_root, keep_days)
    log(f"Published {stamp}; manifest contains {len(runs)} run(s).")
    return runs


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    publish(
        stamp=args.stamp,
        plots_dir=args.plots_dir,
        pages_repo=args.pages_repo,
        keep_days=args.keep_days,
        fourpanel_plots_dir=args.fourpanel_plots_dir,
        lightning_plots_dir=args.lightning_plots_dir,
        danger_plots_dir=args.danger_plots_dir,
        lightning_verif_plots_dir=args.lightning_verif_plots_dir,
        model=args.model,
        partial=args.partial,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
