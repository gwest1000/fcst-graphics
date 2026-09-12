#!/usr/bin/env python3
"""End-to-end health monitor for forecast retrieval and publication pipelines."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo

import telegram_notify
from fire_danger_peak import MAX_CWFIS_ANCHOR_AGE_AT_INIT_HOURS


PUBLIC_BASE_URL = "https://pub-969ec1fc2e19465797efb65b276a58da.r2.dev"
SITE_URL = "https://gwest1000.github.io/fcst-graphics/"
REPO_ROOT = Path(__file__).resolve().parent
STATE_PATH = Path("logs/state/pipeline_health.json")
LATEST_PATH = Path("logs/pipeline_health_latest.json")
HISTORY_PATH = Path("logs/pipeline_health_history.jsonl")
LOCK_PATH = Path("logs/state/pipeline_health.lock")
HISTORY_MAX_BYTES = 25_000_000
MACHINE_DATA_CONFIG = Path("~/.config/project-data.env").expanduser()
LOCAL_TZ = ZoneInfo("America/Vancouver")
PENDING_DURATION = dt.timedelta(minutes=55)
RECOVERY_HOLD = dt.timedelta(minutes=55)
WARNING_REPEAT = dt.timedelta(hours=24)
CRITICAL_REPEAT = dt.timedelta(hours=6)
MAX_REPORTED_PROBLEMS = 8
RUN_DELAY_GRACE = dt.timedelta(hours=1)
RUN_REMINDER = dt.timedelta(hours=4)
MIN_NOTIFICATION_INTERVAL = dt.timedelta(hours=1)
CWFIS_EARLY_WARNING_AGE = dt.timedelta(hours=36)
CWFIS_FIELDS = ("ffmc", "dmc", "dc")
CWFIS_WCS_URL = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
RADARSAT_HEALTH_PATH = Path(
    os.environ.get(
        "RADARSAT_HEALTH_STATUS_PATH",
        str(Path.home() / "projects/radar-sat/var/status/health.json"),
    )
)

SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class ManifestSpec:
    key: str
    label: str
    maximum_run_age: dt.timedelta


@dataclass(frozen=True)
class PipelineStatusSpec:
    model: str
    cycle: str
    label: str


HRDPS_PIPELINE_STATUSES = tuple(
    PipelineStatusSpec("continental", cycle, f"HRDPS 2.5 km {cycle}Z pipeline")
    for cycle in ("00", "06", "12", "18")
)


@dataclass(frozen=True)
class CheckResult:
    key: str
    label: str
    level: str
    summary: str
    immediate: bool = False
    push_eligible: bool = True


MODEL_MANIFESTS = (
    ManifestSpec("continental", "HRDPS 2.5 km", dt.timedelta(hours=12)),
    ManifestSpec("gefs_control", "GEFS Control", dt.timedelta(hours=36)),
    ManifestSpec("ecmwf_control", "ECMWF Control", dt.timedelta(hours=30)),
    ManifestSpec("ecmwf_ensemble", "ECMWF Ensemble", dt.timedelta(hours=30)),
)

@dataclass(frozen=True)
class RunSchedule:
    model: str
    cycle: int
    label: str
    products: tuple[str, ...]
    hours: tuple[int, ...]
    ready_local: tuple[int, int] | None = None


# Website-ready targets, with room for rendering/upload after scheduled downloads.
SHORT_HOURS = tuple(range(0, 49, 3))
LONG_HOURS = tuple(range(0, 145, 3)) + tuple(range(150, 361, 6))
RUN_SCHEDULES = (
    *(RunSchedule("continental", cycle, "HRDPS", (
        "continental_fourpanel", "continental_lightning_twopanel",
        "continental_temperature", "continental_fwi2025_danger",
    ), SHORT_HOURS) for cycle in (0, 6, 12, 18)),
    RunSchedule("gefs_control", 0, "GEFS", ("gefs_control_fourpanel",), tuple(range(3, 49, 3)), (6, 0)),
    RunSchedule("ecmwf_control", 0, "ECMWF control", (
        "ecmwf_control_fourpanel", "ecmwf_control_convective_fourpanel",
    ), LONG_HOURS, (6, 30)),
    RunSchedule("ecmwf_control", 12, "ECMWF control", (
        "ecmwf_control_fourpanel", "ecmwf_control_convective_fourpanel",
    ), LONG_HOURS, (14, 0)),
    RunSchedule("ecmwf_ensemble", 0, "ECMWF ensemble", ("ecmwf_ensemble_spread_500",), LONG_HOURS, (7, 0)),
    RunSchedule("ecmwf_ensemble", 12, "ECMWF ensemble", ("ecmwf_ensemble_spread_500",), LONG_HOURS, (14, 30)),
)

PRODUCT_LABELS = {
    "continental_fourpanel": "convective four-panel",
    "continental_lightning_twopanel": "fire-weather RH/wind/lightning",
    "continental_temperature": "temperature",
    "continental_fwi2025_danger": "fire-danger",
    "gefs_control_fourpanel": "synoptic four-panel",
    "ecmwf_control_fourpanel": "synoptic four-panel",
    "ecmwf_control_convective_fourpanel": "convective four-panel",
    "ecmwf_ensemble_spread_500": "500-hPa ensemble spread",
}


REQUIRED_LAUNCH_AGENTS = (
    "com.greg.hrdps-continental-00",
    "com.greg.hrdps-continental-06",
    "com.greg.hrdps-continental-12",
    "com.greg.hrdps-continental-18",
    "com.greg.gefs-control-fourpanel-00",
    "com.greg.ecmwf-control-fourpanel-00",
    "com.greg.ecmwf-control-fourpanel-12",
    "com.greg.ecmwf-ensemble-spread-00",
    "com.greg.ecmwf-ensemble-spread-12",
    "com.greg.lpi-verification",
    "com.greg.fire-danger-verification",
    "com.greg.fcst-fire-activity-overlay",
    "com.greg.fcst-r2-continental",
    "com.greg.fcst-r2-gefs_control",
    "com.greg.fcst-r2-ecmwf_control",
    "com.greg.fcst-r2-ecmwf_ensemble",
    "com.greg.fcst-r2-usage-monitor",
    "com.greg.fcst-r2-usage-weekly-report",
    "com.greg.fcst-pipeline-health",
    "com.greg.fcst-pipeline-health-daily",
)

LAUNCH_AGENT_LABELS = {
    "com.greg.hrdps-continental-00": "HRDPS 2.5 km 00Z retrieval and plotting",
    "com.greg.hrdps-continental-06": "HRDPS 2.5 km 06Z retrieval and plotting",
    "com.greg.hrdps-continental-12": "HRDPS 2.5 km 12Z retrieval and plotting",
    "com.greg.hrdps-continental-18": "HRDPS 2.5 km 18Z retrieval and plotting",
    "com.greg.gefs-control-fourpanel-00": "GEFS Control retrieval and plotting",
    "com.greg.ecmwf-control-fourpanel-00": "ECMWF Control 00Z retrieval and plotting",
    "com.greg.ecmwf-control-fourpanel-12": "ECMWF Control 12Z retrieval and plotting",
    "com.greg.ecmwf-ensemble-spread-00": "ECMWF Ensemble 00Z retrieval and plotting",
    "com.greg.ecmwf-ensemble-spread-12": "ECMWF Ensemble 12Z retrieval and plotting",
    "com.greg.lpi-verification": "Lightning observation and LPI verification",
    "com.greg.fire-danger-verification": "Fire-danger verification",
    "com.greg.fcst-fire-activity-overlay": "Hourly active-fire overlay",
    "com.greg.fcst-r2-continental": "HRDPS 2.5 km web publisher",
    "com.greg.fcst-r2-gefs_control": "GEFS Control web publisher",
    "com.greg.fcst-r2-ecmwf_control": "ECMWF Control web publisher",
    "com.greg.fcst-r2-ecmwf_ensemble": "ECMWF Ensemble web publisher",
    "com.greg.fcst-r2-usage-monitor": "R2 usage monitor",
    "com.greg.fcst-r2-usage-weekly-report": "R2 weekly usage report",
    "com.greg.fcst-pipeline-health": "Ten-minute forecast health monitor",
    "com.greg.fcst-pipeline-health-daily": "Daily forecast health report",
}

EXIT_CODE_MEANINGS = {
    1: "general program failure",
    2: "invalid command or missing input",
    64: "invalid command-line usage",
    65: "invalid input data",
    66: "required input file was unavailable",
    69: "required network service was unavailable",
    70: "internal software error",
    71: "operating-system error",
    72: "critical operating-system file was unavailable",
    73: "output file could not be created",
    74: "input/output error",
    75: "temporary failure that may succeed on retry",
    76: "communications protocol error",
    77: "permission denied",
    78: "configuration error",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: object) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def age_text(age: dt.timedelta) -> str:
    seconds = max(0, int(age.total_seconds()))
    if seconds < 3600:
        return f"{seconds // 60} min"
    hours = seconds / 3600.0
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


def fetch_json(url: str, timeout: float = 20.0) -> Mapping[str, object]:
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(
        f"{url}{separator}health={int(utc_now().timestamp())}",
        headers={"Cache-Control": "no-cache", "User-Agent": "fcstGraphics-health/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("response is not a JSON object")
    return payload


def level_for_age(age: dt.timedelta, warning_age: dt.timedelta) -> str:
    if age > warning_age * 2:
        return "critical"
    if age > warning_age:
        return "warning"
    return "ok"


def check_model_manifest(
    spec: ManifestSpec,
    now: dt.datetime,
    base_url: str,
    loader: Callable[[str], Mapping[str, object]] = fetch_json,
) -> CheckResult:
    url = f"{base_url.rstrip('/')}/manifests/{spec.key}.json"
    try:
        manifest = loader(url)
        generated = parse_time(manifest["generated"])
        runs = manifest.get("runs")
        if not isinstance(runs, list) or not runs:
            raise RuntimeError("manifest has no published runs")
        latest = runs[0]
        if not isinstance(latest, dict):
            raise RuntimeError("latest run is invalid")
        latest_init = parse_time(latest["init"])
        generated_age = now - generated
        run_age = now - latest_init
        # Manifests are content-addressed operational state, not heartbeats. A
        # healthy publisher intentionally leaves an unchanged manifest alone.
        level = level_for_age(run_age, spec.maximum_run_age)
        summary = f"latest {latest.get('stamp', 'unknown')} ({age_text(run_age)} old); manifest {age_text(generated_age)} old"
        return CheckResult(f"manifest.{spec.key}", spec.label, level, summary)
    except Exception as exc:
        return CheckResult(f"manifest.{spec.key}", spec.label, "critical", f"manifest check failed: {exc}")


def check_fire_manifest(
    now: dt.datetime,
    base_url: str,
    loader: Callable[[str], Mapping[str, object]] = fetch_json,
) -> CheckResult:
    url = f"{base_url.rstrip('/')}/manifests/fire_activity.json"
    try:
        manifest = loader(url)
        if not manifest.get("available"):
            raise RuntimeError("current fire layer is unavailable")
        observation_age = now - parse_time(manifest["observationTime"])
        generated_age = now - parse_time(manifest["generated"])
        level = max(
            (level_for_age(observation_age, dt.timedelta(hours=2)), level_for_age(generated_age, dt.timedelta(hours=2))),
            key=SEVERITY_ORDER.get,
        )
        if manifest.get("stale") and level == "ok":
            level = "warning"
        count = int(manifest.get("observationCount", 0) or 0)
        summary = f"{count} incidents; observations {age_text(observation_age)} old"
        if manifest.get("stale"):
            summary += "; cached fallback"
        return CheckResult("feed.fire_activity", "BCWS fire activity", level, summary)
    except Exception as exc:
        return CheckResult("feed.fire_activity", "BCWS fire activity", "critical", f"feed check failed: {exc}")


def read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return payload


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_hrdps_pipeline_status(
    spec: PipelineStatusSpec,
    now: dt.datetime,
    state_root: Path | None = None,
    pid_checker: Callable[[int], bool] = process_exists,
) -> CheckResult:
    path = (state_root or REPO_ROOT / "logs" / "state") / f"{spec.model}_{spec.cycle}.status.json"
    try:
        status = read_json(path)
        state = str(status.get("status", "unknown"))
        heartbeat = parse_time(status.get("heartbeat_at_utc") or status["updated_at_utc"])
        age = max(dt.timedelta(0), now - heartbeat)
        stamp = str(status.get("stamp") or status.get("expected_stamp") or "unknown run")
        ready = len(status.get("ready_hours") or ())
        rendered = len(status.get("rendered_hours") or ())
        expected = int(status.get("expected_hours") or 0)
        counts = f"; ready {ready}/{expected}, rendered {rendered}/{expected}" if expected else ""

        if state == "failed":
            detail = str(status.get("error") or "unknown error")
            return CheckResult(
                f"pipeline.hrdps.{spec.cycle}",
                spec.label,
                "warning",
                f"{stamp} failed: {detail}",
                push_eligible=False,
            )
        if state in {"degraded", "partial"}:
            return CheckResult(
                f"pipeline.hrdps.{spec.cycle}",
                spec.label,
                "warning",
                f"{stamp} incomplete{counts}",
                push_eligible=False,
            )
        if state in {"starting", "running", "waiting_upstream", "rendering"}:
            pid = int(status.get("pid") or 0)
            if not pid_checker(pid):
                return CheckResult(
                    f"pipeline.hrdps.{spec.cycle}",
                    spec.label,
                    "warning",
                    f"{stamp} was interrupted; its last heartbeat was {age_text(age)} ago{counts}",
                    push_eligible=False,
                )
            level = "critical" if age > dt.timedelta(minutes=90) else "warning" if age > dt.timedelta(minutes=45) else "ok"
            return CheckResult(
                f"pipeline.hrdps.{spec.cycle}",
                spec.label,
                level,
                f"{stamp} {state.replace('_', ' ')}; heartbeat {age_text(age)} old{counts}",
            )
        if state in {"success", "complete"}:
            return CheckResult(
                f"pipeline.hrdps.{spec.cycle}", spec.label, "ok", f"{stamp} complete{counts}"
            )
        return CheckResult(
            f"pipeline.hrdps.{spec.cycle}",
            spec.label,
            "warning",
            f"{stamp} has unknown state {state!r}",
            push_eligible=False,
        )
    except Exception as exc:
        return CheckResult(
            f"pipeline.hrdps.{spec.cycle}",
            spec.label,
            "warning",
            f"status check failed: {exc}",
            push_eligible=False,
        )


def publication_deadline(schedule: RunSchedule, init: dt.datetime) -> dt.datetime:
    if schedule.ready_local is None:
        return init + dt.timedelta(hours=5.5)
    hour, minute = schedule.ready_local
    return dt.datetime.combine(init.date(), dt.time(hour, minute), LOCAL_TZ).astimezone(dt.timezone.utc)


def status_run_stamp(status: Mapping[str, object]) -> str:
    for value in (status.get("stamp"), status.get("expected_stamp"), status.get("error")):
        match = re.search(r"\b(\d{8}T\d{2}Z)\b|(?<!\d)(\d{8}T\d{2}Z)_", str(value or ""))
        if match:
            return match.group(1) or match.group(2)
    return ""


def plain_run_cause(status: Mapping[str, object], interrupted: bool = False) -> str:
    error = str(status.get("error") or "").lower()
    if "404" in error or "not found" in error or "missing" in error:
        return "a required input file could not be found"
    if "timeout" in error or "timed out" in error:
        return "the download server did not respond in time" if any(
            word in error for word in ("http", "download", "connection")
        ) else "the job ran out of time before finishing"
    if any(word in error for word in ("connection", "network", "name resolution", "urlopen")):
        return "the download connection failed"
    if "no space" in error:
        return "the computer ran out of storage space"
    if "permission" in error:
        return "the job could not access a required file or service"
    if status.get("status") == "failed":
        return "the processing job stopped with an error; the exact cause needs investigation"
    if interrupted:
        return "the processing job stopped before finishing"
    if status.get("status") in {"success", "complete"}:
        return "the plots are ready locally but are not all available on the website yet"
    if status.get("status") == "waiting_upstream":
        return "we are still waiting for the model's input files"
    if status.get("status") in {"degraded", "partial"}:
        return "the job finished without producing all of the expected plots"
    if status.get("status") == "rendering":
        return "the forecast plots are still being drawn"
    if status.get("status") in {"starting", "running"}:
        return "downloading or plotting is still in progress; no failure has been reported"
    return "the scheduled job has not reported starting; the cause is not yet known"


def run_is_published(run: Mapping[str, object], schedule: RunSchedule) -> bool:
    return all(product_is_published(run, product, schedule.hours) for product in schedule.products)


def product_is_published(run: Mapping[str, object], product: str, hours: tuple[int, ...]) -> bool:
    if not isinstance(run, dict):
        return False
    products = run.get("products") or {}
    if not isinstance(products, dict):
        return False
    return (
        isinstance(products.get(product), dict)
        and isinstance(products[product].get("hours"), list)
        and set(hours).issubset(products[product]["hours"])
    )


def fire_danger_failure_cause(status: Mapping[str, object]) -> str | None:
    errors = str(status.get("optional_errors") or "").lower()
    if "cwfis" in errors:
        return "usable CWFIS fuel-moisture inputs could not be loaded"
    if "fwi2025" in errors:
        return "the fire-danger calculation failed"
    return None


def check_scheduled_run(
    schedule: RunSchedule,
    now: dt.datetime,
    manifest: Mapping[str, object],
    state_root: Path | None = None,
    pid_checker: Callable[[int], bool] = process_exists,
    *,
    init: dt.datetime | None = None,
) -> CheckResult:
    init = init or now.astimezone(dt.timezone.utc).replace(hour=schedule.cycle, minute=0, second=0, microsecond=0)
    if init > now:
        init -= dt.timedelta(days=1)
    stamp = init.strftime("%Y%m%dT%HZ")
    key = f"run.{schedule.model}.{stamp}"
    label = f"{schedule.label} {schedule.cycle:02d}Z run for {init:%b %d}"
    deadline = publication_deadline(schedule, init)
    published = [run for run in manifest.get("runs", ()) if run_is_published(run, schedule)]
    if any(str(run.get("stamp", "")) >= stamp for run in published):
        return CheckResult(key, label, "ok", "This run or a newer complete run is available on the website.")

    path = (state_root or REPO_ROOT / "logs/state") / f"{schedule.model}_{schedule.cycle:02d}.status.json"
    status = read_state(path)
    if status_run_stamp(status) != stamp:
        status = {}
    state = status.get("status")
    active = state in {"starting", "running", "waiting_upstream", "rendering"}
    pid = int(status.get("pid") or 0)
    interrupted = active and pid > 0 and not pid_checker(pid)
    stopped = state == "failed" or interrupted
    stalled = False
    if active and status.get("heartbeat_at_utc"):
        stalled = now - parse_time(status["heartbeat_at_utc"]) > dt.timedelta(minutes=45)
    delay = now - deadline
    if delay < RUN_DELAY_GRACE:
        previous_init = init - dt.timedelta(days=1)
        previous_stamp = previous_init.strftime("%Y%m%dT%HZ")
        if not any(str(run.get("stamp", "")) >= previous_stamp for run in published):
            # Starting a new cycle must not clear an unresolved missing-run alert.
            return check_scheduled_run(schedule, now, manifest, state_root, pid_checker, init=previous_init)
        return CheckResult(key, label, "ok", f"Expected on the website by {deadline.astimezone(LOCAL_TZ):%H:%M %Z}.")

    cause = plain_run_cause(status, interrupted)
    if stalled and not stopped:
        cause = "the processing job has stopped reporting progress"
    hours = round(delay.total_seconds() / 3600, 1)
    duration = "1 hour" if hours == 1 else f"{hours:.1f} hours"
    available = [product for product in schedule.products if any(
        str(run.get("stamp", "")) >= stamp and product_is_published(run, product, schedule.hours)
        for run in manifest.get("runs", ()) if isinstance(run, dict)
    )]
    missing = [product for product in schedule.products if product not in available]
    if not missing:
        return CheckResult(key, label, "ok", "All expected products are available from this run or newer runs.")
    if available:
        missing_names = ", ".join(PRODUCT_LABELS[product] for product in missing)
        if missing == ["continental_fwi2025_danger"]:
            cause = fire_danger_failure_cause(status) or cause
            summary = f"{label}: weather graphics are available; fire-danger graphics are {duration} late because {cause}."
            if fire_danger_failure_cause(status):
                summary += " Fire-danger shading may also be missing from the fire-weather maps."
        else:
            available_names = ", ".join(PRODUCT_LABELS[product] for product in available)
            summary = f"{label}: {missing_names} graphics are {duration} late because {cause}. Available: {available_names}."
        current = next((run for run in manifest.get("runs", ()) if isinstance(run, dict) and run.get("stamp") == stamp), {})
        for product in missing:
            product_data = (current.get("products") or {}).get(product) or {}
            present_hours = set(product_data.get("hours") or ()) if isinstance(product_data, dict) else set()
            missing_hours = set(schedule.hours) - present_hours
            if present_hours and missing_hours:
                summary += f" {PRODUCT_LABELS[product].capitalize()}: {len(missing_hours)} of {len(schedule.hours)} frames missing."
    else:
        summary = f"{label} is {duration} late because {cause}."
    summary += f" Expected on the website by {deadline.astimezone(LOCAL_TZ):%b %d at %H:%M %Z}."
    if published and not available:
        latest = max(published, key=lambda run: str(run.get("stamp", "")))
        fallback = dt.datetime.strptime(latest["stamp"], "%Y%m%dT%HZ")
        summary += f" The latest complete forecast available is {fallback:%b %d %HZ}."
    elif not available:
        summary += " No complete forecast from this model could be confirmed on the website."
    return CheckResult(key, label, "critical" if delay >= dt.timedelta(hours=4) else "warning", summary, immediate=True)


def project_data_root(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    direct = env.get("FCSTGRAPHICS_DATA_ROOT", "").strip()
    if direct:
        return Path(os.path.expandvars(direct)).expanduser()
    shared = env.get("PROJECT_DATA_ROOT", "").strip()
    if not shared and MACHINE_DATA_CONFIG.exists():
        for raw_line in MACHINE_DATA_CONFIG.read_text().splitlines():
            line = raw_line.strip()
            if line.startswith("PROJECT_DATA_ROOT="):
                shared = line.split("=", 1)[1].strip().strip("\"'")
                break
    if shared:
        return Path(os.path.expandvars(shared)).expanduser() / "fcstGraphics" / "data"
    return Path(__file__).resolve().parent / "data"


def check_lightning_archive(now: dt.datetime, data_root: Path) -> CheckResult:
    path = data_root / "lightning_ml" / "status.json"
    try:
        status = read_json(path)
        updated_age = now - parse_time(status["updated_at_utc"])
        observation_age = now - parse_time(status["latest_observation_block"])
        level = max(
            (level_for_age(updated_age, dt.timedelta(hours=2)), level_for_age(observation_age, dt.timedelta(hours=3))),
            key=SEVERITY_ORDER.get,
        )
        summary = f"latest block {status['latest_observation_block']} ({age_text(observation_age)} old)"
        return CheckResult("feed.lightning_archive", "ECCC lightning archive", level, summary)
    except Exception as exc:
        return CheckResult("feed.lightning_archive", "ECCC lightning archive", "critical", f"archive check failed: {exc}")


def dated_directories(path: Path) -> set[dt.date]:
    dates = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and re.fullmatch(r"\d{8}", child.name):
                try:
                    dates.append(dt.datetime.strptime(child.name, "%Y%m%d").date())
                except ValueError:
                    continue
    return set(dates)


def latest_usable_cwfis_anchor(now: dt.datetime, data_root: Path) -> dt.date | None:
    cache = data_root / "cwfis_fwi"
    dates = set.intersection(*(dated_directories(cache / field) for field in CWFIS_FIELDS))
    if not dates:
        return None

    # Plotting modules resolve volume-backed paths at import time. A missing data
    # volume must remain a reportable health failure, not crash this monitor.
    from PIL import Image
    from make_experimental_danger_class import cwfis_cache_path, validate_cwfis_geotiff
    from make_hrdps_west_convective import MODEL_CONFIGS

    extent = MODEL_CONFIGS["continental"].extent
    for date in sorted(dates, reverse=True):
        if cwfis_anchor_time(date) > now:
            continue
        try:
            for field in CWFIS_FIELDS:
                path = cwfis_cache_path(cache, field, date, extent)
                with Image.open(path) as image:
                    if image.format != "TIFF" or not (
                        image.tag_v2.get(34264) or (image.tag_v2.get(33922) and image.tag_v2.get(33550))
                    ):
                        raise ValueError("Not a georeferenced TIFF")
                validate_cwfis_geotiff(path, field, None)
        except (OSError, RuntimeError, ValueError):
            continue
        return date
    return None


def cwfis_anchor_time(date: dt.date) -> dt.datetime:
    # CWFIS daily fuel-moisture codes represent 20Z in Pacific BC.
    return dt.datetime.combine(date, dt.time(20), dt.timezone.utc)


def check_cwfis_anchors(now: dt.datetime, data_root: Path) -> CheckResult:
    latest = latest_usable_cwfis_anchor(now, data_root)
    if latest is None:
        return CheckResult(
            "feed.cwfis_anchors", "CWFIS fire-danger inputs", "critical",
            "No complete usable CWFIS fuel-moisture dataset could be verified for BC. "
            "New fire-danger guidance cannot be initialized; other HRDPS weather fields do not depend on these inputs.",
        )
    age = now - cwfis_anchor_time(latest)
    expires = cwfis_anchor_time(latest) + dt.timedelta(hours=MAX_CWFIS_ANCHOR_AGE_AT_INIT_HOURS)
    summary = f"The latest usable CWFIS fuel-moisture inputs are from {latest:%b %d} ({age.total_seconds() / 3600:.1f} hours old)."
    if now > expires:
        level = "critical"
        summary += " They are too old to initialize new fire-danger forecasts; other HRDPS weather fields are unaffected."
    elif age >= CWFIS_EARLY_WARNING_AGE:
        level = "warning"
        summary += (
            f" Early warning: no newer complete inputs have been downloaded. They reach the "
            f"{MAX_CWFIS_ANCHOR_AGE_AT_INIT_HOURS}-hour limit on {expires.astimezone(LOCAL_TZ):%b %d at %H:%M %Z} "
            f"(in {(expires - now).total_seconds() / 3600:.1f} hours); later model runs will lose fire-danger guidance unless fresh inputs arrive."
        )
    else:
        level = "ok"
    return CheckResult("feed.cwfis_anchors", "CWFIS fire-danger inputs", level, summary, immediate=True)


def fetch_cwfis_sample(field: str, date: dt.date) -> bytes:
    params = urllib.parse.urlencode({
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": f"public:{field}", "BBOX": "-125,49,-120,54",
        "WIDTH": "16", "HEIGHT": "16", "CRS": "EPSG:4326",
        "FORMAT": "geotiff", "time": date.isoformat(),
    })
    with urllib.request.urlopen(f"{CWFIS_WCS_URL}?{params}", timeout=20) as response:
        return response.read(1_000_000)


def validate_cwfis_sample(content: bytes) -> None:
    from PIL import Image

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        pass
    else:
        detail = " ".join(" ".join(root.itertext()).split())
        if "geotiff writer" in detail.lower():
            raise ValueError("their map server cannot find its GeoTIFF export component")
        raise ValueError("their map server returned an error instead of a data map")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "TIFF":
                raise ValueError("not a TIFF")
            image.load()
    except (OSError, ValueError) as exc:
        raise ValueError("the download was not a readable GeoTIFF data map") from exc


def check_cwfis_service(
    now: dt.datetime, data_root: Path,
    loader: Callable[[str, dt.date], bytes] = fetch_cwfis_sample,
) -> CheckResult:
    # A known usable date tests export availability, not today's normal release latency.
    date = latest_usable_cwfis_anchor(now, data_root) or (now - dt.timedelta(days=1)).date()
    failures: dict[str, str] = {}
    for field in CWFIS_FIELDS:
        try:
            validate_cwfis_sample(loader(field, date))
        except ValueError as exc:
            failures[field] = str(exc)
        except Exception:
            failures[field] = "the download service could not be reached or rejected the request"
    if failures:
        fields = ", ".join(field.upper() for field in failures)
        causes = "; ".join(dict.fromkeys(failures.values()))
        return CheckResult(
            "feed.cwfis_service", "CWFIS map-download service", "warning",
            f"CWFIS fuel-moisture map downloads are failing for {fields} because {causes}. "
            "Fresh fire-danger inputs cannot be retrieved. Existing usable inputs can support new runs only until their "
            f"{MAX_CWFIS_ANCHOR_AGE_AT_INIT_HOURS}-hour limit.",
        )
    return CheckResult("feed.cwfis_service", "CWFIS map-download service", "ok", "All three fuel-moisture map exports are readable.")


def configured_volume(data_root: Path) -> Path:
    parts = data_root.resolve(strict=False).parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path("/") / parts[1] / parts[2]
    return data_root


def check_storage(
    data_root: Path,
    radar_health_path: Path = RADARSAT_HEALTH_PATH,
) -> CheckResult:
    volume = configured_volume(data_root)
    if not volume.exists() or not data_root.exists():
        return CheckResult("storage.runtime", "Forecast data storage", "critical", f"unavailable: {volume}", immediate=True)
    try:
        usage = shutil.disk_usage(volume)
    except OSError as exc:
        return CheckResult("storage.runtime", "Forecast data storage", "critical", f"usage check failed: {exc}", immediate=True)
    free_fraction = usage.free / usage.total if usage.total else 0.0
    free_gb = usage.free / 1_000_000_000
    level = (
        "critical"
        if free_fraction < 0.05 or free_gb < 100
        else "warning"
        if free_fraction < 0.10 or free_gb < 200
        else "ok"
    )
    summary = f"{free_gb:.0f} GB free ({free_fraction:.0%})"
    try:
        radar_health = read_state(radar_health_path)
        storage = radar_health.get("storage", {})
        if isinstance(storage, dict):
            local_bytes = int(storage.get("totalBytes", 0) or 0)
            cache_bytes = int(storage.get("compositeCacheBytes", 0) or 0)
            segment_bytes = int(storage.get("videoSegmentBytes", 0) or 0)
            source_bytes = int(storage.get("sourceFrameBytes", 0) or 0)
            if local_bytes:
                summary += (
                    f"; Radar-Sat {local_bytes / 1_000_000_000:.1f} GB"
                    f" (cache {cache_bytes / 1_000_000_000:.1f},"
                    f" HLS {segment_bytes / 1_000_000_000:.1f},"
                    f" source {source_bytes / 1_000_000_000:.1f})"
                )
                if local_bytes >= 30_000_000_000:
                    level = "critical"
                elif local_bytes >= 20_000_000_000 and level != "critical":
                    level = "warning"
    except (OSError, TypeError, ValueError):
        # The Radar-Sat health service owns status-file availability. This
        # check remains the storage alarm and does not duplicate that service.
        pass
    return CheckResult(
        "storage.runtime",
        "Forecast data storage",
        level,
        summary,
        immediate=True,
    )


def launchctl_print(label: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        check=False,
        capture_output=True,
        text=True,
    )


def reload_launch_agent(label: str) -> bool:
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    if not plist.exists():
        return False
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "enable", f"{domain}/{label}"], check=False, capture_output=True, text=True)
    result = subprocess.run(
        ["launchctl", "bootstrap", domain, str(plist)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 or launchctl_print(label).returncode == 0


def exit_code_description(code: int) -> str:
    if code < 0:
        return f"terminated by signal {-code}"
    if code >= 128:
        return f"terminated by signal {code - 128}"
    return EXIT_CODE_MEANINGS.get(code, "nonzero exit; the program reported a failure")


def check_launch_agent(label: str, auto_repair: bool = True) -> CheckResult:
    short = LAUNCH_AGENT_LABELS.get(label, label.removeprefix("com.greg."))
    result = launchctl_print(label)
    repaired = False
    if result.returncode != 0 and auto_repair:
        repaired = reload_launch_agent(label)
        result = launchctl_print(label)
    if result.returncode != 0:
        return CheckResult(
            f"service.{label}",
            short,
            "critical",
            "schedule is not loaded and automatic repair failed",
            immediate=True,
        )
    if repaired:
        return CheckResult(
            f"service.{label}",
            short,
            "warning",
            "schedule was unloaded and was reloaded automatically",
        )
    state_match = re.search(r"^\s*state = ([^\n]+)", result.stdout, re.MULTILINE)
    if state_match and state_match.group(1).strip() == "running":
        return CheckResult(
            f"service.{label}",
            short,
            "ok",
            "schedule loaded and job currently running",
        )
    match = re.search(r"last exit code = (-?\d+)", result.stdout)
    if match and int(match.group(1)) != 0:
        code = int(match.group(1))
        return CheckResult(
            f"service.{label}",
            short,
            "warning",
            f"last scheduled attempt failed with exit code {code} ({exit_code_description(code)})",
            push_eligible=False,
        )
    return CheckResult(f"service.{label}", short, "ok", "schedule loaded")


def run_checks(
    now: dt.datetime,
    *,
    base_url: str,
    data_root: Path,
    auto_repair: bool = True,
    loader: Callable[[str], Mapping[str, object]] = fetch_json,
    include_services: bool = True,
) -> list[CheckResult]:
    manifests: dict[str, Mapping[str, object]] = {}

    def cached_loader(url: str) -> Mapping[str, object]:
        if url not in manifests:
            manifests[url] = loader(url)
        return manifests[url]

    checks = [check_storage(data_root)]
    checks.extend(check_model_manifest(spec, now, base_url, cached_loader) for spec in MODEL_MANIFESTS)
    readable = {check.key.removeprefix("manifest.") for check in checks if
                check.key.startswith("manifest.") and "manifest check failed" not in check.summary}
    publication_checks = []
    for schedule in RUN_SCHEDULES:
        manifest = manifests.get(f"{base_url.rstrip('/')}/manifests/{schedule.model}.json")
        if schedule.model in readable:
            publication_checks.append(check_scheduled_run(schedule, now, manifest))
    # Per-run checks replace age-only alerts when the public catalogue was readable.
    checks = [check for check in checks if not (
        check.key.removeprefix("manifest.") in readable and "manifest check failed" not in check.summary
    )]
    checks.extend(publication_checks)
    checks.append(check_fire_manifest(now, base_url, loader))
    checks.append(check_lightning_archive(now, data_root))
    checks.append(check_cwfis_anchors(now, data_root))
    checks.append(check_cwfis_service(now, data_root))
    if include_services:
        for label in REQUIRED_LAUNCH_AGENTS:
            check = check_launch_agent(label, auto_repair)
            modeled_job = label.startswith(("com.greg.hrdps-continental-", "com.greg.gefs-control-fourpanel-",
                                            "com.greg.ecmwf-control-fourpanel-", "com.greg.ecmwf-ensemble-spread-"))
            if modeled_job and check.summary.startswith("last scheduled attempt failed"):
                continue  # Current run checks already distinguish active from superseded failures.
            checks.append(check)
    return checks


def read_state(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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


def state_times(state: Mapping[str, object], key: str) -> dict[str, str]:
    value = state.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {str(item_key): str(item_value) for item_key, item_value in value.items()}


def apply_debounce(
    checks: Iterable[CheckResult],
    previous_state: Mapping[str, object],
    now: dt.datetime,
    pending_duration: dt.timedelta = PENDING_DURATION,
) -> tuple[list[CheckResult], dict[str, object]]:
    previous_counts = previous_state.get("failure_counts", {})
    previous_counts = previous_counts if isinstance(previous_counts, dict) else {}
    previous_pending = state_times(previous_state, "pending_since")
    effective: list[CheckResult] = []
    counts: dict[str, int] = {}
    pending_since: dict[str, str] = {}
    for check in checks:
        if check.level == "ok":
            counts[check.key] = 0
            effective.append(check)
            continue
        count = int(previous_counts.get(check.key, 0) or 0) + 1
        counts[check.key] = count
        if check.immediate:
            effective.append(check)
            continue
        try:
            first_seen = parse_time(previous_pending[check.key])
        except (KeyError, TypeError, ValueError):
            first_seen = now
        pending_since[check.key] = first_seen.isoformat().replace("+00:00", "Z")
        pending_age = max(dt.timedelta(0), now - first_seen)
        if pending_age >= pending_duration:
            effective.append(check)
        else:
            effective.append(
                CheckResult(
                    check.key,
                    check.label,
                    "ok",
                    f"pending {age_text(pending_age)} of {age_text(pending_duration)}: {check.summary}",
                )
            )
    return effective, {"failure_counts": counts, "pending_since": pending_since}


def problem_signature(checks: Iterable[CheckResult]) -> str:
    return "|".join(sorted(f"{check.key}:{check.level}" for check in notification_problems(checks)))


def signature_levels(signature: str) -> dict[str, str]:
    levels: dict[str, str] = {}
    for item in signature.split("|"):
        if not item or ":" not in item:
            continue
        key, level = item.rsplit(":", 1)
        if level in SEVERITY_ORDER:
            levels[key] = level
    return levels


def signature_has_new_or_escalated_problem(signature: str, notified_signature: str) -> bool:
    current = signature_levels(signature)
    notified = signature_levels(notified_signature)
    return any(
        key not in notified
        or SEVERITY_ORDER[level] > SEVERITY_ORDER[notified[key]]
        for key, level in current.items()
    )


def overall_level(checks: Iterable[CheckResult]) -> str:
    return max((check.level for check in checks), default="ok", key=SEVERITY_ORDER.get)


def operational_impact(check: CheckResult) -> str:
    if check.key == "storage.runtime":
        return "New downloads and graphics may fail; already-published graphics should remain online."
    if check.key == "group.public_manifests":
        return "The website may be unable to discover current model runs; this may be one shared R2 or network failure."
    if check.key == "group.schedulers":
        return "Multiple forecast update schedules are offline, so several products may stop updating."
    if check.key.startswith("manifest."):
        return "The website may be serving an older model run while the new run is delayed."
    if check.key.startswith("pipeline.hrdps."):
        return "The affected HRDPS cycle may be incomplete; already-published runs remain available."
    if check.key == "feed.fire_activity":
        return "Active-fire symbols may be outdated; the underlying model forecast fields are unaffected."
    if check.key == "feed.lightning_archive":
        return "Lightning verification and later LPI tuning may have a data gap; forecast LPI fields are unaffected."
    if check.key.startswith("feed.cwfis_"):
        return "New fire-danger guidance will become unavailable once usable fuel-moisture inputs exceed the 48-hour limit."
    if check.key.startswith("service."):
        if "reloaded automatically" in check.summary:
            return "The schedule was restored automatically; no missing product has been confirmed."
        if "automatic repair failed" in check.summary:
            return "Future updates for this component will not run until its schedule is restored."
        return "Existing graphics remain online, but the next update from this component may be delayed."
    return "This component may not update on schedule."


def recommended_action(check: CheckResult) -> str:
    if check.key == "storage.runtime":
        return "Confirm the external volume is mounted; if Radar-Sat is oversized, inspect its cache and HLS working set."
    if check.key == "group.public_manifests":
        return "Open the graphics page from another network; if it also fails, inspect R2 and the publishers."
    if check.key == "group.schedulers":
        return "Check launchd and the forecast monitor logs, then reinstall the affected launch agents."
    if check.key.startswith("manifest."):
        return "Check the affected model on the graphics page, then inspect its retrieval and publisher logs."
    if check.key.startswith("pipeline.hrdps."):
        return "Inspect the HRDPS cycle status and log for its upstream-ready and rendered-hour counts."
    if check.key == "feed.fire_activity":
        return "Check the BCWS feed and the hourly fire-overlay log."
    if check.key == "feed.lightning_archive":
        return "Inspect the lightning retrieval log and archive status before the next verification cycle."
    if check.key.startswith("feed.cwfis_"):
        return "Check CWFIS map downloads and the last complete usable fuel-moisture inputs."
    if check.key.startswith("service."):
        if "reloaded automatically" in check.summary:
            return "No immediate action; confirm that its next scheduled update completes."
        return "Inspect this component's launch-agent and log; reload it if the next attempt does not recover."
    return "Inspect the component log and confirm its next scheduled update."


def notification_problems(
    checks: Iterable[CheckResult],
    *,
    include_daily_only: bool = False,
) -> list[CheckResult]:
    problems = [
        check
        for check in checks
        if check.level != "ok" and (check.push_eligible or include_daily_only)
    ]
    manifest_failures = [
        check
        for check in problems
        if check.key.startswith("manifest.") and "manifest check failed" in check.summary
    ]
    grouped: list[CheckResult] = []
    remaining = problems
    if len(manifest_failures) >= 3:
        affected = ", ".join(check.label for check in manifest_failures)
        level = max((check.level for check in manifest_failures), key=SEVERITY_ORDER.get)
        grouped.append(
            CheckResult(
                "group.public_manifests",
                "Public forecast graphics access",
                level,
                f"{len(manifest_failures)}/{len(MODEL_MANIFESTS)} model manifests could not be read; affected: {affected}",
            )
        )
        remaining = [check for check in remaining if check not in manifest_failures]

    scheduler_failures = [
        check
        for check in remaining
        if check.key.startswith("service.") and "automatic repair failed" in check.summary
    ]
    if len(scheduler_failures) >= 3:
        affected = ", ".join(check.label for check in scheduler_failures[:5])
        if len(scheduler_failures) > 5:
            affected += f", and {len(scheduler_failures) - 5} more"
        grouped.append(
            CheckResult(
                "group.schedulers",
                "Forecast update schedules",
                "critical",
                f"{len(scheduler_failures)} schedules are unloaded and could not be repaired; affected: {affected}",
                immediate=True,
            )
        )
        remaining = [check for check in remaining if check not in scheduler_failures]
    return [*grouped, *remaining]


def report_body(checks: list[CheckResult], now: dt.datetime, daily: bool) -> str:
    level = overall_level(checks).upper()
    problems = notification_problems(checks, include_daily_only=daily)
    storage = next((check for check in checks if check.key == "storage.runtime"), None)
    if daily and not problems:
        lines = [f"All {len(checks)}/{len(checks)} checks are healthy."]
        if storage is not None:
            lines.append(f"Disk: {storage.summary}")
        return "\n".join(lines)

    plain_forecast_report = bool(problems) and all(check.key.startswith(("run.", "feed.cwfis_")) for check in problems)
    lines = [] if plain_forecast_report else [f"Status: {level}", f"Checked: {now.astimezone(LOCAL_TZ):%Y-%m-%d %H:%M %Z}"]
    if problems:
        if not plain_forecast_report:
            lines.append(f"Problems: {len(problems)}")
        for check in problems[:MAX_REPORTED_PROBLEMS]:
            if check.key.startswith(("run.", "feed.cwfis_")):
                lines.append(check.summary)
                continue
            lines.append(f"[{check.level.upper()}] {check.label}")
            lines.append(f"Issue: {check.summary}")
            lines.append(f"Impact: {operational_impact(check)}")
            lines.append(f"Action: {recommended_action(check)}")
        if len(problems) > MAX_REPORTED_PROBLEMS:
            lines.append(f"...and {len(problems) - MAX_REPORTED_PROBLEMS} more; see the local health history for details")
    else:
        historical = [
            check for check in checks if check.level != "ok" and not check.push_eligible
        ]
        if historical:
            lines.append(
                f"No active alerting problems; {len(historical)} historical warning"
                f"{'s' if len(historical) != 1 else ''} remain for the daily report."
            )
        else:
            lines.append(f"All {len(checks)} checks are healthy.")
    if daily and storage is not None and storage not in problems:
        lines.append(f"Disk: {storage.summary}")
    return "\n".join(lines)


def notification_decision(
    *,
    signature: str,
    notified_signature: str,
    level: str,
    now: dt.datetime,
    previous_state: Mapping[str, object],
    always_notify: bool,
) -> tuple[str | None, str]:
    try:
        last_notified = parse_time(previous_state["last_notification_at"])
    except (KeyError, TypeError, ValueError):
        last_notified = None
    can_notify = last_notified is None or now - last_notified >= MIN_NOTIFICATION_INTERVAL
    if always_notify and can_notify:
        return "daily", ""

    if signature:
        if signature_has_new_or_escalated_problem(signature, notified_signature):
            return "alert" if can_notify else None, ""
        if signature != notified_signature:
            return None, ""
        if last_notified is None:
            return "reminder", ""
        repeat = RUN_REMINDER if any(key.startswith("run.") for key in signature_levels(signature)) else (
            CRITICAL_REPEAT if level == "critical" else WARNING_REPEAT
        )
        if can_notify and now - last_notified >= repeat:
            return "reminder", ""
        return None, ""

    if not notified_signature:
        return None, ""

    recovery_since = str(previous_state.get("recovery_since", ""))
    try:
        first_healthy = parse_time(recovery_since)
    except (TypeError, ValueError):
        first_healthy = now
        recovery_since = now.isoformat().replace("+00:00", "Z")
    if can_notify and now - first_healthy >= RECOVERY_HOLD:
        return "recovery", recovery_since
    return None, recovery_since


@contextmanager
def monitor_lock(path: Path, timeout_seconds: float = 120.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"health-monitor lock remained busy for {timeout_seconds:.0f} seconds")
                time.sleep(0.5)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--latest-path", type=Path, default=LATEST_PATH)
    parser.add_argument("--history-path", type=Path, default=HISTORY_PATH)
    parser.add_argument("--lock-path", type=Path, default=LOCK_PATH)
    parser.add_argument("--lock-timeout", type=float, default=120.0)
    parser.add_argument("--always-notify", action="store_true", help="Send the daily heartbeat even when healthy.")
    parser.add_argument(
        "--operational",
        action="store_true",
        help="Allow this invocation to update durable monitor state, repair services, and send Telegram notifications.",
    )
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--no-auto-repair", action="store_true")
    parser.add_argument("--no-services", action="store_true", help="Skip launch-agent checks, primarily for tests.")
    return parser.parse_args(list(argv))


def isolate_diagnostic_invocation(args: argparse.Namespace) -> argparse.Namespace:
    """Keep direct/manual checks from mutating the scheduled monitor incident."""
    if args.operational:
        return args
    args.no_notify = True
    args.no_auto_repair = True
    diagnostic_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"fcst-health-diagnostic-{os.getpid()}"
    if args.state_path == STATE_PATH:
        args.state_path = diagnostic_root / "state.json"
    if args.latest_path == LATEST_PATH:
        args.latest_path = diagnostic_root / "latest.json"
    if args.history_path == HISTORY_PATH:
        args.history_path = diagnostic_root / "history.jsonl"
    if args.lock_path == LOCK_PATH:
        args.lock_path = diagnostic_root / "monitor.lock"
    return args


def run_monitor(args: argparse.Namespace) -> int:
    now = utc_now()
    state = read_state(args.state_path)
    base_url = os.environ.get("FCST_R2_PUBLIC_BASE_URL", PUBLIC_BASE_URL).strip() or PUBLIC_BASE_URL
    data_root = project_data_root()
    raw_checks = run_checks(
        now,
        base_url=base_url,
        data_root=data_root,
        auto_repair=not args.no_auto_repair,
        include_services=not args.no_services,
    )
    checks, debounce_state = apply_debounce(raw_checks, state, now)
    signature = problem_signature(checks)
    previous_signature = str(state.get("problem_signature", ""))
    notified_signature = str(state.get("notified_signature", previous_signature))
    payload = {
        "updated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "overall_level": overall_level(checks),
        "problem_signature": signature,
        "notified_signature": notified_signature,
        **debounce_state,
        "checks": [asdict(check) for check in checks],
        "raw_checks": [asdict(check) for check in raw_checks],
    }
    if state.get("last_notification_at"):
        payload["last_notification_at"] = state["last_notification_at"]
    reason, recovery_since = notification_decision(
        signature=signature,
        notified_signature=notified_signature,
        level=str(payload["overall_level"]),
        now=now,
        previous_state=state,
        always_notify=args.always_notify,
    )
    if (
        reason is None and signature and signature != notified_signature
        and not signature_has_new_or_escalated_problem(signature, notified_signature)
    ):
        # A strictly smaller or lower-severity problem set is partial recovery,
        # not a new incident. Keep rate-limited new incidents unacknowledged.
        payload["notified_signature"] = signature
    if recovery_since:
        payload["recovery_since"] = recovery_since
    write_json(args.latest_path, payload)

    notify = reason is not None
    notification_sent = False
    notification_error = ""
    if notify and not args.no_notify:
        if reason == "reminder":
            title = "Forecast Graphics Health Reminder"
        elif signature:
            title = "Forecast Graphics Health Alert"
        elif reason == "recovery":
            title = "Forecast Graphics Health Recovered"
        else:
            title = "Forecast Graphics Daily Health"
        body = report_body(checks, now, args.always_notify)
        payload["notification_title"] = title
        payload["notification_body"] = body
        try:
            receipt = telegram_notify.send_message(
                title,
                body,
                url=SITE_URL if signature else None,
            )
            notification_sent = True
            payload["telegram_receipt"] = receipt
            payload["notified_signature"] = signature
            payload["last_notification_at"] = now.isoformat().replace("+00:00", "Z")
            payload.pop("recovery_since", None)
        except Exception as exc:
            notification_error = str(exc)
            print(f"Telegram notification failed: {exc}", flush=True)

    payload["notification_sent"] = notification_sent
    payload["notification_delivery"] = "accepted_by_telegram" if notification_sent else "failed" if notification_error else "not_attempted"
    payload["notification_error"] = notification_error
    write_json(args.latest_path, payload)
    write_json(args.state_path, payload)
    append_history(args.history_path, payload)
    healthy = sum(check.level == "ok" for check in checks)
    print(
        f"{now.astimezone(LOCAL_TZ):%Y-%m-%d %H:%M:%S %Z} Pipeline health: "
        f"level={payload['overall_level']}, healthy={healthy}/{len(checks)}, "
        f"problems={len(checks) - healthy}, notified={notification_sent}",
        flush=True,
    )
    return 0


def main(argv: Iterable[str]) -> int:
    args = isolate_diagnostic_invocation(parse_args(argv))
    try:
        with monitor_lock(args.lock_path, args.lock_timeout):
            return run_monitor(args)
    except TimeoutError as exc:
        print(f"Pipeline health check skipped: {exc}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
