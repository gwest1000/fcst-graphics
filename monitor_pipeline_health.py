#!/usr/bin/env python3
"""End-to-end health monitor for forecast retrieval and publication pipelines."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import telegram_notify


PUBLIC_BASE_URL = "https://pub-969ec1fc2e19465797efb65b276a58da.r2.dev"
STATE_PATH = Path("logs/state/pipeline_health.json")
LATEST_PATH = Path("logs/pipeline_health_latest.json")
MACHINE_DATA_CONFIG = Path("~/.config/project-data.env").expanduser()
LOCAL_TZ = ZoneInfo("America/Vancouver")

SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class ManifestSpec:
    key: str
    label: str
    maximum_run_age: dt.timedelta


@dataclass(frozen=True)
class CheckResult:
    key: str
    label: str
    level: str
    summary: str
    immediate: bool = False


MODEL_MANIFESTS = (
    ManifestSpec("continental", "HRDPS 2.5 km", dt.timedelta(hours=12)),
    ManifestSpec("west", "HRDPS-West 1 km", dt.timedelta(hours=30)),
    ManifestSpec("gefs_control", "GEFS Control", dt.timedelta(hours=36)),
    ManifestSpec("ecmwf_control", "ECMWF Control", dt.timedelta(hours=30)),
    ManifestSpec("ecmwf_ensemble", "ECMWF Ensemble", dt.timedelta(hours=30)),
)

REQUIRED_LAUNCH_AGENTS = (
    "com.greg.hrdps-west-convective-00",
    "com.greg.hrdps-west-convective-12",
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
    "com.greg.fcst-r2-west",
    "com.greg.fcst-r2-gefs_control",
    "com.greg.fcst-r2-ecmwf_control",
    "com.greg.fcst-r2-ecmwf_ensemble",
    "com.greg.fcst-r2-usage-monitor",
    "com.greg.fcst-r2-usage-weekly-report",
    "com.greg.fcst-pipeline-health",
    "com.greg.fcst-pipeline-health-daily",
)


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
        level = max(
            (level_for_age(generated_age, dt.timedelta(minutes=30)), level_for_age(run_age, spec.maximum_run_age)),
            key=SEVERITY_ORDER.get,
        )
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


def latest_dated_directory(path: Path) -> dt.date | None:
    dates = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and re.fullmatch(r"\d{8}", child.name):
                try:
                    dates.append(dt.datetime.strptime(child.name, "%Y%m%d").date())
                except ValueError:
                    continue
    return max(dates) if dates else None


def check_cwfis_anchors(now: dt.datetime, data_root: Path) -> CheckResult:
    fields = ("ffmc", "dmc", "dc")
    latest = {field: latest_dated_directory(data_root / "cwfis_fwi" / field) for field in fields}
    if any(value is None for value in latest.values()):
        missing = ", ".join(field.upper() for field, value in latest.items() if value is None)
        return CheckResult("feed.cwfis_anchors", "CWFIS FWI anchors", "critical", f"missing {missing} archive")
    oldest = min(value for value in latest.values() if value is not None)
    age_days = (now.astimezone(LOCAL_TZ).date() - oldest).days
    level = "critical" if age_days > 4 else "warning" if age_days > 2 else "ok"
    dates = ", ".join(f"{field.upper()} {value:%Y-%m-%d}" for field, value in latest.items())
    return CheckResult("feed.cwfis_anchors", "CWFIS FWI anchors", level, dates)


def configured_volume(data_root: Path) -> Path:
    parts = data_root.resolve(strict=False).parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path("/") / parts[1] / parts[2]
    return data_root


def check_storage(data_root: Path) -> CheckResult:
    volume = configured_volume(data_root)
    if not volume.exists() or not data_root.exists():
        return CheckResult("storage.runtime", "Forecast data storage", "critical", f"unavailable: {volume}", immediate=True)
    try:
        usage = shutil.disk_usage(volume)
    except OSError as exc:
        return CheckResult("storage.runtime", "Forecast data storage", "critical", f"usage check failed: {exc}", immediate=True)
    free_fraction = usage.free / usage.total if usage.total else 0.0
    free_gb = usage.free / 1_000_000_000
    level = "critical" if free_fraction < 0.05 or free_gb < 50 else "warning" if free_fraction < 0.10 or free_gb < 100 else "ok"
    return CheckResult("storage.runtime", "Forecast data storage", level, f"{free_gb:.0f} GB free ({free_fraction:.0%})", immediate=True)


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


def check_launch_agent(label: str, auto_repair: bool = True) -> CheckResult:
    short = label.removeprefix("com.greg.")
    result = launchctl_print(label)
    repaired = False
    if result.returncode != 0 and auto_repair:
        repaired = reload_launch_agent(label)
        result = launchctl_print(label)
    if result.returncode != 0:
        return CheckResult(f"service.{label}", short, "critical", "launch agent is not loaded", immediate=True)
    if repaired:
        return CheckResult(f"service.{label}", short, "warning", "was unloaded and was reloaded automatically", immediate=True)
    match = re.search(r"last exit code = (-?\d+)", result.stdout)
    if match and int(match.group(1)) != 0:
        return CheckResult(f"service.{label}", short, "warning", f"last exit code {match.group(1)}", immediate=True)
    return CheckResult(f"service.{label}", short, "ok", "loaded", immediate=True)


def run_checks(
    now: dt.datetime,
    *,
    base_url: str,
    data_root: Path,
    auto_repair: bool = True,
    loader: Callable[[str], Mapping[str, object]] = fetch_json,
    include_services: bool = True,
) -> list[CheckResult]:
    checks = [check_storage(data_root)]
    checks.extend(check_model_manifest(spec, now, base_url, loader) for spec in MODEL_MANIFESTS)
    checks.append(check_fire_manifest(now, base_url, loader))
    checks.append(check_lightning_archive(now, data_root))
    checks.append(check_cwfis_anchors(now, data_root))
    if include_services:
        checks.extend(check_launch_agent(label, auto_repair) for label in REQUIRED_LAUNCH_AGENTS)
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


def apply_debounce(
    checks: Iterable[CheckResult],
    previous_state: Mapping[str, object],
) -> tuple[list[CheckResult], dict[str, int]]:
    previous_counts = previous_state.get("failure_counts", {})
    previous_counts = previous_counts if isinstance(previous_counts, dict) else {}
    effective: list[CheckResult] = []
    counts: dict[str, int] = {}
    for check in checks:
        if check.level == "ok":
            counts[check.key] = 0
            effective.append(check)
            continue
        count = int(previous_counts.get(check.key, 0) or 0) + 1
        counts[check.key] = count
        threshold = 1 if check.immediate else 2
        effective.append(check if count >= threshold else CheckResult(check.key, check.label, "ok", f"transient failure 1/{threshold}: {check.summary}"))
    return effective, counts


def problem_signature(checks: Iterable[CheckResult]) -> str:
    return "|".join(sorted(f"{check.key}:{check.level}" for check in checks if check.level != "ok"))


def overall_level(checks: Iterable[CheckResult]) -> str:
    return max((check.level for check in checks), default="ok", key=SEVERITY_ORDER.get)


def report_body(checks: list[CheckResult], now: dt.datetime, daily: bool) -> str:
    level = overall_level(checks).upper()
    problems = [check for check in checks if check.level != "ok"]
    lines = [f"Status: {level}", f"Checked: {now.astimezone(LOCAL_TZ):%Y-%m-%d %H:%M %Z}"]
    if problems:
        lines.append(f"Problems: {len(problems)}")
        for check in problems[:20]:
            lines.append(f"[{check.level.upper()}] {check.label}: {check.summary}")
        if len(problems) > 20:
            lines.append(f"...and {len(problems) - 20} more")
    else:
        lines.append(f"All {len(checks)} checks are healthy.")
    if daily:
        lines.append("")
        lines.append("Feed summary:")
        for check in checks:
            if check.key.startswith(("manifest.", "feed.", "storage.")):
                lines.append(f"{check.label}: {check.summary}")
    return "\n".join(lines)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--latest-path", type=Path, default=LATEST_PATH)
    parser.add_argument("--always-notify", action="store_true", help="Send the daily heartbeat even when healthy.")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--no-auto-repair", action="store_true")
    parser.add_argument("--no-services", action="store_true", help="Skip launch-agent checks, primarily for tests.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
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
    checks, failure_counts = apply_debounce(raw_checks, state)
    signature = problem_signature(checks)
    previous_signature = str(state.get("problem_signature", ""))
    notified_signature = str(state.get("notified_signature", previous_signature))
    payload = {
        "updated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "overall_level": overall_level(checks),
        "problem_signature": signature,
        "notified_signature": notified_signature,
        "failure_counts": failure_counts,
        "checks": [asdict(check) for check in checks],
        "raw_checks": [asdict(check) for check in raw_checks],
    }
    write_json(args.latest_path, payload)

    notify = args.always_notify or signature != notified_signature
    notification_sent = False
    notification_error = ""
    if notify and not args.no_notify:
        if signature:
            title = "Forecast Graphics Health Alert"
        elif previous_signature:
            title = "Forecast Graphics Health Recovered"
        else:
            title = "Forecast Graphics Daily Health"
        try:
            telegram_notify.send_message(title, report_body(checks, now, args.always_notify))
            notification_sent = True
            payload["notified_signature"] = signature
        except Exception as exc:
            notification_error = str(exc)
            print(f"Telegram notification failed: {exc}", flush=True)

    payload["notification_sent"] = notification_sent
    payload["notification_error"] = notification_error
    write_json(args.latest_path, payload)
    write_json(args.state_path, payload)
    healthy = sum(check.level == "ok" for check in checks)
    print(
        f"Pipeline health: level={payload['overall_level']}, healthy={healthy}/{len(checks)}, "
        f"problems={len(checks) - healthy}, notified={notification_sent}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
