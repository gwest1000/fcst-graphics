"""Resolve static and generated Forecast Graphics paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


PROJECT_NAME = "fcstGraphics"
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MACHINE_CONFIG = Path("~/.config/project-data.env").expanduser()


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve(strict=False)


def _runtime_environment() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("PROJECT_DATA_ROOT"):
        return env
    config_path = _expanded_path(
        env.get("PROJECT_DATA_CONFIG", str(DEFAULT_MACHINE_CONFIG))
    )
    try:
        lines = config_path.read_text().splitlines()
    except FileNotFoundError:
        return env
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "PROJECT_DATA_ROOT":
            env[name.strip()] = value.strip().strip("\"'")
            break
    return env


def _configured_root(
    *,
    kind: str,
    environ: Mapping[str, str],
    project_root: Path,
) -> tuple[Path, str | None]:
    override_name = f"FCSTGRAPHICS_{kind.upper()}_ROOT"
    override = environ.get(override_name, "").strip()
    if override:
        return _expanded_path(override), override_name

    shared = environ.get("PROJECT_DATA_ROOT", "").strip()
    if shared:
        shared_root = _expanded_path(shared)
        if not shared_root.is_dir():
            raise RuntimeError(
                f"PROJECT_DATA_ROOT is configured but unavailable: {shared_root}. "
                "Mount the data volume or correct the machine-level setting."
            )
        return shared_root / PROJECT_NAME / kind, "PROJECT_DATA_ROOT"

    return project_root / kind, None


def _validate_configured_root(path: Path, source: str | None) -> Path:
    if source is None:
        return path
    if path.is_dir():
        return path
    raise RuntimeError(
        f"{source} resolved to an unavailable Forecast Graphics directory: {path}. "
        "Mount the data volume or create the configured project directory first."
    )


def data_root(
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Return the generated-data root, honoring project and shared overrides."""

    env = _runtime_environment() if environ is None else environ
    path, source = _configured_root(kind="data", environ=env, project_root=project_root)
    return _validate_configured_root(path, source)


def plot_root(
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Return the generated-plot root, honoring project and shared overrides."""

    env = _runtime_environment() if environ is None else environ
    path, source = _configured_root(kind="plots", environ=env, project_root=project_root)
    return _validate_configured_root(path, source)


def data_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)


def plot_path(*parts: str) -> Path:
    return plot_root().joinpath(*parts)


def static_data_path(*parts: str) -> Path:
    return STATIC_DATA_ROOT.joinpath(*parts)
