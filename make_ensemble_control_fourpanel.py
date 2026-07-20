#!/usr/bin/env python3
"""Make synoptic four-panel sheets for ECMWF/GEFS control members."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from eccodes import codes_get, codes_get_array, codes_grib_new_from_file, codes_release
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

import plot_style
from make_hrdps_west_convective import RunInfo, WATERSHED_CACHE, parse_stamp, subset_slices
from make_hrdps_west_fourpanel import (
    DATA_CRS,
    PANEL_PROJ,
    add_watersheds,
    decimate,
    label_contours,
    load_watersheds,
    make_absv_cmap,
    make_ipw_cmap,
    make_precip_cmap,
    make_rh_cmap,
    plot_barbs,
)

FORECAST_HOURS = tuple(range(0, 49, 3))
GEFS_FORECAST_HOURS = tuple(range(3, 49, 3))
EXTENT = (-138.7, -109.0, 46.0, 58.45)
CONCRETE_REPO = Path("/Users/greg/projects/concrete_fcst")
CONCRETE_DATA_ROOT = Path("/Volumes/Greg1_2tb/concrete_fcst_data")
OMEGA = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0
DRY_AIR_GAS_CONSTANT = 287.05
GRAVITY = 9.80665
WIND_850_TERRAIN_MARGIN_HPA = 35.0
DEFAULT_BARB_STRIDE = {
    "ecmwf_control": 1,
    "gefs_control": 1,
}
VECTOR_DENSITY_BY_MODEL = {
    "gefs_control": 3.0,
    "ecmwf_control": 3.75,
}
IVT_VECTOR_STYLES = (
    (0.0, 250.0, "#777777", 0.00155, "<250"),
    (250.0, 500.0, "#000000", 0.00180, "250-499"),
    (500.0, 750.0, "#000000", 0.00270, "500-749"),
    (750.0, 1000.0, "#ffffff", 0.00310, "750-999"),
    (1000.0, np.inf, "#b000d4", 0.00345, "1000+"),
)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    source_label: str
    output_prefix: str
    default_output_dir: str
    resolution_km: float


MODEL_CONFIGS = {
    "ecmwf_control": ModelConfig(
        key="ecmwf_control",
        label="ECMWF IFS Control",
        source_label="ECMWF Open Data control",
        output_prefix="ecmwf_control",
        default_output_dir="plots/ecmwf_control_fourpanel",
        resolution_km=28.0,
    ),
    "gefs_control": ModelConfig(
        key="gefs_control",
        label="GEFS Control",
        source_label="NOAA GEFS control",
        output_prefix="gefs_control",
        default_output_dir="plots/gefs_control_fourpanel",
        resolution_km=28.0,
    ),
}


def model_hours(model: str) -> tuple[int, ...]:
    return GEFS_FORECAST_HOURS if model == "gefs_control" else FORECAST_HOURS


@dataclass
class Field:
    data: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    step_range: str = ""


def log(message: str) -> None:
    print(message, flush=True)


def grid_stride(config: ModelConfig, target_km: float, minimum: int = 1) -> int:
    return max(minimum, int(round(target_km / config.resolution_km)))


def raw_contour_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return lat[::stride, ::stride], lon[::stride, ::stride], data[::stride, ::stride]


def default_barb_stride(config: ModelConfig) -> int:
    return DEFAULT_BARB_STRIDE.get(config.key, grid_stride(config, 165.0))


def pressure_panel_barb_stride(config: ModelConfig, surface_stride: int) -> int:
    return surface_stride


def barb_row_density(config: ModelConfig) -> float:
    return VECTOR_DENSITY_BY_MODEL[config.key]


def barb_column_density(config: ModelConfig) -> float:
    return VECTOR_DENSITY_BY_MODEL[config.key]


def cycle_date_from_stamp(stamp: str) -> dt.date:
    return parse_stamp(stamp).date()


def cycle_hour_from_stamp(stamp: str) -> int:
    return parse_stamp(stamp).hour


def latest_cycle_stamp(cycle: int = 0, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    candidate = now.replace(hour=cycle, minute=0, second=0, microsecond=0)
    if candidate > now:
        candidate -= dt.timedelta(days=1)
    return f"{candidate:%Y%m%dT%HZ}"


def grib_field_matches(gid, short_name: str, level_type: str, level: int, step: int) -> bool:
    try:
        return (
            str(codes_get(gid, "shortName")).lower() == short_name.lower()
            and str(codes_get(gid, "typeOfLevel")) == level_type
            and int(codes_get(gid, "level")) == int(level)
            and int(codes_get(gid, "step")) == int(step)
        )
    except Exception:
        return False


def grib_values(gid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = int(codes_get(gid, "Ni"))
    ny = int(codes_get(gid, "Nj"))
    data = codes_get_array(gid, "values").reshape(ny, nx).astype(np.float32)
    lat = codes_get_array(gid, "latitudes").reshape(ny, nx).astype(np.float32)
    lon = codes_get_array(gid, "longitudes").reshape(ny, nx).astype(np.float32)
    lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
    data[np.abs(data) > 1e20] = np.nan
    return data, lat, lon


def read_matching_grib(path: Path, short_name: str, level_type: str, level: int, step: int) -> Field:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                if grib_field_matches(gid, short_name, level_type, level, step):
                    data, lat, lon = grib_values(gid)
                    step_range = str(codes_get(gid, "stepRange"))
                    return Field(data=data, lat=lat, lon=lon, step_range=step_range)
            finally:
                codes_release(gid)
    raise KeyError(f"{short_name}:{level_type}:{level}:F{step:03d} not found in {path}")


def crop_extent(field: Field) -> Field:
    yslice, xslice = subset_slices(field.lat, field.lon, EXTENT)
    return Field(
        data=field.data[yslice, xslice],
        lat=field.lat[yslice, xslice],
        lon=field.lon[yslice, xslice],
        step_range=field.step_range,
    )


def parse_step_start(step_range: str) -> int | None:
    if "-" not in step_range:
        return None
    start, _, _ = step_range.partition("-")
    try:
        return int(start)
    except ValueError:
        return None


def parse_step_range(step_range: str) -> tuple[int, int] | None:
    if "-" not in step_range:
        return None
    start, _, end = step_range.partition("-")
    try:
        return int(start), int(end)
    except ValueError:
        return None


def field_like(reference: Field, data: np.ndarray, step_range: str = "") -> Field:
    return Field(data=data.astype(np.float32), lat=reference.lat, lon=reference.lon, step_range=step_range)


def regular_grid_absv(u: Field, v: Field) -> Field:
    lon_rad = np.deg2rad(u.lon.astype(np.float64))
    lat_rad = np.deg2rad(u.lat.astype(np.float64))
    uu = u.data.astype(np.float64)
    vv = v.data.astype(np.float64)
    dvdx = np.full_like(vv, np.nan, dtype=np.float64)
    dudy = np.full_like(uu, np.nan, dtype=np.float64)

    dlon = lon_rad[:, 2:] - lon_rad[:, :-2]
    dx = EARTH_RADIUS_M * np.cos(lat_rad[:, 1:-1]) * dlon
    dvdx[:, 1:-1] = (vv[:, 2:] - vv[:, :-2]) / dx
    dvdx[:, 0] = dvdx[:, 1]
    dvdx[:, -1] = dvdx[:, -2]

    dlat = lat_rad[2:, :] - lat_rad[:-2, :]
    dy = EARTH_RADIUS_M * dlat
    dudy[1:-1, :] = (uu[2:, :] - uu[:-2, :]) / dy
    dudy[0, :] = dudy[1, :]
    dudy[-1, :] = dudy[-2, :]

    coriolis = 2.0 * OMEGA * np.sin(lat_rad)
    absv = (dvdx - dudy + coriolis) * 1.0e5
    return field_like(u, absv)


def precip_3h(current: Field, previous: Field | None, fhour: int) -> Field:
    data = np.maximum(current.data, 0.0)
    current_range = parse_step_range(current.step_range)
    previous_range = parse_step_range(previous.step_range) if previous is not None else None
    should_subtract = False
    if previous is not None:
        if current_range is None:
            should_subtract = True
        elif previous_range is not None:
            current_start, current_end = current_range
            previous_start, previous_end = previous_range
            should_subtract = (
                current_end - current_start > 3
                and previous_start == current_start
                and previous_end < current_end
            )
        else:
            current_start, current_end = current_range
            should_subtract = current_start == 0 and current_end > 3
    if should_subtract:
        data = np.maximum(current.data - previous.data, 0.0)
    return field_like(current, data, current.step_range)


def specific_humidity_from_rh(temp_k: np.ndarray, rh_percent: np.ndarray, pressure_hpa: float) -> np.ndarray:
    temp_c = temp_k - 273.15
    es_hpa = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e_hpa = np.clip(rh_percent, 0.0, 100.0) * 0.01 * es_hpa
    mixing_ratio = 0.622 * e_hpa / np.maximum(pressure_hpa - e_hpa, 1.0)
    return mixing_ratio / (1.0 + mixing_ratio)


def pressure_level_ipw(provider: "BaseProvider", fhour: int) -> Field:
    levels = (925, 850, 700, 500)
    q_fields: list[Field] = []
    for level in levels:
        temp = provider.pressure(fhour, "t", level)
        rh = provider.pressure(fhour, "r", level)
        q_fields.append(field_like(temp, specific_humidity_from_rh(temp.data, rh.data, float(level))))

    ipw = np.zeros(q_fields[0].data.shape, dtype=np.float32)
    for index in range(len(levels) - 1):
        layer_pa = (levels[index] - levels[index + 1]) * 100.0
        ipw += (0.5 * (q_fields[index].data + q_fields[index + 1].data) * layer_pa / 9.80665).astype(np.float32)
    return field_like(q_fields[0], np.maximum(ipw, 0.0))


def optional_pressure(provider: "BaseProvider", fhour: int, short_name: str, level: int) -> Field | None:
    try:
        return provider.pressure(fhour, short_name, level)
    except Exception:
        return None


def interpolate_to_reference(source: Field, reference: Field) -> Field:
    grids_match = (
        source.data.shape == reference.data.shape
        and np.allclose(source.lat, reference.lat)
        and np.allclose(source.lon, reference.lon)
    )
    if grids_match:
        return source

    latitudes = source.lat[:, 0].astype(np.float64)
    longitudes = source.lon[0, :].astype(np.float64)
    values = source.data.astype(np.float64)
    if latitudes[0] > latitudes[-1]:
        latitudes = latitudes[::-1]
        values = values[::-1, :]
    if longitudes[0] > longitudes[-1]:
        longitudes = longitudes[::-1]
        values = values[:, ::-1]
    interpolator = RegularGridInterpolator(
        (latitudes, longitudes),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack((reference.lat.ravel(), reference.lon.ravel()))
    return field_like(reference, interpolator(points).reshape(reference.data.shape))


def full_column_vapor_transport(provider: "BaseProvider", fhour: int) -> tuple[Field, Field]:
    """Estimate full-column IVT using TCW and the moisture-weighted pressure-level wind."""
    levels = (925, 850, 700, 500)
    q_fields: list[Field] = []
    u_fields: list[Field] = []
    v_fields: list[Field] = []
    for level in levels:
        temp = provider.pressure(fhour, "t", level)
        rh = provider.pressure(fhour, "r", level)
        q_fields.append(
            field_like(temp, specific_humidity_from_rh(temp.data, rh.data, float(level)))
        )
        u_fields.append(provider.pressure(fhour, "u", level))
        v_fields.append(provider.pressure(fhour, "v", level))

    layer_water = np.zeros(q_fields[0].data.shape, dtype=np.float32)
    transport_u = np.zeros_like(layer_water)
    transport_v = np.zeros_like(layer_water)
    for index in range(len(levels) - 1):
        layer_pa = (levels[index] - levels[index + 1]) * 100.0
        layer_water += (
            0.5 * (q_fields[index].data + q_fields[index + 1].data) * layer_pa / GRAVITY
        ).astype(np.float32)
        transport_u += (
            0.5
            * (
                q_fields[index].data * u_fields[index].data
                + q_fields[index + 1].data * u_fields[index + 1].data
            )
            * layer_pa
            / GRAVITY
        ).astype(np.float32)
        transport_v += (
            0.5
            * (
                q_fields[index].data * v_fields[index].data
                + q_fields[index + 1].data * v_fields[index + 1].data
            )
            * layer_pa
            / GRAVITY
        ).astype(np.float32)

    total_column_water = interpolate_to_reference(provider.ipw(fhour), q_fields[0]).data
    scale = np.divide(
        total_column_water,
        layer_water,
        out=np.full_like(layer_water, np.nan),
        where=np.isfinite(total_column_water) & (layer_water > 0.1),
    )
    return (
        field_like(q_fields[0], transport_u * scale),
        field_like(q_fields[0], transport_v * scale),
    )


def low_level_vertical_velocity_cm_s(provider: "BaseProvider", fhour: int) -> Field | None:
    omega = optional_pressure(provider, fhour, "w", 850)
    if omega is None:
        return None
    temp = provider.pressure(fhour, "t", 850)
    rho = 85000.0 / (DRY_AIR_GAS_CONSTANT * np.maximum(temp.data, 150.0))
    vertical_velocity = (-omega.data / np.maximum(rho * GRAVITY, 1.0e-6)) * 100.0
    return field_like(omega, vertical_velocity)


def terrain_adjusted_850_wind(
    provider: "BaseProvider",
    fhour: int,
    u850: Field,
    v850: Field,
    u700: Field,
    v700: Field,
) -> tuple[Field, Field]:
    psfc = provider.surface_pressure_hpa(fhour)
    if psfc is None or psfc.data.shape != u850.data.shape:
        return u850, v850
    use700 = np.isfinite(psfc.data) & (psfc.data <= 850.0 + WIND_850_TERRAIN_MARGIN_HPA)
    return field_like(u850, np.where(use700, u700.data, u850.data)), field_like(v850, np.where(use700, v700.data, v850.data))


class BaseProvider:
    def __init__(self, data_root: Path, run: RunInfo):
        self.data_root = data_root
        self.run = run

    def pressure(self, fhour: int, short_name: str, level: int) -> Field:
        raise NotImplementedError

    def surface(self, fhour: int, short_name: str, level_type: str, level: int = 0) -> Field:
        raise NotImplementedError

    def surface_pressure_hpa(self, fhour: int) -> Field | None:
        return None

    def precip(self, fhour: int) -> Field:
        current = self.surface(fhour, "tp", "surface", 0)
        previous = None
        if fhour > 0:
            try:
                previous = self.surface(fhour - 3, "tp", "surface", 0)
            except Exception:
                previous = None
        return precip_3h(current, previous, fhour)


class EcmwfProvider(BaseProvider):
    def cycle_root(self) -> Path:
        return self.data_root / "raw" / "ecmwf" / "realtime" / f"{self.run.init_time:%Y%m%d}" / self.run.cycle

    def pressure(self, fhour: int, short_name: str, level: int) -> Field:
        return crop_extent(read_matching_grib(self.cycle_root() / "pl_cf.grib2", short_name, "isobaricInhPa", level, fhour))

    def surface(self, fhour: int, short_name: str, level_type: str, level: int = 0) -> Field:
        return crop_extent(read_matching_grib(self.cycle_root() / "sfc_cf.grib2", short_name, level_type, level, fhour))

    def surface_pressure_hpa(self, fhour: int) -> Field | None:
        try:
            psfc = self.surface(fhour, "sp", "surface", 0)
        except Exception:
            return None
        return field_like(psfc, psfc.data / 100.0)

    def cape(self, fhour: int, reference: Field) -> Field | None:
        return None

    def precip(self, fhour: int) -> Field:
        precip_metres = super().precip(fhour)
        return field_like(
            precip_metres,
            precip_metres.data * 1000.0,
            precip_metres.step_range,
        )

    def ipw(self, fhour: int) -> Field:
        return self.surface(fhour, "tcwv", "entireAtmosphere", 0)


class GefsProvider(BaseProvider):
    def cycle_root(self) -> Path:
        return self.data_root / "raw" / "gefs" / "realtime" / f"{self.run.init_time:%Y%m%d}" / self.run.cycle / "gec00"

    def gefs_file(self, fhour: int, product: str) -> Path:
        stem = "pgrb2s.0p25" if product == "surface_0p25" else "pgrb2a.0p50"
        return self.cycle_root() / product / f"gec00.t{int(self.run.cycle):02d}z.{stem}.f{fhour:03d}.{product}.grib2"

    def pressure(self, fhour: int, short_name: str, level: int) -> Field:
        return crop_extent(read_matching_grib(self.gefs_file(fhour, "pressure_0p50"), short_name, "isobaricInhPa", level, fhour))

    def surface(self, fhour: int, short_name: str, level_type: str, level: int = 0) -> Field:
        return crop_extent(read_matching_grib(self.gefs_file(fhour, "surface_0p25"), short_name, level_type, level, fhour))

    def surface_pressure_hpa(self, fhour: int) -> Field | None:
        for product in ("pressure_0p50", "surface_0p25"):
            try:
                psfc = crop_extent(read_matching_grib(self.gefs_file(fhour, product), "pres", "surface", 0, fhour))
            except Exception:
                continue
            return field_like(psfc, psfc.data / 100.0)
        return None

    def cape(self, fhour: int, reference: Field) -> Field | None:
        try:
            return self.surface(fhour, "cape", "surface", 0)
        except Exception:
            return None

    def ipw(self, fhour: int) -> Field:
        for level_type in ("atmosphereSingleLayer", "atmosphere", "entireAtmosphere"):
            try:
                return self.surface(fhour, "pwat", level_type, 0)
            except Exception:
                pass
        return pressure_level_ipw(self, fhour)


def provider_for(model: str, data_root: Path, run: RunInfo) -> BaseProvider:
    if model == "ecmwf_control":
        return EcmwfProvider(data_root, run)
    if model == "gefs_control":
        return GefsProvider(data_root, run)
    raise ValueError(f"Unsupported model: {model}")


def required_files_present(model: str, data_root: Path, run: RunInfo, hours: Iterable[int]) -> bool:
    provider = provider_for(model, data_root, run)
    try:
        for fhour in hours:
            provider.pressure(fhour, "gh", 500)
            provider.pressure(fhour, "u", 500)
            provider.pressure(fhour, "v", 500)
            provider.pressure(fhour, "t", 500)
            provider.pressure(fhour, "t", 850)
            provider.pressure(fhour, "r", 850)
            provider.pressure(fhour, "r", 700)
            provider.pressure(fhour, "u", 850)
            provider.pressure(fhour, "v", 850)
            provider.pressure(fhour, "u", 700)
            provider.pressure(fhour, "v", 700)
            provider.ipw(fhour)
            provider.surface(fhour, "msl" if model == "ecmwf_control" else "prmsl", "meanSea", 0)
            provider.surface(fhour, "10u", "heightAboveGround", 10)
            provider.surface(fhour, "10v", "heightAboveGround", 10)
            provider.precip(fhour)
    except Exception as exc:
        log(f"Missing {model} input fields for {run.stamp}: {exc}")
        return False
    return True


def concrete_python(repo_root: Path) -> Path:
    candidate = repo_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def ensure_downloads(
    model: str,
    run: RunInfo,
    hours: Iterable[int],
    concrete_repo: Path,
    data_root: Path,
    force: bool = False,
) -> None:
    hours = tuple(sorted(set(int(hour) for hour in hours)))
    python = concrete_python(concrete_repo)
    if model == "ecmwf_control":
        cycle_root = data_root / "raw" / "ecmwf" / "realtime" / f"{run.init_time:%Y%m%d}" / run.cycle
        include_labels = []
        if force or not (cycle_root / "sfc_cf.grib2").exists():
            include_labels.append("surface_cf")
        if force or not (cycle_root / "pl_cf.grib2").exists():
            include_labels.append("pressure_cf")
        if not include_labels:
            return
        cmd = [
            str(python),
            "-m",
            "concrete_fcst.cli",
            "download-ecmwf-realtime",
            "--repo-root",
            str(concrete_repo),
            "--date",
            f"{run.init_time:%Y%m%d}",
            "--cycle",
            str(int(run.cycle)),
            "--include",
            *include_labels,
        ]
    elif model == "gefs_control":
        cmd = [
            str(python),
            "-m",
            "concrete_fcst.cli",
            "download-gefs-realtime",
            "--repo-root",
            str(concrete_repo),
            "--cycle-date",
            f"{run.init_time:%Y-%m-%d}",
            "--cycle-hour",
            str(int(run.cycle)),
            "--member",
            "gec00",
            "--forecast-hour",
            *(str(hour) for hour in hours),
            "--show-results",
            "5",
            "--continue-on-error",
        ]
        if force:
            cmd.extend(["--no-skip-existing"])
    else:
        raise ValueError(f"Unsupported model: {model}")
    log("Running data download: " + " ".join(cmd))
    subprocess.run(cmd, cwd=concrete_repo, check=True)


def add_base_features(ax: plt.Axes) -> None:
    ax.set_extent(EXTENT, crs=DATA_CRS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#ffffff")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f4ef", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=0.75, zorder=20)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black", linewidth=0.65, zorder=21)
    admin = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_1_states_provinces_lines",
        "50m",
        facecolor="none",
        edgecolor="black",
        linewidth=0.55,
    )
    ax.add_feature(admin, zorder=21)
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        ax.spines["geo"].set_linewidth(1.5)
        ax.spines["geo"].set_edgecolor("black")
    except Exception:
        pass


def unit_vector_components(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude = np.hypot(u, v)
    valid = np.isfinite(magnitude) & (magnitude > 0.0)
    unit_u = np.divide(
        u,
        magnitude,
        out=np.full_like(u, np.nan, dtype=np.float64),
        where=valid,
    )
    unit_v = np.divide(
        v,
        magnitude,
        out=np.full_like(v, np.nan, dtype=np.float64),
        where=valid,
    )
    return unit_u, unit_v, magnitude


def add_ivt_vector_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=color, linewidth=max(1.2, width * 900.0), label=label)
        for _, _, color, width, label in IVT_VECTOR_STYLES
    ]
    legend = ax.legend(
        handles=handles,
        title="IVT kg m$^{-1}$ s$^{-1}$",
        loc="upper left",
        bbox_to_anchor=(0.006, 0.948),
        ncol=5,
        borderaxespad=0.0,
        borderpad=0.25,
        columnspacing=0.75,
        handlelength=1.35,
        handletextpad=0.3,
        frameon=True,
        facecolor="#d9d9d9",
        edgecolor="black",
        framealpha=0.94,
        fontsize=5.5,
        title_fontsize=5.8,
    )
    legend.set_zorder(52)


def plot_transport_vectors(
    ax: plt.Axes,
    u: Field,
    v: Field,
    stride: int,
    row_density: float = 1.0,
    column_density: float = 1.0,
) -> None:
    sample = plot_style.vector_sample_slices(
        ax,
        u.data.shape,
        minimum=stride,
        row_density=row_density,
        column_density=column_density,
    )
    uu = u.data[sample]
    vv = v.data[sample]
    unit_u, unit_v, magnitude = unit_vector_components(uu, vv)
    sample_lon = u.lon[sample]
    sample_lat = u.lat[sample]
    for lower, upper, color, width, _ in IVT_VECTOR_STYLES:
        selected = (
            np.isfinite(unit_u)
            & np.isfinite(unit_v)
            & (magnitude >= lower)
            & (magnitude < upper)
        )
        if not np.any(selected):
            continue
        ax.quiver(
            sample_lon[selected],
            sample_lat[selected],
            unit_u[selected],
            unit_v[selected],
            transform=DATA_CRS,
            color=color,
            width=width,
            scale=4.6,
            scale_units="inches",
            pivot="middle",
            headwidth=3.7,
            headlength=4.8,
            headaxislength=4.1,
            minlength=0.05,
            zorder=24,
        )
    add_ivt_vector_legend(ax)


def plot_fourpanel(
    out_path: Path,
    config: ModelConfig,
    provider: BaseProvider,
    run: RunInfo,
    fhour: int,
    watersheds,
    shade_stride: int,
    contour_stride: int,
    barb_stride: int,
) -> None:
    header = plot_style.valid_header(run, fhour, config.label)
    pressure_barb_stride = pressure_panel_barb_stride(config, barb_stride)
    wind_row_density = barb_row_density(config)
    wind_column_density = barb_column_density(config)
    fig = plt.figure(figsize=plot_style.PLOT_FIGSIZE, dpi=plot_style.PLOT_DPI, facecolor="white")
    axes = [fig.add_axes(position, projection=PANEL_PROJ) for position in plot_style.FOURPANEL_POSITIONS]
    for ax in axes:
        add_base_features(ax)

    # 1) 500 hPa absolute vorticity, 500 hPa height, 500 hPa wind.
    ax = axes[0]
    hgt500 = provider.pressure(fhour, "gh", 500)
    u500 = provider.pressure(fhour, "u", 500)
    v500 = provider.pressure(fhour, "v", 500)
    absv = regular_grid_absv(u500, v500)
    cmap, norm, levels = make_absv_cmap()
    cf = ax.contourf(
        decimate(absv.lon, shade_stride),
        decimate(absv.lat, shade_stride),
        decimate(absv.data, shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    clat, clon, chgt = raw_contour_grid(
        hgt500.lat,
        hgt500.lon,
        hgt500.data / 1000.0,
        stride=contour_stride,
    )
    hgt_ct = ax.contour(clon, clat, chgt, levels=np.arange(4.8, 6.3, 0.12), colors="black", linewidths=1.35, transform=DATA_CRS, zorder=22)
    label_contours(hgt_ct, fontsize=5.4, fmt="%.2g")
    plot_barbs(
        ax,
        u500.lon,
        u500.lat,
        u500.data,
        v500.data,
        pressure_barb_stride,
        color="black",
        row_density=wind_row_density,
        column_density=wind_column_density,
    )
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=[-4, 0, 4, 8, 12, 16, 20, 24], label="$10^{-5}$ s$^{-1}$", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, "50.0kPa AbsVort(shaded), Hgt(cntrd,km), 50.0kPa Wind(hlf brb=10km/h)", run, config.source_label)

    # 2) Total column water, low-level vertical velocity, and integrated vapor transport.
    ax = axes[1]
    ipw = provider.ipw(fhour)
    vt_u, vt_v = full_column_vapor_transport(provider, fhour)
    wvel = low_level_vertical_velocity_cm_s(provider, fhour)
    cmap, norm, levels = make_ipw_cmap()
    cf = ax.contourf(
        decimate(ipw.lon, shade_stride),
        decimate(ipw.lat, shade_stride),
        decimate(ipw.data, shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    footer = "IPW(shaded,mm), IVT(unit vctrs coloured by kg m-1 s-1)"
    if wvel is not None:
        clat, clon, cwvel = raw_contour_grid(
            wvel.lat,
            wvel.lon,
            wvel.data,
            stride=contour_stride,
        )
        positive = ax.contour(
            clon,
            clat,
            cwvel,
            levels=np.arange(5, 55, 5),
            colors="#d00000",
            linewidths=1.1,
            transform=DATA_CRS,
            zorder=23,
        )
        negative = ax.contour(
            clon,
            clat,
            cwvel,
            levels=np.arange(-50, 0, 5),
            colors="#1658d3",
            linewidths=1.0,
            linestyles="dashed",
            transform=DATA_CRS,
            zorder=23,
        )
        label_contours(positive, fontsize=5.0, fmt="%d", colors="#d00000")
        label_contours(negative, fontsize=5.0, fmt="%d", colors="#1658d3")
        footer = "IPW(shaded,mm), LL WVel(cntrd every 5cm/s), IVT(unit vctrs coloured by magnitude)"
    plot_transport_vectors(
        ax,
        vt_u,
        vt_v,
        pressure_barb_stride,
        row_density=wind_row_density,
        column_density=wind_column_density,
    )
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=np.arange(10, 52, 2), label="mm", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, footer, run, config.source_label)

    # 3) 850-700 hPa RH, 850 hPa temperature, 850 hPa wind.
    ax = axes[2]
    rh850 = provider.pressure(fhour, "r", 850)
    rh700 = provider.pressure(fhour, "r", 700)
    rh = field_like(rh850, np.nanmean(np.stack([rh850.data, rh700.data]), axis=0))
    tmp850 = provider.pressure(fhour, "t", 850)
    u850 = provider.pressure(fhour, "u", 850)
    v850 = provider.pressure(fhour, "v", 850)
    u700 = provider.pressure(fhour, "u", 700)
    v700 = provider.pressure(fhour, "v", 700)
    u_panel, v_panel = terrain_adjusted_850_wind(provider, fhour, u850, v850, u700, v700)
    cmap, norm, levels = make_rh_cmap()
    cf = ax.contourf(
        decimate(rh.lon, shade_stride),
        decimate(rh.lat, shade_stride),
        decimate(rh.data, shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    tmp850_c = tmp850.data - 273.15
    clat, clon, ctmp = raw_contour_grid(
        tmp850.lat,
        tmp850.lon,
        tmp850_c,
        stride=contour_stride,
    )
    temp_ct = ax.contour(clon, clat, ctmp, levels=[level for level in np.arange(-32, 34, 2) if level != 0], colors="black", linewidths=1.15, transform=DATA_CRS, zorder=22)
    label_contours(temp_ct, fontsize=5.8, fmt="%d")
    zero_ct = ax.contour(clon, clat, ctmp, levels=[0], colors="#0057ff", linewidths=1.65, transform=DATA_CRS, zorder=23)
    label_contours(zero_ct, fontsize=5.8, fmt="%d", colors="#0057ff")
    warm_ct = ax.contour(clon, clat, ctmp, levels=[16], colors="#ff8c00", linewidths=1.45, transform=DATA_CRS, zorder=23)
    label_contours(warm_ct, fontsize=5.8, fmt="%d", colors="#ff8c00")
    plot_barbs(
        ax,
        u_panel.lon,
        u_panel.lat,
        u_panel.data,
        v_panel.data,
        pressure_barb_stride,
        color="black",
        row_density=wind_row_density,
        column_density=wind_column_density,
    )
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=[10, 15, 20, 25, 30, 70, 75, 80, 85, 90], label="%", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, "85.0-70.0kPa RH(%,shaded), 85.0kPa Temp(C,cntrd), 85/70kPa Wind(hlf brb=10km/h)", run, config.source_label)

    # 4) Three-hour precipitation, MSLP, 10 m wind.
    ax = axes[3]
    precip = provider.precip(fhour)
    msl = provider.surface(fhour, "msl" if config.key == "ecmwf_control" else "prmsl", "meanSea", 0)
    u10 = provider.surface(fhour, "10u", "heightAboveGround", 10)
    v10 = provider.surface(fhour, "10v", "heightAboveGround", 10)
    cmap, norm, levels = make_precip_cmap()
    cf = ax.contourf(
        decimate(precip.lon, shade_stride),
        decimate(precip.lat, shade_stride),
        decimate(precip.data, shade_stride),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=DATA_CRS,
        transform_first=True,
        zorder=3,
    )
    clat, clon, cmslp = raw_contour_grid(
        msl.lat,
        msl.lon,
        msl.data / 1000.0,
        stride=contour_stride,
    )
    mslp_ct = ax.contour(clon, clat, cmslp, levels=np.arange(95.2, 104.8, 0.4), colors="#5f5f5f", linewidths=1.05, transform=DATA_CRS, zorder=22)
    label_contours(mslp_ct, fontsize=5.6, fmt="%.1f")
    major_mslp = ax.contour(clon, clat, cmslp, levels=np.arange(95.2, 104.8, 0.8), colors="#0046ff", linewidths=1.35, transform=DATA_CRS, zorder=23)
    label_contours(major_mslp, fontsize=5.8, fmt="%.1f", colors="black")
    plot_barbs(
        ax,
        u10.lon,
        u10.lat,
        u10.data,
        v10.data,
        barb_stride,
        color="black",
        row_density=wind_row_density,
        column_density=wind_column_density,
    )
    add_watersheds(ax, watersheds)
    plot_style.add_fourpanel_colorbar(fig, ax, cf, ticks=[0.25, 2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 60, 80, 100], label="mm", fmt="%g")
    plot_style.add_fourpanel_text(ax, header, "3h Precip(shaded,mm), MSLP(cntrd,kPa), and 10m Wind(hlf brb=10km/h)", run, config.source_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def make_plots(
    model: str,
    run: RunInfo,
    data_root: Path,
    output_dir: Path,
    watershed_cache: Path,
    refresh_watersheds: bool,
    no_watersheds: bool,
    shade_stride: int | None = None,
    contour_stride: int | None = None,
    barb_stride: int | None = None,
    hours: Iterable[int] = FORECAST_HOURS,
) -> list[Path]:
    config = MODEL_CONFIGS[model]
    hours = tuple(int(hour) for hour in hours)
    if not hours:
        return []
    provider = provider_for(model, data_root, run)
    shade_stride = shade_stride or 1
    contour_stride = contour_stride or 1
    barb_stride = barb_stride or default_barb_stride(config)
    watersheds = [] if no_watersheds else load_watersheds(watershed_cache, refresh=refresh_watersheds)
    plot_dir = output_dir / run.stamp
    out_paths: list[Path] = []
    for fhour in hours:
        log(f"Plotting {config.label} F{fhour:03d}.")
        out_path = plot_dir / f"{config.output_prefix}_fourpanel_{run.stamp}_f{fhour:03d}.png"
        plot_fourpanel(out_path, config, provider, run, fhour, watersheds, shade_stride, contour_stride, barb_stride)
        log(f"  wrote {out_path}")
        out_paths.append(out_path)
    return out_paths


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--stamp", default=None, help="Run stamp, e.g. 20260630T00Z. Defaults to latest 00Z.")
    parser.add_argument("--cycle", type=int, default=0, choices=[0, 6, 12, 18])
    parser.add_argument("--data-root", type=Path, default=CONCRETE_DATA_ROOT)
    parser.add_argument("--concrete-repo", type=Path, default=CONCRETE_REPO)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hours", default=None, help="Comma-separated forecast hours to plot.")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--shade-stride", type=int, default=None)
    parser.add_argument("--contour-stride", type=int, default=None)
    parser.add_argument("--barb-stride", type=int, default=None)
    parser.add_argument("--watershed-cache", type=Path, default=WATERSHED_CACHE)
    parser.add_argument("--refresh-watersheds", action="store_true")
    parser.add_argument("--no-watersheds", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = MODEL_CONFIGS[args.model]
    stamp = args.stamp or latest_cycle_stamp(args.cycle)
    run = RunInfo(cycle=f"{cycle_hour_from_stamp(stamp):02d}", stamp=stamp, init_time=parse_stamp(stamp))
    hours = model_hours(args.model) if args.hours is None else tuple(int(item) for item in args.hours.split(",") if item.strip())
    output_dir = args.output_dir or Path(config.default_output_dir)
    if args.download and (args.force_download or not required_files_present(args.model, args.data_root, run, hours)):
        ensure_downloads(args.model, run, hours, args.concrete_repo, args.data_root, force=args.force_download)
    make_plots(
        args.model,
        run,
        args.data_root,
        output_dir,
        args.watershed_cache,
        args.refresh_watersheds,
        args.no_watersheds,
        args.shade_stride,
        args.contour_stride,
        args.barb_stride,
        hours,
    )
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raise SystemExit(main(sys.argv[1:]))
