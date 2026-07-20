"""Pure diagnostics for the HRDPS all-cause gust estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
GRAVITY_MS2 = 9.80665
DOWNSLOPE_RADIUS_KM = 10.0
DOWNSLOPE_SLX_THRESHOLD_MS = -0.1
DOWNSLOPE_STABLE_LAPSE_K_M = -0.006
DOWNSLOPE_SLOPE_THRESHOLD = 0.01
MAX_PROFILE_HEIGHT_AGL_M = 3000.0
MIN_INVERSION_DEPTH_M = 100.0


@dataclass(frozen=True)
class DownslopeResult:
    gust_kmh: np.ndarray
    mask: np.ndarray
    slope_index_ms: np.ndarray
    lapse_rate_k_m: np.ndarray


@dataclass(frozen=True)
class PcgeResult:
    gust_kmh: np.ndarray
    potential_kmh: np.ndarray
    trigger: np.ndarray


def ramp(data: np.ndarray, low: float, high: float) -> np.ndarray:
    if high == low:
        raise ValueError("ramp bounds must differ")
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def geometric_vertical_velocity_ms(
    omega_pa_s: np.ndarray,
    temp_k: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    valid = np.isfinite(omega_pa_s) & np.isfinite(temp_k) & (temp_k > 150.0)
    density_kg_m3 = (pressure_hpa * 100.0) / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * np.where(valid, temp_k, np.nan)
    )
    return np.where(valid, -omega_pa_s / (density_kg_m3 * GRAVITY_MS2), np.nan)


def nan_uniform_filter(data: np.ndarray, size: int) -> np.ndarray:
    valid = np.isfinite(data)
    numerator = uniform_filter(np.where(valid, data, 0.0), size=size, mode="nearest")
    denominator = uniform_filter(valid.astype(np.float32), size=size, mode="nearest")
    result = np.full(data.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def interpolate_profile_to_height(
    heights_agl_m: np.ndarray,
    values: np.ndarray,
    target_agl_m: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a vertically ordered profile to a 2-D target height."""
    if heights_agl_m.shape != values.shape or heights_agl_m.ndim != 3:
        raise ValueError("profile heights and values must have matching (level, y, x) shapes")
    if target_agl_m.shape != heights_agl_m.shape[1:]:
        raise ValueError("target height must match the profile horizontal shape")

    result = np.full(target_agl_m.shape, np.nan, dtype=np.float32)
    previous_height = heights_agl_m[0]
    previous_value = values[0]
    for level in range(1, heights_agl_m.shape[0]):
        height = heights_agl_m[level]
        value = values[level]
        layer_depth = height - previous_height
        valid = (
            ~np.isfinite(result)
            & np.isfinite(previous_height)
            & np.isfinite(previous_value)
            & np.isfinite(height)
            & np.isfinite(value)
            & (layer_depth > 1.0)
            & (target_agl_m >= previous_height)
            & (target_agl_m <= height)
        )
        weight = np.zeros(target_agl_m.shape, dtype=np.float32)
        np.divide(
            target_agl_m - previous_height,
            layer_depth,
            out=weight,
            where=valid,
        )
        result[valid] = (previous_value + weight * (value - previous_value))[valid]

        usable = np.isfinite(height) & np.isfinite(value) & (height > previous_height)
        previous_height = np.where(usable, height, previous_height)
        previous_value = np.where(usable, value, previous_value)

    above_top = ~np.isfinite(result) & np.isfinite(previous_value) & (target_agl_m >= previous_height)
    result[above_top] = previous_value[above_top]
    return result


def inversion_metrics(
    surface_temp_k: np.ndarray,
    profile_temp_k: np.ndarray,
    profile_height_agl_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lapse rate below the deepest inversion top and inversion presence."""
    if profile_temp_k.shape != profile_height_agl_m.shape or profile_temp_k.ndim != 3:
        raise ValueError("temperature and height profiles must have matching (level, y, x) shapes")

    previous_height = np.full(surface_temp_k.shape, 2.0, dtype=np.float32)
    previous_temp = surface_temp_k.astype(np.float32, copy=False)
    current_depth = np.zeros(surface_temp_k.shape, dtype=np.float32)
    best_depth = np.zeros(surface_temp_k.shape, dtype=np.float32)
    best_top_height = np.full(surface_temp_k.shape, np.nan, dtype=np.float32)
    best_top_temp = np.full(surface_temp_k.shape, np.nan, dtype=np.float32)

    for level in range(profile_temp_k.shape[0]):
        height = profile_height_agl_m[level]
        temp = profile_temp_k[level]
        layer_depth = height - previous_height
        valid = (
            np.isfinite(height)
            & np.isfinite(temp)
            & np.isfinite(previous_temp)
            & (layer_depth >= 20.0)
            & (height <= MAX_PROFILE_HEIGHT_AGL_M)
        )
        inversion = valid & (temp > previous_temp)
        current_depth = np.where(inversion, current_depth + layer_depth, 0.0)
        improved = inversion & (current_depth > best_depth)
        best_depth = np.where(improved, current_depth, best_depth)
        best_top_height = np.where(improved, height, best_top_height)
        best_top_temp = np.where(improved, temp, best_top_temp)

        previous_height = np.where(valid, height, previous_height)
        previous_temp = np.where(valid, temp, previous_temp)

    inversion_present = best_depth >= MIN_INVERSION_DEPTH_M
    lapse_rate = np.where(
        inversion_present & (best_top_height > 2.0),
        (best_top_temp - surface_temp_k) / (best_top_height - 2.0),
        np.nan,
    )
    return lapse_rate.astype(np.float32), inversion_present


def terrain_peak_mask(terrain_m: np.ndarray, grid_spacing_m: float) -> np.ndarray:
    """Apply ECCC's Hessian test for a local topographic maximum."""
    gradient_y, gradient_x = np.gradient(terrain_m, grid_spacing_m, grid_spacing_m)
    dyy = np.gradient(gradient_y, grid_spacing_m, axis=0)
    dxx = np.gradient(gradient_x, grid_spacing_m, axis=1)
    dxy = np.gradient(gradient_x, grid_spacing_m, axis=0)
    determinant = dxx * dyy - dxy * dxy
    return np.isfinite(determinant) & (determinant > 0.0) & (dxx < 0.0) & (dyy < 0.0)


def downslope_adjusted_gust(
    regular_gust_kmh: np.ndarray,
    gust_max_kmh: np.ndarray,
    terrain_m: np.ndarray,
    ridge_u_ms: np.ndarray,
    ridge_v_ms: np.ndarray,
    surface_temp_k: np.ndarray,
    profile_temp_k: np.ndarray,
    profile_height_agl_m: np.ndarray,
    grid_spacing_m: float,
) -> DownslopeResult:
    """Apply the four-mask ECCC WEonG downslope gust formulation."""
    radius_points = max(1, int(np.ceil(DOWNSLOPE_RADIUS_KM * 1000.0 / grid_spacing_m)))
    filter_size = 2 * radius_points + 1

    gradient_y, gradient_x = np.gradient(terrain_m, grid_spacing_m, grid_spacing_m)
    slope = np.hypot(gradient_x, gradient_y)
    slope_index = ridge_u_ms * gradient_x + ridge_v_ms * gradient_y
    cross_barrier_mask = slope_index < DOWNSLOPE_SLX_THRESHOLD_MS
    slope_mask = slope >= DOWNSLOPE_SLOPE_THRESHOLD

    lapse_rate, inversion_present = inversion_metrics(
        surface_temp_k,
        profile_temp_k,
        profile_height_agl_m,
    )
    lapse_rate_sua = nan_uniform_filter(lapse_rate, filter_size)
    stable_mask = lapse_rate_sua >= DOWNSLOPE_STABLE_LAPSE_K_M

    peak_mask = terrain_peak_mask(terrain_m, grid_spacing_m)
    inversion_over_peak = peak_mask & inversion_present
    inversion_near_peak = maximum_filter(
        inversion_over_peak.astype(np.uint8),
        size=filter_size,
        mode="nearest",
    ).astype(bool)

    mask = cross_barrier_mask & slope_mask & stable_mask & inversion_near_peak
    ratio = gust_max_kmh / np.maximum(regular_gust_kmh, 1.0)
    candidate = np.where(
        ratio < 1.5,
        gust_max_kmh,
        0.5 * (regular_gust_kmh + gust_max_kmh),
    )
    adjusted = np.where(mask, np.maximum(regular_gust_kmh, candidate), regular_gust_kmh)
    return DownslopeResult(
        gust_kmh=adjusted.astype(np.float32),
        mask=mask,
        slope_index_ms=slope_index.astype(np.float32),
        lapse_rate_k_m=lapse_rate_sua.astype(np.float32),
    )


def pcge_gust(
    regular_gust_kmh: np.ndarray,
    dcape_j_kg: np.ndarray,
    lifted_index_k: np.ndarray,
    cape_j_kg: np.ndarray,
    pbl_height_m: np.ndarray,
    wind850_ms: np.ndarray,
    wind700_ms: np.ndarray,
    precip_rate_mm_h: np.ndarray,
    omega700_pa_s: np.ndarray,
    temp700_k: np.ndarray,
) -> PcgeResult:
    """Return a continuously triggered potential convective gust estimate."""
    regular_ms = regular_gust_kmh / 3.6
    pbl_factor = ramp(pbl_height_m, 300.0, 1800.0)

    momentum850_ms = (0.70 + 0.15 * pbl_factor) * wind850_ms
    momentum700_ms = (0.45 + 0.20 * pbl_factor) * wind700_ms
    momentum_ms = np.fmax.reduce([regular_ms, momentum850_ms, momentum700_ms])

    downdraft_ms = 0.35 * np.sqrt(np.maximum(0.0, 2.0 * dcape_j_kg))
    potential_ms = np.hypot(momentum_ms, downdraft_ms)
    potential_ms = np.clip(potential_ms, regular_ms, 55.0)

    li_factor = ramp(1.0 - lifted_index_k, 0.0, 5.0)
    cape_factor = ramp(cape_j_kg, 50.0, 800.0)
    environment = li_factor * (0.85 + 0.15 * cape_factor)

    upward700_ms = geometric_vertical_velocity_ms(omega700_pa_s, temp700_k, 700.0)
    # Precipitation can identify an active downdraft directly. A dry-convection
    # trigger requires much stronger 700 hPa ascent than ordinary synoptic lift.
    ascent_factor = ramp(upward700_ms, 0.15, 0.75)
    precip_factor = ramp(precip_rate_mm_h, 0.25, 2.50)
    active_convection = np.fmax(ascent_factor, precip_factor)
    dcape_factor = ramp(dcape_j_kg, 150.0, 750.0)
    trigger = np.clip(environment * active_convection * dcape_factor, 0.0, 1.0)

    potential_kmh = 3.6 * potential_ms
    gust_kmh = regular_gust_kmh + trigger * np.maximum(0.0, potential_kmh - regular_gust_kmh)
    return PcgeResult(
        gust_kmh=gust_kmh.astype(np.float32),
        potential_kmh=potential_kmh.astype(np.float32),
        trigger=trigger.astype(np.float32),
    )
