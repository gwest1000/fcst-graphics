#!/usr/bin/env python3
"""Peak-burn daily summaries and cache helpers for hourly FWI2025 fields."""

from __future__ import annotations

import datetime as dt
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fwi2025


FIRE_DAY_RESET_HOUR = 5
PEAK_GUIDANCE_MIN_LOCAL_HOUR = 17
PEAK_CACHE_VERSION = 1


@dataclass(frozen=True)
class HourlyPeakRecord:
    valid_utc: dt.datetime
    local_hour: int
    wind_kmh: np.ndarray
    ffmc: np.ndarray
    fwi: np.ndarray
    bui: np.ndarray


@dataclass(frozen=True)
class PeakBurnDay:
    fire_date: dt.date
    fwi: np.ndarray
    bui: np.ndarray
    peak_local_hour: np.ndarray
    complete: bool
    hour_count: int
    last_valid_utc: dt.datetime
    cutoff_reached: bool


@dataclass(frozen=True)
class PeakDangerGrid:
    fire_date: dt.date
    source_run_stamp: str
    source_init_utc: dt.datetime
    fwi: np.ndarray
    bui: np.ndarray
    danger: np.ndarray
    peak_local_hour: np.ndarray
    complete: bool
    hour_count: int
    path: Path


def fire_date_for_valid(valid_utc: dt.datetime, local_tz: dt.tzinfo) -> dt.date:
    local = valid_utc.astimezone(local_tz)
    if local.hour < FIRE_DAY_RESET_HOUR:
        return local.date() - dt.timedelta(days=1)
    return local.date()


class PeakBurnAccumulator:
    """Stream hourly grids into NRCan-style daily peak-burn summaries.

    The centered five-point binomial wind smoother is evaluated without
    retaining an entire day of full-resolution model grids.
    """

    def __init__(self, local_tz: dt.tzinfo):
        self.local_tz = local_tz
        self._fire_date: dt.date | None = None
        self._started_at_reset = False
        self._hour_count = 0
        self._records: deque[HourlyPeakRecord] = deque(maxlen=5)
        self._best_score: np.ndarray | None = None
        self._best_fwi: np.ndarray | None = None
        self._best_bui: np.ndarray | None = None
        self._best_hour: np.ndarray | None = None
        self._max_ffmc: np.ndarray | None = None
        self._fallback_fwi: np.ndarray | None = None
        self._fallback_bui: np.ndarray | None = None
        self._fallback_hour: int | None = None
        self._last_valid_utc: dt.datetime | None = None

    @staticmethod
    def _compact(data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float32).copy()

    def _reset(self, fire_date: dt.date, local_hour: int) -> None:
        self._fire_date = fire_date
        self._started_at_reset = local_hour == FIRE_DAY_RESET_HOUR
        self._hour_count = 0
        self._records.clear()
        self._best_score = None
        self._best_fwi = None
        self._best_bui = None
        self._best_hour = None
        self._max_ffmc = None
        self._fallback_fwi = None
        self._fallback_bui = None
        self._fallback_hour = None
        self._last_valid_utc = None

    def _evaluate(self, record: HourlyPeakRecord, smoothed_wind: np.ndarray) -> None:
        score = fwi2025.initial_spread_index(record.ffmc, smoothed_wind)
        valid = np.isfinite(score) & np.isfinite(record.fwi) & np.isfinite(record.bui)
        if self._best_score is None:
            self._best_score = np.where(valid, score, -np.inf).astype(np.float32)
            self._best_fwi = np.where(valid, record.fwi, np.nan).astype(np.float32)
            self._best_bui = np.where(valid, record.bui, np.nan).astype(np.float32)
            self._best_hour = np.where(valid, record.local_hour, -1).astype(np.int8)
            return
        assert self._best_fwi is not None and self._best_bui is not None and self._best_hour is not None
        update = valid & (score > self._best_score)
        self._best_score[update] = score[update]
        self._best_fwi[update] = record.fwi[update]
        self._best_bui[update] = record.bui[update]
        self._best_hour[update] = record.local_hour

    def _append_record(self, record: HourlyPeakRecord) -> None:
        self._records.append(record)
        self._hour_count += 1
        self._last_valid_utc = record.valid_utc
        if self._max_ffmc is None:
            self._max_ffmc = record.ffmc.copy()
        else:
            np.fmax(self._max_ffmc, record.ffmc, out=self._max_ffmc)
        if record.local_hour == FIRE_DAY_RESET_HOUR + 12:
            self._fallback_fwi = record.fwi.copy()
            self._fallback_bui = record.bui.copy()
            self._fallback_hour = record.local_hour

        if self._hour_count == 1:
            self._evaluate(record, record.wind_kmh)
        elif self._hour_count == 3:
            first, centre, third = tuple(self._records)
            smooth = 0.25 * first.wind_kmh + 0.5 * centre.wind_kmh + 0.25 * third.wind_kmh
            self._evaluate(centre, smooth)
        if self._hour_count >= 5:
            records = tuple(self._records)
            smooth = (
                records[0].wind_kmh
                + 4.0 * records[1].wind_kmh
                + 6.0 * records[2].wind_kmh
                + 4.0 * records[3].wind_kmh
                + records[4].wind_kmh
            ) / 16.0
            self._evaluate(records[2], smooth)

    def _finalize(self, ended_at_boundary: bool) -> PeakBurnDay | None:
        if self._fire_date is None or not self._records or self._best_fwi is None:
            return None
        records = tuple(self._records)
        if len(records) >= 3:
            smooth = 0.25 * records[-3].wind_kmh + 0.5 * records[-2].wind_kmh + 0.25 * records[-1].wind_kmh
            self._evaluate(records[-2], smooth)
        self._evaluate(records[-1], records[-1].wind_kmh)

        assert self._best_bui is not None and self._best_hour is not None and self._max_ffmc is not None
        fwi = self._best_fwi.copy()
        bui = self._best_bui.copy()
        peak_hour = self._best_hour.copy()
        if self._fallback_fwi is not None and self._fallback_bui is not None and self._fallback_hour is not None:
            fallback = self._max_ffmc < 85.0
            fwi[fallback] = self._fallback_fwi[fallback]
            bui[fallback] = self._fallback_bui[fallback]
            peak_hour[fallback] = self._fallback_hour

        complete = self._started_at_reset and ended_at_boundary and 23 <= self._hour_count <= 25
        assert self._last_valid_utc is not None
        coverage_end = self._last_valid_utc.astimezone(self.local_tz)
        cutoff_reached = coverage_end.date() > self._fire_date or (
            coverage_end.date() == self._fire_date
            and coverage_end.hour >= PEAK_GUIDANCE_MIN_LOCAL_HOUR
        )
        return PeakBurnDay(
            self._fire_date,
            fwi,
            bui,
            peak_hour,
            complete,
            self._hour_count,
            self._last_valid_utc,
            cutoff_reached,
        )

    def push(
        self,
        valid_utc: dt.datetime,
        wind_kmh: np.ndarray,
        ffmc: np.ndarray,
        fwi: np.ndarray,
        bui: np.ndarray,
    ) -> list[PeakBurnDay]:
        local = valid_utc.astimezone(self.local_tz)
        fire_date = fire_date_for_valid(valid_utc, self.local_tz)
        completed: list[PeakBurnDay] = []
        if self._fire_date is None:
            self._reset(fire_date, local.hour)
        elif fire_date != self._fire_date:
            day = self._finalize(ended_at_boundary=local.hour == FIRE_DAY_RESET_HOUR)
            if day is not None:
                completed.append(day)
            self._reset(fire_date, local.hour)

        self._append_record(
            HourlyPeakRecord(
                valid_utc=valid_utc,
                local_hour=local.hour,
                wind_kmh=self._compact(wind_kmh),
                ffmc=self._compact(ffmc),
                fwi=self._compact(fwi),
                bui=self._compact(bui),
            )
        )
        return completed

    def finish(self) -> PeakBurnDay | None:
        return self._finalize(ended_at_boundary=False)


def peak_cache_directory(cache_dir: Path, model_key: str, fire_date: dt.date) -> Path:
    return cache_dir / "peak_daily" / model_key / f"{fire_date:%Y%m%d}"


def peak_cache_path(cache_dir: Path, model_key: str, fire_date: dt.date, run_stamp: str) -> Path:
    return peak_cache_directory(cache_dir, model_key, fire_date) / f"{run_stamp}.npz"


def reaches_peak_guidance_cutoff(
    day: PeakBurnDay,
    local_tz: dt.tzinfo,
    cutoff_hour: int = PEAK_GUIDANCE_MIN_LOCAL_HOUR,
) -> bool:
    """Return whether a partial fire day includes at least the local cutoff hour."""
    if day.complete:
        return True
    coverage_end = day.last_valid_utc.astimezone(local_tz)
    return coverage_end.date() > day.fire_date or (
        coverage_end.date() == day.fire_date and coverage_end.hour >= cutoff_hour
    )


def peak_run_marker_path(cache_dir: Path, model_key: str, run_stamp: str) -> Path:
    return cache_dir / "peak_daily" / model_key / "runs" / f"{run_stamp}.complete"


def save_peak_danger_grid(
    cache_dir: Path,
    model_key: str,
    run_stamp: str,
    run_init_utc: dt.datetime,
    day: PeakBurnDay,
    danger: np.ndarray,
    allow_partial: bool = False,
) -> Path:
    if not day.complete and not allow_partial:
        raise ValueError("Only complete fire days may be saved as peak-daily guidance.")
    cutoff_reached = day.complete or day.cutoff_reached
    if not day.complete and not cutoff_reached:
        raise ValueError(
            f"Partial peak-daily guidance must extend through "
            f"{PEAK_GUIDANCE_MIN_LOCAL_HOUR:02d}:00 local."
        )
    path = peak_cache_path(cache_dir, model_key, day.fire_date, run_stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    danger_u8 = np.where(np.isfinite(danger), danger, 0).astype(np.uint8)
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray([PEAK_CACHE_VERSION], dtype=np.int16),
                model_key=np.asarray(model_key),
                run_stamp=np.asarray(run_stamp),
                run_init_utc=np.asarray(run_init_utc.isoformat()),
                fire_date=np.asarray(day.fire_date.isoformat()),
                hour_count=np.asarray([day.hour_count], dtype=np.int8),
                complete=np.asarray([day.complete], dtype=np.bool_),
                cutoff_reached=np.asarray([cutoff_reached], dtype=np.bool_),
                coverage_end_utc=np.asarray(day.last_valid_utc.isoformat()),
                fwi=day.fwi.astype(np.float32, copy=False),
                bui=day.bui.astype(np.float32, copy=False),
                danger=danger_u8,
                peak_local_hour=day.peak_local_hour.astype(np.int8, copy=False),
            )
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def mark_peak_run_complete(cache_dir: Path, model_key: str, run_stamp: str, fire_dates: list[dt.date]) -> Path:
    path = peak_run_marker_path(cache_dir, model_key, run_stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(day.isoformat() for day in sorted(fire_dates)) + "\n")
    return path


def load_peak_danger_grid(
    cache_dir: Path,
    model_key: str,
    fire_date: dt.date,
    as_of_utc: dt.datetime,
    expected_shape: tuple[int, int],
) -> PeakDangerGrid | None:
    directory = peak_cache_directory(cache_dir, model_key, fire_date)
    if not directory.exists():
        return None
    candidates: list[PeakDangerGrid] = []
    for path in directory.glob("*.npz"):
        try:
            with np.load(path) as data:
                if int(data["version"][0]) != PEAK_CACHE_VERSION:
                    continue
                source_init = dt.datetime.fromisoformat(str(data["run_init_utc"]))
                if source_init.tzinfo is None:
                    source_init = source_init.replace(tzinfo=dt.timezone.utc)
                if source_init > as_of_utc:
                    continue
                fwi = data["fwi"].astype(np.float32)
                bui = data["bui"].astype(np.float32)
                danger_u8 = data["danger"].astype(np.uint8)
                peak_hour = data["peak_local_hour"].astype(np.int8)
                complete = bool(data["complete"][0]) if "complete" in data else True
                hour_count = int(data["hour_count"][0])
                cutoff_reached = (
                    bool(data["cutoff_reached"][0])
                    if "cutoff_reached" in data
                    else hour_count >= PEAK_GUIDANCE_MIN_LOCAL_HOUR - FIRE_DAY_RESET_HOUR + 1
                )
                if not complete and not cutoff_reached:
                    continue
                if any(array.shape != expected_shape for array in (fwi, bui, danger_u8, peak_hour)):
                    continue
                candidates.append(
                    PeakDangerGrid(
                        fire_date=dt.date.fromisoformat(str(data["fire_date"])),
                        source_run_stamp=str(data["run_stamp"]),
                        source_init_utc=source_init,
                        fwi=fwi,
                        bui=bui,
                        danger=np.where(danger_u8 > 0, danger_u8, np.nan).astype(np.float32),
                        peak_local_hour=peak_hour,
                        complete=complete,
                        hour_count=hour_count,
                        path=path,
                    )
                )
        except (OSError, KeyError, ValueError):
            continue
    return max(candidates, key=lambda item: item.source_init_utc, default=None)


def load_peak_danger_for_display(
    cache_dir: Path,
    model_key: str,
    fire_date: dt.date,
    as_of_utc: dt.datetime,
    expected_shape: tuple[int, int],
    max_previous_days: int = 1,
) -> PeakDangerGrid | None:
    """Load a qualified day, retaining recent guidance when the new day is under-covered."""
    for days_back in range(max(0, int(max_previous_days)) + 1):
        source = load_peak_danger_grid(
            cache_dir,
            model_key,
            fire_date - dt.timedelta(days=days_back),
            as_of_utc,
            expected_shape,
        )
        if source is not None:
            return source
    return None


def prune_peak_cache(cache_dir: Path, model_key: str, keep_days: int = 10) -> None:
    root = cache_dir / "peak_daily" / model_key
    if not root.exists():
        return
    cutoff = dt.date.today() - dt.timedelta(days=max(1, keep_days))
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name == "runs":
            continue
        try:
            fire_date = dt.datetime.strptime(directory.name, "%Y%m%d").date()
        except ValueError:
            continue
        if fire_date < cutoff:
            __import__("shutil").rmtree(directory)
