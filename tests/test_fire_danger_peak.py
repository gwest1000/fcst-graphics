from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np

import fire_danger_peak as peak


LOCAL_TZ = ZoneInfo("America/Vancouver")


class PeakBurnAccumulatorTest(unittest.TestCase):
    @staticmethod
    def complete_day(ffmc_value: float = 90.0) -> peak.PeakBurnDay:
        accumulator = peak.PeakBurnAccumulator(LOCAL_TZ)
        start = dt.datetime(2026, 7, 14, 12, tzinfo=dt.timezone.utc)
        winds = np.full(25, 5.0, dtype=np.float32)
        winds[10] = 60.0
        completed: list[peak.PeakBurnDay] = []
        for index in range(25):
            completed.extend(
                accumulator.push(
                    start + dt.timedelta(hours=index),
                    np.array([[winds[index]]], dtype=np.float32),
                    np.array([[ffmc_value]], dtype=np.float32),
                    np.array([[100.0 + index]], dtype=np.float32),
                    np.array([[40.0 + index]], dtype=np.float32),
                )
            )
        if len(completed) != 1:
            raise AssertionError(f"Expected one completed fire day, found {len(completed)}")
        return completed[0]

    def test_five_point_smoothed_wind_selects_unsmoothed_indices(self) -> None:
        day = self.complete_day()
        self.assertTrue(day.complete)
        self.assertEqual(day.hour_count, 24)
        self.assertEqual(day.fire_date, dt.date(2026, 7, 14))
        self.assertEqual(int(day.peak_local_hour[0, 0]), 15)
        self.assertEqual(float(day.fwi[0, 0]), 110.0)
        self.assertEqual(float(day.bui[0, 0]), 50.0)

    def test_low_ffmc_day_falls_back_to_1700_local(self) -> None:
        day = self.complete_day(ffmc_value=84.0)
        self.assertEqual(int(day.peak_local_hour[0, 0]), 17)
        self.assertEqual(float(day.fwi[0, 0]), 112.0)
        self.assertEqual(float(day.bui[0, 0]), 52.0)

    def test_cache_uses_latest_nonfuture_compatible_run(self) -> None:
        day = self.complete_day()
        danger = np.array([[4.0]], dtype=np.float32)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_init = dt.datetime(2026, 7, 14, 0, tzinfo=dt.timezone.utc)
            second_init = dt.datetime(2026, 7, 14, 12, tzinfo=dt.timezone.utc)
            peak.save_peak_danger_grid(root, "continental", "20260714T00Z", first_init, day, danger)
            peak.save_peak_danger_grid(root, "continental", "20260714T12Z", second_init, day, danger + 1)

            loaded = peak.load_peak_danger_grid(
                root,
                "continental",
                day.fire_date,
                dt.datetime(2026, 7, 14, 6, tzinfo=dt.timezone.utc),
                (1, 1),
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.source_run_stamp, "20260714T00Z")
            self.assertEqual(float(loaded.danger[0, 0]), 4.0)
            self.assertTrue(loaded.complete)
            self.assertEqual(loaded.hour_count, 24)

            self.assertIsNone(
                peak.load_peak_danger_grid(
                    root,
                    "continental",
                    day.fire_date,
                    dt.datetime(2026, 7, 15, tzinfo=dt.timezone.utc),
                    (2, 2),
                )
            )

    def test_terminal_partial_day_can_supply_best_available_peak(self) -> None:
        accumulator = peak.PeakBurnAccumulator(LOCAL_TZ)
        start = dt.datetime(2026, 7, 18, 12, tzinfo=dt.timezone.utc)
        for index in range(13):
            accumulator.push(
                start + dt.timedelta(hours=index),
                np.array([[15.0 + index]], dtype=np.float32),
                np.array([[90.0]], dtype=np.float32),
                np.array([[20.0 + index]], dtype=np.float32),
                np.array([[40.0]], dtype=np.float32),
            )
        day = accumulator.finish()
        self.assertIsNotNone(day)
        assert day is not None
        self.assertFalse(day.complete)
        self.assertEqual(day.hour_count, 13)
        self.assertEqual(day.last_valid_utc.astimezone(LOCAL_TZ).hour, 17)
        self.assertTrue(peak.reaches_peak_guidance_cutoff(day, LOCAL_TZ))

        danger = np.array([[3.0]], dtype=np.float32)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                peak.save_peak_danger_grid(root, "continental", "20260718T00Z", start, day, danger)
            peak.save_peak_danger_grid(
                root,
                "continental",
                "20260718T00Z",
                start,
                day,
                danger,
                allow_partial=True,
            )
            loaded = peak.load_peak_danger_grid(root, "continental", day.fire_date, start, (1, 1))
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertFalse(loaded.complete)
            self.assertEqual(loaded.hour_count, 13)
            self.assertEqual(float(loaded.danger[0, 0]), 3.0)

    def test_terminal_partial_day_before_1700_is_rejected(self) -> None:
        accumulator = peak.PeakBurnAccumulator(LOCAL_TZ)
        start = dt.datetime(2026, 7, 18, 12, tzinfo=dt.timezone.utc)
        for index in range(12):
            accumulator.push(
                start + dt.timedelta(hours=index),
                np.array([[15.0]], dtype=np.float32),
                np.array([[90.0]], dtype=np.float32),
                np.array([[20.0]], dtype=np.float32),
                np.array([[40.0]], dtype=np.float32),
            )
        day = accumulator.finish()
        self.assertIsNotNone(day)
        assert day is not None
        self.assertEqual(day.last_valid_utc.astimezone(LOCAL_TZ).hour, 16)
        self.assertFalse(peak.reaches_peak_guidance_cutoff(day, LOCAL_TZ))

        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "17:00 local"):
                peak.save_peak_danger_grid(
                    Path(tmpdir),
                    "continental",
                    "20260718T00Z",
                    start,
                    day,
                    np.array([[3.0]], dtype=np.float32),
                    allow_partial=True,
                )

    def test_display_loader_retains_previous_qualified_day(self) -> None:
        day = self.complete_day()
        danger = np.array([[4.0]], dtype=np.float32)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            peak.save_peak_danger_grid(
                root,
                "continental",
                "20260714T12Z",
                dt.datetime(2026, 7, 14, 12, tzinfo=dt.timezone.utc),
                day,
                danger,
            )
            next_date = day.fire_date + dt.timedelta(days=1)
            stale_path = peak.peak_cache_path(root, "continental", next_date, "20260715T12Z")
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                stale_path,
                version=np.asarray([peak.PEAK_CACHE_VERSION], dtype=np.int16),
                model_key=np.asarray("continental"),
                run_stamp=np.asarray("20260715T12Z"),
                run_init_utc=np.asarray("2026-07-15T12:00:00+00:00"),
                fire_date=np.asarray(next_date.isoformat()),
                hour_count=np.asarray([1], dtype=np.int8),
                complete=np.asarray([False], dtype=np.bool_),
                fwi=np.asarray([[1.0]], dtype=np.float32),
                bui=np.asarray([[1.0]], dtype=np.float32),
                danger=np.asarray([[1]], dtype=np.uint8),
                peak_local_hour=np.asarray([[5]], dtype=np.int8),
            )
            loaded = peak.load_peak_danger_for_display(
                root,
                "continental",
                next_date,
                dt.datetime(2026, 7, 15, 12, tzinfo=dt.timezone.utc),
                (1, 1),
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.fire_date, day.fire_date)
            self.assertEqual(float(loaded.danger[0, 0]), 4.0)


if __name__ == "__main__":
    unittest.main()
