from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import automate_lpi_verification as verification
import lightning_ml_archive as archive
import make_hrdps_west_convective as hrdps


class LpiVerificationWindowTest(unittest.TestCase):
    def run_info(self, stamp: str) -> hrdps.RunInfo:
        init = hrdps.parse_stamp(stamp)
        return hrdps.RunInfo(cycle=f"{init:%H}", stamp=stamp, init_time=init)

    def test_12z_run_uses_f003_through_f024(self) -> None:
        window = verification.first_full_12z_window(self.run_info("20260708T12Z"))
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window.start, dt.datetime(2026, 7, 8, 12, tzinfo=dt.timezone.utc))
        self.assertEqual(window.end, dt.datetime(2026, 7, 9, 12, tzinfo=dt.timezone.utc))
        self.assertEqual(window.included_hours, (3, 6, 9, 12, 15, 18, 21, 24))
        self.assertEqual(window.end_fhour, 24)

    def test_18z_run_uses_next_days_window(self) -> None:
        window = verification.first_full_12z_window(self.run_info("20260708T18Z"))
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window.start, dt.datetime(2026, 7, 9, 12, tzinfo=dt.timezone.utc))
        self.assertEqual(window.end, dt.datetime(2026, 7, 10, 12, tzinfo=dt.timezone.utc))
        self.assertEqual(window.included_hours, (21, 24, 27, 30, 33, 36, 39, 42))
        self.assertEqual(window.end_fhour, 42)

    def test_daily_window_requires_144_ten_minute_observations(self) -> None:
        start = dt.datetime(2026, 7, 8, 12, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
        timestamps = verification.expected_obs_times(start, end)
        self.assertEqual(len(timestamps), 144)
        self.assertEqual(timestamps[0], start + dt.timedelta(minutes=10))
        self.assertEqual(timestamps[-1], end)

    def test_expected_filename_uses_window_end_forecast_hour(self) -> None:
        self.assertEqual(
            verification.expected_verification_name("west", "20260708T12Z"),
            "hrdps_west_lightning_verif_20260708T12Z_f024.png",
        )
        self.assertEqual(
            verification.expected_verification_name("continental", "20260708T18Z"),
            "hrdps_continental_lightning_verif_20260708T18Z_f042.png",
        )

    def test_observations_use_unfilled_threshold_contours(self) -> None:
        class Collection:
            def set_path_effects(self, effects) -> None:
                self.effects = effects

        class Contours:
            def __init__(self, count: int) -> None:
                self.collections = [Collection() for _ in range(count)]

        class Axes:
            def contour(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return Contours(len(kwargs["levels"]))

        axes = Axes()
        obs = verification.ObsGrid(
            lat=np.array([[49.0, 49.0], [50.0, 50.0]], dtype=np.float32),
            lon=np.array([[-124.0, -123.0], [-124.0, -123.0]], dtype=np.float32),
            flash_km2=np.array([[0.0, 0.1], [0.75, 3.0]], dtype=np.float32),
        )
        contours = verification.add_observed_lightning_contours(axes, obs)
        self.assertIsNotNone(contours)
        self.assertEqual(
            axes.kwargs["levels"],
            [
                verification.OBS_LOW_FLASH_KM2,
                verification.OBS_MED_FLASH_KM2,
                verification.OBS_HIGH_FLASH_KM2,
            ],
        )
        self.assertEqual(axes.kwargs["colors"], list(verification.OBS_CONTOUR_COLORS))
        self.assertFalse(hasattr(axes, "scatter"))

    def test_tuning_readiness_requires_three_weeks_of_well_covered_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            start = dt.datetime(2026, 7, 8, 3, tzinfo=dt.timezone.utc)
            for index in range(21 * 8 + 1):
                timestamp = start + dt.timedelta(hours=3 * index)
                path = archive.observation_aggregate_path(root, timestamp)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                path.with_suffix(".json").write_text(
                    json.dumps({"nonzero_cells": 1 if index < 12 else 0}) + "\n"
                )

            readiness = verification.lpi_tuning_readiness(root)
            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["archive_span_days"], 21.0)
            self.assertEqual(readiness["coverage_fraction"], 1.0)
            self.assertEqual(readiness["active_observation_blocks"], 12)


if __name__ == "__main__":
    unittest.main()
