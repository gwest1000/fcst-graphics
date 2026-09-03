from __future__ import annotations

import datetime as dt
import unittest

import numpy as np

import analyze_lpi_calibration as calibration


class LpiCalibrationTest(unittest.TestCase):
    def test_daily_window_uses_12z_boundaries(self) -> None:
        self.assertEqual(
            calibration.daily_window_start(dt.datetime(2026, 8, 10, 12, tzinfo=dt.timezone.utc)),
            dt.datetime(2026, 8, 9, 12, tzinfo=dt.timezone.utc),
        )
        self.assertEqual(
            calibration.daily_window_start(dt.datetime(2026, 8, 10, 15, tzinfo=dt.timezone.utc)),
            dt.datetime(2026, 8, 10, 12, tzinfo=dt.timezone.utc),
        )

    def test_pava_is_monotone(self) -> None:
        histogram = calibration.Histogram.empty()
        histogram.count[:4] = [10.0, 10.0, 10.0, 10.0]
        histogram.events[:4] = [1.0, 8.0, 4.0, 9.0]
        probabilities = calibration.pava_probabilities(histogram)
        self.assertTrue(np.all(np.diff(probabilities) >= 0.0))
        self.assertAlmostEqual(probabilities[1], probabilities[2])

    def test_cape_ablation_remains_monotone_in_li(self) -> None:
        formula = next(item for item in calibration.candidate_formulas() if item.name == "ablate_cape")
        fields = {
            "mu_li": np.array([1.0, -2.0, -5.0], dtype=np.float32),
            "cape": np.full(3, 500.0, dtype=np.float32),
            "charge_rh": np.full(3, 80.0, dtype=np.float32),
            "charge_depth": np.full(3, 150.0, dtype=np.float32),
            "mid_rh": np.full(3, 75.0, dtype=np.float32),
            "upward_w": np.full(3, 0.05, dtype=np.float32),
            "precip_3h": np.full(3, 1.5, dtype=np.float32),
            "precip_rate": np.full(3, 0.8, dtype=np.float32),
        }
        potential = calibration.compute_formula(fields, formula)
        self.assertTrue(np.all(np.diff(potential) >= 0.0))

    def test_solar_geometry_changes_between_july_and_august(self) -> None:
        latitude = np.array([50.0], dtype=np.float32)
        july, july_hours = calibration.solar_max_elevation_and_daylight_hours(
            dt.datetime(2026, 7, 15, 12, tzinfo=dt.timezone.utc), latitude
        )
        august, august_hours = calibration.solar_max_elevation_and_daylight_hours(
            dt.datetime(2026, 8, 15, 12, tzinfo=dt.timezone.utc), latitude
        )
        self.assertGreater(july[0], august[0])
        self.assertGreater(july_hours[0], august_hours[0])

    def test_bootstrap_difference_is_zero_for_identical_forecasts(self) -> None:
        first = calibration.Histogram.empty()
        second = calibration.Histogram.empty()
        first.count[:2] = [10.0, 10.0]
        first.events[:2] = [2.0, 8.0]
        second.count[:2] = [8.0, 12.0]
        second.events[:2] = [1.0, 9.0]
        days = {
            dt.datetime(2026, 8, 1, 12, tzinfo=dt.timezone.utc): first,
            dt.datetime(2026, 8, 2, 12, tzinfo=dt.timezone.utc): second,
        }
        probability = np.linspace(0.01, 0.99, len(calibration.SCORE_CENTERS))
        result = calibration.bootstrap_difference(days, days, probability, probability, seed=4, draws=20)
        self.assertAlmostEqual(result["brier_difference_low"], 0.0)
        self.assertAlmostEqual(result["brier_difference_high"], 0.0)
        self.assertAlmostEqual(result["auc_difference_low"], 0.0)
        self.assertAlmostEqual(result["auc_difference_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
