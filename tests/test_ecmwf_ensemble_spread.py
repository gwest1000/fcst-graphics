from __future__ import annotations

import unittest
from pathlib import Path

import ecmwf_ensemble_stats_data as stats_data
import make_ecmwf_ensemble_spread as spread
import publish_hrdps_west as publisher
import r2_publish


class EcmwfEnsembleSpreadTests(unittest.TestCase):
    def test_reference_spread_scale_has_expected_boundaries_and_colors(self):
        self.assertEqual(spread.SPREAD_LEVELS_KM[0], 0.005)
        self.assertEqual(spread.SPREAD_LEVELS_KM[-1], 0.30)
        self.assertEqual(len(spread.SPREAD_COLORS), len(spread.SPREAD_LEVELS_KM) - 1)
        self.assertEqual(spread.SPREAD_COLORS[0], "#c9ffc9")
        self.assertEqual(spread.SPREAD_COLORS[5], "#7878ff")
        self.assertEqual(spread.SPREAD_COLORS[15], "#ff4a4a")
        self.assertEqual(spread.SPREAD_COLORS[25], "#b0b0b0")

    def test_schedule_reaches_f252(self):
        self.assertEqual(spread.FORECAST_HOURS[-1], 252)
        self.assertIn(144, spread.FORECAST_HOURS)
        self.assertNotIn(147, spread.FORECAST_HOURS)
        self.assertIn(150, spread.FORECAST_HOURS)

    def test_archive_paths_keep_mean_and_spread_separate(self):
        paths = stats_data.archive_paths("20260817", "00", Path("/tmp/ecmwf-stats-test"))
        self.assertNotEqual(paths.mean, paths.spread)
        self.assertEqual(paths.mean.name, "gh500_ensemble_mean.grib2")
        self.assertEqual(paths.spread.name, "gh500_ensemble_spread.grib2")

    def test_product_is_published_as_its_own_model_group(self):
        key = "ecmwf_ensemble_spread_500"
        self.assertIn(key, publisher.PRODUCTS_BY_MODEL["ecmwf_ensemble"])
        self.assertIn(key, r2_publish.MODEL_PRODUCTS["ecmwf_ensemble"])
        self.assertEqual(publisher.PRODUCTS[key].model, "ECMWF ENS")


if __name__ == "__main__":
    unittest.main()
