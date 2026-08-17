from __future__ import annotations

import unittest
import datetime as dt
from pathlib import Path
import tempfile

import numpy as np

import automate_ecmwf_ensemble_spread as automation
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

    def test_schedule_reaches_f360(self):
        self.assertEqual(spread.FORECAST_HOURS[-1], 360)
        self.assertEqual(len(spread.FORECAST_HOURS), 85)
        self.assertIn(144, spread.FORECAST_HOURS)
        self.assertNotIn(147, spread.FORECAST_HOURS)
        self.assertIn(150, spread.FORECAST_HOURS)

    def test_domain_is_zoomed_and_anchored_near_eastern_newfoundland(self):
        old_width = spread.REFERENCE_EXTENT[1] - spread.REFERENCE_EXTENT[0]
        old_height = spread.REFERENCE_EXTENT[3] - spread.REFERENCE_EXTENT[2]
        self.assertAlmostEqual((spread.EXTENT[1] - spread.EXTENT[0]) / old_width, 0.85)
        self.assertAlmostEqual((spread.EXTENT[3] - spread.EXTENT[2]) / old_height, 0.85)
        self.assertAlmostEqual(spread.EXTENT[1], -52.0)

    def test_source_crop_covers_projected_corners_across_the_dateline(self):
        west, east, south, north = spread.SOURCE_EXTENT
        self.assertLess(west, -205.0)
        self.assertGreater(east, -10.0)
        self.assertLess(south, 0.0)
        self.assertGreater(north, 80.0)
        wrapped = spread.centered_longitudes(
            np.asarray([150.0, -180.0, -10.0]),
            (spread.EXTENT[0] + spread.EXTENT[1]) / 2.0,
        )
        np.testing.assert_allclose(wrapped, [-210.0, -180.0, -10.0])

    def test_mean_height_smoothing_reduces_grid_scale_noise(self):
        field = np.zeros((9, 9), dtype=float)
        field[4, 4] = 1.0
        smoothed = spread.smooth_mean_height(field)
        self.assertLess(smoothed[4, 4], 1.0)
        self.assertGreater(smoothed[4, 5], 0.0)

    def test_green_blue_boundary_is_emphasized(self):
        self.assertEqual(spread.GREEN_BLUE_BOUNDARY_KM, 0.05)
        self.assertEqual(spread.SPREAD_CONTOUR_LINEWIDTH, 0.48)
        self.assertEqual(spread.HEIGHT_CONTOUR_LINEWIDTH, 1.50)
        self.assertEqual(spread.GREEN_BLUE_BOUNDARY_LINEWIDTH, 1.80)

    def test_archive_paths_keep_mean_and_spread_separate(self):
        paths = stats_data.archive_paths("20260817", "00", Path("/tmp/ecmwf-stats-test"))
        self.assertNotEqual(paths.mean, paths.spread)
        self.assertEqual(paths.mean.name, "gh500_ensemble_mean.grib2")
        self.assertEqual(paths.spread.name, "gh500_ensemble_spread.grib2")

    def test_local_plot_pruning_keeps_only_recent_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expired = root / "20260801T00Z"
            recent = root / "20260816T12Z"
            unrelated = root / "scratch"
            for path in (expired, recent, unrelated):
                path.mkdir()

            removed = automation.prune_local_plots(
                root,
                7,
                now=dt.datetime(2026, 8, 17, 18, tzinfo=dt.timezone.utc),
            )

            self.assertEqual(removed, [expired])
            self.assertFalse(expired.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(unrelated.exists())

    def test_product_is_published_as_its_own_model_group(self):
        key = "ecmwf_ensemble_spread_500"
        self.assertIn(key, publisher.PRODUCTS_BY_MODEL["ecmwf_ensemble"])
        self.assertIn(key, r2_publish.MODEL_PRODUCTS["ecmwf_ensemble"])
        self.assertEqual(publisher.PRODUCTS[key].model, "ECMWF ENS")
        self.assertEqual(publisher.PRODUCTS[key].category, "Upper Levels")
        self.assertEqual(publisher.PRODUCTS[key].hours[-1], 360)

    def test_viewer_places_product_under_upper_levels(self):
        site = (Path(__file__).resolve().parents[1] / "site/index.html").read_text()
        self.assertIn('<div class="category-title">Upper Levels</div>', site)
        self.assertIn('<span class="label">50.0kPa Hgt Mn|SD</span>', site)
        self.assertLess(site.index(">Surface</div>"), site.index(">Upper Levels</div>"))
        self.assertLess(site.index(">Upper Levels</div>"), site.index(">Verification</div>"))


if __name__ == "__main__":
    unittest.main()
