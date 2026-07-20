from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import make_experimental_danger_class as danger
import automate_hrdps_west as automation
import make_hrdps_west_convective as hrdps
import make_hrdps_evolved_danger_class as evolved
import publish_hrdps_west as publisher


class DangerRegionTest(unittest.TestCase):
    def tearDown(self) -> None:
        hrdps.set_model("west")

    def test_evolved_danger_is_available_for_both_hrdps_grids(self) -> None:
        self.assertIn("fwi2025_danger", publisher.PRODUCTS_BY_MODEL["west"])
        self.assertIn("continental_fwi2025_danger", publisher.PRODUCTS_BY_MODEL["continental"])

    def test_fire_danger_verification_is_a_static_product(self) -> None:
        product = publisher.PRODUCTS["fire_danger_verif"]
        self.assertEqual(product.hours, (0,))
        self.assertEqual(publisher.PRODUCTS_BY_MODEL["danger_verif"], ("fire_danger_verif",))

    def test_convective_fourpanel_is_continental_only(self) -> None:
        self.assertNotIn("fourpanel", publisher.PRODUCTS_BY_MODEL["west"])
        self.assertIn("continental_fourpanel", publisher.PRODUCTS_BY_MODEL["continental"])

    def test_continental_download_includes_hourly_danger_prerequisites(self) -> None:
        hrdps.set_model("continental")
        run = hrdps.RunInfo("12", "20260710T12Z", hrdps.parse_stamp("20260710T12Z"))
        with TemporaryDirectory() as tmpdir, mock.patch.object(automation.convective, "download_one") as download:
            automation.download_hours(run, Path(tmpdir), (9,), workers=1)

        expected = Path("008") / hrdps.field_name("TMP", "TGL", "2", run.stamp, 8)
        destinations = [call.args[1] for call in download.call_args_list]
        self.assertTrue(any(path.parts[-2:] == expected.parts for path in destinations))

    def test_handoff_checkpoint_matches_model_cycle_interval(self) -> None:
        hrdps.set_model("west")
        self.assertEqual(evolved.handoff_checkpoint_hour(), 12)
        hrdps.set_model("continental")
        self.assertEqual(evolved.handoff_checkpoint_hour(), 6)

    def test_cwfis_drought_code_range_retains_extreme_values(self) -> None:
        self.assertGreaterEqual(danger.CWFIS_VALUE_LIMITS["dc"][1], 1000.0)

    def test_cwfis_no_data_is_filled_from_nearest_analysis(self) -> None:
        source = np.array([[10.0, np.nan, np.nan], [np.nan, np.nan, 40.0]], dtype=np.float32)
        filled = evolved.fill_nearest_valid(source)
        self.assertTrue(np.isfinite(filled).all())
        self.assertEqual(float(filled[0, 0]), 10.0)
        self.assertEqual(float(filled[1, 2]), 40.0)

    def test_display_smoothing_preserves_mask_and_softens_single_cell_peak(self) -> None:
        hrdps.set_model("continental")
        source = np.ones((5, 5), dtype=np.float32)
        source[2, 2] = 5.0
        source[0, :] = np.nan

        smoothed = evolved.smooth_danger_for_display(source)

        self.assertTrue(np.isnan(smoothed[0, :]).all())
        self.assertGreater(float(smoothed[2, 2]), 1.0)
        self.assertLess(float(smoothed[2, 2]), 5.0)

    def test_schedule_two_matrix_examples(self) -> None:
        fwi = np.array([0.0, 40.0, 60.0], dtype=np.float32)
        bui = np.array([0.0, 100.0, 250.0], dtype=np.float32)
        np.testing.assert_array_equal(danger.classify_region(fwi, bui, 1), [1, 5, 5])
        np.testing.assert_array_equal(danger.classify_region(fwi, bui, 2), [1, 4, 5])
        np.testing.assert_array_equal(danger.classify_region(fwi, bui, 3), [1, 4, 5])

    def test_bcgw_district_unions_match_schedule_one_cities(self) -> None:
        points = {
            "Vancouver": (-123.12, 49.28, 1),
            "Prince George": (-122.75, 53.92, 1),
            "Williams Lake": (-122.14, 52.13, 2),
            "Quesnel": (-122.49, 52.98, 2),
            "100 Mile House": (-121.29, 51.64, 2),
            "Kamloops": (-120.33, 50.67, 3),
            "Kelowna": (-119.50, 49.89, 3),
            "Golden": (-116.97, 51.30, 3),
            "Cranbrook": (-115.77, 49.51, 3),
        }
        lon = np.array([[point[0] for point in points.values()]])
        lat = np.array([[point[1] for point in points.values()]])
        actual = danger.danger_regions(lon, lat).ravel()
        expected = np.array([point[2] for point in points.values()])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
