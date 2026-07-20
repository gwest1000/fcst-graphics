from pathlib import Path
import unittest

import automate_hrdps_west as automation
import make_ensemble_control_fourpanel as ensemble
import make_hrdps_west_convective as hrdps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WatershedBoundaryTest(unittest.TestCase):
    def tearDown(self) -> None:
        hrdps.set_model("west")

    def test_all_plot_families_share_bch_watershed_source(self) -> None:
        expected = Path("data/bc_watersheds/bch/AllWatershedsUTM.shp")

        self.assertEqual(hrdps.WATERSHED_CACHE, expected)
        self.assertEqual(ensemble.WATERSHED_CACHE, expected)
        self.assertEqual(automation.FOURPANEL_WATERSHED_CACHE, expected)

    def test_watershed_boundaries_use_thin_dark_blue_style(self) -> None:
        self.assertEqual(hrdps.WATERSHED_EDGE_COLOR, "#173f73")
        self.assertLessEqual(hrdps.WATERSHED_LINEWIDTH, 0.8)
        self.assertGreater(hrdps.WATERSHED_HALO_LINEWIDTH, hrdps.WATERSHED_LINEWIDTH)

    def test_bch_watersheds_load_as_valid_lon_lat_polygons(self) -> None:
        hrdps.set_model("west")
        watersheds = hrdps.load_watersheds(PROJECT_ROOT / hrdps.WATERSHED_CACHE)

        self.assertEqual(len(watersheds), 54)
        self.assertTrue(all(geom.is_valid and not geom.is_empty for geom in watersheds))
        self.assertTrue(all(geom.geom_type in {"Polygon", "MultiPolygon"} for geom in watersheds))

        min_lon = min(geom.bounds[0] for geom in watersheds)
        min_lat = min(geom.bounds[1] for geom in watersheds)
        max_lon = max(geom.bounds[2] for geom in watersheds)
        max_lat = max(geom.bounds[3] for geom in watersheds)
        self.assertTrue(-131.0 < min_lon < -128.0)
        self.assertTrue(47.0 < min_lat < 49.0)
        self.assertTrue(-116.0 < max_lon < -113.0)
        self.assertTrue(57.0 < max_lat < 59.0)


if __name__ == "__main__":
    unittest.main()
