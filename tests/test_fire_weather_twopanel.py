import datetime as dt
import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import automate_hrdps_west as automation
import make_hrdps_fire_weather_twopanel as twopanel
import make_hrdps_west_lightning as lightning
import publish_hrdps_west as publisher


class FireWeatherTwoPanelTests(unittest.TestCase):
    def test_product_asset_version_changes_when_any_frame_changes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "frame_f000.png"
            later = root / "frame_f003.png"
            first.write_bytes(b"first")
            later.write_bytes(b"later")
            original = publisher.product_asset_version("20260717T12Z", "test", [first, later])

            later.write_bytes(b"updated-later-frame")
            updated = publisher.product_asset_version("20260717T12Z", "test", [first, later])

        self.assertNotEqual(original, updated)

    def test_product_is_operational_for_continental_and_west_regions(self):
        self.assertIn("continental_lightning_twopanel", publisher.PRODUCTS_BY_MODEL["continental"])
        self.assertNotIn("continental_lightning", publisher.PRODUCTS)
        self.assertNotIn("continental_convective", publisher.PRODUCTS)
        self.assertNotIn("convective", publisher.PRODUCTS)
        self.assertNotIn("lightning", publisher.PRODUCTS)
        self.assertEqual(
            set(automation.lightning_product_keys("west")),
            {"lightning_sw", "lightning_se", "lightning_ne"},
        )
        for key in automation.lightning_product_keys("west"):
            self.assertEqual(publisher.PRODUCTS[key].plot_type, "Fire Weather")

    def test_product_metadata_and_automation_family(self):
        product = publisher.PRODUCTS["continental_lightning_twopanel"]
        self.assertEqual(product.prefix, twopanel.OUTPUT_PREFIX)
        self.assertEqual(product.label, "Fire Weather")
        self.assertIn(
            "continental_lightning_twopanel",
            automation.lightning_product_keys("continental"),
        )

    def test_continental_render_skips_retired_one_panel(self):
        run = lightning.RunInfo(
            cycle="12",
            stamp="20260720T12Z",
            init_time=dt.datetime(2026, 7, 20, 12, tzinfo=dt.timezone.utc),
        )
        lat, lon = np.meshgrid(
            np.linspace(48.0, 59.0, 4, dtype=np.float32),
            np.linspace(-139.0, -114.0, 5, dtype=np.float32),
            indexing="ij",
        )
        fields = mock.Mock()
        with TemporaryDirectory() as tmpdir:
            lightning.set_model("continental")
            try:
                with (
                    mock.patch.object(lightning, "load_transmission_lines", return_value=[]),
                    mock.patch.object(lightning, "read_grib", side_effect=[(None, lat, lon), (np.zeros_like(lat), None, None)]),
                    mock.patch.object(lightning, "subset_slices", return_value=(slice(None), slice(None))),
                    mock.patch.object(lightning, "compute_lightning_fields", return_value=fields),
                    mock.patch.object(lightning, "save_lpi_cache", return_value=Path(tmpdir) / "lpi.npz"),
                    mock.patch.object(lightning, "plot_lightning") as legacy_plot,
                    mock.patch.object(twopanel, "plot_twopanel") as two_panel_plot,
                ):
                    paths = lightning.make_region_plots(
                        run,
                        Path(tmpdir) / "data",
                        Path(tmpdir) / "plots",
                        2,
                        4,
                        6,
                        hours=(0,),
                        no_fwi=True,
                        region_keys=("bc",),
                    )
            finally:
                lightning.set_model("west")

        legacy_plot.assert_not_called()
        two_panel_plot.assert_called_once()
        self.assertEqual(len(paths), 1)

    def test_two_panel_vectors_are_fifty_percent_denser_than_previous_layout(self):
        self.assertEqual(twopanel.VECTOR_DENSITY_MULTIPLIER, 1.875)
        self.assertGreater(twopanel.VECTOR_SIZE_MULTIPLIER, 1.0)
        self.assertGreater(twopanel.VECTOR_BOLD_MULTIPLIER, 1.0)

    def test_regional_vector_and_dry_lightning_sampling(self):
        self.assertAlmostEqual(
            twopanel.REGIONAL_VECTOR_COLUMN_DENSITY_MULTIPLIER,
            twopanel.REGIONAL_VECTOR_ROW_DENSITY_MULTIPLIER * 1.20,
        )
        self.assertEqual(twopanel.REGIONAL_DRY_LIGHTNING_DENSITY_MULTIPLIER, 1.875)
        self.assertEqual(twopanel.DRY_LIGHTNING_AREA_MULTIPLIER, 1.25)
        self.assertEqual(twopanel.PRECIP_DOT_MODERATE_MM, 2.5)
        self.assertEqual(twopanel.PRECIP_DOT_HEAVY_MM, 10.0)
        self.assertEqual(twopanel.REGIONAL_PRECIP_DOT_AREA_MULTIPLIER, 1.25)

    def test_precipitation_colors_are_distinct_from_danger_and_dry_lightning(self):
        self.assertNotIn(twopanel.PRECIP_DOT_MODERATE_COLOR, twopanel.DANGER_COLORS)
        self.assertNotIn(twopanel.PRECIP_DOT_HEAVY_COLOR, twopanel.DANGER_COLORS)
        self.assertNotEqual(twopanel.PRECIP_DOT_HEAVY_COLOR, twopanel.lightning.DRY_LIGHTNING_COLOR)

    def test_regional_colorbars_touch_requested_upper_corners(self):
        for key in ("sw", "ne"):
            layout = twopanel.regional_colorbar_layout(key)
            self.assertLessEqual(layout["backdrop"][0], 0.001)
            self.assertAlmostEqual(layout["backdrop"][1] + layout["backdrop"][3], 0.997)
        southeast = twopanel.regional_colorbar_layout("se")
        self.assertAlmostEqual(southeast["backdrop"][0] + southeast["backdrop"][2], 0.999)
        self.assertAlmostEqual(southeast["backdrop"][1] + southeast["backdrop"][3], 0.997)

    def test_bc_colorbars_reach_panel_edges(self):
        for layout in (twopanel.BC_GUST_COLORBAR_LAYOUT, twopanel.BC_DANGER_COLORBAR_LAYOUT):
            self.assertAlmostEqual(layout["backdrop"][0] + layout["backdrop"][2], 1.0)
            self.assertLessEqual(layout["cax_bounds"][0] + layout["cax_bounds"][2], 0.973)
            self.assertGreater(layout["cax_bounds"][0], layout["backdrop"][0])
            self.assertEqual(layout["backdrop_edgecolor"], "black")
            self.assertAlmostEqual(layout["backdrop_linewidth"], 0.65)

    def test_regional_colorbar_backdrops_have_thin_black_borders(self):
        for key in ("sw", "se", "ne"):
            layout = twopanel.regional_colorbar_layout(key)
            self.assertEqual(layout["backdrop_edgecolor"], "black")
            self.assertAlmostEqual(layout["backdrop_linewidth"], 0.65)

    def test_edge_footers_use_sentence_case(self):
        left, right = twopanel.edge_panel_footers(6)

        self.assertIn("brown", left)
        self.assertIn("3-h max gust", left)
        self.assertIn("Danger", right)
        self.assertIn("dry lightning", right)
        self.assertNotEqual(left, left.upper())
        self.assertNotEqual(right, right.upper())

    def test_projection_is_clockwise_and_projected_crop_is_portrait(self):
        self.assertEqual(twopanel.PLOT_CRS.proj4_params["lon_0"], -98.0)
        width = twopanel.PROJECTED_X_LIMITS[1] - twopanel.PROJECTED_X_LIMITS[0]
        height = twopanel.PROJECTED_Y_LIMITS[1] - twopanel.PROJECTED_Y_LIMITS[0]
        self.assertLess(width, height)

    def test_edge_text_bands_are_operational_with_requested_type_scale(self):
        self.assertTrue(inspect.signature(twopanel.plot_twopanel).parameters["edge_bands"].default)
        self.assertTrue(inspect.signature(twopanel.plot_regional_twopanel).parameters["edge_bands"].default)
        self.assertAlmostEqual(twopanel.EDGE_HEADER_FONTSIZE, 11.6 * 1.25 * 0.90)
        self.assertAlmostEqual(twopanel.EDGE_FOOTER_FONTSIZE, 8.0 * 2.0 * 0.90)
        self.assertLess(twopanel.EDGE_HEADER_HEIGHT, 0.050)
        self.assertLess(twopanel.EDGE_FOOTER_HEIGHT, 0.038)


if __name__ == "__main__":
    unittest.main()
