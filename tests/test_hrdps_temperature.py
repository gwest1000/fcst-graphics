from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import automate_hrdps_west as automation
import make_hrdps_temperature as temperature
import publish_hrdps_west as publisher
import r2_publish


class HrdpsTemperatureTests(unittest.TestCase):
    def tearDown(self) -> None:
        temperature.set_model("continental")

    def test_reference_palette_has_two_degree_intervals_and_four_degree_ticks(self) -> None:
        self.assertEqual(temperature.TEMPERATURE_LEVELS_C[0], -58)
        self.assertEqual(temperature.TEMPERATURE_LEVELS_C[-1], 48)
        self.assertTrue(
            all(
                right - left == 2
                for left, right in zip(
                    temperature.TEMPERATURE_LEVELS_C,
                    temperature.TEMPERATURE_LEVELS_C[1:],
                )
            )
        )
        self.assertEqual(
            len(temperature.TEMPERATURE_COLORS),
            len(temperature.TEMPERATURE_LEVELS_C) - 1,
        )
        self.assertEqual(temperature.TEMPERATURE_TICKS_C, tuple(range(-50, 41, 10)))
        self.assertEqual(temperature.ISOTHERM_LEVELS_C, tuple(range(-60, 51, 10)))

    def test_reference_palette_places_operational_colors_at_key_temperatures(self) -> None:
        intervals = {
            low: color
            for low, color in zip(
                temperature.TEMPERATURE_LEVELS_C[:-1],
                temperature.TEMPERATURE_COLORS,
            )
        }
        self.assertEqual(intervals[-2], "#00ab00")
        self.assertEqual(intervals[10], "#ffc300")
        self.assertEqual(intervals[20], "#af0000")
        self.assertEqual(intervals[30], "#fd7bfd")
        self.assertEqual(intervals[40], "#444343")

    def test_model_region_assignments_match_requested_products(self) -> None:
        self.assertEqual(temperature.region_keys_for_model("continental"), ("bc",))
        temperature.set_model("continental")
        self.assertEqual(
            temperature.output_prefix("bc"),
            "hrdps_continental_temperature",
        )

    def test_temperature_regions_use_wide_product_specific_domains(self) -> None:
        self.assertEqual(
            temperature.region_config("bc").extent,
            (-138.2, -108.03, 47.55, 60.0),
        )

    def test_temperature_colorbar_fills_map_height_and_reaches_right_border(self) -> None:
        backdrop = temperature.TEMPERATURE_COLORBAR_BACKDROP
        colorbar = temperature.TEMPERATURE_COLORBAR_AX
        map_height = (
            1.0
            - temperature.plot_style.SINGLE_HEADER_BAND_HEIGHT
            - temperature.plot_style.SINGLE_FOOTER_BAND_HEIGHT
        )

        self.assertAlmostEqual(backdrop[0] + backdrop[2], 1.0)
        self.assertAlmostEqual(colorbar[0] + colorbar[2], 0.9955)
        self.assertAlmostEqual(colorbar[1], temperature.plot_style.SINGLE_FOOTER_BAND_HEIGHT)
        self.assertAlmostEqual(colorbar[3], map_height)
        self.assertGreater(temperature.TEMPERATURE_COLORBAR_TICK_FONTSIZE, 6.8)
        self.assertGreater(temperature.TEMPERATURE_COLORBAR_LEFT_EDGE_WIDTH, 0.0)

    def test_temperature_products_are_registered_for_automation_and_r2(self) -> None:
        self.assertEqual(
            automation.temperature_product_keys("continental"),
            ("continental_temperature",),
        )
        self.assertNotIn("west", r2_publish.MODEL_PRODUCTS)
        self.assertIn("continental_temperature", r2_publish.MODEL_PRODUCTS["continental"])
        product = publisher.PRODUCTS["continental_temperature"]
        self.assertEqual(product.plot_type, "2 m Temperature")
        self.assertEqual(product.hours, temperature.FORECAST_HOURS)

    def test_sidebar_exposes_temperature_area_model_hierarchy(self) -> None:
        site = (Path(__file__).parents[1] / "site" / "index.html").read_text()
        self.assertIn('data-plot-toggle="temperature"', site)
        self.assertIn('data-area-toggle="temperature_bc"', site)
        self.assertIn('data-product-select="continental_temperature"', site)
        self.assertNotIn("HRDPS-West", site)


if __name__ == "__main__":
    unittest.main()
