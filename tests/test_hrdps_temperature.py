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
        temperature.set_model("west")

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
        self.assertEqual(temperature.TEMPERATURE_TICKS_C, tuple(range(-56, 49, 4)))
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
        self.assertEqual(temperature.region_keys_for_model("west"), ("south", "north"))
        self.assertEqual(temperature.region_keys_for_model("continental"), ("bc",))

        temperature.set_model("west")
        self.assertEqual(
            temperature.output_prefix("south"),
            "hrdps_west_temperature_south",
        )
        temperature.set_model("continental")
        self.assertEqual(
            temperature.output_prefix("bc"),
            "hrdps_continental_temperature",
        )

    def test_temperature_regions_use_wide_product_specific_domains(self) -> None:
        self.assertEqual(
            temperature.region_config("south").extent,
            (-129.5, -113.1, 48.0, 54.08),
        )
        self.assertEqual(
            temperature.region_config("north").extent,
            (-133.06, -112.44, 51.7, 59.2),
        )

    def test_temperature_products_are_registered_for_automation_and_r2(self) -> None:
        west_keys = {"temperature_south", "temperature_north"}
        self.assertEqual(set(automation.temperature_product_keys("west")), west_keys)
        self.assertEqual(
            automation.temperature_product_keys("continental"),
            ("continental_temperature",),
        )
        self.assertTrue(west_keys.issubset(r2_publish.MODEL_PRODUCTS["west"]))
        self.assertIn("continental_temperature", r2_publish.MODEL_PRODUCTS["continental"])
        for key in (*sorted(west_keys), "continental_temperature"):
            self.assertEqual(publisher.PRODUCTS[key].plot_type, "2 m Temperature")
            self.assertEqual(publisher.PRODUCTS[key].hours, temperature.FORECAST_HOURS)
        self.assertTrue(
            {"temperature_sw", "temperature_se", "temperature_ne"}.issubset(
                r2_publish.RETIRED_PRODUCTS["west"]
            )
        )

    def test_one_west_field_is_reused_for_both_regional_frames(self) -> None:
        run = temperature.hrdps.RunInfo(
            cycle="00",
            stamp="20260724T00Z",
            init_time=dt.datetime(2026, 7, 24, tzinfo=dt.timezone.utc),
        )
        lat, lon = np.meshgrid(
            np.linspace(47.0, 60.0, 8, dtype=np.float32),
            np.linspace(-140.0, -113.0, 10, dtype=np.float32),
            indexing="ij",
        )
        temp_k = np.full(lat.shape, 293.15, dtype=np.float32)

        with TemporaryDirectory() as tmpdir:
            temperature.set_model("west")
            with (
                mock.patch.object(
                    temperature.hrdps,
                    "read_grib",
                    side_effect=[(None, lat, lon), (temp_k, None, None)],
                ) as read_grib,
                mock.patch.object(temperature, "render_temperature") as render,
            ):
                outputs = temperature.make_plots(
                    run,
                    Path(tmpdir) / "data",
                    Path(tmpdir) / "plots",
                    hours=(0,),
                    no_watersheds=True,
                )

        self.assertEqual(read_grib.call_count, 2)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(
            {path.name for path in outputs},
            {
                "hrdps_west_temperature_south_20260724T00Z_f000.png",
                "hrdps_west_temperature_north_20260724T00Z_f000.png",
            },
        )

    def test_sidebar_exposes_temperature_area_model_hierarchy(self) -> None:
        site = (Path(__file__).parents[1] / "site" / "index.html").read_text()
        self.assertIn('data-plot-toggle="temperature"', site)
        for area in ("bc", "south", "north"):
            self.assertIn(f'data-area-toggle="temperature_{area}"', site)
        for product_key in (
            "continental_temperature",
            "temperature_south",
            "temperature_north",
        ):
            self.assertIn(f'data-product-select="{product_key}"', site)
        for retired_key in ("temperature_sw", "temperature_se", "temperature_ne"):
            self.assertNotIn(f'data-product-select="{retired_key}"', site)


if __name__ == "__main__":
    unittest.main()
