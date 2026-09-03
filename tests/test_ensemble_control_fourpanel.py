from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import automate_ensemble_control_fourpanel as automation
import make_ensemble_control_fourpanel as ensemble
import publish_hrdps_west as publisher
import r2_publish


def field(value: float, step_range: str = "") -> ensemble.Field:
    data = np.full((2, 2), value, dtype=np.float32)
    lat = np.array([[50.0, 50.0], [51.0, 51.0]], dtype=np.float32)
    lon = np.array([[-125.0, -124.0], [-125.0, -124.0]], dtype=np.float32)
    return ensemble.Field(data=data, lat=lat, lon=lon, step_range=step_range)


class FakeEcmwfProvider(ensemble.EcmwfProvider):
    def __init__(self) -> None:
        pass

    def surface(self, fhour: int, short_name: str, level_type: str, level: int = 0) -> ensemble.Field:
        values = {
            0: (0.001, "0"),
            3: (0.004, "0-3"),
            144: (0.100, "0-144"),
            150: (0.106, "0-150"),
        }
        value, step_range = values[fhour]
        return field(value, step_range)


class EnsembleControlFourPanelTest(unittest.TestCase):
    def test_500_hpa_height_contours_are_six_dam_and_anchored_at_600_dam(self) -> None:
        np.testing.assert_allclose(np.diff(ensemble.HGT500_LEVELS_KM), 0.06)
        self.assertIn(6.00, ensemble.HGT500_LEVELS_KM)
        self.assertEqual(ensemble.HGT500_LABEL_FORMAT % 6.00, "6.00")

    def test_unit_vector_components_preserve_direction_and_normalize_length(self) -> None:
        u = np.array([[3.0, 0.0, np.nan]], dtype=np.float32)
        v = np.array([[4.0, -2.0, 1.0]], dtype=np.float32)

        unit_u, unit_v, magnitude = ensemble.unit_vector_components(u, v)

        np.testing.assert_allclose(magnitude[0, :2], [5.0, 2.0])
        np.testing.assert_allclose(np.hypot(unit_u[0, :2], unit_v[0, :2]), [1.0, 1.0])
        self.assertAlmostEqual(float(unit_u[0, 0]), 0.6)
        self.assertAlmostEqual(float(unit_v[0, 0]), 0.8)
        self.assertTrue(np.isnan(unit_u[0, 2]))

    def test_ecmwf_precip_is_converted_from_metres_to_three_hour_mm(self) -> None:
        precip = FakeEcmwfProvider().precip(3)

        np.testing.assert_allclose(precip.data, 4.0)

    def test_ecmwf_precip_switches_to_six_hour_accumulations_after_f144(self) -> None:
        precip = FakeEcmwfProvider().precip(150)

        np.testing.assert_allclose(precip.data, 6.0, atol=1.0e-5)

    def test_ecmwf_forecast_schedule_reaches_fifteen_days(self) -> None:
        hours = ensemble.model_hours("ecmwf_control")

        self.assertEqual(hours[0], 0)
        self.assertEqual(hours[-1], 360)
        self.assertIn(144, hours)
        self.assertIn(150, hours)
        self.assertNotIn(147, hours)
        self.assertTrue(all(b - a == 3 for a, b in zip(hours, hours[1:]) if b <= 144))
        self.assertTrue(all(b - a == 6 for a, b in zip(hours, hours[1:]) if a >= 144))

    def test_ecmwf_geopotential_is_converted_to_height_metres(self) -> None:
        geopotential = field(2.0 * ensemble.GRAVITY)

        terrain = ensemble.geopotential_height_m(geopotential)

        np.testing.assert_allclose(terrain.data, 2.0)

    def test_ecmwf_source_stamp_includes_initialization(self) -> None:
        self.assertEqual(ensemble.MODEL_CONFIGS["ecmwf_control"].source_label, "ECMWF")
        run = ensemble.RunInfo(
            cycle="00",
            stamp="20260807T00Z",
            init_time=ensemble.parse_stamp("20260807T00Z"),
        )

        source = ensemble.panel_source_text(run, ensemble.MODEL_CONFIGS["ecmwf_control"])

        self.assertEqual(source, "Data: ECMWF | Init:2026080700")
        self.assertEqual(ensemble.MODEL_CONFIGS["ecmwf_control"].label, "ECMWF Ctl")

    def test_ecmwf_preflight_builds_one_inventory_for_all_hours(self) -> None:
        pressure, surface = ensemble.required_ecmwf_grib_keys((0, 360))

        self.assertIn(("gh", "isobaricInhPa", 500, 360), pressure)
        self.assertIn(("r", "isobaricInhPa", 925, 0), pressure)
        self.assertIn(("2d", "heightAboveGround", 2, 360), surface)
        self.assertIn(("tp", "surface", 0, 0), surface)

    def test_incomplete_convective_archive_skips_large_grib_scan(self) -> None:
        run = ensemble.RunInfo(
            cycle="12",
            stamp="20260820T12Z",
            init_time=ensemble.parse_stamp("20260820T12Z"),
        )
        with (
            mock.patch.object(ensemble, "provider_for"),
            mock.patch.object(ensemble.ecmwf_convective_data, "archive_has_hours", return_value=False),
            mock.patch.object(ensemble, "grib_inventory") as inventory,
        ):
            self.assertFalse(
                ensemble.required_files_present("ecmwf_control", Path("/data"), run, (0, 360))
            )
        inventory.assert_not_called()

    def test_ecmwf_vector_density_is_25_percent_above_gefs(self) -> None:
        gefs = ensemble.MODEL_CONFIGS["gefs_control"]
        ecmwf = ensemble.MODEL_CONFIGS["ecmwf_control"]

        self.assertEqual(ensemble.default_barb_stride(gefs), ensemble.default_barb_stride(ecmwf))
        self.assertAlmostEqual(
            ensemble.barb_row_density(ecmwf),
            ensemble.barb_row_density(gefs) * 1.25,
        )
        self.assertAlmostEqual(
            ensemble.barb_column_density(ecmwf),
            ensemble.barb_column_density(gefs) * 1.25,
        )

    def test_surface_based_lifted_index_matches_reference_parcel_values(self) -> None:
        actual = ensemble.surface_based_lifted_index_values(
            np.array([303.15, 293.15, 288.15], dtype=np.float32),
            np.array([293.15, 283.15, 268.15], dtype=np.float32),
            np.array([1000.0, 900.0, 800.0], dtype=np.float32),
            np.array([263.15, 258.15, 255.15], dtype=np.float32),
        )

        np.testing.assert_allclose(actual, [-6.2164, -2.6215, -0.1292], atol=0.04)

    def test_ecmwf_convective_product_is_published_with_synoptic_product(self) -> None:
        key = "ecmwf_control_convective_fourpanel"

        self.assertIn(key, publisher.PRODUCTS_BY_MODEL["ecmwf_control"])
        self.assertIn(key, r2_publish.MODEL_PRODUCTS["ecmwf_control"])
        self.assertEqual(publisher.PRODUCTS[key].prefix, "ecmwf_control_convective_fourpanel")

        site_html = (Path(__file__).parents[1] / "site" / "index.html").read_text()
        self.assertIn(f'data-product-select="{key}"', site_html)

    def test_ecmwf_plot_set_requires_both_four_panel_products(self) -> None:
        config = ensemble.MODEL_CONFIGS["ecmwf_control"]
        stamp = "20260807T12Z"
        with TemporaryDirectory() as directory:
            plot_dir = Path(directory) / stamp
            plot_dir.mkdir()
            (plot_dir / automation.image_name(config, stamp, 0)).touch()
            self.assertFalse(automation.plot_set_complete(Path(directory), config, stamp, (0,)))

            (plot_dir / automation.convective_image_name(config, stamp, 0)).touch()
            self.assertTrue(automation.plot_set_complete(Path(directory), config, stamp, (0,)))

    def test_raw_contour_grid_does_not_modify_values(self) -> None:
        source = np.arange(16, dtype=np.float32).reshape(4, 4)
        lat = source + 40.0
        lon = source - 130.0

        sampled_lat, sampled_lon, sampled_data = ensemble.raw_contour_grid(
            lat,
            lon,
            source,
            stride=2,
        )

        np.testing.assert_array_equal(sampled_lat, lat[::2, ::2])
        np.testing.assert_array_equal(sampled_lon, lon[::2, ::2])
        np.testing.assert_array_equal(sampled_data, source[::2, ::2])


if __name__ == "__main__":
    unittest.main()
