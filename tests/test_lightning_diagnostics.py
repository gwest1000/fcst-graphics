from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import make_hrdps_west_lightning as lightning


class LightningDiagnosticTest(unittest.TestCase):
    def tearDown(self) -> None:
        lightning.set_model("west")

    def test_regular_gust_field_names(self) -> None:
        lightning.set_model("west")
        self.assertEqual(
            lightning.gust_field_name("20260710T00Z", 3),
            "CMC_hrdps_west_GUST_TGL_10_rotated_latlon0.009x0.009_20260710T00Z_P003-00.grib2",
        )
        lightning.set_model("continental")
        self.assertEqual(
            lightning.gust_field_name("20260710T12Z", 3),
            "20260710T12Z_MSC_HRDPS_GUST_AGL-10m_RLatLon0.0225_PT003H.grib2",
        )

    def test_dry_lightning_markers_use_compact_black_asterisks(self) -> None:
        lightning.set_model("west")
        self.assertEqual(lightning.dry_lightning_marker_area("bc"), 9.0)
        self.assertEqual(lightning.dry_lightning_marker_area("sw"), 14.0)

        lightning.set_model("continental")
        self.assertEqual(lightning.dry_lightning_marker_area("bc"), 9.0)
        self.assertEqual(lightning.dry_lightning_marker_area("sw"), 14.0)
        self.assertEqual(lightning.DRY_LIGHTNING_MARKER, (5, 2, 0))
        self.assertEqual(lightning.DRY_LIGHTNING_COLOR, "#161616")

    def test_fire_weather_region_expansions(self) -> None:
        self.assertEqual(lightning.FIRE_WEATHER_REGIONS["bc"].extent, (-138.2, -109.5, 46.0, 58.45))
        self.assertEqual(lightning.FIRE_WEATHER_REGIONS["sw"].extent, (-128.5, -120.0, 48.0, 55.0))
        self.assertEqual(lightning.FIRE_WEATHER_REGIONS["se"].extent, (-121.25, -114.2, 48.0, 53.7))
        self.assertEqual(lightning.FIRE_WEATHER_REGIONS["ne"].extent, (-130.0, -118.5, 51.0, 59.2))

    def test_transmission_lines_load_from_a_cached_geojson(self) -> None:
        lightning.set_model("continental")
        collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-130.0, 50.0], [-120.0, 52.0]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-100.0, 40.0], [-99.0, 41.0]],
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "transmission.geojson"
            cache_path.write_text(json.dumps(collection))
            lines = lightning.load_transmission_lines(cache_path)

        self.assertEqual(len(lines), 1)
        west, east, south, north = lightning.model_config().extent
        self.assertTrue(lines[0].intersects(lightning.box(west, south, east, north)))

    def test_rh_hatch_categories_are_non_overlapping(self) -> None:
        rh = np.array([np.nan, 19.9, 20.0, 20.1, 30.0, 30.1, 60.0, 60.1, 80.0, 80.1])
        low30, low20, high60, high80 = lightning.rh_hatch_masks(rh)
        np.testing.assert_array_equal(low30, [False, False, True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(low20, [False, True, False, False, False, False, False, False, False, False])
        np.testing.assert_array_equal(high60, [False, False, False, False, False, False, True, True, True, False])
        np.testing.assert_array_equal(high80, [False, False, False, False, False, False, False, False, False, True])

    def test_fire_weather_footer_uses_simple_category_labels(self) -> None:
        footer = lightning.fire_weather_footer(6)
        self.assertIn("3-h max LPI", footer)
        self.assertIn("3-h max Gust", footer)
        self.assertIn("valid-time 10m RH", footer)
        self.assertIn("3-h max dry lightning", footer)

    def test_three_hour_window_prerequisites_include_hourly_inputs(self) -> None:
        self.assertEqual(lightning.diagnostic_window_hours(0), (0,))
        self.assertEqual(lightning.diagnostic_window_hours(3), (1, 2, 3))
        self.assertEqual(lightning.diagnostic_window_hours(6), (4, 5, 6))
        self.assertEqual(lightning.prerequisite_hours((3,)), (1, 2, 3))
        self.assertEqual(lightning.prerequisite_hours((6,)), tuple(range(1, 7)))

        requirements = lightning.required_names_by_hour("20260710T00Z", (12,))
        self.assertEqual(
            requirements[7],
            {lightning.field_name("APCP", "SFC", "0", "20260710T00Z", 7)},
        )
        self.assertIn(
            lightning.field_name("MU-VT-LI", "ISBL", "500", "20260710T00Z", 10),
            requirements[10],
        )

    def test_three_hour_hazards_use_maxima_and_full_window_rain(self) -> None:
        def snapshot(
            potential: tuple[float, float],
            gust: tuple[float, float],
            u10: tuple[float, float],
            precip: tuple[float, float] = (0.0, 0.0),
        ) -> lightning.LightningFields:
            potential_array = np.asarray([potential], dtype=np.float32)
            ones = np.ones_like(potential_array)
            return lightning.LightningFields(
                potential=potential_array,
                dry_potential=np.zeros_like(potential_array),
                li=-ones,
                cape=500.0 * ones,
                precip_3h=np.asarray([precip], dtype=np.float32),
                charge_rh=70.0 * ones,
                trigger=potential_array / 100.0,
                subcloud_rh=20.0 * ones,
                surface_rh=20.0 * ones,
                gust_kmh=np.asarray([gust], dtype=np.float32),
                u10_ms=np.asarray([u10], dtype=np.float32),
                v10_ms=np.asarray([[value + 100.0 for value in u10]], dtype=np.float32),
            )

        snapshots = (
            snapshot((80.0, 20.0), (30.0, 80.0), (1.0, 10.0)),
            snapshot((40.0, 90.0), (50.0, 40.0), (2.0, 20.0)),
            snapshot((30.0, 50.0), (40.0, 60.0), (3.0, 30.0), precip=(0.0, 2.5)),
        )
        with patch.object(
            lightning,
            "compute_instantaneous_lightning_fields",
            side_effect=snapshots,
        ) as compute:
            result = lightning.compute_lightning_fields(
                Path("data"),
                Mock(),
                3,
                slice(None),
                slice(None),
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
                1,
            )

        self.assertEqual([call.args[2] for call in compute.call_args_list], [1, 2, 3])
        np.testing.assert_allclose(result.potential, [[80.0, 90.0]])
        np.testing.assert_allclose(result.dry_potential, [[80.0, 0.0]])
        np.testing.assert_allclose(result.gust_kmh, [[50.0, 80.0]])
        np.testing.assert_allclose(result.u10_ms, [[2.0, 10.0]])
        np.testing.assert_allclose(result.v10_ms, [[102.0, 110.0]])
        np.testing.assert_allclose(result.surface_rh, snapshots[-1].surface_rh)

    def test_rh_hatches_use_stroke_alpha_without_face_shading(self) -> None:
        self.assertEqual(lightning.RH_HATCH_ALPHA, 0.50)
        self.assertEqual(lightning.RECOVERY_RH_HATCH_ALPHA, 0.55)
        self.assertEqual(lightning.RH_HATCH_LINEWIDTH, 0.55)
        dry_collection = Mock()
        lightning.style_rh_hatch_collection(dry_collection, lightning.LOW_RH_HATCH_COLOR)
        dry_collection.set_facecolor.assert_called_once_with((0.0, 0.0, 0.0, 0.0))
        dry_collection.set_edgecolor.assert_called_once_with(
            lightning.mcolors.to_rgba(lightning.LOW_RH_HATCH_COLOR, lightning.RH_HATCH_ALPHA),
        )
        recovery_collection = Mock()
        lightning.style_rh_hatch_collection(recovery_collection, lightning.GOOD_RECOVERY_HATCH_COLOR)
        recovery_collection.set_facecolor.assert_called_once_with((0.0, 0.0, 0.0, 0.0))
        recovery_collection.set_edgecolor.assert_called_once_with(
            lightning.mcolors.to_rgba(
                lightning.GOOD_RECOVERY_HATCH_COLOR,
                lightning.RECOVERY_RH_HATCH_ALPHA,
            ),
        )

    def test_peak_danger_contours_and_vector_edges_avoid_rh_browns(self) -> None:
        self.assertEqual(lightning.PEAK_DANGER_CONTOUR_LEVELS, (1.5, 2.5, 3.5, 4.5))
        self.assertEqual(lightning.PEAK_DANGER_CONTOUR_COLORS, ("#666666", "#252525", "#d36b00", "#a20d18"))
        self.assertEqual(lightning.PEAK_DANGER_CONTOUR_LINEWIDTHS, (1.65, 1.9, 2.3, 2.75))
        self.assertEqual(
            lightning.PEAK_DANGER_CONTOUR_LABELS,
            {1.5: "LOW", 2.5: "MODERATE", 3.5: "HIGH", 4.5: "EXTREME"},
        )
        self.assertTrue(set(lightning.PEAK_DANGER_CONTOUR_COLORS).isdisjoint({
            lightning.LOW_RH_HATCH_COLOR,
            lightning.VERY_LOW_RH_HATCH_COLOR,
        }))
        self.assertEqual(lightning.GUST_VECTOR_EDGE_WIDTH, 0.22)
        self.assertEqual(lightning.GUST_VECTOR_EDGE_ALPHA, 0.55)

    def test_pressure_layer_edges_use_midpoints(self) -> None:
        actual = lightning.pressure_layer_edges_hpa((1000, 900, 800))
        np.testing.assert_allclose(actual, [1050.0, 950.0, 850.0, 750.0])

    def test_pressure_layer_thickness_clips_below_ground_portion(self) -> None:
        surface_pressure = np.array([1020.0, 925.0, 825.0, 700.0], dtype=np.float32)
        np.testing.assert_allclose(
            lightning.pressure_layer_thickness_hpa(surface_pressure, 950.0, 850.0),
            [100.0, 75.0, 0.0, 0.0],
        )

    def test_omega_conversion_is_upward_positive(self) -> None:
        omega = np.array([-1.0, 1.0], dtype=np.float32)
        temperature = np.array([280.0, 280.0], dtype=np.float32)
        actual = lightning.geometric_vertical_velocity_ms(omega, temperature, 700.0)
        density = 70000.0 / (lightning.DRY_AIR_GAS_CONSTANT_J_KG_K * 280.0)
        expected = np.array(
            [1.0 / (density * lightning.GRAVITY_MS2), -1.0 / (density * lightning.GRAVITY_MS2)]
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6)

    def test_omega_conversion_masks_invalid_temperature(self) -> None:
        actual = lightning.geometric_vertical_velocity_ms(
            np.array([-1.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            700.0,
        )
        self.assertTrue(np.isnan(actual[0]))


if __name__ == "__main__":
    unittest.main()
