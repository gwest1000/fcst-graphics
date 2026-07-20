from __future__ import annotations

import unittest

import numpy as np

import make_ensemble_control_fourpanel as ensemble


def field(value: float, step_range: str = "") -> ensemble.Field:
    data = np.full((2, 2), value, dtype=np.float32)
    lat = np.array([[50.0, 50.0], [51.0, 51.0]], dtype=np.float32)
    lon = np.array([[-125.0, -124.0], [-125.0, -124.0]], dtype=np.float32)
    return ensemble.Field(data=data, lat=lat, lon=lon, step_range=step_range)


class FakeEcmwfProvider(ensemble.EcmwfProvider):
    def __init__(self) -> None:
        pass

    def surface(self, fhour: int, short_name: str, level_type: str, level: int = 0) -> ensemble.Field:
        values = {0: (0.001, "0"), 3: (0.004, "0-3")}
        value, step_range = values[fhour]
        return field(value, step_range)


class EnsembleControlFourPanelTest(unittest.TestCase):
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
