from __future__ import annotations

import unittest

import numpy as np

import gust_diagnostics as gust


class GustDiagnosticTest(unittest.TestCase):
    def test_profile_interpolation(self) -> None:
        heights = np.array([[[10.0]], [[1000.0]]], dtype=np.float32)
        values = np.array([[[2.0]], [[12.0]]], dtype=np.float32)
        actual = gust.interpolate_profile_to_height(
            heights,
            values,
            np.array([[505.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(actual, [[7.0]])

    def test_inversion_metrics_use_deepest_inversion_top(self) -> None:
        surface = np.array([[290.0]], dtype=np.float32)
        heights = np.array([[[500.0]], [[1000.0]], [[1500.0]]], dtype=np.float32)
        temps = np.array([[[289.0]], [[291.0]], [[290.0]]], dtype=np.float32)
        lapse, present = gust.inversion_metrics(surface, temps, heights)
        self.assertTrue(bool(present[0, 0]))
        self.assertAlmostEqual(float(lapse[0, 0]), 1.0 / 998.0, places=6)

    def test_downslope_mask_uses_ratio_blend(self) -> None:
        yy, xx = np.mgrid[-3:4, -3:4]
        terrain = (1000.0 - 20.0 * (xx * xx + yy * yy)).astype(np.float32)
        regular = np.full(terrain.shape, 40.0, dtype=np.float32)
        gust_max = np.full(terrain.shape, 80.0, dtype=np.float32)
        ridge_u = np.full(terrain.shape, 10.0, dtype=np.float32)
        ridge_v = np.zeros(terrain.shape, dtype=np.float32)
        surface = np.full(terrain.shape, 290.0, dtype=np.float32)
        heights = np.stack(
            [np.full(terrain.shape, 500.0), np.full(terrain.shape, 1000.0)]
        ).astype(np.float32)
        temps = np.stack(
            [np.full(terrain.shape, 289.0), np.full(terrain.shape, 291.0)]
        ).astype(np.float32)

        result = gust.downslope_adjusted_gust(
            regular,
            gust_max,
            terrain,
            ridge_u,
            ridge_v,
            surface,
            temps,
            heights,
            1000.0,
        )
        self.assertTrue(bool(result.mask[3, 4]))
        self.assertAlmostEqual(float(result.gust_kmh[3, 4]), 60.0)
        self.assertFalse(bool(result.mask[3, 2]))
        self.assertAlmostEqual(float(result.gust_kmh[3, 2]), 40.0)

    def test_pcge_requires_active_convection(self) -> None:
        one = np.ones((1, 1), dtype=np.float32)
        active = gust.pcge_gust(
            36.0 * one,
            750.0 * one,
            -4.0 * one,
            800.0 * one,
            1800.0 * one,
            20.0 * one,
            25.0 * one,
            2.5 * one,
            -1.0 * one,
            280.0 * one,
        )
        self.assertAlmostEqual(float(active.trigger[0, 0]), 1.0)
        self.assertGreater(float(active.gust_kmh[0, 0]), 75.0)

        inactive = gust.pcge_gust(
            36.0 * one,
            750.0 * one,
            -4.0 * one,
            800.0 * one,
            1800.0 * one,
            20.0 * one,
            25.0 * one,
            np.zeros_like(one),
            one,
            280.0 * one,
        )
        self.assertAlmostEqual(float(inactive.trigger[0, 0]), 0.0)
        self.assertAlmostEqual(float(inactive.gust_kmh[0, 0]), 36.0)


if __name__ == "__main__":
    unittest.main()
