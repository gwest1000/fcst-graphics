from __future__ import annotations

import unittest

import numpy as np

import analyze_lpi_smoothing as smoothing


class LpiSmoothingTest(unittest.TestCase):
    def test_zero_smoothing_preserves_field(self) -> None:
        source = np.array([[0.0, 2.0], [4.0, np.nan]], dtype=np.float32)
        np.testing.assert_equal(smoothing.smooth_nan(source, 0.0), source)

    def test_smoothing_reduces_isolated_peak(self) -> None:
        source = np.zeros((9, 9), dtype=np.float32)
        source[4, 4] = 100.0
        result = smoothing.smooth_nan(source, 10.0)
        self.assertLess(result[4, 4], 100.0)
        self.assertGreater(result[4, 3], 0.0)

    def test_spatial_detail_declines_after_smoothing(self) -> None:
        source = np.indices((20, 20)).sum(axis=0) % 2 * 100.0
        mask = np.ones(source.shape, dtype=bool)
        raw = smoothing.spatial_detail(source.astype(np.float32), mask)
        smooth = smoothing.spatial_detail(smoothing.smooth_nan(source, 10.0), mask)
        self.assertLess(smooth["mean_abs_neighbor_difference"], raw["mean_abs_neighbor_difference"])


if __name__ == "__main__":
    unittest.main()
