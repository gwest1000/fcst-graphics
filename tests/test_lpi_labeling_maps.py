from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path

import numpy as np

import make_lpi_labeling_maps as labeling


class LpiLabelingMapsTest(unittest.TestCase):
    def test_complete_days_requires_all_eight_blocks(self) -> None:
        start = dt.datetime(2026, 8, 1, 12, tzinfo=dt.timezone.utc)
        paths = {start + dt.timedelta(hours=hour): Path(str(hour)) for hour in labeling.PERIOD_HOURS}
        self.assertEqual(labeling.complete_days(paths), [start])
        paths.pop(start + dt.timedelta(hours=9))
        self.assertEqual(labeling.complete_days(paths), [])

    def test_case_ids_are_stable_and_blind(self) -> None:
        start = dt.datetime(2026, 8, 1, 12, tzinfo=dt.timezone.utc)
        identifier = labeling.case_id(start)
        self.assertEqual(identifier, labeling.case_id(start))
        self.assertNotIn("2026", identifier)

    def test_synthetic_points_are_deterministic(self) -> None:
        density = np.array([[0.0, 0.2], [0.5, 1.0]], dtype=np.float32)
        lon = np.array([[-124.0, -123.9], [-124.0, -123.9]], dtype=np.float32)
        lat = np.array([[50.0, 50.0], [49.9, 49.9]], dtype=np.float32)
        first = labeling.synthetic_strike_points(density, lon, lat, 42)
        second = labeling.synthetic_strike_points(density, lon, lat, 42)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
