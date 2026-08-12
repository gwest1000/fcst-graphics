from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import ecmwf_convective_data as convective


class EcmwfConvectiveDataTest(unittest.TestCase):
    def test_regional_slice_discards_global_grid(self) -> None:
        latitude = np.repeat(np.arange(90.0, -90.25, -0.25)[:, None], 1440, axis=1)
        longitude_row = np.arange(-180.0, 180.0, 0.25)
        longitude = np.repeat(longitude_row[None, :], latitude.shape[0], axis=0)

        yslice, xslice = convective._regional_slices(latitude, longitude)

        retained_fraction = (
            latitude[yslice, xslice].size / latitude.size
        )
        self.assertLess(retained_fraction, 0.02)
        self.assertGreaterEqual(float(longitude[yslice, xslice].min()), convective.REGIONAL_EXTENT[0])
        self.assertLessEqual(float(longitude[yslice, xslice].max()), convective.REGIONAL_EXTENT[1])

    def test_archive_round_trip_and_hour_check(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / convective.ARCHIVE_NAME
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    version=np.asarray([convective.ARCHIVE_VERSION], dtype=np.int16),
                    steps=np.asarray([0, 3], dtype=np.int16),
                    lat=np.full((2, 2), 50.0, dtype=np.float32),
                    lon=np.full((2, 2), -120.0, dtype=np.float32),
                    mucape=np.asarray([np.zeros((2, 2)), np.ones((2, 2))], dtype=np.float32),
                    surface_pressure_pa=np.full((2, 2, 2), 90000.0, dtype=np.float32),
                )

            archive = convective.load_archive(path)
            values, _, _ = archive.field("mucape", 3)

            np.testing.assert_array_equal(values, np.ones((2, 2)))
            self.assertTrue(convective.archive_has_hours(path, (0, 3)))
            self.assertFalse(convective.archive_has_hours(path, (0, 6)))


if __name__ == "__main__":
    unittest.main()
