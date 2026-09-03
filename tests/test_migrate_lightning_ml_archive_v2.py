from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import lightning_ml_archive as archive
import migrate_lightning_ml_archive_v2 as migration


class LightningArchiveMigrationTest(unittest.TestCase):
    @staticmethod
    def write_npz(path: Path, forecast_hour: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray([1], dtype=np.int16),
            forecast_hour=np.asarray([forecast_hour], dtype=np.int16),
            field=np.arange(12, dtype=np.int16).reshape(3, 4),
        )

    def test_stages_validates_and_only_retires_large_v1_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_model = archive.model_archive_dir(root, schema_version=1)
            old_hourly = archive.hourly_lpi_archive_dir(root, schema_version=1)
            lat = np.full((3, 4), 50.0, dtype=np.float32)
            lon = np.full((3, 4), -125.0, dtype=np.float32)
            grid_path = old_model / "static/grid.npz"
            grid_path.parent.mkdir(parents=True)
            np.savez_compressed(grid_path, lat=lat, lon=lon)

            self.write_npz(old_model / "2026/20260801T00Z/f015.npz", 15)
            self.write_npz(old_model / "2026/20260801T00Z/f003.npz", 3)
            self.write_npz(old_hourly / "2026/20260801T06Z/f007.npz", 7)
            self.write_npz(old_hourly / "2026/20260801T06Z/f001.npz", 1)

            observation = archive.observation_archive_dir(root) / "keep.txt"
            baseline = archive.baseline_archive_dir(root) / "keep.txt"
            observation.parent.mkdir(parents=True)
            baseline.parent.mkdir(parents=True)
            observation.write_text("keep\n")
            baseline.write_text("keep\n")

            mask = np.asarray(
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, False, False],
                ]
            )
            with mock.patch.object(archive, "archive_domain_mask", return_value=mask):
                staged = migration.migrate(root, delete_v1=False)
            self.assertEqual(staged["status"], "verified")
            self.assertTrue(old_model.exists())
            self.assertTrue(old_hourly.exists())

            migrated_model = archive.model_hour_archive_path(root, "20260801T00Z", 15)
            migrated_hourly = archive.hourly_lpi_hour_archive_path(root, "20260801T06Z", 7)
            self.assertTrue(migrated_model.exists())
            self.assertTrue(migrated_hourly.exists())
            self.assertFalse(archive.model_hour_archive_path(root, "20260801T00Z", 3).exists())
            self.assertFalse(archive.hourly_lpi_hour_archive_path(root, "20260801T06Z", 1).exists())
            with np.load(migrated_model) as packed:
                np.testing.assert_array_equal(packed["field"][mask], np.arange(12).reshape(3, 4)[mask])
                self.assertTrue(np.all(packed["field"][~mask] == archive.FILL_VALUE))

            completed = migration.migrate(root, delete_v1=True)
            self.assertEqual(completed["status"], "complete")
            self.assertFalse(old_model.exists())
            self.assertFalse(old_hourly.exists())
            self.assertTrue(observation.exists())
            self.assertTrue(baseline.exists())


if __name__ == "__main__":
    unittest.main()
