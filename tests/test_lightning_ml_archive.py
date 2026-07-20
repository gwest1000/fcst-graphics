from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, TiffImagePlugin

import automate_hrdps_west as automation
import lightning_ml_archive as archive


UTC = dt.timezone.utc


class ObservationArchiveTest(unittest.TestCase):
    def test_archive_write_probe_succeeds_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            self.assertEqual(archive.verify_archive_writable(root), root)
            self.assertEqual(list(root.glob(".lightning_ml_write_probe.*")), [])

    def test_archive_write_probe_explains_macos_permission_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            with mock.patch.object(Path, "open", side_effect=PermissionError(1, "denied")):
                with self.assertRaisesRegex(RuntimeError, "Removable Volumes or Full Disk Access"):
                    archive.verify_archive_writable(root)

    def test_block_end_includes_right_boundary(self) -> None:
        self.assertEqual(
            archive.observation_block_end(dt.datetime(2026, 7, 9, 0, 0, tzinfo=UTC)),
            dt.datetime(2026, 7, 9, 0, 0, tzinfo=UTC),
        )
        self.assertEqual(
            archive.observation_block_end(dt.datetime(2026, 7, 9, 0, 10, tzinfo=UTC)),
            dt.datetime(2026, 7, 9, 3, 0, tzinfo=UTC),
        )
        self.assertEqual(
            archive.observation_block_end(dt.datetime(2026, 7, 9, 2, 50, tzinfo=UTC)),
            dt.datetime(2026, 7, 9, 3, 0, tzinfo=UTC),
        )

    def test_block_has_eighteen_ten_minute_sources(self) -> None:
        end = dt.datetime(2026, 7, 9, 3, 0, tzinfo=UTC)
        timestamps = archive.observation_block_times(end)
        self.assertEqual(len(timestamps), 18)
        self.assertEqual(timestamps[0], dt.datetime(2026, 7, 9, 0, 10, tzinfo=UTC))
        self.assertEqual(timestamps[-1], end)

    def write_source(self, path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (0.0335, 0.023, 0.0)
        tags[33922] = (0.0, 0.0, 0.0, -140.0, 60.0, 0.0)
        tags[42113] = "-999"
        data = np.asarray([[value, 0.0], [-999.0, value]], dtype=np.float32)
        Image.fromarray(data).save(path, format="TIFF", compression="tiff_adobe_deflate", tiffinfo=tags)

    def test_aggregate_writes_three_hour_density_then_removes_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            obs_dir = base / "obs"
            root = base / "archive"
            end = dt.datetime(2026, 7, 9, 3, 0, tzinfo=UTC)
            sources = []
            for timestamp in archive.observation_block_times(end):
                source = archive.observation_source_path(obs_dir, timestamp)
                self.write_source(source, 0.1)
                sources.append(source)

            output = archive.aggregate_observation_block(obs_dir, root, end, delete_sources=True)
            self.assertIsNotNone(output)
            assert output is not None
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
            self.assertFalse(any(path.exists() for path in sources))

            with Image.open(output) as image:
                actual = np.asarray(image, dtype=np.float32)
            np.testing.assert_allclose(actual, [[18.0, 0.0], [-999.0, 18.0]])
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["source_count"], 18)
            self.assertEqual(sidecar["window_end_utc"], "2026-07-09T03:00:00Z")


class ModelArchiveTest(unittest.TestCase):
    def test_baseline_backfill_command_uses_default_cache_directory(self) -> None:
        args = archive.parse_args(["archive-baseline", "--run", "20260711T00Z"])
        self.assertEqual(args.command, "archive-baseline")
        self.assertEqual(args.cache_dir, archive.DEFAULT_LPI_CACHE_DIR)

    def test_only_twice_daily_continental_runs_are_eligible(self) -> None:
        self.assertTrue(archive.should_archive_model_run("continental", "00"))
        self.assertTrue(archive.should_archive_model_run("continental", "12"))
        self.assertFalse(archive.should_archive_model_run("continental", "06"))
        self.assertFalse(archive.should_archive_model_run("west", "12"))

    def test_pack_round_trip_and_missing_value(self) -> None:
        spec = archive.FieldSpec("temperature", "TMP", "Sfc", "K", 273.15, 0.05)
        source = np.asarray([270.0, 273.15, 280.0, np.nan], dtype=np.float32)
        packed, clipped = archive._pack_field(source, spec)
        actual = archive.unpack_field(packed, spec)
        self.assertEqual(clipped, 0)
        np.testing.assert_allclose(actual[:3], source[:3], atol=0.026)
        self.assertTrue(np.isnan(actual[3]))

    def test_f000_excludes_precipitation_fields(self) -> None:
        keys = {spec.key for spec in archive.model_field_specs(0)}
        self.assertNotIn("precip_rate", keys)
        self.assertNotIn("precip_accum", keys)
        later_keys = {spec.key for spec in archive.model_field_specs(3)}
        self.assertIn("precip_rate", later_keys)
        self.assertIn("precip_accum", later_keys)

    def test_cleanup_preserves_unarchived_twice_daily_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            for stamp in ("20260710T00Z", "20260710T06Z", "20260710T12Z"):
                (data_dir / stamp).mkdir(parents=True)
            original_model = automation.convective.model_config().key
            try:
                automation.convective.set_model("continental")
                automation.cleanup_model_data(data_dir, "20260710T12Z", root / "archive")
            finally:
                automation.convective.set_model(original_model)
            self.assertTrue((data_dir / "20260710T00Z").exists())
            self.assertFalse((data_dir / "20260710T06Z").exists())
            self.assertTrue((data_dir / "20260710T12Z").exists())


if __name__ == "__main__":
    unittest.main()
