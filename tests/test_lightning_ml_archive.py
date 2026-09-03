from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
        self.assertEqual(args.model, "continental")

    def test_only_twice_daily_continental_runs_are_eligible(self) -> None:
        self.assertTrue(archive.should_archive_model_run("continental", "00"))
        self.assertTrue(archive.should_archive_model_run("continental", "12"))
        self.assertFalse(archive.should_archive_model_run("continental", "06"))
        self.assertFalse(archive.should_archive_model_run("west", "12"))
        for cycle in ("00", "06", "12", "18"):
            self.assertTrue(archive.should_archive_hourly_lpi_run("continental", cycle))
        self.assertFalse(archive.should_archive_hourly_lpi_run("west", "12"))
        for cycle in ("00", "06", "12", "18"):
            self.assertTrue(archive.should_archive_lpi_baseline_run("continental", cycle))
            self.assertFalse(archive.should_archive_lpi_baseline_run("west", cycle))

    def test_archive_windows_cover_first_complete_12z_to_12z_day(self) -> None:
        self.assertEqual(archive.model_forecast_hours("00"), tuple(range(15, 37, 3)))
        self.assertEqual(archive.model_forecast_hours("12"), tuple(range(3, 25, 3)))
        self.assertEqual(archive.model_forecast_hours("06"), ())
        self.assertEqual(archive.hourly_lpi_forecast_hours("00"), tuple(range(13, 37)))
        self.assertEqual(archive.hourly_lpi_forecast_hours("06"), tuple(range(7, 31)))
        self.assertEqual(archive.hourly_lpi_forecast_hours("12"), tuple(range(1, 25)))
        self.assertEqual(archive.hourly_lpi_forecast_hours("18"), tuple(range(19, 43)))

    def test_west_baseline_archive_is_decommissioned(self) -> None:
        self.assertFalse(archive.should_archive_lpi_baseline_run("west", "00"))
        self.assertFalse(archive.should_archive_lpi_baseline_run("west", "12"))

    def test_pack_round_trip_and_missing_value(self) -> None:
        spec = archive.FieldSpec("temperature", "TMP", "Sfc", "K", 273.15, 0.05)
        source = np.asarray([270.0, 273.15, 280.0, np.nan], dtype=np.float32)
        packed, clipped = archive._pack_field(source, spec)
        actual = archive.unpack_field(packed, spec)
        self.assertEqual(clipped, 0)
        np.testing.assert_allclose(actual[:3], source[:3], atol=0.026)
        self.assertTrue(np.isnan(actual[3]))

    def test_pack_masks_values_outside_archive_domain(self) -> None:
        spec = archive.FieldSpec("temperature", "TMP", "Sfc", "K", 273.15, 0.05)
        source = np.asarray([[273.15, 274.15], [275.15, 276.15]], dtype=np.float32)
        domain_mask = np.asarray([[True, False], [False, True]])
        packed, clipped = archive._pack_field(source, spec, domain_mask)
        self.assertEqual(clipped, 0)
        self.assertEqual(packed[0, 1], archive.FILL_VALUE)
        self.assertEqual(packed[1, 0], archive.FILL_VALUE)
        np.testing.assert_allclose(
            archive.unpack_field(packed, spec)[domain_mask],
            source[domain_mask],
            atol=0.026,
        )

    def test_f000_excludes_precipitation_fields(self) -> None:
        keys = {spec.key for spec in archive.model_field_specs(0)}
        self.assertNotIn("precip_rate", keys)
        self.assertNotIn("precip_accum", keys)
        later_keys = {spec.key for spec in archive.model_field_specs(3)}
        self.assertIn("precip_rate", later_keys)
        self.assertIn("precip_accum", later_keys)

    def test_cleanup_preserves_any_continental_run_with_incomplete_archives(self) -> None:
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
            self.assertTrue((data_dir / "20260710T06Z").exists())
            self.assertTrue((data_dir / "20260710T12Z").exists())

    def test_hourly_lpi_archive_packs_raw_tuning_ingredients(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            run = archive.hrdps.RunInfo(
                cycle="06",
                stamp="20260710T06Z",
                init_time=dt.datetime(2026, 7, 10, 6, tzinfo=UTC),
            )
            lat = np.arange(16, dtype=np.float32).reshape(4, 4) + 45.0
            lon = np.arange(16, dtype=np.float32).reshape(4, 4) - 140.0
            fields = SimpleNamespace(
                **{
                    spec.attribute: np.full((4, 4), index + 0.25, dtype=np.float32)
                    for index, spec in enumerate(archive.HOURLY_LPI_INGREDIENT_SPECS)
                }
            )

            output = archive.archive_hourly_lpi_ingredients(
                root,
                run,
                7,
                lat,
                lon,
                fields,
                stride=2,
                formula_version="test_formula",
            )

            self.assertTrue(output.exists())
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["shape"], [2, 2])
            self.assertEqual(sidecar["formula_version"], "test_formula")
            with np.load(output) as packed:
                self.assertEqual(int(packed["forecast_hour"][0]), 7)
                self.assertEqual(str(packed["formula_version"].item()), "test_formula")
                self.assertEqual(packed["mu_li"].shape, (2, 2))
            manifest = json.loads((output.parent / "manifest.json").read_text())
            self.assertEqual(manifest["archived_hours"], [7])
            self.assertFalse(manifest["complete"])

    def test_hourly_precipitation_rate_range_covers_extreme_convection(self) -> None:
        spec = next(spec for spec in archive.HOURLY_LPI_INGREDIENT_SPECS if spec.key == "precip_rate")
        self.assertGreaterEqual(32767 * spec.scale + spec.offset, 300.0)


if __name__ == "__main__":
    unittest.main()
