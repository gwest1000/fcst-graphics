from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import make_experimental_danger_class as danger
import automate_hrdps_west as automation
import make_hrdps_west_convective as hrdps
import make_hrdps_evolved_danger_class as evolved
import publish_hrdps_west as publisher


class DangerRegionTest(unittest.TestCase):
    def tearDown(self) -> None:
        hrdps.set_model("continental")

    def test_evolved_danger_is_continental_only(self) -> None:
        self.assertNotIn("west", publisher.PRODUCTS_BY_MODEL)
        self.assertIn("continental_fwi2025_danger", publisher.PRODUCTS_BY_MODEL["continental"])

    def test_fire_danger_verification_is_a_static_product(self) -> None:
        product = publisher.PRODUCTS["fire_danger_verif"]
        self.assertEqual(product.hours, (0,))
        self.assertEqual(publisher.PRODUCTS_BY_MODEL["danger_verif"], ("fire_danger_verif",))

    def test_convective_fourpanel_is_continental_only(self) -> None:
        self.assertNotIn("west", publisher.PRODUCTS_BY_MODEL)
        self.assertIn("continental_fourpanel", publisher.PRODUCTS_BY_MODEL["continental"])

    def test_continental_download_includes_hourly_danger_prerequisites(self) -> None:
        hrdps.set_model("continental")
        run = hrdps.RunInfo("12", "20260710T12Z", hrdps.parse_stamp("20260710T12Z"))
        with TemporaryDirectory() as tmpdir, mock.patch.object(automation.convective, "download_one") as download:
            automation.download_hours(run, Path(tmpdir), (9,), workers=1)

        expected = Path("008") / hrdps.field_name("TMP", "TGL", "2", run.stamp, 8)
        destinations = [call.args[1] for call in download.call_args_list]
        self.assertTrue(any(path.parts[-2:] == expected.parts for path in destinations))

    def test_handoff_checkpoint_matches_model_cycle_interval(self) -> None:
        hrdps.set_model("continental")
        self.assertEqual(evolved.handoff_checkpoint_hour(), 6)

    def test_cwfis_anchor_age_is_limited_only_at_run_initialization(self) -> None:
        init = hrdps.parse_stamp("20260902T12Z")
        self.assertEqual(
            tuple(evolved.candidate_anchor_dates(init)),
            (dt.date(2026, 9, 1), dt.date(2026, 8, 31)),
        )
        self.assertGreater(
            evolved.anchor_age_hours(dt.date(2026, 8, 30), init),
            evolved.MAX_CWFIS_ANCHOR_AGE_AT_INIT_HOURS,
        )

    def test_cwfis_drought_code_range_retains_extreme_values(self) -> None:
        self.assertGreaterEqual(danger.CWFIS_VALUE_LIMITS["dc"][1], 1000.0)

    def test_cwfis_coverage_rejects_partial_analysis(self) -> None:
        reference = np.ones((10, 10), dtype=bool)
        partial = reference.copy()
        partial[:, :3] = False

        with self.assertRaisesRegex(RuntimeError, "retained only 70.0%"):
            danger.validate_cwfis_coverage(partial, reference)

    def test_cwfis_coverage_accepts_small_daily_mask_change(self) -> None:
        reference = np.ones((10, 10), dtype=bool)
        current = reference.copy()
        current[:2, :2] = False

        danger.validate_cwfis_coverage(current, reference)

    def test_cwfis_coverage_rejects_ten_percent_loss(self) -> None:
        reference = np.ones((10, 10), dtype=bool)
        current = reference.copy()
        current[0, :] = False

        with self.assertRaisesRegex(RuntimeError, "minimum 95%"):
            danger.validate_cwfis_coverage(current, reference)

    def test_cwfis_reference_coverage_uses_bc_analysis_domain(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "ffmc"
            previous = root / "20260824" / "cwfis_ffmc_20260824_domain.tif"
            current = root / "20260831" / "cwfis_ffmc_20260831_domain.tif"
            previous.parent.mkdir(parents=True)
            current.parent.mkdir(parents=True)
            previous.touch()
            current.touch()
            bc_mask = np.array([[False, True], [False, True]])
            with mock.patch.object(
                danger,
                "cwfis_valid_mask",
                return_value=bc_mask,
            ) as valid_mask:
                reference = danger.cwfis_reference_mask(current, "ffmc", dt.date(2026, 8, 31))

        np.testing.assert_array_equal(reference, bc_mask)
        valid_mask.assert_called_once_with(previous, "ffmc", analysis_domain_only=True)

    def test_cwfis_validation_compares_current_bc_footprint(self) -> None:
        full_domain = np.ones((2, 2), dtype=bool)
        bc_domain = np.array([[False, True], [False, True]])
        with mock.patch.object(
            danger,
            "cwfis_valid_mask",
            side_effect=(full_domain, bc_domain),
        ) as valid_mask:
            danger.validate_cwfis_geotiff(Path("current.tif"), "ffmc", bc_domain)

        self.assertEqual(valid_mask.call_args_list[1].kwargs, {"analysis_domain_only": True})

    def test_optional_danger_failure_is_reported_without_raising(self) -> None:
        with mock.patch.object(
            automation,
            "render_danger_worker",
            side_effect=RuntimeError("partial CWFIS analysis"),
        ):
            count, error = automation.render_optional_danger_worker(
                "west",
                hrdps.RunInfo("12", "20260831T12Z", hrdps.parse_stamp("20260831T12Z")),
                Path("/data"),
                Path("/plots"),
                (0, 3),
            )

        self.assertEqual(count, 0)
        self.assertEqual(error, "RuntimeError: partial CWFIS analysis")

    def test_incomplete_hourly_bridge_allows_bootstrap_fallback(self) -> None:
        run = hrdps.RunInfo("18", "20260901T18Z", hrdps.parse_stamp("20260901T18Z"))
        state = mock.sentinel.anchor_state
        grid = mock.sentinel.grid
        with mock.patch.object(
            evolved,
            "advance_state",
            side_effect=FileNotFoundError("missing bridge hour"),
        ):
            result = evolved.try_advance_bridge(
                state,
                run,
                Path("/data"),
                grid,
                dt.date(2026, 8, 31),
                hrdps.parse_stamp("20260901T18Z"),
            )

        self.assertIsNone(result)

    def test_missing_cwfis_hours_receive_placeholder_frames(self) -> None:
        hrdps.set_model("continental")
        run = hrdps.RunInfo("12", "20260902T12Z", hrdps.parse_stamp("20260902T12Z"))
        grid = evolved.ModelGrid(
            lat=np.zeros((2, 2), dtype=np.float32),
            lon=np.zeros((2, 2), dtype=np.float32),
            yslice=slice(None),
            xslice=slice(None),
        )
        with (
            TemporaryDirectory() as tmpdir,
            mock.patch.object(evolved, "load_model_grid", return_value=grid),
            mock.patch.object(evolved, "load_transmission_lines", return_value=[]),
            mock.patch.object(evolved, "iter_evolved_fields", return_value=iter(())),
            mock.patch.object(evolved, "plot_unavailable_danger") as placeholder,
        ):
            paths = evolved.make_plots(
                run,
                Path(tmpdir) / "data",
                Path(tmpdir) / "plots",
                Path(tmpdir) / "cache",
                (0, 3),
                watersheds=[],
                allow_bootstrap=True,
            )

        self.assertEqual(len(paths), 2)
        self.assertEqual(placeholder.call_count, 2)

    def test_cwfis_no_data_is_filled_from_nearest_analysis(self) -> None:
        source = np.array([[10.0, np.nan, np.nan], [np.nan, np.nan, 40.0]], dtype=np.float32)
        filled = evolved.fill_nearest_valid(source)
        self.assertTrue(np.isfinite(filled).all())
        self.assertEqual(float(filled[0, 0]), 10.0)
        self.assertEqual(float(filled[1, 2]), 40.0)

    def test_display_smoothing_precedes_schedule_two_classification(self) -> None:
        hrdps.set_model("continental")
        fwi = np.full((7, 7), 46.0, dtype=np.float32)
        fwi[3, 3] = 100.0
        bui = np.full((7, 7), 150.0, dtype=np.float32)
        regions = np.full((7, 7), 3, dtype=np.uint8)

        danger_class = danger.classify_smoothed_danger(fwi, bui, regions, sigma=2.0)

        self.assertTrue(np.equal(danger_class, np.floor(danger_class)).all())
        self.assertTrue(set(np.unique(danger_class)).issubset({4.0, 5.0}))

    def test_schedule_two_matrix_examples(self) -> None:
        fwi = np.array([0.0, 40.0, 60.0], dtype=np.float32)
        bui = np.array([0.0, 100.0, 250.0], dtype=np.float32)
        np.testing.assert_array_equal(danger.classify_region(fwi, bui, 1), [1, 5, 5])
        np.testing.assert_array_equal(danger.classify_region(fwi, bui, 2), [1, 4, 5])
        np.testing.assert_array_equal(danger.classify_region(fwi, bui, 3), [1, 4, 5])

    def test_bcgw_district_unions_match_schedule_one_cities(self) -> None:
        points = {
            "Vancouver": (-123.12, 49.28, 1),
            "Prince George": (-122.75, 53.92, 1),
            "Williams Lake": (-122.14, 52.13, 2),
            "Quesnel": (-122.49, 52.98, 2),
            "100 Mile House": (-121.29, 51.64, 2),
            "Kamloops": (-120.33, 50.67, 3),
            "Kelowna": (-119.50, 49.89, 3),
            "Golden": (-116.97, 51.30, 3),
            "Cranbrook": (-115.77, 49.51, 3),
        }
        lon = np.array([[point[0] for point in points.values()]])
        lat = np.array([[point[1] for point in points.values()]])
        actual = danger.danger_regions(lon, lat).ravel()
        expected = np.array([point[2] for point in points.values()])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
