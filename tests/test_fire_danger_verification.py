from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image

import fire_danger_verification as verification
import automate_fire_danger_verification as automation
import sync_fire_danger_bcws_inputs as mirror


class FireDangerVerificationTest(unittest.TestCase):
    def test_recent_bcws_inputs_are_mirrored_for_launch_agents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            station = source / "observations/bcws/datamart/stations/current_stations.csv"
            recent = source / "observations/bcws/datamart/daily/2026/2026-07-13.csv"
            old = source / "observations/bcws/datamart/daily/2026/2026-05-01.csv"
            snapshot = source / "observations/bcws/danger_summaries/snapshots/2026/07/14/test.json"
            for path in (station, recent, old, snapshot):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name)

            copied, considered = mirror.sync_inputs(
                source,
                target,
                as_of=dt.date(2026, 7, 14),
                keep_days=45,
            )
            self.assertEqual((copied, considered), (3, 3))
            self.assertTrue((target / recent.relative_to(source)).exists())
            self.assertTrue((target / snapshot.relative_to(source)).exists())
            self.assertFalse((target / old.relative_to(source)).exists())

    def test_dashboard_source_signature_tracks_archived_inputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "archive"
            concrete = root / "concrete"
            before = automation.source_signature("20260714T12Z", archive, concrete)
            forecast = archive / "forecasts/continental/20260714T12Z/2026-07-14.csv"
            forecast.parent.mkdir(parents=True)
            forecast.write_text("station_code\n001\n")
            after = automation.source_signature("20260714T12Z", archive, concrete)
            self.assertEqual(before["forecast_mtime_ns"], 0)
            self.assertGreater(after["forecast_mtime_ns"], 0)

            state_path = root / "state.json"
            automation.save_state(state_path, after)
            self.assertEqual(automation.load_state(state_path), after)

    def test_station_mapping_is_rebuilt_when_grid_shape_changes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            concrete = root / "concrete"
            station_path = concrete / "observations/bcws/datamart/stations/current_stations.csv"
            station_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "STATION_CODE": ["001"],
                    "STATION_NAME": ["Test Station"],
                    "LATITUDE": [49.0],
                    "LONGITUDE": [-123.0],
                }
            ).to_csv(station_path, index=False)
            cache = root / "cache"

            mapping = verification.station_grid_mapping(
                "continental",
                np.array([[49.0, 49.1], [49.2, 49.3]]),
                np.array([[-123.0, -122.9], [-122.8, -122.7]]),
                cache,
                concrete,
            )
            self.assertEqual((mapping.grid_shape_y.iloc[0], mapping.grid_shape_x.iloc[0]), (2, 2))

            mapping = verification.station_grid_mapping(
                "continental",
                np.full((3, 3), 49.0),
                np.full((3, 3), -123.0),
                cache,
                concrete,
            )
            self.assertEqual((mapping.grid_shape_y.iloc[0], mapping.grid_shape_x.iloc[0]), (3, 3))

    def test_forecast_observation_matching_and_dashboard(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "archive"
            concrete = root / "concrete"
            forecast_path = archive / "forecasts/continental/20260712T12Z/2026-07-13.csv"
            forecast_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "station_code": ["001"],
                    "model_key": ["continental"],
                    "model_label": ["HRDPS 2.5 km"],
                    "run_stamp": ["20260712T12Z"],
                    "run_init_utc": ["2026-07-12T12:00:00+00:00"],
                    "fire_date": ["2026-07-13"],
                    "lead_hours": [32.0],
                    "forecast_danger_class": [4],
                }
            ).to_csv(forecast_path, index=False)
            observed_path = concrete / "observations/bcws/datamart/daily/2026/2026-07-13.csv"
            observed_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "DATE_TIME": [2026071312],
                    "STATION_CODE": ["001"],
                    "STATION_NAME": ["Test Station"],
                    "DANGER_RATING": [3],
                    "FIRE_WEATHER_INDEX": [18.0],
                    "BUILDUP_INDEX": [65.0],
                }
            ).to_csv(observed_path, index=False)

            matched = verification.matched_verification_frame(archive, concrete)
            self.assertEqual(len(matched), 1)
            self.assertEqual(matched.lead_group.iloc[0], "Day 1")
            self.assertEqual(float(matched.error.iloc[0]), 1.0)

            dashboard = root / "dashboard.png"
            verification.render_dashboard(dashboard, matched, as_of=dt.date(2026, 7, 14))
            self.assertTrue(dashboard.exists())
            with Image.open(dashboard) as image:
                self.assertEqual(image.size, (1440, 900))


if __name__ == "__main__":
    unittest.main()
