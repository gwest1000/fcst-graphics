import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests

import fire_activity


class FireActivityTests(unittest.TestCase):
    def test_fire_status_labels_normalize_to_radar_sat_categories(self):
        self.assertEqual(fire_activity.fire_status_key("Out of Control"), "out_of_control")
        self.assertEqual(fire_activity.fire_status_key("BH"), "being_held")
        self.assertEqual(fire_activity.fire_status_key("Under_Control"), "under_control")
        self.assertEqual(fire_activity.fire_status_key("Status unavailable"), "unknown")

    def test_active_fire_parser_excludes_out_incidents_and_marks_fires_of_note(self):
        features = [
            {
                "geometry": {"type": "Point", "coordinates": [-121.45, 49.89]},
                "properties": {
                    "FIRE_STATUS": "Fire of Note",
                    "FIRE_NUMBER": "V10742",
                    "FIRE_OF_NOTE_IND": "Y",
                    "INCIDENT_NAME": "Brunswick Creek",
                    "CURRENT_SIZE": 5389.6,
                },
            },
            {
                "geometry": {"type": "Point", "coordinates": [-120.0, 50.0]},
                "properties": {
                    "FIRE_STATUS": "Out",
                    "FIRE_NUMBER": "K00001",
                },
            },
        ]

        observations = fire_activity.parse_active_fire_features(features)

        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].fire_of_note)
        self.assertEqual(observations[0].name, "Brunswick Creek")
        self.assertAlmostEqual(observations[0].size_hectares, 5389.6)
        self.assertEqual(observations[0].agency, "BCWS")

    def test_us_active_fire_parser_marks_current_ics209_large_incidents(self):
        features = [
            {
                "geometry": None,
                "properties": {
                    "InitialLongitude": -120.5,
                    "InitialLatitude": 48.2,
                    "IncidentName": "Example Fire",
                    "UniqueFireIdentifier": "2026-WA-ABC-123",
                    "IncidentSize": 100.0,
                    "ModifiedOnDateTime_dt": 1780000000000,
                    "ICS209ReportStatus": "U",
                },
            }
        ]

        observations = fire_activity.parse_us_active_fire_features(features)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].agency, "NIFC")
        self.assertTrue(observations[0].is_highlighted)
        self.assertAlmostEqual(
            observations[0].size_hectares,
            100.0 * fire_activity.ACRES_TO_HECTARES,
        )
        self.assertEqual(fire_activity.fire_status_key(observations[0].status), "unknown")

    def test_active_fire_download_combines_bcws_and_nifc(self):
        now = dt.datetime(2026, 9, 9, 18, tzinfo=dt.timezone.utc)
        bc = fire_activity.FireActivity(
            source="bcws_active_fires",
            retrieved_at=now,
            observations=(fire_activity.FireObservation(-121.7, 51.3, "active_fire"),),
        )
        us = fire_activity.FireActivity(
            source="nifc_active_fires",
            retrieved_at=now,
            observations=(fire_activity.FireObservation(-120.5, 48.2, "active_fire"),),
        )
        with (
            mock.patch.object(fire_activity, "download_bc_active_fires", return_value=bc),
            mock.patch.object(fire_activity, "download_us_active_fires", return_value=us),
        ):
            activity = fire_activity.download_active_fires(now)

        self.assertEqual(activity.source, "bcws_nifc_active_fires")
        self.assertEqual(len(activity.observations), 2)
        self.assertTrue(activity.is_active_fire_feed)

    def test_nifc_failure_does_not_suppress_bcws_fires(self):
        now = dt.datetime(2026, 9, 9, 18, tzinfo=dt.timezone.utc)
        bc = fire_activity.FireActivity(
            source="bcws_active_fires",
            retrieved_at=now,
            observations=(fire_activity.FireObservation(-121.7, 51.3, "active_fire"),),
        )
        messages = []
        with (
            mock.patch.object(fire_activity, "download_bc_active_fires", return_value=bc),
            mock.patch.object(
                fire_activity,
                "download_us_active_fires",
                side_effect=requests.Timeout("offline"),
            ),
        ):
            activity = fire_activity.download_active_fires(now, logger=messages.append)

        self.assertEqual(activity, bc)
        self.assertTrue(any("continuing with BCWS" in message for message in messages))

    def test_hotspots_are_clustered_into_plot_scale_cells(self):
        features = [
            {
                "geometry": {"type": "Point", "coordinates": [-121.75, 51.32]},
                "properties": {"frp": 10.0, "rep_date": "2026-07-22T01:00:00Z"},
            },
            {
                "geometry": {"type": "Point", "coordinates": [-121.74, 51.33]},
                "properties": {"frp": 20.0, "rep_date": "2026-07-22T02:00:00Z"},
            },
        ]

        observations = fire_activity.cluster_hotspot_features(features)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].detection_count, 2)
        self.assertAlmostEqual(observations[0].frp_mw, 30.0)
        self.assertEqual(observations[0].observed_at, "2026-07-22T02:00:00Z")

    def test_loader_falls_back_to_hotspots_and_caches_them(self):
        now = dt.datetime(2026, 7, 22, 15, tzinfo=dt.timezone.utc)
        fallback = fire_activity.FireActivity(
            source="cwfis_hotspots_24h",
            retrieved_at=now,
            observations=(
                fire_activity.FireObservation(-121.7, 51.3, "hotspot", detection_count=3),
            ),
        )
        with TemporaryDirectory() as tmpdir:
            active_cache = Path(tmpdir) / "active.json"
            hotspot_cache = Path(tmpdir) / "hotspots.json"
            with (
                mock.patch.object(
                    fire_activity,
                    "download_active_fires",
                    side_effect=requests.Timeout("offline"),
                ),
                mock.patch.object(fire_activity, "download_hotspots", return_value=fallback),
            ):
                result = fire_activity.load_fire_activity(
                    active_cache,
                    hotspot_cache,
                    now=now,
                )

            self.assertEqual(result, fallback)
            self.assertTrue(hotspot_cache.exists())

    def test_fresh_active_cache_avoids_network_request(self):
        now = dt.datetime(2026, 7, 22, 15, tzinfo=dt.timezone.utc)
        activity = fire_activity.FireActivity(
            source="bcws_active_fires",
            retrieved_at=now - dt.timedelta(minutes=10),
            observations=(fire_activity.FireObservation(-121.7, 51.3, "active_fire"),),
        )
        with TemporaryDirectory() as tmpdir:
            active_cache = Path(tmpdir) / "active.json"
            fire_activity.write_cache(activity, active_cache)
            with mock.patch.object(fire_activity, "download_active_fires") as download:
                result = fire_activity.load_fire_activity(
                    active_cache,
                    Path(tmpdir) / "hotspots.json",
                    now=now,
                )

        download.assert_not_called()
        self.assertTrue(result.cached)
        self.assertEqual(len(result.observations), 1)

    def test_recent_active_cache_precedes_hotspots_when_live_feed_fails(self):
        now = dt.datetime(2026, 7, 22, 15, tzinfo=dt.timezone.utc)
        cached = fire_activity.FireActivity(
            source="bcws_active_fires",
            retrieved_at=now - dt.timedelta(hours=2),
            observations=(fire_activity.FireObservation(-121.7, 51.3, "active_fire"),),
        )
        with TemporaryDirectory() as tmpdir:
            active_cache = Path(tmpdir) / "active.json"
            fire_activity.write_cache(cached, active_cache)
            with (
                mock.patch.object(
                    fire_activity,
                    "download_active_fires",
                    side_effect=requests.Timeout("offline"),
                ),
                mock.patch.object(fire_activity, "download_hotspots") as hotspots,
            ):
                result = fire_activity.load_fire_activity(
                    active_cache,
                    Path(tmpdir) / "hotspots.json",
                    now=now,
                )

        hotspots.assert_not_called()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.is_active_fire_feed)
        self.assertTrue(result.cached)
        self.assertTrue(result.stale)


if __name__ == "__main__":
    unittest.main()
