from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from unittest import mock

from r2_publish import (
    FORECAST_KEEP_DAYS,
    FORECAST_FULL_RUN_KEEP_HOURS,
    FORECAST_RETAINED_CYCLES,
    PNG_END_MARKER,
    PublishState,
    R2Config,
    RETIRED_PRODUCTS,
    VERIFICATION_KEEP_DAYS,
    build_manifest,
    delete_object_keys,
    discover_frames,
    is_complete_png,
    manifest_content_sha256,
    object_key_for,
    publish_model,
    purge_retired_objects,
    run_is_retained,
)
from publish_hrdps_west import PRODUCTS, image_name_for_hour, minimum_manifest_hours


class R2PublishTests(unittest.TestCase):
    def test_public_retention_is_seven_day_forecast_and_two_week_verification(self):
        self.assertEqual(FORECAST_KEEP_DAYS, 7)
        self.assertEqual(VERIFICATION_KEEP_DAYS, 14)

    def test_extended_ecmwf_product_retains_legacy_complete_runs(self):
        self.assertEqual(PRODUCTS["ecmwf_control_fourpanel"].hours[-1], 360)
        self.assertEqual(minimum_manifest_hours("ecmwf_control_fourpanel"), 17)

    def test_object_keys_separate_retention_classes(self):
        forecast = object_key_for("continental", "continental_fourpanel", "20260720T12Z", "frame.png")
        verification = object_key_for("continental", "continental_lightning_verif", "20260720T12Z", "frame.png")
        self.assertIn("/forecast/", forecast)
        self.assertIn("/verification/", verification)

    def test_empty_manifest_is_valid(self):
        manifest = build_manifest(
            "continental",
            [],
            "https://assets.example.com",
            generated=dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc),
        )
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["model"], "continental")
        self.assertEqual(manifest["runs"], [])

    def test_manifest_hash_ignores_only_generation_time(self):
        first = {"generated": "2026-01-01T00:00:00Z", "runs": [{"init": "a"}]}
        second = {"generated": "2026-01-01T00:03:00Z", "runs": [{"init": "a"}]}
        changed = {"generated": "2026-01-01T00:03:00Z", "runs": [{"init": "b"}]}
        self.assertEqual(manifest_content_sha256(first), manifest_content_sha256(second))
        self.assertNotEqual(manifest_content_sha256(first), manifest_content_sha256(changed))

    def test_noop_publish_does_not_reupload_unchanged_manifest(self):
        client = mock.Mock()
        config = R2Config("account", "access", "secret", "bucket", "https://assets.example.com")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "r2_publish.candidate_stamps", return_value=[]
        ):
            state_path = Path(tmp) / "state.sqlite3"
            first = publish_model(
                "continental", sync_retained=True, config=config, state_path=state_path, client=client
            )
            second = publish_model(
                "continental", sync_retained=True, config=config, state_path=state_path, client=client
            )
        self.assertTrue(first["manifest_uploaded"])
        self.assertFalse(second["manifest_uploaded"])
        self.assertEqual(client.put_object.call_count, 1)

    def test_publish_state_initializes_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = PublishState(Path(tmp) / "state.sqlite3")
            try:
                self.assertEqual(state.retained_rows("west"), [])
            finally:
                state.close()

    def test_publish_state_resets_when_bucket_scope_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite3"
            first = PublishState(path, storage_scope="account/first")
            first.connection.execute(
                """
                INSERT INTO artifacts (
                    object_key, model, product_key, stamp, forecast_hour, source_path,
                    size_bytes, mtime_ns, sha256, format_version, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("key", "west", "lightning_sw", "20260720T12Z", 0, "x", 1, 1, "hash", "v", "now"),
            )
            first.connection.commit()
            first.close()

            second = PublishState(path, storage_scope="account/second")
            try:
                count = second.connection.execute("SELECT count(*) FROM artifacts").fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                second.close()

    def test_complete_png_requires_iend_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            path.write_bytes(b"not-finished")
            self.assertFalse(is_complete_png(path))
            path.write_bytes(b"png-payload" + PNG_END_MARKER)
            self.assertTrue(is_complete_png(path))

    def test_retired_products_are_removed_from_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = PublishState(Path(tmp) / "state.sqlite3")
            try:
                for object_key, product_key in (("active", "continental_fourpanel"), ("old", "continental_convective")):
                    state.connection.execute(
                        """
                        INSERT INTO artifacts (
                            object_key, model, product_key, stamp, forecast_hour, source_path,
                            size_bytes, mtime_ns, sha256, format_version, uploaded_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            object_key,
                            "continental",
                            product_key,
                            "20260720T12Z",
                            0,
                            "x",
                            1,
                            1,
                            "hash",
                            "v",
                            "now",
                        ),
                    )
                state.connection.commit()
                removed = state.prune_inactive_products("continental", ("continental_fourpanel",))
                products = {
                    row[0]
                    for row in state.connection.execute(
                        "SELECT product_key FROM artifacts WHERE model = 'continental'"
                    )
                }
            finally:
                state.close()
        self.assertEqual(removed, 1)
        self.assertEqual(products, {"continental_fourpanel"})

    def test_retired_r2_prefixes_are_deleted(self):
        client = mock.Mock()
        client.list_objects_v2.side_effect = lambda **request: {
            "Contents": [{"Key": f"{request['Prefix']}old.png"}],
            "IsTruncated": False,
        }
        config = R2Config("account", "access", "secret", "bucket", "https://assets.example.com")

        deleted = purge_retired_objects(client, config, "continental")

        self.assertEqual(deleted, len(RETIRED_PRODUCTS["continental"]))
        self.assertEqual(
            {
                call.kwargs["Prefix"]
                for call in client.list_objects_v2.call_args_list
            },
            {
                f"models/continental/forecast/{product_key}/"
                for product_key in RETIRED_PRODUCTS["continental"]
            },
        )
        self.assertEqual(
            client.delete_objects.call_count,
            len(RETIRED_PRODUCTS["continental"]),
        )

    def test_delete_object_keys_batches_and_rejects_partial_failures(self):
        config = R2Config("account", "access", "secret", "bucket", "https://assets.example.com")
        client = mock.Mock()
        client.delete_objects.return_value = {}

        self.assertEqual(delete_object_keys(client, config, (f"key-{i}" for i in range(1001))), 1001)
        self.assertEqual(client.delete_objects.call_count, 2)

        client.reset_mock()
        client.delete_objects.return_value = {
            "Errors": [{"Key": "bad", "Code": "AccessDenied"}],
        }
        with self.assertRaisesRegex(RuntimeError, "AccessDenied"):
            delete_object_keys(client, config, ["bad"])

    def test_manifest_is_committed_before_expired_objects_are_deleted(self):
        events = []
        client = mock.Mock()
        client.put_object.side_effect = lambda **kwargs: events.append(("put", kwargs["Key"])) or {}
        client.delete_objects.side_effect = (
            lambda **kwargs: events.append(
                ("delete", tuple(item["Key"] for item in kwargs["Delete"]["Objects"]))
            )
            or {}
        )
        config = R2Config("account", "access", "secret", "bucket", "https://assets.example.com")
        now = dt.datetime.now(dt.timezone.utc)
        old_stamp = (now - dt.timedelta(days=FORECAST_KEEP_DAYS + 2)).strftime("%Y%m%dT%HZ")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.sqlite3"
            state = PublishState(state_path, storage_scope="account/bucket")
            state.connection.execute(
                """
                INSERT INTO artifacts (
                    object_key, model, product_key, stamp, forecast_hour, source_path,
                    size_bytes, mtime_ns, sha256, format_version, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "models/continental/forecast/continental_fourpanel/old/frame.png",
                    "continental",
                    "continental_fourpanel",
                    old_stamp,
                    0,
                    "x",
                    1,
                    1,
                    "hash",
                    "v",
                    "now",
                ),
            )
            state.connection.commit()
            state.close()

            with mock.patch("r2_publish.candidate_stamps", return_value=[]):
                result = publish_model(
                    "continental",
                    sync_retained=True,
                    config=config,
                    state_path=state_path,
                    client=client,
                )

            state = PublishState(state_path, storage_scope="account/bucket")
            try:
                self.assertEqual(state.connection.execute("SELECT count(*) FROM artifacts").fetchone()[0], 0)
            finally:
                state.close()

        self.assertEqual(result["remote_deleted"], 1)
        self.assertLess(events.index(("put", "manifests/continental.json")), next(
            index for index, event in enumerate(events) if event[0] == "delete"
        ))

    def test_retained_sync_filters_each_product_independently(self):
        now = dt.datetime(2026, 7, 20, 12, tzinfo=dt.timezone.utc)
        stamp = "20260710T12Z"
        with tempfile.TemporaryDirectory() as tmp:
            roots = {
                "continental_fourpanel": Path(tmp) / "forecast",
                "continental_lightning_verif": Path(tmp) / "verification",
            }
            for key, root in roots.items():
                hour = PRODUCTS[key].hours[0]
                run_dir = root / stamp
                run_dir.mkdir(parents=True)
                (run_dir / image_name_for_hour(stamp, key, hour)).touch()
            with mock.patch("r2_publish.MODEL_PRODUCTS", {"continental": tuple(roots)}), mock.patch(
                "r2_publish.source_root", side_effect=lambda key: roots[key]
            ):
                frames = discover_frames(
                    "continental",
                    [stamp],
                    enforce_retention=True,
                    now=now,
                )
        self.assertEqual([frame.product_key for frame in frames], ["continental_lightning_verif"])

    def test_forecast_runs_are_thinned_by_cycle_after_72_hours(self):
        now = dt.datetime(2026, 9, 3, 12, tzinfo=dt.timezone.utc)
        self.assertEqual(FORECAST_FULL_RUN_KEEP_HOURS, 72)
        self.assertEqual(FORECAST_RETAINED_CYCLES, {0, 6, 12})
        self.assertTrue(run_is_retained("continental_fourpanel", now - dt.timedelta(hours=70), now))
        self.assertTrue(run_is_retained("continental_fourpanel", dt.datetime(2026, 8, 30, 6, tzinfo=dt.timezone.utc), now))
        self.assertFalse(run_is_retained("continental_fourpanel", dt.datetime(2026, 8, 30, 18, tzinfo=dt.timezone.utc), now))
        self.assertTrue(run_is_retained("continental_lightning_verif", dt.datetime(2026, 8, 25, 18, tzinfo=dt.timezone.utc), now))


if __name__ == "__main__":
    unittest.main()
