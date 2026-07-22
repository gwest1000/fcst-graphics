from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from unittest import mock

from r2_publish import (
    PNG_END_MARKER,
    PublishState,
    R2Config,
    build_manifest,
    discover_frames,
    is_complete_png,
    object_key_for,
    purge_retired_objects,
)
from publish_hrdps_west import PRODUCTS, image_name_for_hour


class R2PublishTests(unittest.TestCase):
    def test_object_keys_separate_retention_classes(self):
        forecast = object_key_for("west", "lightning_sw", "20260720T12Z", "frame.png")
        verification = object_key_for("west", "lightning_verif", "20260720T12Z", "frame.png")
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
                for object_key, product_key in (("active", "lightning_sw"), ("old", "convective")):
                    state.connection.execute(
                        """
                        INSERT INTO artifacts (
                            object_key, model, product_key, stamp, forecast_hour, source_path,
                            size_bytes, mtime_ns, sha256, format_version, uploaded_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            object_key,
                            "west",
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
                removed = state.prune_inactive_products("west", ("lightning_sw",))
                products = {
                    row[0]
                    for row in state.connection.execute(
                        "SELECT product_key FROM artifacts WHERE model = 'west'"
                    )
                }
            finally:
                state.close()
        self.assertEqual(removed, 1)
        self.assertEqual(products, {"lightning_sw"})

    def test_retired_r2_prefixes_are_deleted(self):
        client = mock.Mock()
        client.list_objects_v2.return_value = {
            "Contents": [{"Key": "models/west/forecast/convective/old.png"}],
            "IsTruncated": False,
        }
        config = R2Config("account", "access", "secret", "bucket", "https://assets.example.com")

        deleted = purge_retired_objects(client, config, "west")

        self.assertEqual(deleted, 1)
        client.list_objects_v2.assert_called_once_with(
            Bucket="bucket", Prefix="models/west/forecast/convective/"
        )
        client.delete_objects.assert_called_once_with(
            Bucket="bucket",
            Delete={
                "Objects": [{"Key": "models/west/forecast/convective/old.png"}],
                "Quiet": True,
            },
        )

    def test_retained_sync_filters_each_product_independently(self):
        now = dt.datetime(2026, 7, 20, 12, tzinfo=dt.timezone.utc)
        stamp = "20260710T12Z"
        with tempfile.TemporaryDirectory() as tmp:
            roots = {
                "lightning_sw": Path(tmp) / "forecast",
                "lightning_verif": Path(tmp) / "verification",
            }
            for key, root in roots.items():
                hour = PRODUCTS[key].hours[0]
                run_dir = root / stamp
                run_dir.mkdir(parents=True)
                (run_dir / image_name_for_hour(stamp, key, hour)).touch()
            with mock.patch("r2_publish.MODEL_PRODUCTS", {"west": tuple(roots)}), mock.patch(
                "r2_publish.source_root", side_effect=lambda key: roots[key]
            ):
                frames = discover_frames(
                    "west",
                    [stamp],
                    enforce_retention=True,
                    now=now,
                )
        self.assertEqual([frame.product_key for frame in frames], ["lightning_verif"])


if __name__ == "__main__":
    unittest.main()
