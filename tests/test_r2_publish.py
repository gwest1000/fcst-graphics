from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from r2_publish import PNG_END_MARKER, PublishState, build_manifest, is_complete_png, object_key_for


class R2PublishTests(unittest.TestCase):
    def test_object_keys_separate_retention_classes(self):
        forecast = object_key_for("west", "convective", "20260720T12Z", "frame.png")
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
                ("key", "west", "convective", "20260720T12Z", 0, "x", 1, 1, "hash", "v", "now"),
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


if __name__ == "__main__":
    unittest.main()
