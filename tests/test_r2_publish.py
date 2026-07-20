from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from r2_publish import PublishState, build_manifest, object_key_for


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


if __name__ == "__main__":
    unittest.main()
