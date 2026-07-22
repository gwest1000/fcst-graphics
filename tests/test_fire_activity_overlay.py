import datetime as dt
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

import fire_activity
import fire_activity_overlay as overlay
from r2_publish import R2Config


class FireActivityOverlayTests(unittest.TestCase):
    def sample_activity(self) -> fire_activity.FireActivity:
        return fire_activity.FireActivity(
            source="bcws_active_fires",
            retrieved_at=dt.datetime(2026, 7, 22, 17, tzinfo=dt.timezone.utc),
            observations=(
                fire_activity.FireObservation(-122.75, 53.92, "active_fire"),
                fire_activity.FireObservation(-123.12, 49.28, "active_fire"),
                fire_activity.FireObservation(-116.0, 50.0, "active_fire", fire_of_note=True),
                fire_activity.FireObservation(-121.0, 56.0, "active_fire"),
            ),
        )

    def test_overlay_matches_frame_dimensions_and_uses_right_panel(self):
        with TemporaryDirectory() as tmpdir:
            path = overlay.render_overlay(
                overlay.OVERLAY_SPECS[0],
                self.sample_activity(),
                Path(tmpdir),
            )
            image = Image.open(path).convert("RGBA")
            alpha = np.asarray(image)[:, :, 3]

        self.assertEqual(image.size, (1440, 900))
        y_pixels, x_pixels = np.nonzero(alpha)
        self.assertGreater(len(x_pixels), 0)
        self.assertGreater(x_pixels.min(), 700)
        self.assertEqual(alpha[:, :700].max(), 0)

    def test_manifest_describes_only_available_live_products(self):
        config = R2Config("account", "access", "secret", "bucket", "https://example.r2.dev")
        activity = self.sample_activity()
        versions = {spec.product_key: f"sha-{index}" for index, spec in enumerate(overlay.OVERLAY_SPECS)}
        manifest = overlay.build_manifest(config, activity, versions)

        self.assertTrue(manifest["available"])
        self.assertEqual(manifest["source"], "bcws_active_fires")
        self.assertEqual(manifest["observationCount"], 4)
        self.assertEqual(set(manifest["products"]), set(versions))
        self.assertTrue(
            manifest["products"]["lightning_sw"]["image"].endswith("/live/fire_activity/lightning_sw.png")
        )
        self.assertEqual(manifest["products"]["lightning_sw"]["minimumRunStamp"], "20260721T12Z")

    def test_unavailable_manifest_clears_products(self):
        config = R2Config("account", "access", "secret", "bucket", "https://example.r2.dev")
        manifest = overlay.build_manifest(config, None, {})

        self.assertFalse(manifest["available"])
        self.assertEqual(manifest["products"], {})
        self.assertIsNone(manifest["observationTime"])

    def test_state_write_is_atomic_and_readable(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            overlay.write_state({"versions": {"lightning_sw": "abc"}}, path)

            self.assertEqual(overlay.read_state(path)["versions"]["lightning_sw"], "abc")
            self.assertEqual(json.loads(path.read_text())["versions"]["lightning_sw"], "abc")

    def test_second_publish_skips_unchanged_pngs_but_updates_manifest(self):
        class FakeClient:
            def __init__(self):
                self.puts = []

            def put_object(self, **kwargs):
                body = kwargs["Body"]
                payload = body.read() if hasattr(body, "read") else body
                self.puts.append((kwargs["Key"], payload))

        config = R2Config("account", "access", "secret", "bucket", "https://example.r2.dev")
        client = FakeClient()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = overlay.publish_overlays(
                self.sample_activity(),
                output_dir=root / "overlays",
                state_path=root / "state.json",
                config=config,
                client=client,
            )
            second = overlay.publish_overlays(
                self.sample_activity(),
                output_dir=root / "overlays",
                state_path=root / "state.json",
                config=config,
                client=client,
            )

        self.assertEqual(first["uploaded"], 4)
        self.assertEqual(second["uploaded"], 0)
        self.assertEqual(second["unchanged"], 4)
        self.assertEqual(sum(key == overlay.MANIFEST_KEY for key, _ in client.puts), 2)

    def test_site_layers_live_fires_only_over_latest_products(self):
        html = (Path(__file__).parents[1] / "site" / "index.html").read_text()

        self.assertIn('id="fireActivityOverlay"', html)
        self.assertIn("followLatestRun &&", html)
        self.assertIn("fireActivityManifest?.available", html)
        self.assertIn("60 * 60 * 1000", html)


if __name__ == "__main__":
    unittest.main()
