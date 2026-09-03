from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from reconcile_r2_objects import classify_object, reconcile
from r2_publish import R2Config


class FakeR2:
    def __init__(self, objects: dict[str, int]):
        self.objects = dict(objects)

    def list_objects_v2(self, **_kwargs):
        return {
            "Contents": [{"Key": key, "Size": size} for key, size in self.objects.items()],
            "IsTruncated": False,
        }

    def delete_objects(self, **kwargs):
        for item in kwargs["Delete"]["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}


class ReconcileR2ObjectsTests(unittest.TestCase):
    def setUp(self):
        self.now = dt.datetime(2026, 9, 2, 12, tzinfo=dt.timezone.utc)

    def test_policy_uses_model_initialization_not_upload_age(self):
        self.assertEqual(
            classify_object(
                "models/continental/forecast/continental_fourpanel/20260820T12Z/frame.png", self.now
            ),
            "expired",
        )
        self.assertEqual(
            classify_object(
                "models/continental/verification/continental_lightning_verif/20260820T12Z/frame.png", self.now
            ),
            "retained",
        )

    def test_old_18z_forecast_is_thinned_but_12z_is_retained(self):
        self.assertEqual(
            classify_object(
                "models/continental/forecast/continental_fourpanel/20260829T18Z/frame.png",
                self.now,
            ),
            "expired",
        )
        self.assertEqual(
            classify_object(
                "models/continental/forecast/continental_fourpanel/20260829T12Z/frame.png",
                self.now,
            ),
            "retained",
        )

    def test_unknown_products_are_reported_but_not_deleted(self):
        unknown = "models/future/forecast/future_product/20260801T12Z/frame.png"
        expired = "models/continental/forecast/continental_fourpanel/20260801T12Z/frame.png"
        client = FakeR2({unknown: 5, expired: 7})
        config = R2Config("account", "access", "secret", "bucket", "https://example.com")

        with tempfile.TemporaryDirectory() as temporary:
            result = reconcile(
                client,
                config,
                apply=True,
                now=self.now,
                status_path=Path(temporary) / "status.json",
            )

        self.assertEqual(result["deleted"], 1)
        self.assertIn(unknown, client.objects)
        self.assertNotIn(expired, client.objects)

    def test_decommissioned_west_objects_are_deleted(self):
        key = "models/west/forecast/lightning_sw/20260901T12Z/frame.png"
        client = FakeR2({key: 7})
        config = R2Config("account", "access", "secret", "bucket", "https://example.com")
        with tempfile.TemporaryDirectory() as temporary:
            result = reconcile(
                client,
                config,
                apply=True,
                now=self.now,
                status_path=Path(temporary) / "status.json",
            )
        self.assertEqual(result["deleted"], 1)
        self.assertNotIn(key, client.objects)


if __name__ == "__main__":
    unittest.main()
