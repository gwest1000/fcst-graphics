from __future__ import annotations

import datetime as dt
import unittest

from monitor_r2_usage import (
    assess_usage,
    billing_period,
    classify_operations,
    latest_storage,
)


class R2UsageMonitorTests(unittest.TestCase):
    def test_billing_period_uses_twentieth_anchor(self):
        before = billing_period(dt.datetime(2026, 7, 19, 12, tzinfo=dt.timezone.utc), 20)
        after = billing_period(dt.datetime(2026, 7, 20, 12, tzinfo=dt.timezone.utc), 20)
        self.assertEqual(before.start, dt.datetime(2026, 6, 20, tzinfo=dt.timezone.utc))
        self.assertEqual(before.end, dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc))
        self.assertEqual(after.start, dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc))
        self.assertEqual(after.end, dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc))

    def test_operations_are_classified_for_pricing(self):
        groups = [
            {"dimensions": {"actionType": "PutObject"}, "sum": {"requests": 4}},
            {"dimensions": {"actionType": "GetObject"}, "sum": {"requests": 8}},
            {"dimensions": {"actionType": "DeleteObject"}, "sum": {"requests": 2}},
            {"dimensions": {"actionType": "FutureOperation"}, "sum": {"requests": 1}},
        ]
        result = classify_operations(groups)
        self.assertEqual(result["class_a"], 4)
        self.assertEqual(result["class_b"], 8)
        self.assertEqual(result["free"], 2)
        self.assertEqual(result["unknown"], 1)

    def test_projected_usage_warns_before_free_tier(self):
        period = billing_period(dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc), 20)
        midpoint = period.start + (period.end - period.start) / 2
        result = assess_usage(2.0, 400_000, 1_000_000, period, midpoint)
        self.assertEqual(result["level"], "warning")
        self.assertAlmostEqual(result["fractions"]["class_a_projected"], 0.8, places=2)

    def test_latest_storage_sums_the_latest_value_for_each_bucket(self):
        result = latest_storage(
            [
                {
                    "max": {
                        "objectCount": 12,
                        "uploadCount": 1,
                        "payloadSize": 5000,
                        "metadataSize": 300,
                    },
                    "dimensions": {
                        "bucketName": "fcst-graphics",
                        "datetime": "2026-07-20T21:00:00Z",
                    },
                },
                {
                    "max": {
                        "objectCount": 8,
                        "uploadCount": 0,
                        "payloadSize": 2000,
                        "metadataSize": 100,
                    },
                    "dimensions": {
                        "bucketName": "other",
                        "datetime": "2026-07-20T20:00:00Z",
                    },
                },
                {
                    "max": {
                        "objectCount": 5,
                        "uploadCount": 0,
                        "payloadSize": 1000,
                        "metadataSize": 50,
                    },
                    "dimensions": {
                        "bucketName": "fcst-graphics",
                        "datetime": "2026-07-20T19:00:00Z",
                    },
                },
            ]
        )
        self.assertEqual(result["payload_bytes"], 7000)
        self.assertEqual(result["metadata_bytes"], 400)
        self.assertEqual(result["object_count"], 20)
        self.assertEqual(result["pending_uploads"], 1)
        self.assertEqual(set(result["buckets"]), {"fcst-graphics", "other"})


if __name__ == "__main__":
    unittest.main()
