from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import patch

from monitor_r2_usage import (
    BillingPeriod,
    assess_usage,
    billing_period,
    classify_operations,
    daily_storage_peaks,
    latest_storage,
    main,
    projected_storage_gb_month,
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
            {"dimensions": {"actionType": "PutObject", "bucketName": "forecast"}, "sum": {"requests": 4}},
            {"dimensions": {"actionType": "GetObject", "bucketName": "radar"}, "sum": {"requests": 8}},
            {"dimensions": {"actionType": "DeleteObjects"}, "sum": {"requests": 2}},
            {"dimensions": {"actionType": "FutureOperation"}, "sum": {"requests": 1}},
        ]
        result = classify_operations(groups)
        self.assertEqual(result["class_a"], 4)
        self.assertEqual(result["class_b"], 8)
        self.assertEqual(result["free"], 2)
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["buckets"]["forecast"]["class_a"], 4)
        self.assertEqual(result["buckets"]["radar"]["class_b"], 8)
        self.assertEqual(result["buckets"]["account-level"]["free"], 2)

    def test_projected_usage_warns_before_free_tier(self):
        period = billing_period(dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc), 20)
        midpoint = period.start + (period.end - period.start) / 2
        result = assess_usage(2.0, 400_000, 1_000_000, period, midpoint)
        self.assertEqual(result["level"], "warning")
        self.assertAlmostEqual(result["fractions"]["class_a_projected"], 0.8, places=2)

    def test_storage_has_explicit_operating_headroom(self):
        period = billing_period(dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc), 20)
        now = period.start + dt.timedelta(days=1)
        self.assertEqual(assess_usage(8.4, 0, 0, period, now)["level"], "ok")
        self.assertEqual(assess_usage(8.5, 0, 0, period, now)["level"], "warning")
        self.assertEqual(assess_usage(9.8, 0, 0, period, now)["level"], "critical")

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

    def test_storage_billing_uses_account_daily_peak(self):
        groups = [
            {
                "max": {"payloadSize": 4_000_000_000, "metadataSize": 0},
                "dimensions": {"bucketName": "forecast", "datetime": "2026-07-20T12:00:00Z"},
            },
            {
                "max": {"payloadSize": 5_000_000_000, "metadataSize": 0},
                "dimensions": {"bucketName": "radar", "datetime": "2026-07-20T12:00:00Z"},
            },
            {
                "max": {"payloadSize": 6_000_000_000, "metadataSize": 0},
                "dimensions": {"bucketName": "forecast", "datetime": "2026-07-20T18:00:00Z"},
            },
            {
                "max": {"payloadSize": 5_000_000_000, "metadataSize": 0},
                "dimensions": {"bucketName": "radar", "datetime": "2026-07-20T18:00:00Z"},
            },
        ]
        self.assertEqual(daily_storage_peaks(groups), {"2026-07-20": 11_000_000_000})

    def test_storage_projection_holds_current_footprint_for_future_days(self):
        period = BillingPeriod(
            dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 7, 24, tzinfo=dt.timezone.utc),
        )
        observed, projected = projected_storage_gb_month(
            {"2026-07-20": 12_000_000_000, "2026-07-21": 10_000_000_000},
            period,
            dt.datetime(2026, 7, 21, 20, tzinfo=dt.timezone.utc),
            8_000_000_000,
        )
        self.assertEqual(observed, 11.0)
        self.assertEqual(projected, 9.5)

    @patch("monitor_r2_usage.write_json")
    @patch("monitor_r2_usage.append_history")
    @patch("monitor_r2_usage.notify", return_value=True)
    @patch(
        "monitor_r2_usage.graphql_usage",
        return_value={"operations": [], "storage": []},
    )
    def test_weekly_report_notifies_when_usage_is_ok(
        self,
        _usage,
        notify_mock,
        _append_history,
        _write_json,
    ):
        with patch.dict(
            "os.environ",
            {
                "FCST_R2_ACCOUNT_ID": "account",
                "FCST_R2_BUCKET": "bucket",
                "FCST_CLOUDFLARE_API_TOKEN": "token",
            },
            clear=False,
        ):
            self.assertEqual(main(["--always-notify"]), 0)
        notify_mock.assert_called_once()
        self.assertIn("R2 weekly report", notify_mock.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
