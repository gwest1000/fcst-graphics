from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest import mock

import monitor_pipeline_health as health
import telegram_notify


NOW = dt.datetime(2026, 8, 17, 22, tzinfo=dt.timezone.utc)


class PipelineHealthTests(unittest.TestCase):
    def test_model_manifest_uses_run_and_generation_age(self):
        spec = health.ManifestSpec("continental", "HRDPS", dt.timedelta(hours=12))
        payload = {
            "generated": "2026-08-17T21:55:00Z",
            "runs": [{"stamp": "20260817T18Z", "init": "2026-08-17T18:00:00Z"}],
        }
        result = health.check_model_manifest(spec, NOW, "https://example.test", lambda _url: payload)
        self.assertEqual(result.level, "ok")
        self.assertIn("20260817T18Z", result.summary)

        payload["runs"][0]["init"] = "2026-08-16T00:00:00Z"
        result = health.check_model_manifest(spec, NOW, "https://example.test", lambda _url: payload)
        self.assertEqual(result.level, "critical")

    def test_fire_manifest_flags_stale_observations(self):
        payload = {
            "available": True,
            "generated": "2026-08-17T21:55:00Z",
            "observationTime": "2026-08-17T17:00:00Z",
            "observationCount": 120,
            "stale": False,
        }
        result = health.check_fire_manifest(NOW, "https://example.test", lambda _url: payload)
        self.assertEqual(result.level, "critical")

    def test_remote_failures_are_debounced_but_services_are_immediate(self):
        remote = health.CheckResult("manifest.west", "West", "critical", "offline")
        service = health.CheckResult("service.x", "Service", "critical", "missing", immediate=True)
        checks, counts = health.apply_debounce([remote, service], {})
        self.assertEqual(checks[0].level, "ok")
        self.assertEqual(checks[1].level, "critical")
        checks, _ = health.apply_debounce([remote], {"failure_counts": counts})
        self.assertEqual(checks[0].level, "critical")

    def test_problem_signature_is_stable(self):
        checks = [
            health.CheckResult("b", "B", "warning", "late"),
            health.CheckResult("a", "A", "critical", "missing"),
        ]
        self.assertEqual(health.problem_signature(checks), "a:critical|b:warning")

    def test_healthy_daily_report_only_includes_count_and_disk(self):
        checks = [
            health.CheckResult("storage.runtime", "Storage", "ok", "887 GB free (44%)"),
            health.CheckResult("feed.fire_activity", "Fires", "ok", "20 incidents"),
        ]
        body = health.report_body(checks, NOW, daily=True)
        self.assertEqual(body, "All 2/2 checks are healthy.\nDisk: 887 GB free (44%)")

    def test_unhealthy_daily_report_includes_problem_details(self):
        checks = [
            health.CheckResult("storage.runtime", "Storage", "ok", "887 GB free (44%)"),
            health.CheckResult("feed.fire_activity", "Fires", "critical", "feed unavailable"),
        ]
        body = health.report_body(checks, NOW, daily=True)
        self.assertIn("[CRITICAL] Fires: feed unavailable", body)
        self.assertIn("Disk: 887 GB free (44%)", body)

    @mock.patch("telegram_notify.urllib.request.urlopen")
    def test_telegram_client_uses_existing_environment_contract(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        urlopen.return_value = response
        with mock.patch("telegram_notify.json.load", return_value={"ok": True}):
            telegram_notify.send_message(
                "Title",
                "Body",
                environ={"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_CHAT_ID": "123"},
            )
        request = urlopen.call_args.args[0]
        self.assertIn("/botsecret/sendMessage", request.full_url)
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_id"], "123")
        self.assertEqual(payload["text"], "Title\n\nBody")


if __name__ == "__main__":
    unittest.main()
