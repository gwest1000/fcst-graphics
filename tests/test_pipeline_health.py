from __future__ import annotations

import datetime as dt
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import monitor_pipeline_health as health
import telegram_notify


NOW = dt.datetime(2026, 8, 17, 22, tzinfo=dt.timezone.utc)


class PipelineHealthTests(unittest.TestCase):
    def test_manual_invocation_is_isolated_from_operational_state(self):
        with mock.patch.dict("os.environ", {"TMPDIR": "/tmp"}):
            args = health.isolate_diagnostic_invocation(health.parse_args([]))
        self.assertTrue(args.no_notify)
        self.assertTrue(args.no_auto_repair)
        self.assertNotEqual(args.state_path, health.STATE_PATH)
        self.assertIn("fcst-health-diagnostic-", str(args.state_path))

    def test_operational_invocation_keeps_configured_state(self):
        args = health.isolate_diagnostic_invocation(
            health.parse_args(["--operational"])
        )
        self.assertFalse(args.no_notify)
        self.assertEqual(args.state_path, health.STATE_PATH)

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

    def test_unchanged_manifest_is_not_mistaken_for_failed_publishing(self):
        spec = health.ManifestSpec("ecmwf_control", "ECMWF", dt.timedelta(hours=30))
        payload = {
            "generated": "2026-08-15T00:00:00Z",
            "runs": [{"stamp": "20260817T00Z", "init": "2026-08-17T00:00:00Z"}],
        }
        result = health.check_model_manifest(
            spec,
            dt.datetime(2026, 8, 17, 20, tzinfo=dt.timezone.utc),
            "https://example.test",
            lambda _url: payload,
        )
        self.assertEqual(result.level, "ok")
        self.assertIn("manifest 2.8 d old", result.summary)

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

    def test_transient_failures_are_debounced_but_unrepaired_services_are_immediate(self):
        remote = health.CheckResult("manifest.west", "West", "critical", "offline")
        service = health.CheckResult("service.x", "Service", "critical", "missing", immediate=True)
        checks, debounce = health.apply_debounce([remote, service], {}, NOW)
        self.assertEqual(checks[0].level, "ok")
        self.assertEqual(checks[1].level, "critical")
        checks, _ = health.apply_debounce(
            [remote],
            debounce,
            NOW + dt.timedelta(minutes=34),
        )
        self.assertEqual(checks[0].level, "ok")
        checks, _ = health.apply_debounce(
            [remote],
            debounce,
            NOW + dt.timedelta(minutes=56),
        )
        self.assertEqual(checks[0].level, "critical")

    @mock.patch("monitor_pipeline_health.launchctl_print")
    def test_service_exit_code_is_explained_and_debounced(self, launchctl_print):
        launchctl_print.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="state = not running\nlast exit code = 75\n",
            stderr="",
        )
        result = health.check_launch_agent("com.greg.fcst-fire-activity-overlay")
        self.assertEqual(result.label, "Hourly active-fire overlay")
        self.assertFalse(result.immediate)
        self.assertFalse(result.push_eligible)
        self.assertIn("temporary failure", result.summary)

    @mock.patch("monitor_pipeline_health.launchctl_print")
    def test_running_service_overrides_previous_exit_failure(self, launchctl_print):
        launchctl_print.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="state = running\nlast exit code = 1\n",
            stderr="",
        )
        result = health.check_launch_agent(
            "com.greg.hrdps-continental-06",
            auto_repair=False,
        )
        self.assertEqual(result.level, "ok")
        self.assertIn("currently running", result.summary)

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
        self.assertIn("[CRITICAL] Fires", body)
        self.assertIn("Issue: feed unavailable", body)
        self.assertIn("Active-fire symbols may be outdated", body)
        self.assertIn("Action: Check the BCWS feed", body)
        self.assertIn("Disk: 887 GB free (44%)", body)

    def test_shared_manifest_failure_is_grouped(self):
        checks = [
            health.CheckResult(f"manifest.{key}", label, "critical", "manifest check failed: timeout")
            for key, label in (("west", "West"), ("continental", "Continental"), ("gefs", "GEFS"))
        ]
        body = health.report_body(checks, NOW, daily=False)
        self.assertIn("[CRITICAL] Public forecast graphics access", body)
        self.assertIn("3/4 model manifests could not be read", body)
        self.assertNotIn("[CRITICAL] West", body)
        four_checks = [
            *checks,
            health.CheckResult("manifest.ecmwf", "ECMWF", "critical", "manifest check failed: timeout"),
        ]
        self.assertEqual(health.problem_signature(checks), health.problem_signature(four_checks))

    def test_hrdps_pipeline_reports_partial_as_degraded(self):
        spec = health.PipelineStatusSpec("continental", "12", "HRDPS 12Z")
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "continental_12.status.json"
            path.write_text(json.dumps({
                "status": "degraded",
                "updated_at_utc": NOW.isoformat(),
                "heartbeat_at_utc": NOW.isoformat(),
                "stamp": "20260817T12Z",
                "expected_hours": 17,
                "ready_hours": list(range(12)),
                "rendered_hours": list(range(10)),
            }))
            result = health.check_hrdps_pipeline_status(
                spec, NOW, Path(tmpdir), pid_checker=lambda _pid: True
            )
        self.assertEqual(result.level, "warning")
        self.assertIn("rendered 10/17", result.summary)

    def test_hrdps_pipeline_flags_stalled_heartbeat(self):
        spec = health.PipelineStatusSpec("continental", "12", "HRDPS 12Z")
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "continental_12.status.json"
            path.write_text(json.dumps({
                "status": "waiting_upstream",
                "updated_at_utc": (NOW - dt.timedelta(hours=2)).isoformat(),
                "heartbeat_at_utc": (NOW - dt.timedelta(hours=2)).isoformat(),
                "expected_stamp": "20260817T12Z",
            }))
            result = health.check_hrdps_pipeline_status(
                spec, NOW, Path(tmpdir), pid_checker=lambda _pid: True
            )
        self.assertEqual(result.level, "critical")
        self.assertIn("heartbeat 2.0 h old", result.summary)

    def test_interrupted_hrdps_attempt_is_daily_diagnostic(self):
        spec = health.PipelineStatusSpec("continental", "06", "HRDPS 06Z")
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "continental_06.status.json"
            path.write_text(json.dumps({
                "status": "waiting_upstream",
                "pid": 1234,
                "updated_at_utc": (NOW - dt.timedelta(hours=4)).isoformat(),
                "heartbeat_at_utc": (NOW - dt.timedelta(hours=4)).isoformat(),
                "expected_stamp": "20260817T06Z",
            }))
            result = health.check_hrdps_pipeline_status(
                spec, NOW, Path(tmpdir), pid_checker=lambda _pid: False
            )
        self.assertEqual(result.level, "warning")
        self.assertFalse(result.push_eligible)
        self.assertIn("was interrupted", result.summary)

    def test_systemic_scheduler_failure_is_grouped(self):
        checks = [
            health.CheckResult(
                f"service.{number}",
                f"Service {number}",
                "critical",
                "schedule is not loaded and automatic repair failed",
                immediate=True,
            )
            for number in range(4)
        ]
        problems = health.notification_problems(checks)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].key, "group.schedulers")
        self.assertIn("4 schedules", problems[0].summary)

    def test_notification_reminders_and_recovery_hold(self):
        state = {
            "last_notification_at": (NOW - dt.timedelta(hours=5)).isoformat(),
        }
        reason, _ = health.notification_decision(
            signature="a:critical",
            notified_signature="a:critical",
            level="critical",
            now=NOW,
            previous_state=state,
            always_notify=False,
        )
        self.assertIsNone(reason)

        reason, _ = health.notification_decision(
            signature="a:critical",
            notified_signature="a:critical",
            level="critical",
            now=NOW + dt.timedelta(hours=2),
            previous_state=state,
            always_notify=False,
        )
        self.assertEqual(reason, "reminder")

        reason, recovering = health.notification_decision(
            signature="",
            notified_signature="a:critical",
            level="ok",
            now=NOW,
            previous_state={},
            always_notify=False,
        )
        self.assertIsNone(reason)
        reason, _ = health.notification_decision(
            signature="",
            notified_signature="a:critical",
            level="ok",
            now=NOW + dt.timedelta(minutes=56),
            previous_state={"recovery_since": recovering},
            always_notify=False,
        )
        self.assertEqual(reason, "recovery")

    def test_partial_recovery_and_severity_decrease_do_not_alert(self):
        reason, _ = health.notification_decision(
            signature="b:warning",
            notified_signature="a:critical|b:warning",
            level="warning",
            now=NOW,
            previous_state={"last_notification_at": NOW.isoformat()},
            always_notify=False,
        )
        self.assertIsNone(reason)

        reason, _ = health.notification_decision(
            signature="a:warning",
            notified_signature="a:critical",
            level="warning",
            now=NOW,
            previous_state={"last_notification_at": NOW.isoformat()},
            always_notify=False,
        )
        self.assertIsNone(reason)

    def test_new_problem_and_severity_increase_alert(self):
        reason, _ = health.notification_decision(
            signature="a:warning|b:warning",
            notified_signature="a:warning",
            level="warning",
            now=NOW,
            previous_state={"last_notification_at": NOW.isoformat()},
            always_notify=False,
        )
        self.assertEqual(reason, "alert")

        reason, _ = health.notification_decision(
            signature="a:critical",
            notified_signature="a:warning",
            level="critical",
            now=NOW,
            previous_state={"last_notification_at": NOW.isoformat()},
            always_notify=False,
        )
        self.assertEqual(reason, "alert")

    def test_historical_service_failure_is_daily_only(self):
        checks = [
            health.CheckResult(
                "service.a",
                "Scheduled job",
                "warning",
                "last scheduled attempt failed",
                push_eligible=False,
            )
        ]
        self.assertEqual(health.problem_signature(checks), "")
        self.assertNotIn("Scheduled job", health.report_body(checks, NOW, daily=False))
        self.assertIn("Scheduled job", health.report_body(checks, NOW, daily=True))

    def test_history_keeps_an_auditable_json_line(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            health.append_history(path, {"updated": "now", "problem": "example"})
            self.assertEqual(json.loads(path.read_text()), {"updated": "now", "problem": "example"})

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
