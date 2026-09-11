from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import monitor_pipeline_health as health
import telegram_notify


NOW = dt.datetime(2026, 8, 17, 22, tzinfo=dt.timezone.utc)


class PipelineHealthTests(unittest.TestCase):
    def scheduled_check(self, now, status=None, runs=(), schedule=None, alive=True):
        schedule = schedule or health.RUN_SCHEDULES[2]
        with TemporaryDirectory() as directory:
            if status is not None:
                path = Path(directory) / f"{schedule.model}_{schedule.cycle:02d}.status.json"
                path.write_text(json.dumps(status))
            return health.check_scheduled_run(
                schedule, now, {"runs": list(runs)}, Path(directory), lambda _pid: alive
            )

    def published_run(self, stamp, schedule=None):
        schedule = schedule or health.RUN_SCHEDULES[2]
        return {"stamp": stamp, "products": {key: {"hours": list(schedule.hours)} for key in schedule.products}}

    def test_schedule_products_and_hours_match_operational_products(self):
        import publish_hrdps_west as publishing
        for schedule in health.RUN_SCHEDULES:
            for product in schedule.products:
                self.assertIn(product, publishing.PRODUCTS_BY_MODEL[schedule.model])
                self.assertEqual(schedule.hours, tuple(publishing.PRODUCTS[product].hours))

    def test_website_deadlines_respect_model_schedule_and_daylight_saving(self):
        init = dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc)
        self.assertEqual(health.publication_deadline(health.RUN_SCHEDULES[2], init), init + dt.timedelta(hours=5.5))
        ecmwf = next(s for s in health.RUN_SCHEDULES if s.model == "ecmwf_control" and s.cycle == 0)
        for month, expected_utc_hour in ((1, 14), (9, 13)):
            init = dt.datetime(2026, month, 11, tzinfo=dt.timezone.utc)
            deadline = health.publication_deadline(ecmwf, init)
            self.assertEqual((deadline.hour, deadline.minute), (expected_utc_hour, 30))

    def test_delayed_run_alerts_even_with_a_fresh_heartbeat(self):
        now = NOW.replace(hour=18)
        status = {"stamp": "20260817T12Z", "status": "waiting_upstream", "pid": 1, "heartbeat_at_utc": now.isoformat()}
        check = self.scheduled_check(now, status)
        self.assertEqual(check.level, "warning")
        self.assertTrue(check.immediate)
        self.assertIn("0.5 hours late", check.summary)
        self.assertIn("waiting for the model's input files", check.summary)
        effective, _ = health.apply_debounce([check], {}, now)
        self.assertEqual(effective[0].level, "warning")

    def test_normal_model_latency_and_short_delays_do_not_alert(self):
        for now in (NOW.replace(hour=15), NOW.replace(hour=17, minute=40)):
            self.assertEqual(self.scheduled_check(now, runs=[self.published_run("20260817T06Z")]).level, "ok")

    def test_new_cycle_does_not_clear_an_unresolved_missing_run(self):
        now = NOW.replace(hour=12, minute=10)
        check = self.scheduled_check(now, runs=[self.published_run("20260816T06Z")])
        self.assertEqual(check.level, "critical")
        self.assertIn("20260816T12Z", check.key)

    def test_stopped_run_alerts_before_deadline_with_plain_cause_and_recovered_stamp(self):
        check = self.scheduled_check(NOW.replace(hour=15), {
            "stamp": None, "status": "failed",
            "error": "Failed https://example.test/20260817T12Z_MSC_HRDPS_DPT.grib2: 404 Client Error",
        })
        self.assertEqual(check.level, "warning")
        self.assertTrue(check.immediate)
        self.assertIn("HRDPS 12Z run for Aug 17 has stopped", check.summary)
        self.assertIn("required input file could not be found", check.summary)
        body = health.report_body([check], NOW, daily=False)
        for jargon in ("404", "https://example", "manifest", "Issue:", "pipeline"):
            self.assertNotIn(jargon, body)

    def test_old_error_is_not_used_as_the_cause_for_a_new_missing_run(self):
        check = self.scheduled_check(NOW.replace(hour=18), {"stamp": "20260816T12Z", "status": "failed", "error": "404"})
        self.assertIn("has not reported starting", check.summary)
        self.assertNotIn("input file", check.summary)

    def test_newer_complete_publication_suppresses_historical_failure(self):
        check = self.scheduled_check(NOW, {"stamp": "20260817T12Z", "status": "failed"},
                                     [self.published_run("20260817T18Z")])
        self.assertEqual(check.level, "ok")

    def test_partial_publication_and_local_success_do_not_hide_missing_images(self):
        run = self.published_run("20260817T12Z")
        run["products"]["continental_fourpanel"]["hours"].remove(24)
        check = self.scheduled_check(NOW.replace(hour=18), {"stamp": "20260817T12Z", "status": "success"}, [run])
        self.assertEqual(check.level, "warning")
        self.assertIn("ready locally", check.summary)
        self.assertIn("not all available on the website", check.summary)

    def test_interrupted_current_run_alerts_and_each_cycle_has_its_own_incident(self):
        first = self.scheduled_check(NOW.replace(hour=15), {"stamp": "20260817T12Z", "status": "running", "pid": 123}, alive=False)
        second = self.scheduled_check(NOW.replace(hour=15) + dt.timedelta(days=1), {"stamp": "20260818T12Z", "status": "failed"})
        self.assertTrue(first.immediate)
        self.assertIn("stopped before finishing", first.summary)
        reason, _ = health.notification_decision(
            signature=health.problem_signature([second]), notified_signature=health.problem_signature([first]),
            level="warning", now=NOW, previous_state={"last_notification_at": NOW.isoformat()}, always_notify=False,
        )
        self.assertEqual(reason, "alert")

    def test_model_run_reminders_wait_four_hours(self):
        signature = "run.continental.20260817T12Z:warning"
        for hours, expected in ((1, None), (4, "reminder")):
            reason, _ = health.notification_decision(
                signature=signature, notified_signature=signature, level="warning", now=NOW,
                previous_state={"last_notification_at": (NOW - dt.timedelta(hours=hours)).isoformat()}, always_notify=False,
            )
            self.assertEqual(reason, expected)

    @mock.patch("monitor_pipeline_health.telegram_notify.send_message")
    @mock.patch("monitor_pipeline_health.run_checks")
    @mock.patch("monitor_pipeline_health.utc_now", return_value=NOW)
    def test_notification_audit_records_receipt_and_retries_unaccepted_alert(self, _now, checks, send):
        checks.return_value = [health.CheckResult("run.continental.20260817T12Z", "HRDPS", "warning", "HRDPS is late.", immediate=True)]
        with TemporaryDirectory() as directory:
            args = health.parse_args([
                "--operational", "--state-path", f"{directory}/state.json",
                "--latest-path", f"{directory}/latest.json", "--history-path", f"{directory}/history.jsonl",
            ])
            send.side_effect = RuntimeError("connection failed")
            health.run_monitor(args)
            failed = json.loads(args.state_path.read_text())
            self.assertEqual(failed["notification_delivery"], "failed")
            self.assertEqual(failed["notified_signature"], "")
            send.side_effect = None
            send.return_value = {"status": "accepted_by_telegram", "message_id": 123, "date": 100}
            health.run_monitor(args)
            accepted = json.loads(args.state_path.read_text())
            self.assertEqual(accepted["notification_delivery"], "accepted_by_telegram")
            self.assertEqual(accepted["telegram_receipt"]["message_id"], 123)
            self.assertIn("HRDPS is late.", accepted["notification_body"])
            self.assertEqual(send.call_count, 2)

    @mock.patch("monitor_pipeline_health.check_storage", return_value=health.CheckResult("storage", "Storage", "ok", "ok"))
    @mock.patch("monitor_pipeline_health.check_lightning_archive", return_value=health.CheckResult("lightning", "Lightning", "ok", "ok"))
    @mock.patch("monitor_pipeline_health.check_cwfis_anchors", return_value=health.CheckResult("fwi", "FWI", "ok", "ok"))
    def test_unreadable_manifest_does_not_create_speculative_run_delays(self, *_mocks):
        checks = health.run_checks(NOW, base_url="https://example.test", data_root=Path("/tmp"),
                                   include_services=False, loader=mock.Mock(side_effect=RuntimeError("offline")))
        self.assertEqual(sum(c.key.startswith("manifest.") and c.level == "critical" for c in checks), 4)
        self.assertFalse(any(c.key.startswith("run.") for c in checks))

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

    @mock.patch("monitor_pipeline_health.shutil.disk_usage")
    def test_storage_includes_radarsat_working_set_and_alarms(self, disk_usage):
        disk_usage.return_value = shutil._ntuple_diskusage(
            2_000_000_000_000,
            1_300_000_000_000,
            700_000_000_000,
        )
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            radar_health = root / "radar-health.json"
            radar_health.write_text(json.dumps({
                "storage": {
                    "totalBytes": 21_000_000_000,
                    "compositeCacheBytes": 6_000_000_000,
                    "videoSegmentBytes": 5_000_000_000,
                    "sourceFrameBytes": 2_000_000_000,
                }
            }))
            result = health.check_storage(root, radar_health)
        self.assertEqual(result.level, "warning")
        self.assertIn("Radar-Sat 21.0 GB", result.summary)
        self.assertIn("cache 6.0", result.summary)

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
        with mock.patch("telegram_notify.json.load", return_value={"ok": True, "result": {"message_id": 42, "date": 123}}):
            receipt = telegram_notify.send_message(
                "Title",
                "Body",
                environ={"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_CHAT_ID": "123"},
            )
        request = urlopen.call_args.args[0]
        self.assertIn("/botsecret/sendMessage", request.full_url)
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_id"], "123")
        self.assertEqual(payload["text"], "Title\n\nBody")
        self.assertFalse(payload["disable_notification"])
        self.assertEqual(receipt, {"status": "accepted_by_telegram", "message_id": 42, "date": 123})

    @mock.patch("telegram_notify.urllib.request.urlopen", side_effect=RuntimeError("could not connect to /botsecret/sendMessage"))
    def test_delivery_errors_do_not_leak_bot_token(self, _urlopen):
        with self.assertRaisesRegex(RuntimeError, r"\[redacted\]") as caught:
            telegram_notify.send_message("Title", "Body", environ={"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_CHAT_ID": "123"})
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
