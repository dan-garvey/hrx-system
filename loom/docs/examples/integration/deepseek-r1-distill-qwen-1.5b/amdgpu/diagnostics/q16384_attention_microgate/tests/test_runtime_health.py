#!/usr/bin/env python3
"""Host-only tests for GPU health and quiescence evidence parsing."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import datetime as dt
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_health as health  # noqa: E402


def process_payload() -> bytes:
    return json.dumps([
        {
            "gpu": 0,
            "process_list": [{"process_info": "No running processes detected"}],
        }
    ]).encode("ascii")


def metric_payload(*, correctable: int = 0) -> bytes:
    counters = {
        "correctable_count": 0,
        "deferred_count": 0,
        "uncorrectable_count": 0,
    }
    return json.dumps({
        "gpu_data": [{
            "ecc": {
                "total_correctable_count": correctable,
                "total_deferred_count": 0,
                "total_uncorrectable_count": 0,
            },
            "ecc_blocks": {name: counters for name in health.ECC_BLOCKS},
            "gpu": 0,
            "mem_usage": {"free_vram": {"unit": "MB", "value": 294613}},
            "perf_level": health.EXPECTED_PERFORMANCE_LEVEL,
            "usage": {
                "gfx_activity": {"unit": "%", "value": 0},
                "umc_activity": {"unit": "%", "value": 0},
            },
        }]
    }).encode("ascii")


def monitor_payload() -> bytes:
    expected = health.EXPECTED_MONITOR_SNAPSHOT
    return json.dumps([{
        "Config": {"Image": expected["config_image"]},
        "HostConfig": {
            "Binds": expected["binds"],
            "DeviceCgroupRules": expected["device_cgroup_rules"],
            "DeviceRequests": expected["device_requests"],
            "Devices": expected["devices"],
            "Privileged": expected["privileged"],
        },
        "Id": expected["id"],
        "Image": expected["image_id"],
        "Name": expected["name"],
        "RestartCount": expected["restart_count"],
        "State": {
            "Running": expected["running"],
            "StartedAt": expected["started_at"],
            "Status": expected["status"],
        },
    }]).encode("ascii")


def quiescence_payload(stage: str) -> dict[str, object]:
    polls = []
    for number in (1, 2):
        polls.append({
            "captured_utc": f"2026-08-30T23:00:{number:02d}Z",
            "clean": True,
            "consecutive_clean": number,
            "containerd_tasks": [health.EXPECTED_MONITOR_TASK],
            "device_user_pids": [],
            "kfd_pids": [],
            "poll": number,
            "unexpected_containerd_tasks": [],
        })
    return {
        "pass": True,
        "polls": polls,
        "required_consecutive_clean": health.REQUIRED_CONSECUTIVE_CLEAN_POLLS,
        "schema": "loom-aiter-attention-host-quiescence-v1",
        "snapshot_timeout_seconds": health.SNAPSHOT_TIMEOUT_SECONDS,
        "stage": stage,
    }


class RuntimeHealthTests(unittest.TestCase):
    def test_clean_snapshot_and_monitor_are_accepted(self) -> None:
        result = health.validate_amd_snapshot(process_payload(), metric_payload())
        self.assertTrue(result["pass"])
        self.assertEqual(result["free_vram_mb"], 294613)
        self.assertEqual(health.validate_monitor(monitor_payload()), health.EXPECTED_MONITOR_SNAPSHOT)
        self.assertTrue(health.validate_quiescence(quiescence_payload("preflight"), "preflight")["pass"])

    def test_nonzero_ecc_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ECC total"):
            health.validate_amd_snapshot(process_payload(), metric_payload(correctable=1))

    def test_unexpected_container_task_is_rejected(self) -> None:
        payload = quiescence_payload("postflight")
        poll = payload["polls"][-1]
        assert isinstance(poll, dict)
        poll["containerd_tasks"] = [health.EXPECTED_MONITOR_TASK, "other/task"]
        poll["unexpected_containerd_tasks"] = ["other/task"]
        poll["clean"] = False
        poll["consecutive_clean"] = 0
        with self.assertRaisesRegex(RuntimeError, "did not end"):
            health.validate_quiescence(payload, "postflight")

    def test_reservation_completion_uses_explicit_manifest_binding(self) -> None:
        start = dt.datetime(2026, 8, 31, 4, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 8, 31, 20, 0, tzinfo=dt.timezone.utc)
        reservation = {
            "batch_opt_out": True,
            "date_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "id": "06a94f67-275d-78f6-8000-3680fb507fd5",
            "member_emails": ["operator@example.com"],
            "owner_email": "operator@example.com",
            "target": {"id": "target-id", "name": "target-host"},
            "title": "test reservation",
        }
        result = health.validate_reservation_completion(
            reservation,
            reservation["id"],
            "target-id",
            "target-host",
            "operator@example.com",
            now_epoch=int(dt.datetime(2026, 8, 31, 18, 0,
                                      tzinfo=dt.timezone.utc).timestamp()),
        )
        self.assertTrue(result["active_at_gate_completion"])
        with self.assertRaisesRegex(RuntimeError, "target changed"):
            health.validate_reservation_completion(
                reservation,
                reservation["id"],
                "other-target",
                "target-host",
                "operator@example.com",
                now_epoch=result["validated_epoch"],
            )


if __name__ == "__main__":
    unittest.main()
