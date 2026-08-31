#!/usr/bin/env python3
"""Host-only tests for the one-shot launcher and receipt lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import run_once as launcher  # noqa: E402
from support import canonical, write_manifest  # noqa: E402


class RunOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q16384_run_once_")
        self.root = Path(self.temporary.name).resolve()
        self.package = self.root / "package"
        self.live = self.root / "live"
        self.live.mkdir(mode=0o700)
        self.evidence = self.live / "evidence"
        self.state = self.live / "state"
        account = pwd.getpwuid(os.geteuid())
        hostname = os.uname().nodename
        self.manifest = write_manifest(
            self.package,
            self.evidence,
            self.state,
            target={
                "fqdn": f"{hostname}.test",
                "hostname": hostname,
                "id": "b2d65e95-3ff7-464d-93f6-bea8bab3a365",
            },
            operator={
                "email": "operator@example.com",
                "id": "0ebd461d-faeb-42b6-a50e-15c8cbfc8f8b",
                "short_id": account.pw_name,
            },
        )
        self.admission = self.live / "admission"
        self.admission.mkdir(mode=0o700)
        for name, data in {
            "captured-utc.txt": b"2026-08-31T18:00:00Z\n",
            "jobs.json": canonical({"count": 0, "jobs": []}),
            "reservation.json": canonical({"fixture": True}),
            "whoami.json": canonical(self.manifest["operator"]),
        }.items():
            path = self.admission / name
            path.write_bytes(data)
            path.chmod(0o600)
        self.authorization = self.live / "authorization.json"
        self.authorization.write_bytes(canonical({"fixture": True}))
        self.authorization.chmod(0o600)
        self.root_digest = "7" * 64
        self.validation = {
            "admission_age_seconds": 1.0,
            "authorization_expires_epoch": 1788206400,
            "authorization_id": "q16384-review-20260831-001",
            "authorization_issued_epoch": 1788192000,
            "authorization_sha256": "8" * 64,
            "authorized_case_count": 1,
            "authorized_cases": ["q16384-first-attention"],
            "consumption_receipt_path": self.manifest["execution"]["receipt_path"],
            "host": self.manifest["target"]["hostname"],
            "invocation_count": 1,
            "maximum_gpu_invocations": 1,
            "microgate_root_sha256": self.root_digest,
            "pass": True,
            "reservation_end_epoch": 1788206400,
            "reservation_id": self.manifest["reservation"]["id"],
            "reservation_remaining_seconds": 3600.0,
            "reservation_start_epoch": 1788148800,
            "result_path": self.manifest["execution"]["result_path"],
            "schema": "loom-q16384-first-attention-launch-abi-validation-v1",
            "target": self.manifest["target"]["fqdn"],
            "validated_epoch": 1788202800,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _patches(self) -> list[mock._patch]:
        def snapshot(path: Path, _stage: str) -> dict[str, object]:
            path.mkdir(mode=0o700)
            return {"pass": True}

        return [
            mock.patch.object(launcher, "PACKAGE_ROOT", self.package),
            mock.patch.object(launcher, "RUNNER", self.package / "bin/q16384_attention_microgate"),
            mock.patch.object(launcher, "CAMPAIGN_LOCK", self.live / "campaign.lock"),
            mock.patch.object(launcher, "verify_package", return_value=(self.root_digest, 42)),
            mock.patch.object(launcher, "validate", return_value=self.validation),
            mock.patch.object(launcher, "capture_snapshot", side_effect=snapshot),
            mock.patch.object(launcher, "_capture_postflight", return_value=[]),
            mock.patch.object(launcher, "validate_success_evidence", return_value={"pass": True}),
        ]

    def test_runner_is_invoked_once_after_receipt_creation_and_claims_it(self) -> None:
        calls = 0

        def invoke(command: list[str], environment: dict[str, str]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            receipt = self.state / launcher.RECEIPT_NAME
            consumed = Path(f"{receipt}.consumed")
            self.assertTrue(receipt.is_file())
            self.assertTrue((self.state / launcher.ATTEMPT_NAME).is_file())
            os.rename(receipt, consumed)
            result = self.evidence / "RESULT.json"
            result.write_text('{"status":"pass"}\n', encoding="ascii")
            result.chmod(0o600)
            self.assertEqual(environment["MICROGATE_AUTHORIZATION_CONSUMED"], "1")
            return {
                "finished_epoch": 1788202802,
                "outer_timeout": False,
                "returncode": 0,
                "started_epoch": 1788202801,
                "stderr": b"",
                "stdout": b"q16384 first-attention launch-ABI microgate: PASS\n",
                "supervised_command": [launcher.TIMEOUT, "fixture"],
                "timed_out": False,
            }

        patches = self._patches()
        patches.append(mock.patch.object(launcher, "_invoke_runner_once", side_effect=invoke))
        with mock.patch.dict(os.environ, {"PROCEED_GPU": "YES"}, clear=False):
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7], patches[8]:
                result = launcher.run_once(
                    self.evidence,
                    self.admission,
                    self.authorization,
                    self.state,
                    process_exit=None,
                )
        self.assertEqual(result, 0)
        self.assertEqual(calls, 1)
        self.assertFalse((self.state / launcher.RECEIPT_NAME).exists())
        self.assertTrue(Path(f"{self.state / launcher.RECEIPT_NAME}.consumed").is_file())
        self.assertTrue((self.evidence / "PASS.json").is_file())
        self.assertFalse((self.evidence / "FAIL.json").exists())
        self.assertTrue((self.evidence / "ROOT_SHA256").is_file())

    def test_proceed_interlock_blocks_before_package_or_runner(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(launcher, "verify_package") as verify, \
             mock.patch.object(launcher, "_invoke_runner_once") as invoke:
            with self.assertRaisesRegex(RuntimeError, "PROCEED_GPU=YES"):
                launcher.run_once(
                    self.evidence,
                    self.admission,
                    self.authorization,
                    self.state,
                    process_exit=None,
                )
        verify.assert_not_called()
        invoke.assert_not_called()

    def test_receipt_is_canonical_single_case_and_one_use(self) -> None:
        self.state.mkdir(mode=0o700)
        receipt = self.state / launcher.RECEIPT_NAME
        launcher.create_receipt(
            receipt,
            self.validation,
            self.authorization,
            self.evidence / "RESULT.json",
        )
        raw = receipt.read_text(encoding="ascii")
        self.assertIn("authorized_cases=q16384-first-attention\n", raw)
        self.assertIn("authorized_case_count=1\n", raw)
        self.assertIn("maximum_gpu_invocations=1\n", raw)
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o400)
        self.assertTrue(launcher.retire_unclaimed_receipt(receipt))
        self.assertFalse(receipt.exists())
        self.assertTrue(Path(f"{receipt}.consumed").is_file())
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            launcher.create_receipt(
                receipt,
                self.validation,
                self.authorization,
                self.evidence / "RESULT.json",
            )


if __name__ == "__main__":
    unittest.main()
