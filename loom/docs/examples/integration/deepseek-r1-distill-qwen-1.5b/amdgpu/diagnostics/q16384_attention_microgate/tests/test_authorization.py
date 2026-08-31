#!/usr/bin/env python3
"""Adversarial host-only tests for the one-run authorization validator."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import validate_authorization as validator  # noqa: E402
from support import canonical, write_manifest  # noqa: E402


ROOT_DIGEST = "7" * 64
NOW = dt.datetime(2026, 8, 31, 18, 0, 0, tzinfo=dt.timezone.utc)


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q16384_auth_")
        self.root = Path(self.temporary.name).resolve()
        self.package = self.root / "package"
        self.live = self.root / "live"
        self.live.mkdir(mode=0o700)
        self.evidence = self.live / "evidence"
        self.state = self.live / "state"
        self.evidence.mkdir(mode=0o700)
        self.state.mkdir(mode=0o700)
        self.manifest = write_manifest(self.package, self.evidence, self.state)
        self.admission = self.live / "admission"
        self.admission.mkdir(mode=0o700)
        self.authorization_path = self.live / "authorization.json"
        self.receipt_path = Path(self.manifest["execution"]["receipt_path"])
        self.result_path = Path(self.manifest["execution"]["result_path"])
        reservation = self.manifest["reservation"]
        target = self.manifest["target"]
        operator = self.manifest["operator"]
        self.authorization = {
            "authorization_id": "q16384-review-20260831-001",
            "authorized_case_count": 1,
            "authorized_cases": ["q16384-first-attention"],
            "consumption_receipt_path": str(self.receipt_path),
            "decision": validator.DECISION,
            "expires_utc": "2026-08-31T19:30:00Z",
            "host": target["hostname"],
            "invocation_count": 1,
            "issued_utc": "2026-08-31T17:50:00Z",
            "maximum_gpu_invocations": 1,
            "microgate_root_sha256": ROOT_DIGEST,
            "pack_and_attention_contract_satisfied": False,
            "reservation_id": reservation["id"],
            "result_path": str(self.result_path),
            "schema": validator.SCHEMA,
            "scope": validator.SCOPE,
            "target": target["fqdn"],
        }
        self.reservation = {
            "batch_opt_out": True,
            "date_end": reservation["end_utc"],
            "date_start": reservation["start_utc"],
            "id": reservation["id"],
            "member_emails": [operator["email"]],
            "owner_email": operator["email"],
            "target": {"id": target["id"], "name": target["hostname"]},
            "title": "q16384 launch ABI reservation",
        }
        self._write_admission()
        self._write_authorization()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_private(path: Path, data: bytes) -> None:
        path.write_bytes(data)
        path.chmod(0o600)

    def _write_admission(self) -> None:
        payloads = {
            "jobs.json": {"count": 0, "jobs": []},
            "reservation.json": self.reservation,
            "whoami.json": self.manifest["operator"],
        }
        for name, payload in payloads.items():
            self._write_private(self.admission / name, canonical(payload))
        self._write_private(
            self.admission / "captured-utc.txt", b"2026-08-31T17:59:00Z\n"
        )

    def _write_authorization(self, raw: bytes | None = None) -> None:
        self._write_private(
            self.authorization_path,
            canonical(self.authorization) if raw is None else raw,
        )

    def _validate(self) -> dict[str, object]:
        return validator.validate(
            self.admission,
            self.authorization_path,
            ROOT_DIGEST,
            self.receipt_path,
            self.result_path,
            self.manifest["target"]["hostname"],
            now=NOW,
            package_root=self.package,
        )

    def test_valid_canonical_authorization(self) -> None:
        result = self._validate()
        self.assertTrue(result["pass"])
        self.assertEqual(result["authorized_cases"], ["q16384-first-attention"])
        self.assertEqual(result["authorized_case_count"], 1)

    def test_count_scope_root_and_paths_are_strict(self) -> None:
        mutations = (
            ("authorized_case_count", 2),
            ("authorized_cases", ["q512", "q16384-first-attention"]),
            ("maximum_gpu_invocations", True),
            ("microgate_root_sha256", "8" * 64),
            ("scope", "attention-only-raw-hsa"),
            ("consumption_receipt_path", str(self.state / "other.receipt")),
            ("result_path", str(self.evidence / "other.json")),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                original = self.authorization[field]
                self.authorization[field] = value
                self._write_authorization()
                with self.assertRaises(RuntimeError):
                    self._validate()
                self.authorization[field] = original

    def test_noncanonical_symlink_and_hardlink_authorizations_are_rejected(self) -> None:
        self._write_authorization(
            (json.dumps(self.authorization, indent=2, sort_keys=True) + "\n").encode("ascii")
        )
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            self._validate()

        target = self.live / "authorization-target.json"
        self._write_private(target, canonical(self.authorization))
        self.authorization_path.unlink()
        self.authorization_path.symlink_to(target)
        with self.assertRaises(RuntimeError):
            self._validate()
        self.authorization_path.unlink()
        os.link(target, self.authorization_path)
        with self.assertRaisesRegex(RuntimeError, "hardlinked"):
            self._validate()

    def test_stale_admission_and_expired_authorization_are_rejected(self) -> None:
        self._write_private(
            self.admission / "captured-utc.txt", b"2026-08-31T17:00:00Z\n"
        )
        with self.assertRaisesRegex(RuntimeError, "capture age"):
            self._validate()

        self._write_admission()
        self.authorization["expires_utc"] = "2026-08-31T17:59:59Z"
        self._write_authorization()
        with self.assertRaisesRegex(RuntimeError, "time window"):
            self._validate()


if __name__ == "__main__":
    unittest.main()
