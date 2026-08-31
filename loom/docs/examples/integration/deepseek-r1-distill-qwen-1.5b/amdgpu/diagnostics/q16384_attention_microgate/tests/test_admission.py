#!/usr/bin/env python3
"""Host-only tests for manifest-bound Conductor admission normalization."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import normalize_admission as admission  # noqa: E402
from support import write_manifest  # noqa: E402


class AdmissionNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q16384_admission_")
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "output"
        self.output.mkdir(mode=0o700)
        self.package = self.root / "package"
        self.manifest = write_manifest(
            self.package, self.root / "evidence", self.root / "state"
        )
        self.whoami = self.root / "whoami.raw"
        self.reservation = self.root / "reservation.raw"
        self.jobs = self.root / "jobs.raw"
        self.whoami.write_text(
            "conduct 4.2.0\n" + json.dumps(self.manifest["operator"]) + "\n",
            encoding="utf-8",
        )
        target = self.manifest["target"]
        reservation = self.manifest["reservation"]
        self.row = {
            "batch_opt_out": True,
            "creator": {"email": self.manifest["operator"]["email"]},
            "date_end": reservation["end_utc"],
            "date_start": reservation["start_utc"],
            "id": reservation["id"],
            "target_info": {
                "children": [],
                "entity_type": "standalone_system",
                "id": target["id"],
                "name": target["hostname"],
            },
            "title": "q16384 launch ABI reservation",
            "users": [{"email": self.manifest["operator"]["email"]}],
        }
        self._write_reservation(self.row)
        self._write_jobs({
            "count": 0,
            "has_more": False,
            "jobs": [],
            "page_size": admission.EXPECTED_JOBS_PAGE_SIZE,
        })
        self.now = dt.datetime(2026, 8, 31, 18, 0, tzinfo=dt.timezone.utc)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_reservation(self, row: dict[str, object]) -> None:
        self.reservation.write_text(
            "version notice\n" + json.dumps({"reservations": [row]}) + "\n",
            encoding="utf-8",
        )

    def _write_jobs(self, payload: dict[str, object]) -> None:
        self.jobs.write_text(
            "version notice\n" + json.dumps(payload) + "\n", encoding="utf-8"
        )

    def _normalize(self) -> dict[str, object]:
        return admission.normalize(
            self.whoami,
            self.reservation,
            self.jobs,
            self.output,
            "2026-08-31T18:00:00Z",
            now=self.now,
            package_root=self.package,
        )

    def test_matching_standalone_reservation_is_normalized(self) -> None:
        result = self._normalize()
        self.assertEqual(result["reservation_id"], self.manifest["reservation"]["id"])
        normalized = json.loads((self.output / "reservation.json").read_text())
        self.assertEqual(
            normalized["target"],
            {
                "id": self.manifest["target"]["id"],
                "name": self.manifest["target"]["hostname"],
            },
        )
        self.assertEqual({path.name for path in self.output.iterdir()}, admission.OUTPUT_FILES)

    def test_jobs_and_target_drift_are_rejected(self) -> None:
        self._write_jobs({"count": 0, "has_more": True, "jobs": [], "page_size": 25})
        with self.assertRaisesRegex(RuntimeError, "pagination"):
            self._normalize()

        self._write_jobs({"count": 0, "has_more": False, "jobs": [], "page_size": 25})
        changed = dict(self.row)
        changed["target_info"] = dict(self.row["target_info"], children=["unexpected"])
        self._write_reservation(changed)
        with self.assertRaisesRegex(RuntimeError, "standalone"):
            self._normalize()

    def test_wrong_reservation_and_insufficient_time_are_rejected(self) -> None:
        changed = dict(self.row, id="00000000-0000-0000-0000-000000000000")
        self._write_reservation(changed)
        with self.assertRaisesRegex(RuntimeError, "exactly one matching"):
            self._normalize()

        self._write_reservation(self.row)
        self.now = dt.datetime(2026, 8, 31, 19, 45, 1, tzinfo=dt.timezone.utc)
        with self.assertRaisesRegex(RuntimeError, "under 1200"):
            self._normalize()


if __name__ == "__main__":
    unittest.main()
