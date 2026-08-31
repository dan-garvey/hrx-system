#!/usr/bin/python3 -I
"""Normalize Conductor admission against the sealed package manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_manifest import load_manifest  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_JOBS_PAGE_SIZE = 25
EXPECTED_JOBS_KEYS = {"count", "has_more", "jobs", "page_size"}
OUTPUT_FILES = {"captured-utc.txt", "jobs.json", "reservation.json", "whoami.json"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_last_json(path: Path) -> Any:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"no JSON object in {path}")


def _write_canonical(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    path.chmod(0o600)


def normalize(
    whoami_path: Path,
    reservation_path: Path,
    jobs_path: Path,
    output: Path,
    captured_utc: str,
    *,
    now: dt.datetime | None = None,
    package_root: Path | None = None,
) -> dict[str, object]:
    manifest = load_manifest((package_root or PACKAGE_ROOT).resolve(strict=True))
    expected_reservation = manifest["reservation"]
    expected_target = manifest["target"]
    expected_operator = manifest["operator"]
    whoami_raw = load_last_json(whoami_path)
    reservation_raw = load_last_json(reservation_path)
    jobs_raw = load_last_json(jobs_path)
    require(isinstance(whoami_raw, dict), "Conductor identity response is invalid")
    whoami = {
        "email": whoami_raw.get("email"),
        "id": whoami_raw.get("id"),
        "short_id": whoami_raw.get("short_id"),
    }
    require(whoami == expected_operator, "Conductor identity changed")
    require(isinstance(reservation_raw, dict), "reservation response is invalid")
    reservations = reservation_raw.get("reservations", [])
    require(isinstance(reservations, list), "reservation rows are invalid")
    rows = [
        row for row in reservations
        if isinstance(row, dict) and row.get("id") == expected_reservation["id"]
    ]
    require(len(rows) == 1, "expected exactly one matching active reservation")
    row = rows[0]
    require(row.get("date_start") == expected_reservation["start_utc"],
            "reservation start changed")
    require(row.get("date_end") == expected_reservation["end_utc"],
            "reservation end changed")
    target_info = row.get("target_info")
    require(isinstance(target_info, dict), "reservation target is invalid")
    require(target_info.get("id") == expected_target["id"] and
            target_info.get("name") == expected_target["hostname"],
            "reservation target changed")
    require(target_info.get("entity_type") == "standalone_system" and
            target_info.get("children") == [],
            "reservation target is not a standalone system")
    require(row.get("batch_opt_out") is True, "reservation is not exclusive")
    users = row.get("users", [])
    require(isinstance(users, list) and all(isinstance(user, dict) for user in users),
            "reservation users are invalid")
    members = sorted(user.get("email") for user in users)
    require(all(isinstance(email, str) for email in members),
            "reservation member email is invalid")
    require(members == sorted(set(members)), "reservation members are duplicated")
    require(expected_operator["email"] in members,
            "operator is not a reservation member")
    creator = row.get("creator")
    require(isinstance(creator, dict)
            and creator.get("email") == expected_operator["email"],
            "reservation creator changed")
    require(isinstance(row.get("title"), str) and row["title"],
            "reservation title is invalid")
    current = now or dt.datetime.now(dt.timezone.utc)
    current = current.astimezone(dt.timezone.utc)
    start = dt.datetime.fromisoformat(
        expected_reservation["start_utc"].replace("Z", "+00:00")
    )
    end = dt.datetime.fromisoformat(
        expected_reservation["end_utc"].replace("Z", "+00:00")
    )
    require(start <= current < end, "reservation is not active")
    require((end - current).total_seconds() >= 1200,
            "reservation has under 1200 seconds remaining")
    require(isinstance(jobs_raw, dict), "Conductor jobs response is invalid")
    require(set(jobs_raw) == EXPECTED_JOBS_KEYS,
            "Conductor jobs response schema changed")
    require(type(jobs_raw["count"]) is int and jobs_raw["count"] == 0,
            "Conductor reports nonterminal jobs")
    require(type(jobs_raw["jobs"]) is list and jobs_raw["jobs"] == [],
            "Conductor reports nonterminal jobs")
    require(jobs_raw["has_more"] is False,
            "Conductor jobs pagination is incomplete")
    require(type(jobs_raw["page_size"]) is int
            and jobs_raw["page_size"] == EXPECTED_JOBS_PAGE_SIZE,
            "Conductor jobs page size changed")
    captured = dt.datetime.fromisoformat(captured_utc.replace("Z", "+00:00"))
    require(captured.tzinfo is not None
            and captured.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            == captured_utc,
            "capture time is not canonical UTC")
    payloads = {
        "jobs.json": {"count": 0, "jobs": []},
        "reservation.json": {
            "batch_opt_out": row["batch_opt_out"],
            "date_end": row["date_end"],
            "date_start": row["date_start"],
            "id": row["id"],
            "member_emails": members,
            "owner_email": creator["email"],
            "target": {
                "id": row["target_info"]["id"],
                "name": row["target_info"]["name"],
            },
            "title": row["title"],
        },
        "whoami.json": whoami,
    }
    require(output.is_dir() and not output.is_symlink(),
            "admission output directory is invalid")
    require(not any(output.iterdir()), "admission output directory is not empty")
    for name, payload in payloads.items():
        _write_canonical(output / name, payload)
    (output / "captured-utc.txt").write_text(captured_utc + "\n", encoding="ascii")
    (output / "captured-utc.txt").chmod(0o600)
    require({path.name for path in output.iterdir()} == OUTPUT_FILES,
            "admission file set changed")
    return {
        "captured_utc": captured_utc,
        "nonterminal_job_count": 0,
        "reservation_end": expected_reservation["end_utc"],
        "reservation_id": expected_reservation["id"],
        "target_entity": expected_target["id"],
        "target_host": expected_target["hostname"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("whoami", type=Path)
    parser.add_argument("reservation", type=Path)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("captured_utc")
    args = parser.parse_args()
    result = normalize(
        args.whoami,
        args.reservation,
        args.jobs,
        args.output,
        args.captured_utc,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
