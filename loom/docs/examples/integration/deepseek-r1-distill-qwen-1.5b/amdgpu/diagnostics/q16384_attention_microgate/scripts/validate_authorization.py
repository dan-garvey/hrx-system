#!/usr/bin/python3 -I
"""Validate the sealed attention microgate authorization and admission set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_manifest import load_manifest  # noqa: E402

SCHEMA = "loom-q16384-first-attention-launch-abi-one-run-authorization-v1"
DECISION = "AUTHORIZE_ONE_GPU_INVOCATION"
SCOPE = "q16384-first-attention-raw-hsa-vs-hip"
EXPECTED_CASES = ["q16384-first-attention"]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_KEYS = {
    "schema",
    "decision",
    "scope",
    "microgate_root_sha256",
    "reservation_id",
    "host",
    "target",
    "authorized_cases",
    "authorized_case_count",
    "issued_utc",
    "expires_utc",
    "maximum_gpu_invocations",
    "invocation_count",
    "consumption_receipt_path",
    "result_path",
    "authorization_id",
    "pack_and_attention_contract_satisfied",
}
ADMISSION_FILES = {
    "captured-utc.txt",
    "jobs.json",
    "reservation.json",
    "whoami.json",
}
RESERVATION_KEYS = {
    "batch_opt_out",
    "date_end",
    "date_start",
    "id",
    "member_emails",
    "owner_email",
    "target",
    "title",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_utc(value: Any, description: str) -> dt.datetime:
    require(isinstance(value, str), f"{description} is not a string")
    require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value)
        is not None,
        f"{description} is not canonical UTC",
    )
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, f"{description} has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def require_safe_absolute_path(path: Path, description: str, *, exists: bool) -> None:
    text = str(path)
    pure = PurePosixPath(text)
    require(path.is_absolute(), f"{description} must be absolute")
    require(
        re.fullmatch(r"/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+", text) is not None,
        f"{description} contains unsafe characters",
    )
    require(
        pure.as_posix() == text and all(part not in ("", ".", "..") for part in pure.parts[1:]),
        f"{description} is not canonical",
    )
    try:
        resolved = path.resolve(strict=exists)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve {description}: {path}") from exc
    require(resolved == path, f"{description} is not a canonical physical path")
    parent = path.parent
    parent_stat = parent.stat()
    require(stat.S_ISDIR(parent_stat.st_mode), f"{description} parent is not a directory")
    require(parent_stat.st_uid == os.geteuid(), f"{description} parent owner changed")
    require(
        parent_stat.st_mode & 0o022 == 0,
        f"{description} parent is group/world writable",
    )


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def secure_read(path: Path, maximum_size: int = 1 << 20) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot securely open file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {path}")
        require(metadata.st_nlink == 1, f"hardlinked file is forbidden: {path}")
        require(metadata.st_uid == os.geteuid(), f"file owner changed: {path}")
        require(metadata.st_mode & 0o077 == 0, f"file is not mode-private: {path}")
        require(metadata.st_size <= maximum_size, f"file is too large: {path}")
        chunks = []
        remaining = maximum_size + 1
        while remaining:
            block = os.read(descriptor, min(65536, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
        require(len(data) <= maximum_size, f"file is too large: {path}")
        return data
    finally:
        os.close(descriptor)


def read_json(path: Path) -> tuple[Any, bytes]:
    raw = secure_read(path)
    return json.loads(raw.decode("utf-8")), raw


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def read_canonical_json(path: Path, description: str) -> Any:
    value, raw = read_json(path)
    require(raw == canonical_json(value), f"{description} is not canonical JSON")
    return value


def validate(
    admission: Path,
    authorization_path: Path,
    root_digest: str,
    receipt_path: Path,
    result_path: Path,
    hostname: str,
    now: dt.datetime | None = None,
    minimum_remaining_seconds: int = 1050,
    package_root: Path | None = None,
) -> dict[str, Any]:
    package = (package_root or PACKAGE_ROOT).resolve(strict=True)
    manifest = load_manifest(package)
    expected_operator = manifest["operator"]
    expected_reservation = manifest["reservation"]
    expected_target = manifest["target"]
    expected_execution = manifest["execution"]
    require(re.fullmatch(r"[0-9a-f]{64}", root_digest) is not None,
            "microgate root digest is malformed")
    require_safe_absolute_path(
        authorization_path, "authorization path", exists=True
    )
    require_safe_absolute_path(
        receipt_path, "consumption receipt path", exists=False
    )
    require_safe_absolute_path(result_path, "result path", exists=False)
    require(not result_path.exists(), "result path already exists")
    claimed_receipt_path = Path(f"{receipt_path}.consumed")
    require_safe_absolute_path(
        claimed_receipt_path, "claimed receipt path", exists=False
    )
    mutable_paths = {
        authorization_path,
        receipt_path,
        claimed_receipt_path,
        result_path,
    }
    require(len(mutable_paths) == 4, "authorization, receipt, and result paths collide")
    require(
        all(not is_within(path, package) for path in mutable_paths),
        "live mutable paths must be outside the sealed package",
    )
    admission_stat = admission.lstat()
    require(stat.S_ISDIR(admission_stat.st_mode), "admission path is not a directory")
    require(not admission.is_symlink(), "admission directory may not be a symlink")
    require(admission_stat.st_uid == os.geteuid(), "admission owner changed")
    require(admission_stat.st_mode & 0o077 == 0, "admission is not mode-private")
    require(admission.resolve(strict=True) == admission,
            "admission is not a canonical physical path")
    require(not is_within(admission, package),
            "admission must be outside the sealed package")
    require(
        {entry.name for entry in admission.iterdir()} == ADMISSION_FILES,
        "admission file set changed",
    )

    authorization, authorization_raw = read_json(authorization_path)
    require(isinstance(authorization, dict), "authorization is not an object")
    canonical = canonical_json(authorization)
    require(authorization_raw == canonical, "authorization is not canonical")
    require(set(authorization) == AUTHORIZATION_KEYS, "authorization fields changed")
    require(authorization["schema"] == SCHEMA, "authorization schema changed")
    require(authorization["decision"] == DECISION, "GPU invocation is not authorized")
    require(authorization["scope"] == SCOPE, "authorization scope changed")
    require(
        authorization["microgate_root_sha256"] == root_digest,
        "authorization binds another microgate root",
    )
    require(
        authorization["reservation_id"] == expected_reservation["id"],
        "authorization binds another reservation",
    )
    require(authorization["host"] == expected_target["hostname"],
            "authorization host changed")
    require(authorization["target"] == expected_target["fqdn"],
            "authorization target changed")
    require(hostname in (expected_target["hostname"], expected_target["fqdn"]),
            "execution host mismatch")
    require(authorization["authorized_cases"] == EXPECTED_CASES,
            "authorization case order or membership changed")
    require(type(authorization["authorized_case_count"]) is int
            and authorization["authorized_case_count"] == 1,
            "authorization case count is not exactly one")
    require(type(authorization["maximum_gpu_invocations"]) is int
            and authorization["maximum_gpu_invocations"] == 1,
            "maximum GPU invocation count is not one")
    require(type(authorization["invocation_count"]) is int
            and authorization["invocation_count"] == 1,
            "requested invocation count is not one")
    require(authorization["pack_and_attention_contract_satisfied"] is False,
            "authorization must acknowledge the deferred pack contract")
    require(
        isinstance(authorization["consumption_receipt_path"], str)
        and authorization["consumption_receipt_path"] == str(receipt_path),
        "authorization binds another consumption receipt",
    )
    require(
        isinstance(authorization["result_path"], str)
        and authorization["result_path"] == str(result_path),
        "authorization binds another result path",
    )
    require(str(receipt_path) == expected_execution["receipt_path"],
            "receipt path differs from package manifest")
    require(str(result_path) == expected_execution["result_path"],
            "result path differs from package manifest")
    authorization_id = authorization["authorization_id"]
    require(
        isinstance(authorization_id, str)
        and 16 <= len(authorization_id) <= 128
        and re.fullmatch(r"[A-Za-z0-9._:-]+", authorization_id) is not None,
        "authorization ID is invalid",
    )

    whoami = read_canonical_json(admission / "whoami.json", "whoami admission")
    reservation = read_canonical_json(
        admission / "reservation.json", "reservation admission"
    )
    jobs = read_canonical_json(admission / "jobs.json", "jobs admission")
    captured_raw = secure_read(admission / "captured-utc.txt", 128)
    captured_text = captured_raw.decode("ascii")
    require(captured_text.endswith("\n") and captured_text.count("\n") == 1,
            "admission capture is not canonical")
    captured = parse_utc(captured_text[:-1], "admission capture")
    require(whoami == expected_operator, "Conductor identity changed")
    require(jobs == {"count": 0, "jobs": []}, "Conductor reports nonterminal jobs")
    require(isinstance(reservation, dict), "reservation admission is not an object")
    require(set(reservation) == RESERVATION_KEYS, "reservation admission fields changed")
    require(reservation["id"] == expected_reservation["id"],
            "admission binds another reservation")
    require(reservation["id"] == authorization["reservation_id"],
            "authorization and admission reservation differ")
    require(reservation["date_start"] == expected_reservation["start_utc"],
            "reservation start changed")
    require(reservation["date_end"] == expected_reservation["end_utc"],
            "reservation end changed")
    require(
        reservation["target"]
        == {"id": expected_target["id"], "name": expected_target["hostname"]},
        "reservation target changed",
    )
    require(reservation["batch_opt_out"] is True, "reservation is not exclusive")
    require(reservation["owner_email"] == expected_operator["email"],
            "reservation owner changed")
    require(
        isinstance(reservation["member_emails"], list)
        and all(isinstance(email, str) for email in reservation["member_emails"])
        and reservation["member_emails"] == sorted(set(reservation["member_emails"]))
        and expected_operator["email"] in reservation["member_emails"],
        "reservation membership changed",
    )
    require(isinstance(reservation["title"], str) and reservation["title"],
            "reservation title is invalid")

    current = now or dt.datetime.now(dt.timezone.utc)
    current = current.astimezone(dt.timezone.utc)
    start = parse_utc(reservation["date_start"], "reservation start")
    end = parse_utc(reservation["date_end"], "reservation end")
    issued = parse_utc(authorization["issued_utc"], "authorization issue")
    expires = parse_utc(authorization["expires_utc"], "authorization expiry")
    age = (current - captured).total_seconds()
    remaining = (end - current).total_seconds()
    require(-30 <= age <= 300, f"admission capture age is invalid: {age}")
    require(start <= current < end, "reservation is not active")
    require(remaining >= minimum_remaining_seconds,
            f"reservation has under {minimum_remaining_seconds} seconds remaining")
    require(start <= issued <= current < expires <= end,
            "authorization time window is invalid")

    return {
        "schema": "loom-q16384-first-attention-launch-abi-validation-v1",
        "pass": True,
        "authorization_id": authorization_id,
        "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
        "microgate_root_sha256": root_digest,
        "authorized_cases": EXPECTED_CASES,
        "authorized_case_count": 1,
        "reservation_id": expected_reservation["id"],
        "reservation_start_epoch": int(start.timestamp()),
        "reservation_end_epoch": int(end.timestamp()),
        "authorization_issued_epoch": int(issued.timestamp()),
        "authorization_expires_epoch": int(expires.timestamp()),
        "host": expected_target["hostname"],
        "target": expected_target["fqdn"],
        "maximum_gpu_invocations": 1,
        "invocation_count": 1,
        "consumption_receipt_path": str(receipt_path),
        "result_path": str(result_path),
        "validated_epoch": int(current.timestamp()),
        "admission_age_seconds": age,
        "reservation_remaining_seconds": remaining,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("admission", type=Path)
    parser.add_argument("authorization", type=Path)
    parser.add_argument("root_digest")
    parser.add_argument("receipt_path", type=Path)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--hostname", default=os.uname().nodename)
    parser.add_argument("--minimum-remaining-seconds", type=int, default=1050)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(
        args.admission,
        args.authorization,
        args.root_digest,
        args.receipt_path,
        args.result_path,
        args.hostname,
        minimum_remaining_seconds=args.minimum_remaining_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
