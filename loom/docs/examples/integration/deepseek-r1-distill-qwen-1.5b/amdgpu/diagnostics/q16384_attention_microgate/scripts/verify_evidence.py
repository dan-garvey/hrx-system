#!/usr/bin/python3 -I
"""Semantically verify successful q16384 raw-HSA-versus-HIP evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_manifest import load_manifest  # noqa: E402
from runtime_health import (  # noqa: E402
    KERNEL_FILES,
    SNAPSHOT_FILES,
    canonical_json,
    compare_amd_health,
    parse_json,
    parse_utc,
    validate_kernel_evidence,
    validate_reservation_completion,
    validate_snapshot_directory,
)
from seal_lib import collect_files, verify_tree  # noqa: E402
from validate_authorization import (  # noqa: E402
    AUTHORIZATION_KEYS,
    DECISION,
    RESERVATION_KEYS,
    SCHEMA as AUTHORIZATION_SCHEMA,
    SCOPE,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TIMEOUT_SECONDS = 600
TERMINATION_GRACE_SECONDS = 30
INITIAL_MINIMUM_REMAINING_SECONDS = 1200
PRELAUNCH_MINIMUM_REMAINING_SECONDS = 1050
RESULT_SCHEMA = "loom-q16384-first-attention-launch-abi-result-v2"
RECEIPT_SCHEMA = "loom-q16384-first-attention-launch-abi-consumption-v1"
RECEIPT_KEYS = (
    "schema",
    "authorization_id",
    "authorization_sha256",
    "authorization_path",
    "microgate_root_sha256",
    "authorized_cases",
    "authorized_case_count",
    "reservation_id",
    "consumption_receipt_path",
    "result_path",
    "reservation_start_epoch",
    "reservation_end_epoch",
    "authorization_issued_epoch",
    "authorization_expires_epoch",
    "host",
    "target",
    "maximum_gpu_invocations",
    "invocation_index",
    "invocation_count",
    "consumed_epoch",
)
VALIDATION_KEYS = {
    "admission_age_seconds",
    "authorization_expires_epoch",
    "authorization_id",
    "authorization_issued_epoch",
    "authorization_sha256",
    "authorized_case_count",
    "authorized_cases",
    "consumption_receipt_path",
    "host",
    "invocation_count",
    "maximum_gpu_invocations",
    "microgate_root_sha256",
    "pass",
    "reservation_end_epoch",
    "reservation_id",
    "reservation_remaining_seconds",
    "reservation_start_epoch",
    "result_path",
    "schema",
    "target",
    "validated_epoch",
}
RUN_RECORD_KEYS = {
    "authorization_id",
    "command",
    "environment",
    "evidence_path",
    "invocation_count",
    "kill_after_seconds",
    "microgate_root_sha256",
    "package_root",
    "postflight_pass",
    "receipt_claimed_or_retired",
    "receipt_retired_by_launcher",
    "returncode",
    "runner_finished_epoch",
    "runner_started_epoch",
    "runtime_timeout_seconds",
    "schema",
    "supervised_command",
    "timed_out",
}
BASE_SUCCESS_FILES = {
    "PASS.json",
    "RESULT.json",
    "admission-captured-utc.txt",
    "admission-jobs.json",
    "admission-reservation.json",
    "admission-whoami.json",
    "attempt.json",
    "authorization-consumed.txt",
    "authorization.json",
    "initial-validation.json",
    "package-verification.json",
    "pre-post-health-comparison.json",
    "prelaunch-validation.json",
    "reservation-completion-validation.json",
    "run-record.json",
    "runner.stderr.txt",
    "runner.stdout.txt",
    *KERNEL_FILES,
}
EXPECTED_SUCCESS_FILES = BASE_SUCCESS_FILES | {
    f"{stage}/{name}"
    for stage in ("preflight", "prelaunch", "postflight")
    for name in SNAPSHOT_FILES
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _exact_contract_value(actual: Any, expected: Any) -> bool:
    """Compare fixed contract data without Python's bool/int coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_contract_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_contract_value(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return bool(actual == expected)


def _read_limited(path: Path, maximum_size: int = 16 << 20) -> bytes:
    metadata = path.stat()
    require(metadata.st_size <= maximum_size, f"evidence file is too large: {path.name}")
    return path.read_bytes()


def _read_json(path: Path, description: str, *, canonical: bool = False) -> Any:
    raw = _read_limited(path)
    value = parse_json(raw, description)
    if canonical:
        require(raw == canonical_json(value), f"{description} is not canonical JSON")
    return value


def _safe_absolute_path(value: Any, description: str) -> str:
    require(isinstance(value, str), f"{description} is not a string")
    require(re.fullmatch(r"/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+", value) is not None,
            f"{description} is not a safe absolute path")
    pure = PurePosixPath(value)
    require(pure.as_posix() == value and all(part not in {"", ".", ".."}
                                             for part in pure.parts[1:]),
            f"{description} is not canonical")
    return value


def _lower_digest(value: Any, description: str, *, nonzero: bool = False) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{description} is not a lowercase SHA-256 digest")
    if nonzero:
        require(value != "0" * 64, f"{description} is the zero digest")
    return value


def _epoch(value: dt.datetime) -> int:
    return int(value.timestamp())


def _validate_admission(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    captured_raw = _read_limited(root / "admission-captured-utc.txt", 128)
    require(captured_raw.endswith(b"\n") and captured_raw.count(b"\n") == 1,
            "admission capture is not one canonical line")
    captured = parse_utc(captured_raw[:-1].decode("ascii"), "admission capture")
    whoami = _read_json(root / "admission-whoami.json", "whoami admission", canonical=True)
    jobs = _read_json(root / "admission-jobs.json", "jobs admission", canonical=True)
    reservation = _read_json(
        root / "admission-reservation.json", "reservation admission", canonical=True
    )
    operator = manifest["operator"]
    expected_reservation = manifest["reservation"]
    target = manifest["target"]
    require(whoami == operator, "Conductor identity changed in evidence")
    require(jobs == {"count": 0, "jobs": []}, "Conductor jobs are not empty in evidence")
    require(isinstance(reservation, dict) and set(reservation) == RESERVATION_KEYS,
            "reservation evidence fields changed")
    require(reservation["id"] == expected_reservation["id"], "reservation ID changed")
    require(reservation["date_start"] == expected_reservation["start_utc"],
            "reservation start changed")
    require(reservation["date_end"] == expected_reservation["end_utc"],
            "reservation end changed")
    require(reservation["target"] == {"id": target["id"], "name": target["hostname"]},
            "reservation target changed")
    require(reservation["batch_opt_out"] is True, "reservation is not exclusive")
    require(reservation["owner_email"] == operator["email"],
            "reservation owner changed")
    members = reservation["member_emails"]
    require(isinstance(members, list) and members == sorted(set(members))
            and operator["email"] in members,
            "reservation membership changed")
    require(isinstance(reservation["title"], str) and reservation["title"],
            "reservation title is invalid")
    return {"captured": captured, "reservation": reservation}


def _validate_authorization(
    root: Path, evidence_path: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    raw = _read_limited(root / "authorization.json", 16384)
    authorization = parse_json(raw, "authorization JSON")
    require(raw == canonical_json(authorization), "authorization evidence is not canonical JSON")
    require(isinstance(authorization, dict) and set(authorization) == AUTHORIZATION_KEYS,
            "authorization evidence fields changed")
    target = manifest["target"]
    reservation = manifest["reservation"]
    execution = manifest["execution"]
    require(authorization["schema"] == AUTHORIZATION_SCHEMA, "authorization schema changed")
    require(authorization["decision"] == DECISION, "authorization decision is not a grant")
    require(authorization["scope"] == SCOPE, "authorization scope changed")
    root_digest = _lower_digest(
        authorization["microgate_root_sha256"], "authorization microgate root"
    )
    require(authorization["reservation_id"] == reservation["id"],
            "authorization reservation changed")
    require(authorization["host"] == target["hostname"], "authorization host changed")
    require(authorization["target"] == target["fqdn"], "authorization target changed")
    require(authorization["authorized_cases"] == ["q16384-first-attention"],
            "authorization cases changed")
    for field in ("authorized_case_count", "maximum_gpu_invocations", "invocation_count"):
        require(type(authorization[field]) is int and authorization[field] == 1,
                f"authorization {field} is not one")
    require(authorization["pack_and_attention_contract_satisfied"] is False,
            "authorization omitted the deferred semantic scope")
    authorization_id = authorization["authorization_id"]
    require(isinstance(authorization_id, str) and 16 <= len(authorization_id) <= 128
            and re.fullmatch(r"[A-Za-z0-9._:-]+", authorization_id) is not None,
            "authorization ID is invalid")
    receipt_path = _safe_absolute_path(
        authorization["consumption_receipt_path"], "authorization receipt path"
    )
    result_path = _safe_absolute_path(authorization["result_path"], "authorization result path")
    require(receipt_path == execution["receipt_path"], "authorization receipt binding changed")
    require(result_path == execution["result_path"] == f"{evidence_path}/RESULT.json",
            "authorization result path does not bind the evidence directory")
    issued = parse_utc(authorization["issued_utc"], "authorization issue")
    expires = parse_utc(authorization["expires_utc"], "authorization expiry")
    return {
        "authorization": authorization,
        "authorization_id": authorization_id,
        "authorization_sha256": hashlib.sha256(raw).hexdigest(),
        "expires": expires,
        "issued": issued,
        "receipt_path": receipt_path,
        "result_path": result_path,
        "root_digest": root_digest,
    }


def _validate_validation_record(
    payload: Any,
    description: str,
    authorization: dict[str, Any],
    admission: dict[str, Any],
    manifest: dict[str, Any],
    *,
    minimum_remaining_seconds: int,
) -> dict[str, Any]:
    require(isinstance(payload, dict) and set(payload) == VALIDATION_KEYS,
            f"{description} fields changed")
    require(payload["schema"] == "loom-q16384-first-attention-launch-abi-validation-v1",
            f"{description} schema changed")
    require(payload["pass"] is True, f"{description} did not pass")
    auth = authorization["authorization"]
    reservation = manifest["reservation"]
    target = manifest["target"]
    expected = {
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "authorized_case_count": 1,
        "authorized_cases": ["q16384-first-attention"],
        "consumption_receipt_path": authorization["receipt_path"],
        "host": target["hostname"],
        "invocation_count": 1,
        "maximum_gpu_invocations": 1,
        "microgate_root_sha256": authorization["root_digest"],
        "reservation_end_epoch": _epoch(parse_utc(reservation["end_utc"], "reservation end")),
        "reservation_id": reservation["id"],
        "reservation_start_epoch": _epoch(parse_utc(reservation["start_utc"], "reservation start")),
        "result_path": authorization["result_path"],
        "target": target["fqdn"],
        "authorization_issued_epoch": _epoch(authorization["issued"]),
        "authorization_expires_epoch": _epoch(authorization["expires"]),
    }
    for key, value in expected.items():
        require(payload[key] == value, f"{description} {key} changed")
    validated = payload["validated_epoch"]
    require(type(validated) is int and expected["reservation_start_epoch"] <= validated <
            expected["reservation_end_epoch"], f"{description} timestamp is invalid")
    remaining = payload["reservation_remaining_seconds"]
    age = payload["admission_age_seconds"]
    require(isinstance(remaining, (int, float)) and not isinstance(remaining, bool)
            and math.isfinite(float(remaining)), f"{description} remaining time is invalid")
    require(isinstance(age, (int, float)) and not isinstance(age, bool)
            and math.isfinite(float(age)), f"{description} admission age is invalid")
    require(float(remaining) >= minimum_remaining_seconds,
            f"{description} lacks the required reservation margin")
    require(-30 <= float(age) <= 300, f"{description} admission age is invalid")
    require(abs(float(remaining) - (expected["reservation_end_epoch"] - validated)) <= 1.1,
            f"{description} remaining time is inconsistent")
    captured_epoch = _epoch(admission["captured"])
    require(abs(float(age) - (validated - captured_epoch)) <= 1.1,
            f"{description} admission age is inconsistent")
    require(_epoch(authorization["issued"]) <= validated < _epoch(authorization["expires"]),
            f"{description} is outside the authorization window")
    require(auth["result_path"] == payload["result_path"],
            f"{description} result binding changed")
    return payload


def _parse_receipt(root: Path) -> dict[str, str]:
    raw = _read_limited(root / "authorization-consumed.txt", 4096)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("receipt is not ASCII") from exc
    require(text.endswith("\n"), "receipt lacks final newline")
    lines = text.splitlines()
    require(len(lines) == len(RECEIPT_KEYS), "receipt field count changed")
    result: dict[str, str] = {}
    for expected_key, line in zip(RECEIPT_KEYS, lines):
        require(line.count("=") == 1, "receipt line is malformed")
        key, value = line.split("=", 1)
        require(key == expected_key and value, "receipt field order or value changed")
        result[key] = value
    return result


def _validate_receipt(
    receipt: dict[str, str], authorization: dict[str, Any], validation: dict[str, Any]
) -> dict[str, str]:
    auth = authorization["authorization"]
    expected = {
        "schema": RECEIPT_SCHEMA,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_path": receipt["authorization_path"],
        "microgate_root_sha256": authorization["root_digest"],
        "authorized_cases": "q16384-first-attention",
        "authorized_case_count": "1",
        "reservation_id": auth["reservation_id"],
        "consumption_receipt_path": authorization["receipt_path"],
        "result_path": authorization["result_path"],
        "reservation_start_epoch": str(validation["reservation_start_epoch"]),
        "reservation_end_epoch": str(validation["reservation_end_epoch"]),
        "authorization_issued_epoch": str(validation["authorization_issued_epoch"]),
        "authorization_expires_epoch": str(validation["authorization_expires_epoch"]),
        "host": auth["host"],
        "target": auth["target"],
        "maximum_gpu_invocations": "1",
        "invocation_index": "1",
        "invocation_count": "1",
        "consumed_epoch": str(validation["validated_epoch"]),
    }
    _safe_absolute_path(receipt["authorization_path"], "receipt authorization path")
    require(receipt == expected, "consumption receipt differs from authorization")
    return receipt


def _validate_package_record(payload: Any, root_digest: str) -> dict[str, Any]:
    require(isinstance(payload, dict) and set(payload) == {
        "closure_file_count", "microgate_root_sha256", "pass", "schema"
    }, "package verification fields changed")
    require(payload["schema"] ==
            "loom-q16384-first-attention-launch-abi-package-verification-v1",
            "package verification schema changed")
    require(payload["pass"] is True and payload["microgate_root_sha256"] == root_digest,
            "package verification did not bind the authorized root")
    require(type(payload["closure_file_count"]) is int and
            payload["closure_file_count"] > 0,
            "package closure file count is invalid")
    return payload


def _validate_attempt(payload: Any, authorization: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(payload, dict) and set(payload) == {
        "authorization_id", "invocation_count", "microgate_root_sha256",
        "result_path", "schema", "started_utc"
    }, "attempt record fields changed")
    require(payload["schema"] == "loom-q16384-first-attention-launch-abi-attempt-v1",
            "attempt schema changed")
    require(payload["authorization_id"] == authorization["authorization_id"],
            "attempt authorization changed")
    require(payload["microgate_root_sha256"] == authorization["root_digest"],
            "attempt root changed")
    require(payload["result_path"] == authorization["result_path"],
            "attempt result binding changed")
    require(type(payload["invocation_count"]) is int and payload["invocation_count"] == 1,
            "attempt invocation count is not one")
    parse_utc(payload["started_utc"], "attempt start")
    return payload


def _validate_run_record(
    payload: Any,
    authorization: dict[str, Any],
    receipt: dict[str, str],
    manifest: dict[str, Any],
    evidence_path: str,
    package_root: Path,
) -> dict[str, Any]:
    require(isinstance(payload, dict) and set(payload) == RUN_RECORD_KEYS,
            "run record fields changed")
    require(payload["schema"] == "loom-q16384-first-attention-launch-abi-run-record-v1",
            "run record schema changed")
    require(payload["authorization_id"] == authorization["authorization_id"] and
            payload["microgate_root_sha256"] == authorization["root_digest"],
            "run record authorization binding changed")
    require(payload["evidence_path"] == evidence_path,
            "run record evidence path changed")
    require(payload["package_root"] == str(package_root),
            "run record package root changed")
    for field in ("invocation_count",):
        require(type(payload[field]) is int and payload[field] == 1,
                f"run record {field} is not one")
    require(payload["runtime_timeout_seconds"] == RUNTIME_TIMEOUT_SECONDS,
            "runner timeout changed")
    require(payload["kill_after_seconds"] == TERMINATION_GRACE_SECONDS,
            "runner termination grace changed")
    require(payload["returncode"] == 0 and payload["timed_out"] is False,
            "runner did not exit successfully")
    require(payload["receipt_claimed_or_retired"] is True and
            payload["receipt_retired_by_launcher"] is False,
            "runner did not claim the receipt")
    require(payload["postflight_pass"] is True, "postflight did not pass")
    command = [
        f"{package_root}/bin/q16384_attention_microgate",
        "--run",
        str(package_root),
        authorization["result_path"],
        authorization["receipt_path"],
        receipt["authorization_path"],
    ]
    require(payload["command"] == command, "runner command changed")
    require(payload["supervised_command"] == [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=30s", "600s", *command
    ], "supervised runner command changed")
    operator = manifest["operator"]
    require(payload["environment"] == {
        "HOME": f"/home/{operator['short_id']}",
        "LANG": "C",
        "LC_ALL": "C",
        "MICROGATE_AUTHORIZATION_CONSUMED": "1",
        "PATH": "/usr/bin:/bin",
        "ROCR_VISIBLE_DEVICES": "0",
    }, "runner environment changed")
    started = payload["runner_started_epoch"]
    finished = payload["runner_finished_epoch"]
    require(type(started) is int and type(finished) is int and 0 <= started <= finished,
            "runner timing is invalid")
    require(finished - started <= RUNTIME_TIMEOUT_SECONDS + TERMINATION_GRACE_SECONDS,
            "runner duration exceeded the supervised bound")
    return payload


def _finite(value: Any, description: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)), f"{description} is not finite")
    return float(value)


def _validate_queue_contract(queue: Any) -> None:
    require(isinstance(queue, dict) and set(queue) == {
        "agent", "contract_satisfied", "create", "expected_size",
        "predicates", "request", "returned",
    }, "result queue contract fields changed")
    agent = queue["agent"]
    require(isinstance(agent, dict) and set(agent) == {
        "max_query_completed", "max_query_status", "max_size",
        "min_query_completed", "min_query_status", "min_size",
        "type", "type_query_completed", "type_query_status",
    }, "result queue agent fields changed")
    require(
        agent["min_query_completed"] is True
        and agent["max_query_completed"] is True
        and agent["type_query_completed"] is True
        and type(agent["min_query_status"]) is int
        and type(agent["max_query_status"]) is int
        and type(agent["type_query_status"]) is int
        and agent["min_query_status"] == 0
        and agent["max_query_status"] == 0
        and agent["type_query_status"] == 0,
        "result queue agent queries did not succeed",
    )
    minimum = agent["min_size"]
    maximum = agent["max_size"]
    require(
        type(minimum) is int
        and type(maximum) is int
        and minimum > 0
        and minimum & (minimum - 1) == 0
        and maximum >= minimum,
        "result queue agent bounds are invalid",
    )
    require(
        type(agent["type"]) is int and agent["type"] in (0, 1),
        "result queue agent type is not ordinary",
    )
    require(
        _exact_contract_value(queue["request"], {"size": 64, "type": 1}),
        "result queue request changed",
    )
    expected_size = max(64, minimum)
    require(
        type(queue["expected_size"]) is int
        and queue["expected_size"] == expected_size
        and expected_size <= maximum,
        "result expected queue size changed",
    )
    require(
        _exact_contract_value(
            queue["create"], {"attempted": True, "status": 0}
        ),
        "result queue creation did not succeed",
    )
    returned = queue["returned"]
    require(isinstance(returned, dict) and set(returned) == {
        "base_address", "doorbell_signal_handle", "features", "id",
        "pointer", "size", "type",
    }, "result returned queue fields changed")
    pointer_pattern = re.compile(r"0x[0-9a-f]{16}")
    require(
        isinstance(returned["pointer"], str)
        and pointer_pattern.fullmatch(returned["pointer"]) is not None
        and returned["pointer"] != "0x0000000000000000",
        "result queue pointer is invalid",
    )
    require(
        isinstance(returned["base_address"], str)
        and pointer_pattern.fullmatch(returned["base_address"]) is not None
        and returned["base_address"] != "0x0000000000000000",
        "result queue base address is invalid",
    )
    require(
        isinstance(returned["doorbell_signal_handle"], str)
        and pointer_pattern.fullmatch(returned["doorbell_signal_handle"]) is not None,
        "result queue doorbell handle is invalid",
    )
    returned_size = returned["size"]
    require(
        type(returned_size) is int
        and returned_size == expected_size
        and minimum <= returned_size <= maximum
        and returned_size & (returned_size - 1) == 0,
        "result returned queue size is invalid",
    )
    require(
        type(returned["type"]) is int and returned["type"] in (0, 1),
        "result returned queue type is not ordinary",
    )
    require(
        type(returned["features"]) is int
        and returned["features"] >= 0
        and returned["features"] & 1 != 0,
        "result queue lacks kernel dispatch",
    )
    require(
        type(returned["id"]) is int and returned["id"] >= 0,
        "result queue ID is invalid",
    )
    require(_exact_contract_value(queue["predicates"], {
        "base_nonnull": True,
        "contract_satisfied": True,
        "kernel_dispatch_supported": True,
        "pointer_nonnull": True,
        "size_matches_expected": True,
        "size_power_of_two": True,
        "size_within_agent_bounds": True,
        "type_ordinary": True,
    }), "result queue predicates changed")
    require(queue["contract_satisfied"] is True,
            "result queue contract is not satisfied")


def _validate_result(
    root: Path, authorization: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    result = _read_json(root / "RESULT.json", "result JSON")
    require(isinstance(result, dict) and set(result) == {
        "authorization_id", "comparison", "dispatch", "geometry",
        "gpu_execution_performed", "inputs", "microgate_root_sha256",
        "outputs", "queue_contract", "runtime", "schema", "scope", "status",
    }, "result fields changed")
    require(result["schema"] == RESULT_SCHEMA and result["status"] == "pass",
            "result is not a launch-ABI pass")
    require(result["gpu_execution_performed"] is True,
            "result does not attest GPU execution")
    require(result["authorization_id"] == authorization["authorization_id"] and
            result["microgate_root_sha256"] == authorization["root_digest"],
            "result authorization binding changed")
    require(result["scope"] == {
        "attention_only": True,
        "case": "q16384-first-attention",
        "dependency_ordering_resolved": False,
        "launch_abi_cleared": True,
        "live_tensor_semantics_resolved": False,
        "pack_semantics_resolved": False,
    }, "result semantic scope changed")

    runtime = result["runtime"]
    require(isinstance(runtime, dict) and set(runtime) == {
        "cpu_agent", "gpu_agent", "hip_runtime_sha256", "hsa_runtime_sha256",
        "isa", "rocprofiler_register_sha256", "visible_gpu_count",
    }, "result runtime fields changed")
    require(isinstance(runtime["cpu_agent"], str) and runtime["cpu_agent"],
            "result CPU agent is invalid")
    require(isinstance(runtime["gpu_agent"], str) and runtime["gpu_agent"],
            "result GPU agent is invalid")
    require(isinstance(runtime["isa"], str) and "gfx950" in runtime["isa"],
            "result ISA is not gfx950")
    require(type(runtime["visible_gpu_count"]) is int and runtime["visible_gpu_count"] == 1,
            "result visible GPU count is not one")
    for name, result_key in (
        ("hsa_runtime", "hsa_runtime_sha256"),
        ("hip_runtime", "hip_runtime_sha256"),
        ("rocprofiler_register", "rocprofiler_register_sha256"),
    ):
        require(runtime[result_key] == manifest["runtime"][name]["sha256"],
                f"result {name} digest changed")

    _validate_queue_contract(result["queue_contract"])

    require(_exact_contract_value(result["geometry"], {
        "group_segment_bytes": 26112,
        "head_dimension": 128,
        "hip_explicit_argument_bytes": 168,
        "hip_grid_blocks": [12, 1, 128],
        "kernarg_segment_bytes": 424,
        "key_end": 16768,
        "kv_heads": 2,
        "metadata_i32": [0, 16768, 1, 0, 0, 16384],
        "position_base": 384,
        "private_segment_bytes": 0,
        "query_count": 16384,
        "query_heads": 12,
        "raw_grid_workitems": [3072, 1, 128],
        "workgroup": [256, 1, 1],
    }), "result geometry changed")

    dispatch = result["dispatch"]
    require(isinstance(dispatch, dict) and set(dispatch) == {
        "hip_completed_ns", "hip_device_synchronizes", "hip_started_ns",
        "q512_code_object_loaded", "q512_dispatches", "raw_completed_ns",
        "raw_completion_waits", "raw_doorbell_written", "raw_full_header",
        "raw_hsa_first", "raw_kernarg_slot", "raw_packet_id", "raw_started_ns",
        "warmup_dispatches",
    }, "result dispatch fields changed")
    expected_dispatch = {
        "raw_hsa_first": True,
        "q512_code_object_loaded": False,
        "q512_dispatches": 0,
        "warmup_dispatches": 0,
        "raw_completion_waits": 1,
        "hip_device_synchronizes": 1,
        "raw_kernarg_slot": 0,
        "raw_doorbell_written": 1,
        "raw_full_header": "0x00031502",
    }
    for key, value in expected_dispatch.items():
        require(type(dispatch[key]) is type(value) and dispatch[key] == value,
                f"result dispatch {key} changed")
    require(type(dispatch["raw_packet_id"]) is int and dispatch["raw_packet_id"] >= 0,
            "raw packet ID is invalid")
    for key in ("raw_started_ns", "raw_completed_ns", "hip_started_ns", "hip_completed_ns"):
        require(type(dispatch[key]) is int and dispatch[key] > 0,
                f"result dispatch {key} is invalid")
    require(dispatch["raw_started_ns"] <= dispatch["raw_completed_ns"] <=
            dispatch["hip_started_ns"] <= dispatch["hip_completed_ns"],
            "raw/HIP launch ordering changed")

    inputs = result["inputs"]
    require(isinstance(inputs, dict) and set(inputs) == {
        "guard_mismatches_after_hip", "guard_mismatches_after_raw", "post_hip",
        "post_raw", "pre_raw", "unchanged_after_hip", "unchanged_after_raw",
    }, "result input fields changed")
    hashes: list[dict[str, Any]] = []
    for stage in ("pre_raw", "post_raw", "post_hip"):
        values = inputs[stage]
        require(isinstance(values, dict) and set(values) == {
            "k", "metadata", "page_indices", "q", "v"
        }, f"result {stage} input hash fields changed")
        for name, digest in values.items():
            _lower_digest(digest, f"{stage} {name} hash", nonzero=True)
        hashes.append(values)
    require(hashes[0] == hashes[1] == hashes[2], "immutable input hashes changed")
    require(inputs["unchanged_after_raw"] is True and inputs["unchanged_after_hip"] is True,
            "result reports mutated immutable inputs")
    require(type(inputs["guard_mismatches_after_raw"]) is int and
            inputs["guard_mismatches_after_raw"] == 0 and
            type(inputs["guard_mismatches_after_hip"]) is int and
            inputs["guard_mismatches_after_hip"] == 0,
            "input guard mismatch was reported")

    outputs = result["outputs"]
    require(isinstance(outputs, dict) and set(outputs) == {
        "hip", "hip_before_launch", "raw_hsa", "raw_hsa_after_hip",
        "raw_hsa_unchanged_after_hip",
    },
            "result output fields changed")
    require(_exact_contract_value(outputs["hip_before_launch"], {
        "expected_poison_values": 16384 * 12 * 128,
        "guard_mismatches": 0,
        "poison_values_present": 16384 * 12 * 128,
    }), "raw HSA modified the reserved HIP output")
    require(outputs["raw_hsa_unchanged_after_hip"] is True,
            "HIP modified the completed raw-HSA output")
    for name in ("raw_hsa", "raw_hsa_after_hip", "hip"):
        observation = outputs[name]
        require(isinstance(observation, dict) and set(observation) == {
            "guard_mismatches", "max_absolute_error", "max_tolerance_ratio",
            "nonfinite_values", "poison_values_remaining", "reference_violations",
            "sampled_elements_compared", "sha256",
        }, f"{name} observation fields changed")
        _lower_digest(observation["sha256"], f"{name} output hash", nonzero=True)
        for field in (
            "guard_mismatches", "nonfinite_values", "poison_values_remaining",
            "reference_violations",
        ):
            require(type(observation[field]) is int and observation[field] == 0,
                    f"{name} {field} is nonzero")
        require(type(observation["sampled_elements_compared"]) is int and
                observation["sampled_elements_compared"] == 8 * 12 * 128,
                f"{name} sampled element count changed")
        require(_finite(observation["max_absolute_error"],
                        f"{name} maximum absolute error") >= 0,
                f"{name} maximum absolute error is negative")
        ratio = _finite(observation["max_tolerance_ratio"],
                        f"{name} maximum tolerance ratio")
        require(0 <= ratio <= 1, f"{name} reference tolerance was exceeded")
    require(outputs["raw_hsa"]["sha256"] == outputs["hip"]["sha256"],
            "raw-HSA and HIP output hashes differ")
    require(outputs["raw_hsa_after_hip"] == outputs["raw_hsa"],
            "raw-HSA output or guards changed after HIP")

    require(_exact_contract_value(result["comparison"], {
        "byte_equal": True,
        "bytes_compared": 16384 * 12 * 128 * 2,
        "first_mismatch_byte": None,
        "first_mismatch_dimension": None,
        "first_mismatch_element": None,
        "first_mismatch_head": None,
        "first_mismatch_row": None,
        "mismatch_bytes": 0,
    }), "raw-HSA and HIP output comparison changed")
    return result


def _validate_pass_record(
    payload: Any,
    root: Path,
    authorization: dict[str, Any],
    run_record: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, Any]:
    require(isinstance(payload, dict) and set(payload) == {
        "authorization_id", "completed_epoch", "invocation_count",
        "microgate_root_sha256", "postflight_pass", "result_sha256",
        "runner_returncode", "schema", "status"
    }, "PASS record fields changed")
    require(payload["schema"] == "loom-q16384-first-attention-launch-abi-evidence-pass-v1",
            "PASS record schema changed")
    require(payload["status"] == "pass" and payload["postflight_pass"] is True,
            "PASS record is not a postflight pass")
    require(payload["authorization_id"] == authorization["authorization_id"] and
            payload["microgate_root_sha256"] == authorization["root_digest"],
            "PASS authorization binding changed")
    require(type(payload["invocation_count"]) is int and
            payload["invocation_count"] == 1 and
            type(payload["runner_returncode"]) is int and
            payload["runner_returncode"] == 0,
            "PASS invocation result changed")
    require(payload["result_sha256"] == hashlib.sha256(
        _read_limited(root / "RESULT.json")
    ).hexdigest(), "PASS result digest changed")
    completed = payload["completed_epoch"]
    require(type(completed) is int and completed >= run_record["runner_finished_epoch"]
            and completed >= completion["validated_epoch"],
            "PASS completion time is too early")
    return payload


def validate_success_evidence(
    evidence: Path,
    *,
    verify_outer_seal: bool = True,
    package_root: Path | None = None,
) -> dict[str, Any]:
    package = (package_root or PACKAGE_ROOT).resolve(strict=True)
    manifest = load_manifest(package)
    root = evidence.resolve(strict=True)
    if verify_outer_seal:
        evidence_digest, evidence_count = verify_tree(root, exclude_build=False)
    else:
        evidence_digest = "unsealed"
        evidence_count = len(collect_files(root, exclude_build=False))
    actual_files = {path.as_posix() for path in collect_files(root, exclude_build=False)}
    require(actual_files == EXPECTED_SUCCESS_FILES,
            f"successful evidence file set changed: "
            f"missing={sorted(EXPECTED_SUCCESS_FILES - actual_files)} "
            f"extra={sorted(actual_files - EXPECTED_SUCCESS_FILES)}")
    require(evidence_count == len(EXPECTED_SUCCESS_FILES),
            "evidence manifest count differs from exact success set")

    evidence_path = _safe_absolute_path(str(root), "evidence path")
    require(evidence_path == manifest["execution"]["evidence_path"],
            "evidence directory differs from package manifest")
    admission = _validate_admission(root, manifest)
    authorization = _validate_authorization(root, evidence_path, manifest)
    reservation_start = parse_utc(manifest["reservation"]["start_utc"], "reservation start")
    reservation_end = parse_utc(manifest["reservation"]["end_utc"], "reservation end")
    require(reservation_start <= authorization["issued"] < authorization["expires"]
            <= reservation_end, "authorization window is outside the reservation")

    initial = _validate_validation_record(
        _read_json(root / "initial-validation.json", "initial validation", canonical=True),
        "initial validation", authorization, admission, manifest,
        minimum_remaining_seconds=INITIAL_MINIMUM_REMAINING_SECONDS,
    )
    prelaunch = _validate_validation_record(
        _read_json(root / "prelaunch-validation.json", "prelaunch validation", canonical=True),
        "prelaunch validation", authorization, admission, manifest,
        minimum_remaining_seconds=PRELAUNCH_MINIMUM_REMAINING_SECONDS,
    )
    require(initial["validated_epoch"] <= prelaunch["validated_epoch"],
            "prelaunch validation predates initial validation")
    receipt = _validate_receipt(_parse_receipt(root), authorization, prelaunch)
    package_record = _validate_package_record(
        _read_json(root / "package-verification.json", "package verification", canonical=True),
        authorization["root_digest"],
    )
    attempt = _validate_attempt(
        _read_json(root / "attempt.json", "attempt record", canonical=True), authorization
    )
    run_record = _validate_run_record(
        _read_json(root / "run-record.json", "run record", canonical=True),
        authorization, receipt, manifest, evidence_path, package,
    )
    attempt_epoch = _epoch(parse_utc(attempt["started_utc"], "attempt start"))
    require(attempt_epoch <= run_record["runner_started_epoch"] + 1,
            "attempt timestamp follows runner start")
    require(prelaunch["validated_epoch"] <= run_record["runner_started_epoch"] + 1,
            "runner predates prelaunch validation")
    require(_read_limited(root / "runner.stdout.txt") ==
            b"q16384 first-attention launch-ABI microgate: PASS\n",
            "runner stdout does not contain the exact success line")
    require(_read_limited(root / "runner.stderr.txt") == b"",
            "runner stderr is not empty on success")
    result = _validate_result(root, authorization, manifest)

    snapshots = {
        stage: validate_snapshot_directory(root / stage, stage)
        for stage in ("preflight", "prelaunch", "postflight")
    }
    for stage in snapshots:
        for name in (
            "allowed-monitor-inspect.stderr.txt",
            "amd-smi-metric.stderr.txt",
            "amd-smi-process.stderr.txt",
        ):
            require(_read_limited(root / stage / name) == b"",
                    f"{stage} command emitted stderr: {name}")
    comparison = compare_amd_health(
        _read_limited(root / "preflight" / "amd-smi-metric.json"),
        _read_limited(root / "postflight" / "amd-smi-metric.json"),
    )
    require(_read_json(root / "pre-post-health-comparison.json",
                       "pre/post health comparison", canonical=True) == comparison,
            "pre/post health comparison differs")
    require(snapshots["preflight"]["monitor"] == snapshots["postflight"]["monitor"],
            "allowed monitor changed during the run")

    completion = _read_json(
        root / "reservation-completion-validation.json",
        "reservation completion validation",
        canonical=True,
    )
    require(isinstance(completion, dict) and type(completion.get("validated_epoch")) is int,
            "reservation completion validation is invalid")
    target = manifest["target"]
    operator = manifest["operator"]
    recomputed_completion = validate_reservation_completion(
        admission["reservation"],
        manifest["reservation"]["id"],
        target["id"],
        target["hostname"],
        operator["email"],
        now_epoch=completion["validated_epoch"],
    )
    require(completion == recomputed_completion,
            "reservation completion validation differs from admission")
    require(completion["validated_epoch"] >= run_record["runner_finished_epoch"],
            "reservation completion predates runner finish")
    validate_kernel_evidence(root, run_record)
    require(_read_limited(root / "kernel-log-during-run.stderr.txt") == b"",
            "kernel log capture emitted stderr")

    pass_record = _validate_pass_record(
        _read_json(root / "PASS.json", "PASS record", canonical=True),
        root, authorization, run_record, completion,
    )
    require(pass_record["completed_epoch"] < _epoch(reservation_end),
            "PASS was recorded after reservation expiry")
    return {
        "authorization_id": authorization["authorization_id"],
        "evidence_file_count": evidence_count,
        "evidence_root_sha256": evidence_digest,
        "microgate_root_sha256": authorization["root_digest"],
        "package_closure_file_count": package_record["closure_file_count"],
        "pass": True,
        "result_status": result["status"],
        "schema": "loom-q16384-first-attention-launch-abi-evidence-verification-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    result = validate_success_evidence(args.evidence)
    print(
        f"evidence semantic verification: PASS "
        f"({result['evidence_file_count']} files, {result['evidence_root_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
