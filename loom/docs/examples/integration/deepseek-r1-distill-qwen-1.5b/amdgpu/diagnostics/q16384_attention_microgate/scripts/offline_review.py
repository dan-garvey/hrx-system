#!/usr/bin/python3 -I
"""Run the complete host-only review and write OFFLINE_REVIEW.json."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_manifest import load_manifest  # noqa: E402
from run_once import (  # noqa: E402
    MINIMUM_POSTFLIGHT_RESERVE_SECONDS,
    PRELAUNCH_MINIMUM_REMAINING_SECONDS,
    RUNTIME_TIMEOUT_SECONDS,
    TERMINATION_GRACE_SECONDS,
)
from seal_lib import collect_files, sha256_file, verify_bound_inputs  # noqa: E402
from verify_evidence import EXPECTED_SUCCESS_FILES  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def command_record(argv: list[str]) -> dict[str, object]:
    started = dt.datetime.now(dt.timezone.utc)
    process = subprocess.run(
        argv,
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = (process.stdout + process.stderr).encode("utf-8")
    return {
        "argv": argv,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "returncode": process.returncode,
        "started_utc": started.isoformat(),
        "stderr": process.stderr,
        "stdout": process.stdout,
    }


def main() -> int:
    manifest = load_manifest(ROOT)
    records = [command_record(["make", "clean"]), command_record(["make", "check"])]
    binary = ROOT / manifest["artifacts"]["runner"]["path"]
    input_report = verify_bound_inputs(ROOT)
    nm = subprocess.run(
        ["nm", "-D", "--undefined-only", str(binary)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    symbols = [line.split()[-1].split("@", 1)[0] for line in nm.splitlines() if line.split()]
    hsa_imports = sorted({symbol for symbol in symbols if symbol.startswith("hsa_")})
    dynamic = subprocess.run(
        ["readelf", "-d", str(binary)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    needed = sorted(re.findall(r"Shared library: \[([^]]+)\]", dynamic))
    runpaths = re.findall(r"Library runpath: \[([^]]+)\]", dynamic)
    passed = (
        all(record["returncode"] == 0 for record in records)
        and input_report["pass"] is True
        and binary.is_file()
        and sha256_file(binary) == manifest["artifacts"]["runner"]["sha256"]
        and "libhsa-runtime64.so.1" in needed
        and "libamdhip64.so" not in needed
        and runpaths == ["/opt/rocm/lib"]
        and "BIND_NOW" in dynamic
        and RUNTIME_TIMEOUT_SECONDS
        + TERMINATION_GRACE_SECONDS
        + MINIMUM_POSTFLIGHT_RESERVE_SECONDS
        == PRELAUNCH_MINIMUM_REMAINING_SECONDS
    )
    report = {
        "authorization_consumed": False,
        "commands": records,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_execution_performed": False,
        "hsa_imports": hsa_imports,
        "hsa_initialized": False,
        "input_verification": input_report,
        "live_execution_contract": {
            "maximum_runner_seconds": RUNTIME_TIMEOUT_SECONDS,
            "minimum_postflight_reserve_seconds": MINIMUM_POSTFLIGHT_RESERVE_SECONDS,
            "prelaunch_minimum_remaining_seconds": PRELAUNCH_MINIMUM_REMAINING_SECONDS,
            "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
        },
        "outer_closure_file_count_before_review": len(collect_files(ROOT)),
        "package_root": str(ROOT),
        "runner_dynamic_contract": {
            "bind_now": "BIND_NOW" in dynamic,
            "direct_needed": needed,
            "hip_loaded_with_dlopen": "libamdhip64.so" not in needed,
            "runpath": runpaths,
        },
        "runner_sha256": sha256_file(binary) if binary.is_file() else None,
        "schema": "loom-q16384-first-attention-launch-abi-offline-review-v1",
        "semantic_evidence_contract": {
            "exact_success_payload_file_count": len(EXPECTED_SUCCESS_FILES),
            "requires_raw_hsa_hip_byte_equality": True,
            "requires_result_status_pass": True,
            "requires_runner_claimed_receipt": True,
        },
        "status": "pass" if passed else "fail",
    }
    temporary = ROOT / f"OFFLINE_REVIEW.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, ROOT / "OFFLINE_REVIEW.json")
    print(json.dumps({"checks": len(records), "pass": passed}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
