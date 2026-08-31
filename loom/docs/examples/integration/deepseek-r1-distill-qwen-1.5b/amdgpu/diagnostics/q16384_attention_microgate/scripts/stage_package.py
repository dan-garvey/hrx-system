#!/usr/bin/python3 -I
"""Construct, host-test, and optionally seal a minimal runnable package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_manifest import canonical_json, promote_manifest, validate_manifest  # noqa: E402
from seal_lib import (  # noqa: E402
    BOUND_INPUTS_SCHEMA,
    EXPECTED_ARCHIVED_ROOT,
    EXPECTED_BUILD_HSA_SHA256,
    EXPECTED_BUILD_ROCPROFILER_SHA256,
    sha256_file,
    tree_digest,
    verify_tree,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_INTEGRATION_ROOT = SOURCE_ROOT.parents[1]
DEFAULT_ARCHIVE = Path(
    "/home/dan/codex-work/loom-aiter-queue-native-attention-microgate-20260830T213507Z"
)
OUTPUT_PARENT = Path("/home/dan/codex-work")
TOP_LEVEL_FILES = (
    "CONTRACT.json",
    "Makefile",
    "PACKAGE_MANIFEST.template.json",
    "README.md",
)
SOURCE_FILES = (
    "q16384_attention_microgate.c",
    "reference.c",
    "reference.h",
    "sha256.c",
    "sha256.h",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _copy_file(source: Path, destination: Path) -> None:
    require(source.is_file() and not source.is_symlink(), f"invalid source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _copy_source_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def _render_deployment_id(path: Path, deployment_id: str) -> None:
    template = path.read_text(encoding="ascii")
    require("DEPLOYMENT_ID" in template, f"deployment placeholder is missing: {path}")
    path.write_text(template.replace("DEPLOYMENT_ID", deployment_id), encoding="ascii")


def _record(path: Path, relative: str) -> dict[str, Any]:
    source = path / relative
    return {
        "path": relative,
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _write_bound_inputs(output: Path) -> None:
    header_digest, header_count = tree_digest(output / "toolchain/include/hsa")
    payload = {
        "archived_input": {
            "role": "build-toolchain-and-kernel-source",
            "root_sha256": EXPECTED_ARCHIVED_ROOT,
        },
        "build_toolchain": {
            "hsa_headers": {
                "file_count": header_count,
                "path": "toolchain/include/hsa",
                "tree_sha256": header_digest,
            },
            "hsa_runtime": _record(output, "toolchain/lib/libhsa-runtime64.so.1"),
            "rocprofiler_register": _record(
                output, "toolchain/lib/librocprofiler-register.so.0"
            ),
        },
        "code_object": _record(
            output, "artifacts/aiter_prefill_q16384_gfx950.hsaco"
        ),
        "schema": BOUND_INPUTS_SCHEMA,
        "target_runtime": json.loads(
            (SOURCE_ROOT / "PACKAGE_MANIFEST.template.json").read_text(encoding="ascii")
        )["runtime"],
    }
    require(payload["build_toolchain"]["hsa_runtime"]["sha256"] ==
            EXPECTED_BUILD_HSA_SHA256, "archived HSA runtime hash changed")
    require(payload["build_toolchain"]["rocprofiler_register"]["sha256"] ==
            EXPECTED_BUILD_ROCPROFILER_SHA256,
            "archived rocprofiler-register hash changed")
    (output / "BOUND_INPUTS.json").write_bytes(canonical_json(payload))


def _write_manifest(output: Path, deployment_id: str) -> None:
    template = json.loads(
        (SOURCE_ROOT / "PACKAGE_MANIFEST.template.json").read_text(encoding="ascii")
    )
    replacement = json.loads(json.dumps(template).replace("DEPLOYMENT_ID", deployment_id))
    commit = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "--short=12", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    replacement["source"]["git_commit"] = commit
    validate_manifest(
        replacement, require_artifact_hashes=False, allow_template=True
    )
    (output / "PACKAGE_MANIFEST.json").write_bytes(canonical_json(replacement))


def _write_authorization_template(output: Path, manifest: dict[str, Any]) -> None:
    authorization = {
        "authorization_id": "REPLACE_WITH_REVIEWER_GRANT_ID",
        "authorized_case_count": 1,
        "authorized_cases": ["q16384-first-attention"],
        "consumption_receipt_path": manifest["execution"]["receipt_path"],
        "decision": "DO_NOT_RUN_TEMPLATE",
        "expires_utc": manifest["reservation"]["end_utc"],
        "host": manifest["target"]["hostname"],
        "invocation_count": 1,
        "issued_utc": manifest["reservation"]["start_utc"],
        "maximum_gpu_invocations": 1,
        "microgate_root_sha256": "REPLACE_WITH_FINAL_SEALED_ROOT_SHA256",
        "pack_and_attention_contract_satisfied": False,
        "reservation_id": manifest["reservation"]["id"],
        "result_path": manifest["execution"]["result_path"],
        "schema": "loom-q16384-first-attention-launch-abi-one-run-authorization-v1",
        "scope": "q16384-first-attention-raw-hsa-vs-hip",
        "target": manifest["target"]["fqdn"],
    }
    raw = (json.dumps(authorization, sort_keys=True, separators=(",", ":")) + "\n")
    (output / "AUTHORIZATION_TEMPLATE.json").write_text(raw, encoding="ascii")


def _run(output: Path, argv: list[str]) -> None:
    subprocess.run(
        argv,
        cwd=output,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def stage(output: Path, archive: Path, deployment_id: str, *, seal: bool) -> dict[str, Any]:
    require(re.fullmatch(r"[A-Za-z0-9._-]+", deployment_id) is not None,
            "deployment ID is unsafe")
    output = output.absolute()
    require(output.parent.resolve(strict=True) == OUTPUT_PARENT,
            f"output must be a direct child of {OUTPUT_PARENT}")
    require(not output.exists(), f"output already exists: {output}")
    archive = archive.resolve(strict=True)
    archived_digest, _ = verify_tree(archive)
    require(archived_digest == EXPECTED_ARCHIVED_ROOT, "archived input root changed")

    output.mkdir(mode=0o755)
    for relative in TOP_LEVEL_FILES:
        _copy_file(SOURCE_ROOT / relative, output / relative)
    _render_deployment_id(output / "README.md", deployment_id)
    _copy_source_tree(SOURCE_ROOT / "scripts", output / "scripts")
    _copy_source_tree(SOURCE_ROOT / "tests", output / "tests")
    for name in SOURCE_FILES:
        _copy_file(SOURCE_ROOT / "src" / name, output / "src" / name)
    for name in ("q16k_aiter_integration.c", "q16k_aiter_integration.h"):
        _copy_file(CANONICAL_INTEGRATION_ROOT / name, output / "src" / name)

    _copy_file(
        archive / "vendor/queue_native/artifacts/aiter_prefill_q16384_gfx950.hsaco",
        output / "artifacts/aiter_prefill_q16384_gfx950.hsaco",
    )
    shutil.copytree(
        archive / "toolchain/include/hsa",
        output / "toolchain/include/hsa",
        copy_function=shutil.copy2,
    )
    for name in ("libhsa-runtime64.so.1", "librocprofiler-register.so.0"):
        _copy_file(archive / "toolchain/lib" / name, output / "toolchain/lib" / name)

    _write_bound_inputs(output)
    _write_manifest(output, deployment_id)
    _run(output, ["make", "all"])
    manifest = promote_manifest(output)
    _write_authorization_template(output, manifest)
    _run(output, ["make", "check"])
    _run(output, [sys.executable, "-I", "scripts/offline_review.py"])
    if seal:
        _run(output, [sys.executable, "-I", "scripts/seal.py"])
    return {
        "deployment_id": deployment_id,
        "package_root": str(output),
        "sealed": seal,
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("deployment_id")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    result = stage(args.output, args.archive, args.deployment_id, seal=args.seal)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
