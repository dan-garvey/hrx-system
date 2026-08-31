#!/usr/bin/python3 -I
"""Create and strictly validate the q16384 launch-ABI package manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "loom-q16384-first-attention-launch-abi-package-v1"
READY_STATUS = "READY_FOR_AUTHORIZATION"
TEMPLATE_STATUS = "STAGING_TEMPLATE"
EXPECTED_CASES = ["q16384-first-attention"]
EXPECTED_KERNEL_SHA256 = (
    "0e90cc246649934b27c886187dea4d7b960266e921edbb0b9b7d0b38eb5e4edc"
)
EXPECTED_RUNTIME_HASHES = {
    "hsa_runtime": "d41abc620d2f228995f809b55c7e4183f513501cc070416316479f5eeaa58253",
    "hip_runtime": "1e9c69bed92a2cd458468e6af29707b4ebbbb94a8361adbb95d30d38b82074d4",
    "rocprofiler_register": "8faae1b02f834857d75b1f318bf826d371ff9eb30b3af5b2d26a279f4525a953",
}
EXPECTED_RUNTIME_PATHS = {
    "hsa_runtime": "/opt/rocm/lib/libhsa-runtime64.so.1",
    "hip_runtime": "/opt/rocm/lib/libamdhip64.so",
    "rocprofiler_register": "/opt/rocm/lib/librocprofiler-register.so.0",
}
ARTIFACT_PATHS = {
    "integration_header": "src/q16k_aiter_integration.h",
    "integration_source": "src/q16k_aiter_integration.c",
    "q16384_kernel": "artifacts/aiter_prefill_q16384_gfx950.hsaco",
    "reference_source": "src/reference.c",
    "runner": "bin/q16384_attention_microgate",
    "runner_source": "src/q16384_attention_microgate.c",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: Any, description: str) -> dt.datetime:
    require(isinstance(value, str), f"{description} is not a string")
    require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value)
        is not None,
        f"{description} is not canonical UTC",
    )
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_absolute(value: Any, description: str) -> str:
    require(isinstance(value, str), f"{description} is not a string")
    path = PurePosixPath(value)
    require(path.is_absolute() and path.as_posix() == value,
            f"{description} is not an absolute canonical path")
    require(all(part not in ("", ".", "..") for part in path.parts[1:]),
            f"{description} contains an unsafe component")
    require(re.fullmatch(r"/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+", value) is not None,
            f"{description} contains unsafe characters")
    return value


def _safe_relative(value: Any, description: str) -> str:
    require(isinstance(value, str) and value, f"{description} is invalid")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value,
            f"{description} is not canonical relative")
    require(all(part not in ("", ".", "..") for part in path.parts),
            f"{description} contains an unsafe component")
    return value


def validate_manifest(
    payload: Any,
    *,
    require_artifact_hashes: bool = True,
    allow_template: bool = False,
) -> dict[str, Any]:
    require(isinstance(payload, dict), "package manifest is not an object")
    require(
        set(payload) == {
            "artifacts", "execution", "operator", "reservation", "runtime",
            "schema", "scope", "source", "status", "target",
        },
        "package manifest fields changed",
    )
    require(payload["schema"] == SCHEMA, "package manifest schema changed")
    expected_statuses = {READY_STATUS, TEMPLATE_STATUS} if allow_template else {READY_STATUS}
    require(payload["status"] in expected_statuses, "package is not ready for authorization")

    target = payload["target"]
    require(isinstance(target, dict) and set(target) == {"fqdn", "hostname", "id"},
            "target manifest is invalid")
    require(all(isinstance(target[key], str) and target[key] for key in target),
            "target manifest contains an empty value")

    operator = payload["operator"]
    require(isinstance(operator, dict) and set(operator) == {"email", "id", "short_id"},
            "operator manifest is invalid")
    require(all(isinstance(operator[key], str) and operator[key] for key in operator),
            "operator manifest contains an empty value")

    reservation = payload["reservation"]
    require(isinstance(reservation, dict) and set(reservation) == {
        "batch_opt_out", "end_utc", "id", "start_utc"
    }, "reservation manifest is invalid")
    require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                         str(reservation["id"])) is not None,
            "reservation ID is invalid")
    require(parse_utc(reservation["start_utc"], "reservation start") <
            parse_utc(reservation["end_utc"], "reservation end"),
            "reservation window is empty")
    require(reservation["batch_opt_out"] is True, "reservation is not exclusive")

    execution = payload["execution"]
    require(isinstance(execution, dict) and set(execution) == {
        "authorized_cases", "evidence_path", "maximum_gpu_invocations",
        "maximum_retries", "receipt_path", "result_path", "state_path",
    }, "execution manifest is invalid")
    require(execution["authorized_cases"] == EXPECTED_CASES,
            "authorized case set changed")
    require(type(execution["maximum_gpu_invocations"]) is int and
            execution["maximum_gpu_invocations"] == 1,
            "maximum GPU invocation count is not one")
    require(type(execution["maximum_retries"]) is int and
            execution["maximum_retries"] == 0,
            "maximum retry count is not zero")
    evidence = _safe_absolute(execution["evidence_path"], "evidence path")
    state = _safe_absolute(execution["state_path"], "state path")
    require(_safe_absolute(execution["result_path"], "result path") ==
            f"{evidence}/RESULT.json", "result path is not evidence-bound")
    require(_safe_absolute(execution["receipt_path"], "receipt path") ==
            f"{state}/q16384-first-attention-one-run.receipt",
            "receipt path is not state-bound")

    require(payload["scope"] == {
        "attention_only": True,
        "dependency_ordering_resolved": False,
        "live_tensor_semantics_resolved": False,
        "pack_semantics_resolved": False,
        "test": "raw-hsa-vs-hip-launch-abi",
    }, "scope manifest changed")

    runtime = payload["runtime"]
    require(isinstance(runtime, dict) and set(runtime) == set(EXPECTED_RUNTIME_HASHES),
            "runtime manifest changed")
    for name, digest in EXPECTED_RUNTIME_HASHES.items():
        item = runtime[name]
        require(isinstance(item, dict) and set(item) == {"path", "sha256"},
                f"{name} runtime manifest is invalid")
        require(item["path"] == EXPECTED_RUNTIME_PATHS[name],
                f"{name} runtime path changed")
        require(item["sha256"] == digest, f"{name} runtime hash changed")

    artifacts = payload["artifacts"]
    require(isinstance(artifacts, dict) and set(artifacts) == set(ARTIFACT_PATHS),
            "artifact manifest is invalid")
    for name, expected_path in ARTIFACT_PATHS.items():
        artifact = artifacts[name]
        require(isinstance(artifact, dict) and set(artifact) == {
            "path", "sha256", "size_bytes"
        }, f"{name} artifact manifest is invalid")
        require(_safe_relative(artifact["path"], f"{name} artifact path") == expected_path,
                f"{name} artifact path changed")
        require(type(artifact["size_bytes"]) is int and artifact["size_bytes"] > 0,
                f"{name} artifact size is invalid")
        digest = artifact["sha256"]
        if require_artifact_hashes or payload["status"] == READY_STATUS:
            require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                    f"{name} artifact SHA-256 is invalid")
        else:
            require(digest == "GENERATED" or
                    re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                    f"{name} artifact SHA-256 placeholder is invalid")
    if payload["status"] == READY_STATUS:
        require(artifacts["q16384_kernel"]["sha256"] == EXPECTED_KERNEL_SHA256,
                "q16384 kernel digest changed")
    require(all("q512" not in artifact["path"].lower()
                for artifact in artifacts.values()),
            "q512 artifact dependency is forbidden")

    source = payload["source"]
    require(isinstance(source, dict) and set(source) == {
        "archived_input_root_sha256", "canonical_path", "git_commit"
    }, "source manifest is invalid")
    require(re.fullmatch(r"[0-9a-f]{64}", source["archived_input_root_sha256"]) is not None,
            "archived input root is invalid")
    require(isinstance(source["canonical_path"], str) and source["canonical_path"],
            "canonical source path is invalid")
    require(isinstance(source["git_commit"], str) and source["git_commit"],
            "source commit is invalid")
    return payload


def load_manifest(
    root: Path,
    *,
    require_artifact_hashes: bool = True,
    allow_template: bool = False,
) -> dict[str, Any]:
    path = root / "PACKAGE_MANIFEST.json"
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("package manifest is invalid JSON") from exc
    require(raw == canonical_json(payload), "package manifest is not canonical JSON")
    return validate_manifest(
        payload,
        require_artifact_hashes=require_artifact_hashes,
        allow_template=allow_template,
    )


def artifact_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    require(path.is_file() and not path.is_symlink(), f"missing artifact: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def promote_manifest(root: Path) -> dict[str, Any]:
    manifest = load_manifest(
        root, require_artifact_hashes=False, allow_template=True
    )
    manifest["artifacts"] = {
        name: artifact_record(root, relative)
        for name, relative in ARTIFACT_PATHS.items()
    }
    manifest["status"] = READY_STATUS
    validate_manifest(manifest)
    (root / "PACKAGE_MANIFEST.json").write_bytes(canonical_json(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-template", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--capture-fields", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = promote_manifest(root) if args.promote else load_manifest(
        root,
        require_artifact_hashes=not args.allow_template,
        allow_template=args.allow_template,
    )
    if args.capture_fields:
        print(manifest["reservation"]["id"])
        print(manifest["target"]["id"])
        print(manifest["target"]["hostname"])
        print(manifest["operator"]["email"])
    else:
        print(json.dumps({"schema": manifest["schema"], "status": manifest["status"]},
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
