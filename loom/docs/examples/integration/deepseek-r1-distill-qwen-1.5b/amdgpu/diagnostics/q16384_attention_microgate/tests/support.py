"""Shared manifest fixtures for host-only tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KERNEL_SHA256 = "0e90cc246649934b27c886187dea4d7b960266e921edbb0b9b7d0b38eb5e4edc"


def write_manifest(
    package_root: Path,
    evidence: Path,
    state: Path,
    *,
    target: dict[str, str] | None = None,
    operator: dict[str, str] | None = None,
) -> dict[str, Any]:
    template = json.loads(
        (ROOT / "PACKAGE_MANIFEST.template.json").read_text(encoding="ascii")
    )
    template["status"] = "READY_FOR_AUTHORIZATION"
    template["source"]["git_commit"] = "test-commit"
    template["execution"].update({
        "evidence_path": str(evidence),
        "receipt_path": str(state / "q16384-first-attention-one-run.receipt"),
        "result_path": str(evidence / "RESULT.json"),
        "state_path": str(state),
    })
    if target is not None:
        template["target"] = target
    if operator is not None:
        template["operator"] = operator
    for name, item in template["artifacts"].items():
        item["sha256"] = KERNEL_SHA256 if name == "q16384_kernel" else "1" * 64
        item["size_bytes"] = 1
    package_root.mkdir(mode=0o700)
    (package_root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return template


def canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
