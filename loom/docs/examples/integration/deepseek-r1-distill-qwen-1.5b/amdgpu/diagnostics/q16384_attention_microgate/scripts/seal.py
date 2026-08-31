#!/usr/bin/python3 -I
"""Seal a reviewed q16384 launch-ABI package exactly once."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seal_lib import seal_package, verify_bound_inputs  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    review = json.loads((root / "OFFLINE_REVIEW.json").read_text(encoding="ascii"))
    if (
        review.get("status") != "pass"
        or review.get("gpu_execution_performed") is not False
        or review.get("hsa_initialized") is not False
    ):
        raise RuntimeError("passing no-GPU offline review is required before sealing")
    verify_bound_inputs(root)
    digest, count = seal_package(root)
    print(f"sealed {count} files: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
