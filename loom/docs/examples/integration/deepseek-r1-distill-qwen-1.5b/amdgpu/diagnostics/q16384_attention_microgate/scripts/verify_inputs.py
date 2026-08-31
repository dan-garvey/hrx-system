#!/usr/bin/python3 -I
"""Verify staged artifacts and build inputs without initializing HSA."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seal_lib import verify_bound_inputs  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify_bound_inputs(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
