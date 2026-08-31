#!/usr/bin/python3 -I
"""Verify the sealed q16384 launch-ABI package and its bound inputs."""

from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seal_lib import verify_package


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    digest, count = verify_package(root)
    print(f"seal verification: PASS ({count} files, {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
