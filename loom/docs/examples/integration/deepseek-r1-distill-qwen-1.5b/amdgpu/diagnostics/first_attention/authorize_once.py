#!/usr/bin/env python3
"""Creates one exclusive, short-lived authorization for a diagnostic POST."""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from authorization import atomic_json_exclusive, make_authorization


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--body", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--sealed-package", required=True, type=Path)
    parser.add_argument("--sealed-package-root-sha256", required=True)
    parser.add_argument("--deployment-manifest", required=True, type=Path)
    parser.add_argument("--reservation-snapshot", required=True, type=Path)
    parser.add_argument("--target-hostname", required=True)
    parser.add_argument("--reservation-id", required=True)
    parser.add_argument("--reservation-end-utc", required=True)
    parser.add_argument("--diagnostic-output", required=True, type=Path)
    parser.add_argument("--attempt-marker", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--response-body", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authorization-id")
    parser.add_argument("--valid-for-seconds", type=int, default=3600)
    args = parser.parse_args()

    if args.valid_for_seconds <= 0 or args.valid_for_seconds > 86400:
        raise ValueError("valid-for-seconds must be in [1, 86400]")
    if any(
        not path.is_file()
        for path in (
            args.body,
            args.adapter,
            args.worker,
            args.deployment_manifest,
            args.reservation_snapshot,
        )
    ) or not args.sealed_package.is_dir():
        raise FileNotFoundError(
            "body, adapter, worker, deployment manifest, reservation snapshot, "
            "and sealed package must exist"
        )
    for label, path in (
        ("body", args.body),
        ("adapter", args.adapter),
        ("worker", args.worker),
        ("sealed package", args.sealed_package),
        ("deployment manifest", args.deployment_manifest),
        ("reservation snapshot", args.reservation_snapshot),
        ("diagnostic output", args.diagnostic_output),
        ("attempt marker", args.attempt_marker),
        ("result", args.result),
        ("response body", args.response_body),
        ("authorization output", args.output),
    ):
        if not path.is_absolute():
            raise ValueError(f"{label} path must be absolute: {path}")
    for path in (
        args.diagnostic_output,
        args.attempt_marker,
        args.result,
        args.response_body,
    ):
        if path.exists():
            raise FileExistsError(f"authorized output already exists: {path}")
    authorization_id = args.authorization_id or (
        "loom-q16k-first-attention-" + secrets.token_hex(16)
    )
    manifest = make_authorization(
        authorization_id,
        args.url,
        args.body,
        args.adapter,
        args.worker,
        args.sealed_package,
        args.sealed_package_root_sha256,
        args.deployment_manifest,
        args.reservation_snapshot,
        args.target_hostname,
        args.reservation_id,
        args.reservation_end_utc,
        args.diagnostic_output,
        args.attempt_marker,
        args.result,
        args.response_body,
        args.valid_for_seconds,
    )
    atomic_json_exclusive(args.output, manifest)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
