#!/usr/bin/env python3
"""Issues one authorized diagnostic POST after one endpoint identity GET."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from authorization import (
    FORBIDDEN_ENV_NAMES,
    FORBIDDEN_ENV_PREFIXES,
    atomic_json_exclusive,
    health_url_for,
    publish_bytes_exclusive,
    require_absolute,
    sha256_bytes,
    sha256_file,
    validate_authorization,
)


MAX_HEALTH_BYTES = 8 * 1024 * 1024


def reject_profiler_environment(
    environment: dict[str, str], description: str
) -> None:
    forbidden = []
    for name, value in environment.items():
        if not value:
            continue
        if name in FORBIDDEN_ENV_NAMES or name.startswith(FORBIDDEN_ENV_PREFIXES):
            forbidden.append(name)
    if forbidden:
        raise RuntimeError(
            f"{description} has forbidden profiler/injection environment: "
            + ", ".join(sorted(forbidden))
        )


def local_hostname() -> str:
    hostname = socket.gethostname().strip().rstrip(".").lower()
    if not hostname:
        raise RuntimeError("socket.gethostname() returned an empty hostname")
    return hostname


def validate_local_hostname(expected: str) -> str:
    actual = local_hostname()
    if actual != expected.strip().rstrip(".").lower():
        raise RuntimeError(
            f"actual socket hostname is {actual!r}, expected {expected!r}"
        )
    return actual


def parse_process_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    environment: dict[str, str] = {}
    for field in raw.split(b"\0"):
        if not field:
            continue
        name_bytes, separator, value_bytes = field.partition(b"=")
        if not separator:
            raise RuntimeError(f"worker PID {pid} has malformed environment data")
        try:
            name = name_bytes.decode("ascii")
            value = value_bytes.decode("utf-8", errors="surrogateescape")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"worker PID {pid} has a non-ASCII environment name"
            ) from exc
        if name in environment:
            raise RuntimeError(f"worker PID {pid} has duplicate environment {name}")
        environment[name] = value
    return environment


def attest_worker_process(
    pid: int,
    worker: Path,
    expected_worker_sha256: str,
    required_environment: dict[str, str],
) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise RuntimeError(f"health reported invalid worker PID: {pid!r}")
    proc_root = Path(f"/proc/{pid}")
    proc_exe = proc_root / "exe"
    try:
        raw_exe = os.readlink(proc_exe)
    except OSError as exc:
        raise RuntimeError(f"cannot inspect worker PID {pid} executable") from exc
    if raw_exe.endswith(" (deleted)"):
        raise RuntimeError(f"worker PID {pid} executable has been deleted")
    expected_exe = worker.resolve(strict=True)
    actual_exe = Path(raw_exe).resolve(strict=True)
    if actual_exe != expected_exe:
        raise RuntimeError(
            f"worker PID {pid} executable is {actual_exe}, expected {expected_exe}"
        )
    actual_sha256 = sha256_file(proc_exe)
    if actual_sha256 != expected_worker_sha256:
        raise RuntimeError(
            f"worker PID {pid} executable hash is {actual_sha256}, "
            f"expected {expected_worker_sha256}"
        )
    environment = parse_process_environment(pid)
    reject_profiler_environment(environment, f"worker PID {pid}")
    for name, expected_value in required_environment.items():
        if environment.get(name) != expected_value:
            raise RuntimeError(
                f"worker PID {pid} environment {name} is "
                f"{environment.get(name)!r}, expected {expected_value!r}"
            )
    return {
        "pid": pid,
        "exe": str(actual_exe),
        "exe_sha256": actual_sha256,
        "required_environment": dict(sorted(required_environment.items())),
        "profiler_environment_present": False,
    }


def response_header_map(headers: list[tuple[str, str]]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for name, value in headers:
        normalized = name.lower()
        if normalized in mapped:
            raise RuntimeError(f"duplicate HTTP response header: {name}")
        mapped[normalized] = value
    return mapped


def validate_response_headers(
    status: int,
    headers: list[tuple[str, str]],
    contract: dict[str, Any],
    description: str,
) -> dict[str, str]:
    expected_status = contract["status"]
    if status != expected_status:
        raise RuntimeError(
            f"{description} status is {status}, expected {expected_status}"
        )
    observed = response_header_map(headers)
    expected_headers = {
        name.lower(): value for name, value in contract["headers"].items()
    }
    for name, expected_value in expected_headers.items():
        if observed.get(name) != expected_value:
            raise RuntimeError(
                f"{description} header {name} is {observed.get(name)!r}, "
                f"expected {expected_value!r}"
            )
    return {name: observed[name] for name in sorted(expected_headers)}


def require_json_subset(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeError(f"{path} is not an object")
        for key, expected_value in expected.items():
            if key not in actual:
                raise RuntimeError(f"{path}.{key} is missing")
            require_json_subset(actual[key], expected_value, f"{path}.{key}")
        return
    if actual != expected or type(actual) is not type(expected):
        raise RuntimeError(f"{path} is {actual!r}, expected {expected!r}")


def validate_health_body(
    body: bytes, contract: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    if len(body) > MAX_HEALTH_BYTES:
        raise RuntimeError("health response is too large")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("health response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("health response is not an object")
    expected_body = contract["body"]
    require_json_subset(payload, expected_body, "health")
    worker_pid = payload.get("worker_pid")
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 1:
        raise RuntimeError("health worker_pid is not a positive process ID")
    return payload, worker_pid


def request_once(
    method: str, url: str, timeout: float, body: bytes | None = None
) -> tuple[int, list[tuple[str, str]], bytes]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("URL must be an absolute http:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials in the URL are forbidden")
    port = parsed.port or 80
    connection = http.client.HTTPConnection(parsed.hostname, port, timeout=timeout)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    headers = {"Connection": "close"}
    if body is not None:
        headers.update(
            {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            }
        )
    try:
        # Each caller invokes this once; there is no retry or redirect path.
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
        return response.status, response.getheaders(), response_body
    finally:
        connection.close()


def create_attempt_marker(
    path: Path,
    authorization_id: str,
    authorization_sha256: str,
    url: str,
    body_sha256: str,
    actual_hostname: str,
    reservation_id: str,
    sealed_package_root_sha256: str,
    health: dict[str, Any],
    worker_attestation: dict[str, Any],
) -> None:
    marker = {
        "schema": "loom.q16k.one_http_post.attempt.v2",
        "authorization_id": authorization_id,
        "authorization_sha256": authorization_sha256,
        "url": url,
        "actual_hostname": actual_hostname,
        "reservation_id": reservation_id,
        "sealed_package_root_sha256": sealed_package_root_sha256,
        "body_sha256": body_sha256,
        "health": health,
        "worker": worker_attestation,
        "created_unix_ns": time.time_ns(),
        "maximum_health_gets": 1,
        "maximum_posts": 1,
        "maximum_retries": 0,
    }
    atomic_json_exclusive(path, marker)


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
    parser.add_argument("--authorization-file", required=True, type=Path)
    parser.add_argument("--attempt-marker", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--response-body", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    for label, path in (
        ("body", args.body),
        ("adapter", args.adapter),
        ("worker", args.worker),
        ("sealed package", args.sealed_package),
        ("deployment manifest", args.deployment_manifest),
        ("reservation snapshot", args.reservation_snapshot),
        ("diagnostic output", args.diagnostic_output),
        ("authorization file", args.authorization_file),
        ("attempt marker", args.attempt_marker),
        ("result", args.result),
        ("response body", args.response_body),
    ):
        require_absolute(path, label)
    reject_profiler_environment(dict(os.environ), "one-POST client")
    actual_hostname = validate_local_hostname(args.target_hostname)
    for path in (
        args.diagnostic_output,
        args.attempt_marker,
        args.result,
        args.response_body,
    ):
        if path.exists():
            raise FileExistsError(f"authorized output already exists: {path}")

    body = args.body.read_bytes()
    body_sha256 = sha256_bytes(body)
    authorization_bytes = args.authorization_file.read_bytes()
    authorization_sha256 = sha256_bytes(authorization_bytes)
    try:
        authorization = json.loads(authorization_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authorization is not valid UTF-8 JSON") from exc
    authorization_id, deployment = validate_authorization(
        authorization,
        args.url,
        args.body,
        body_sha256,
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
    )

    health_url = health_url_for(args.url)
    health_status, health_headers, health_bytes = request_once(
        "GET", health_url, args.timeout
    )
    observed_health_headers = validate_response_headers(
        health_status,
        health_headers,
        deployment["health_contract"],
        "health response",
    )
    health_payload, worker_pid = validate_health_body(
        health_bytes, deployment["health_contract"]
    )
    worker_attestation = attest_worker_process(
        worker_pid,
        args.worker,
        authorization["inputs"]["worker_sha256"],
        deployment["environment"]["required"],
    )
    health_evidence = {
        "url": health_url,
        "status": health_status,
        "headers": observed_health_headers,
        "body_sha256": sha256_bytes(health_bytes),
        "worker_pid": worker_pid,
        "busy": health_payload["busy"],
    }
    create_attempt_marker(
        args.attempt_marker,
        authorization_id,
        authorization_sha256,
        args.url,
        body_sha256,
        actual_hostname,
        args.reservation_id,
        args.sealed_package_root_sha256,
        health_evidence,
        worker_attestation,
    )

    started_ns = time.monotonic_ns()
    status: int | None = None
    response_headers: list[tuple[str, str]] = []
    response_data = b""
    observed_post_headers: dict[str, str] = {}
    error: str | None = None
    try:
        status, response_headers, response_data = request_once(
            "POST", args.url, args.timeout, body
        )
        observed_post_headers = validate_response_headers(
            status,
            response_headers,
            deployment["post_contract"],
            "POST response",
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ns = time.monotonic_ns() - started_ns

    publish_bytes_exclusive(args.response_body, response_data)
    result = {
        "schema": "loom.q16k.one_http_post.result.v2",
        "authorization_id": authorization_id,
        "actual_hostname": actual_hostname,
        "reservation_id": args.reservation_id,
        "sealed_package_root_sha256": args.sealed_package_root_sha256,
        "body_sha256": body_sha256,
        "health_get_count": 1,
        "post_count": 1,
        "retry_count": 0,
        "redirect_count": 0,
        "health": health_evidence,
        "worker": worker_attestation,
        "status": status,
        "headers": response_headers,
        "identity_headers": observed_post_headers,
        "response_bytes": len(response_data),
        "response_sha256": sha256_bytes(response_data),
        "elapsed_ns": elapsed_ns,
        "error": error,
    }
    atomic_json_exclusive(args.result, result)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
