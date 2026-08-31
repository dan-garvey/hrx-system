"""Authorization and sealed-deployment helpers for the one-POST diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


AUTHORIZATION_SCHEMA = "loom.q16k.first_attention.authorization.v2"
DEPLOYMENT_SCHEMA = "loom.q16k.first_attention.deployment.v1"
DEPLOYMENT_STATUS = "READY_FOR_AUTHORIZATION"
EXPECTED_BACKEND = "loom-raw-hsa"
EXPECTED_LLAMA_CPP_ROLE = "cpu-only-http-utility"
EXPECTED_LLAMA_CPP_URL = "http://127.0.0.1:8082"
EXPECTED_LLAMA_CPP_SOURCE_PATH = (
    "/home/dagarvey/codex-work/llama.cpp-hrx-loom-reorg-9a1e95c-gJmqWw"
)
EXPECTED_LLAMA_CPP_SOURCE_COMMIT = "9a1e95c124ce47f0d1425f0d068d8eb85f671f26"
EXPECTED_LLAMA_CPP_SERVER_PATH = (
    EXPECTED_LLAMA_CPP_SOURCE_PATH + "/build-cpu/bin/llama-server"
)
EXPECTED_LLAMA_CPP_SERVER_SHA256 = (
    "7c9795c29858a2a3aaee7d8ca5789137731a753e011f054cc84c4af6d9006acd"
)
EXPECTED_LLAMA_CPP_MODEL_PATH = (
    EXPECTED_LLAMA_CPP_SOURCE_PATH
    + "/models/DeepSeek-R1-Distill-Qwen-1.5B-F16.gguf"
)
EXPECTED_LLAMA_CPP_MODEL_SHA256 = (
    "0f1621d5a06bc1d80f3c43616f5baaf36c0e72a833c701908c9fbee08f6cde5d"
)
EXPECTED_LLAMA_CPP_CPU_ONLY_FLAGS = (
    "--device",
    "none",
    "--n-gpu-layers",
    "0",
    "--no-kv-offload",
    "--no-op-offload",
    "--no-warmup",
    "--no-webui",
    "--no-slots",
    "--ctx-size",
    "512",
    "--parallel",
    "1",
    "--host",
    "127.0.0.1",
    "--port",
    "8082",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA256SUM_LINE_RE = re.compile(r"^([0-9a-f]{64})  (\./[^\n\r]+)$")
IDENTITY_SHA256_HEADERS = frozenset(
    {
        "x-loom-worker-sha256",
        "x-loom-runtime-hsaco-sha256",
        "x-loom-checkpoint-sha256",
        "x-loom-package-root-sha256",
        "x-loom-package-manifest-sha256",
        "x-loom-q16k-asset-manifest-sha256",
        "x-loom-q16k-dense-sha256",
        "x-loom-q16k-shadow-pack-sha256",
        "x-loom-q16k-aiter-sha256",
    }
)

FORBIDDEN_ENV_NAMES = frozenset(
    {"HSA_TOOLS_LIB", "LD_PRELOAD", "ROCP_TOOL_LIBRARIES"}
)
FORBIDDEN_ENV_PREFIXES = ("ROCPROF", "ROCPROFILER")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} is not a lowercase SHA-256 digest")
    return value


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("reservation end must be a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("reservation end must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def require_absolute(path: Path, description: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{description} path must be absolute: {path}")
    return path.resolve(strict=False)


def require_exact_keys(
    value: Any, expected: set[str], description: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{description} keys mismatch; missing={missing!r} extra={extra!r}"
        )
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes_exclusive(path: Path, data: bytes) -> None:
    """Publishes complete bytes atomically and refuses to replace a name."""
    path = require_absolute(path, "output")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {path.parent}")
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        for _ in range(128):
            candidate = path.with_name(
                f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(candidate, flags, 0o600)
                temporary = candidate
                break
            except FileExistsError:
                continue
        if descriptor is None or temporary is None:
            raise FileExistsError("could not reserve a unique temporary output")
        with os.fdopen(descriptor, "wb") as file:
            descriptor = None
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def atomic_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    publish_bytes_exclusive(path, data)


def _reject_symlink_components(root: Path, member: Path) -> None:
    relative = member.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"sealed package member traverses a symlink: {member}")


def verify_sealed_package(path: Path) -> tuple[str, dict[str, str]]:
    package = require_absolute(path, "sealed package")
    if path.is_symlink() or not package.is_dir():
        raise FileNotFoundError(f"sealed package is not a plain directory: {path}")
    root_file = package / "ROOT_SHA256"
    sums_file = package / "SHA256SUMS"
    for seal_file in (root_file, sums_file):
        if seal_file.is_symlink() or not seal_file.is_file():
            raise FileNotFoundError(f"sealed package file is missing: {seal_file}")

    root_text = root_file.read_text(encoding="ascii")
    root_fields = root_text.strip().split()
    if len(root_fields) != 1:
        raise ValueError("ROOT_SHA256 must contain exactly one digest")
    expected_root = validate_sha256(root_fields[0], "sealed package root")
    sums_bytes = sums_file.read_bytes()
    actual_root = sha256_bytes(sums_bytes)
    if actual_root != expected_root:
        raise ValueError(
            f"ROOT_SHA256 does not hash SHA256SUMS: {actual_root} != {expected_root}"
        )
    try:
        lines = sums_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("SHA256SUMS is not ASCII") from exc
    if not lines:
        raise ValueError("SHA256SUMS is empty")

    entries: dict[str, str] = {}
    previous_path = ""
    for line_number, line in enumerate(lines, 1):
        match = SHA256SUM_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        expected_digest, member_name = match.groups()
        relative = PurePosixPath(member_name[2:])
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.as_posix() in ("ROOT_SHA256", "SHA256SUMS")
        ):
            raise ValueError(f"unsafe SHA256SUMS member: {member_name}")
        if member_name in entries:
            raise ValueError(f"duplicate SHA256SUMS member: {member_name}")
        if previous_path and member_name <= previous_path:
            raise ValueError("SHA256SUMS members are not strictly sorted")
        previous_path = member_name
        member = package.joinpath(*relative.parts)
        _reject_symlink_components(package, member)
        if not member.is_file():
            raise FileNotFoundError(f"sealed package member is missing: {member}")
        actual_digest = sha256_file(member)
        if actual_digest != expected_digest:
            raise ValueError(
                f"sealed package member hash mismatch for {member_name}: "
                f"{actual_digest} != {expected_digest}"
            )
        entries[member_name] = expected_digest
    return expected_root, entries


def sealed_package_root(path: Path) -> str:
    root, _ = verify_sealed_package(path)
    return root


def _member_name(package: Path, path: Path, description: str) -> str:
    resolved_package = require_absolute(package, "sealed package")
    resolved_path = require_absolute(path, description)
    try:
        relative = resolved_path.relative_to(resolved_package)
    except ValueError as exc:
        raise ValueError(f"{description} is not inside the sealed package") from exc
    return "./" + relative.as_posix()


def health_url_for(post_url: str) -> str:
    parsed = urllib.parse.urlsplit(post_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1/chat/completions"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "POST URL must be loopback HTTP with path /v1/chat/completions"
        )
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/health", "", "")
    )


def _validate_header_contract(
    value: Any, description: str, worker_sha256: str, content_type: str
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{description} must be a nonempty object")
    headers: dict[str, str] = {}
    for name, header_value in value.items():
        if (
            not isinstance(name, str)
            or not name
            or name.lower() in headers
            or not isinstance(header_value, str)
            or not header_value
        ):
            raise ValueError(f"{description} contains an invalid header")
        headers[name.lower()] = header_value
    required = {
        "content-type": content_type,
        "x-inference-backend": EXPECTED_BACKEND,
        "x-loom-worker-sha256": worker_sha256,
    }
    for name, expected in required.items():
        if headers.get(name) != expected:
            raise ValueError(f"{description} does not pin {name}={expected}")
    missing_identity = sorted(IDENTITY_SHA256_HEADERS - set(headers))
    if missing_identity:
        raise ValueError(
            f"{description} omits identity headers: {missing_identity!r}"
        )
    for name in IDENTITY_SHA256_HEADERS:
        validate_sha256(headers[name], f"{description} {name}")
    return headers


def _validate_llama_cpp(value: Any) -> dict[str, Any]:
    llama_cpp = require_exact_keys(
        value,
        {"role", "url", "source", "server", "model", "cpu_only_flags"},
        "llama.cpp utility",
    )
    source = require_exact_keys(
        llama_cpp["source"], {"path", "commit", "clean"},
        "llama.cpp source"
    )
    server = require_exact_keys(
        llama_cpp["server"], {"path", "sha256"}, "llama.cpp server"
    )
    model = require_exact_keys(
        llama_cpp["model"], {"path", "sha256"}, "llama.cpp model"
    )
    if (
        llama_cpp["role"] != EXPECTED_LLAMA_CPP_ROLE
        or llama_cpp["url"] != EXPECTED_LLAMA_CPP_URL
        or source["path"] != EXPECTED_LLAMA_CPP_SOURCE_PATH
        or source["commit"] != EXPECTED_LLAMA_CPP_SOURCE_COMMIT
        or source["clean"] is not True
        or server
        != {
            "path": EXPECTED_LLAMA_CPP_SERVER_PATH,
            "sha256": EXPECTED_LLAMA_CPP_SERVER_SHA256,
        }
        or model
        != {
            "path": EXPECTED_LLAMA_CPP_MODEL_PATH,
            "sha256": EXPECTED_LLAMA_CPP_MODEL_SHA256,
        }
        or llama_cpp["cpu_only_flags"]
        != list(EXPECTED_LLAMA_CPP_CPU_ONLY_FLAGS)
    ):
        raise ValueError("llama.cpp CPU utility contract mismatch")
    return llama_cpp


def load_and_validate_deployment(
    deployment_manifest: Path,
    sealed_package: Path,
    expected_package_root: str,
    url: str,
    body: Path,
    adapter: Path,
    worker: Path,
    reservation_snapshot: Path,
    target_hostname: str,
    reservation_id: str,
    reservation_end_utc: str,
    diagnostic_output: Path,
    attempt_marker: Path,
    result: Path,
    response_body: Path,
) -> dict[str, Any]:
    package = require_absolute(sealed_package, "sealed package")
    package_root, entries = verify_sealed_package(package)
    if package_root != validate_sha256(expected_package_root, "requested package root"):
        raise ValueError("sealed package root does not match the requested pin")
    manifest_path = require_absolute(deployment_manifest, "deployment manifest")
    if manifest_path != package / "DEPLOYMENT.json":
        raise ValueError("deployment manifest must be <sealed-package>/DEPLOYMENT.json")
    if entries.get("./DEPLOYMENT.json") != sha256_file(manifest_path):
        raise ValueError("deployment manifest is not a validated sealed member")
    try:
        deployment = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("deployment manifest is not valid UTF-8 JSON") from exc
    deployment = require_exact_keys(
        deployment,
        {
            "schema",
            "status",
            "target",
            "reservation",
            "endpoints",
            "llama_cpp",
            "artifacts",
            "environment",
            "outputs",
            "health_contract",
            "post_contract",
            "execution",
        },
        "deployment manifest",
    )
    if deployment["schema"] != DEPLOYMENT_SCHEMA:
        raise ValueError("deployment schema mismatch")
    if deployment["status"] != DEPLOYMENT_STATUS:
        raise ValueError("deployment is not ready for authorization")

    _validate_llama_cpp(deployment["llama_cpp"])

    target = require_exact_keys(deployment["target"], {"hostname"}, "target")
    if target != {"hostname": target_hostname}:
        raise ValueError("deployment target hostname mismatch")

    snapshot_path = require_absolute(reservation_snapshot, "reservation snapshot")
    snapshot_sha256 = sha256_file(snapshot_path)
    reservation = require_exact_keys(
        deployment["reservation"], {"id", "end_utc", "snapshot"}, "reservation"
    )
    snapshot = require_exact_keys(
        reservation["snapshot"], {"path", "sha256"}, "reservation snapshot"
    )
    expected_reservation_end = format_utc(parse_utc(reservation_end_utc))
    if reservation["id"] != reservation_id:
        raise ValueError("deployment reservation ID mismatch")
    if reservation["end_utc"] != expected_reservation_end:
        raise ValueError("deployment reservation end mismatch")
    if snapshot != {
        "path": str(snapshot_path),
        "sha256": snapshot_sha256,
    }:
        raise ValueError("deployment reservation snapshot mismatch")

    artifacts = require_exact_keys(
        deployment["artifacts"], {"adapter", "worker", "request"}, "artifacts"
    )
    artifact_inputs = {
        "adapter": require_absolute(adapter, "adapter"),
        "worker": require_absolute(worker, "worker"),
        "request": require_absolute(body, "request body"),
    }
    artifact_hashes: dict[str, str] = {}
    for name, input_path in artifact_inputs.items():
        artifact = require_exact_keys(
            artifacts[name], {"path", "sha256"}, f"{name} artifact"
        )
        actual_hash = sha256_file(input_path)
        expected_artifact = {"path": str(input_path), "sha256": actual_hash}
        if artifact != expected_artifact:
            raise ValueError(f"deployment {name} artifact mismatch")
        member_name = _member_name(package, input_path, f"{name} artifact")
        if entries.get(member_name) != actual_hash:
            raise ValueError(f"deployment {name} is not a validated sealed member")
        artifact_hashes[name] = actual_hash
    snapshot_member = _member_name(package, snapshot_path, "reservation snapshot")
    if entries.get(snapshot_member) != snapshot_sha256:
        raise ValueError("reservation snapshot is not a validated sealed member")

    post_url = url
    health_url = health_url_for(post_url)
    endpoints = require_exact_keys(
        deployment["endpoints"], {"health_url", "post_url"}, "endpoints"
    )
    if endpoints != {"health_url": health_url, "post_url": post_url}:
        raise ValueError("deployment endpoint mismatch")

    diagnostic_path = require_absolute(diagnostic_output, "diagnostic output")
    environment = require_exact_keys(
        deployment["environment"],
        {"required", "forbidden_names", "forbidden_prefixes"},
        "environment",
    )
    required_environment = {
        "DEEPSEEK_Q16K_AITER": "1",
        "DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC": str(diagnostic_path),
    }
    if environment["required"] != required_environment:
        raise ValueError("deployment required worker environment mismatch")
    if environment["forbidden_names"] != sorted(FORBIDDEN_ENV_NAMES):
        raise ValueError("deployment forbidden environment names mismatch")
    if environment["forbidden_prefixes"] != list(FORBIDDEN_ENV_PREFIXES):
        raise ValueError("deployment forbidden environment prefixes mismatch")

    expected_outputs = {
        "diagnostic": str(diagnostic_path),
        "attempt": str(require_absolute(attempt_marker, "attempt marker")),
        "result": str(require_absolute(result, "result")),
        "response_body": str(require_absolute(response_body, "response body")),
    }
    outputs = require_exact_keys(
        deployment["outputs"], set(expected_outputs), "outputs"
    )
    if outputs != expected_outputs:
        raise ValueError("deployment output paths mismatch")

    health_contract = require_exact_keys(
        deployment["health_contract"], {"status", "headers", "body"},
        "health contract"
    )
    if health_contract["status"] != 200:
        raise ValueError("health contract must require HTTP 200")
    health_headers = _validate_header_contract(
        health_contract["headers"], "health headers", artifact_hashes["worker"],
        "application/json"
    )
    health_body = require_exact_keys(
        health_contract["body"],
        {"status", "inference_backend", "busy", "backend"},
        "health body contract",
    )
    required_backend = {
        "name": EXPECTED_BACKEND,
        "worker_sha256": artifact_hashes["worker"],
        "worker_protocol": 3,
        "worker_capabilities": [
            "generate_batch",
            "generate_stream",
            "long_context_131072_q16k_aiter",
            "token_chunks=256",
        ],
        "max_prompt_tokens": 131071,
        "max_context_tokens": 131072,
        "max_output_tokens": 383,
        "worker_restarts": 0,
    }
    if (
        health_body["status"] != "ok"
        or health_body["inference_backend"] != EXPECTED_BACKEND
        or health_body["busy"] is not False
        or not isinstance(health_body["backend"], dict)
        or any(
            health_body["backend"].get(name) != expected
            for name, expected in required_backend.items()
        )
    ):
        raise ValueError("health body identity contract mismatch")

    post_contract = require_exact_keys(
        deployment["post_contract"], {"status", "headers"}, "POST contract"
    )
    if post_contract["status"] != 200:
        raise ValueError("POST contract must require HTTP 200")
    post_headers = _validate_header_contract(
        post_contract["headers"], "POST headers", artifact_hashes["worker"],
        "text/event-stream"
    )
    if post_headers.get("cache-control") != "no-cache":
        raise ValueError("POST headers must pin cache-control=no-cache")
    if post_headers.get("connection") != "close":
        raise ValueError("POST headers must pin connection=close")
    for name in IDENTITY_SHA256_HEADERS | {"x-inference-backend"}:
        if post_headers[name] != health_headers[name]:
            raise ValueError(f"health and POST identity differ for {name}")

    expected_execution = {
        "health_gets": 1,
        "worker_requests": 1,
        "http_posts": 1,
        "retries": 0,
        "redirects": 0,
        "profiled": False,
    }
    execution = require_exact_keys(
        deployment["execution"], set(expected_execution), "execution"
    )
    if execution != expected_execution:
        raise ValueError("deployment execution limits mismatch")
    return deployment


def make_authorization(
    authorization_id: str,
    url: str,
    body: Path,
    adapter: Path,
    worker: Path,
    sealed_package: Path,
    sealed_package_root_sha256: str,
    deployment_manifest: Path,
    reservation_snapshot: Path,
    target_hostname: str,
    reservation_id: str,
    reservation_end_utc: str,
    diagnostic_output: Path,
    attempt_marker: Path,
    result: Path,
    response_body: Path,
    valid_for_seconds: int,
) -> dict[str, Any]:
    if not target_hostname or not reservation_id:
        raise ValueError("target hostname and reservation ID are required")
    paths = {
        "body": require_absolute(body, "request body"),
        "adapter": require_absolute(adapter, "adapter"),
        "worker": require_absolute(worker, "worker"),
        "sealed_package": require_absolute(sealed_package, "sealed package"),
        "deployment_manifest": require_absolute(
            deployment_manifest, "deployment manifest"
        ),
        "reservation_snapshot": require_absolute(
            reservation_snapshot, "reservation snapshot"
        ),
        "diagnostic_output": require_absolute(
            diagnostic_output, "diagnostic output"
        ),
        "attempt_marker": require_absolute(attempt_marker, "attempt marker"),
        "result": require_absolute(result, "result"),
        "response_body": require_absolute(response_body, "response body"),
    }
    if len({str(path) for path in paths.values()}) != len(paths):
        raise ValueError("authorization paths must be distinct")
    requested_root = validate_sha256(
        sealed_package_root_sha256, "requested sealed package root"
    )
    load_and_validate_deployment(
        paths["deployment_manifest"], paths["sealed_package"], requested_root,
        url, paths["body"], paths["adapter"], paths["worker"],
        paths["reservation_snapshot"], target_hostname, reservation_id,
        reservation_end_utc, paths["diagnostic_output"],
        paths["attempt_marker"], paths["result"], paths["response_body"]
    )

    now_ns = time.time_ns()
    reservation_end = parse_utc(reservation_end_utc)
    reservation_end_ns = int(reservation_end.timestamp() * 1_000_000_000)
    expires_ns = min(
        now_ns + valid_for_seconds * 1_000_000_000,
        reservation_end_ns,
    )
    if expires_ns <= now_ns:
        raise ValueError("reservation has ended")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_id": authorization_id,
        "authorized": True,
        "decision": "GO",
        "created_unix_ns": now_ns,
        "expires_unix_ns": expires_ns,
        "url": url,
        "target": {"hostname": target_hostname},
        "reservation": {
            "id": reservation_id,
            "end_utc": format_utc(reservation_end),
            "snapshot_path": str(paths["reservation_snapshot"]),
            "snapshot_sha256": sha256_file(paths["reservation_snapshot"]),
        },
        "sealed_package": {
            "path": str(paths["sealed_package"]),
            "root_sha256": requested_root,
        },
        "inputs": {
            "body": str(paths["body"]),
            "body_sha256": sha256_file(paths["body"]),
            "adapter": str(paths["adapter"]),
            "adapter_sha256": sha256_file(paths["adapter"]),
            "worker": str(paths["worker"]),
            "worker_sha256": sha256_file(paths["worker"]),
            "deployment_manifest": str(paths["deployment_manifest"]),
            "deployment_manifest_sha256": sha256_file(
                paths["deployment_manifest"]
            ),
        },
        "paths": {
            "diagnostic_output": str(paths["diagnostic_output"]),
            "attempt_marker": str(paths["attempt_marker"]),
            "result": str(paths["result"]),
            "response_body": str(paths["response_body"]),
        },
        "execution": {
            "health_gets": 1,
            "worker_requests": 1,
            "http_posts": 1,
            "retries": 0,
            "redirects": 0,
            "profiled": False,
        },
    }


def validate_authorization(
    manifest: dict[str, Any],
    url: str,
    body: Path,
    body_sha256: str,
    adapter: Path,
    worker: Path,
    sealed_package: Path,
    sealed_package_root_sha256: str,
    deployment_manifest: Path,
    reservation_snapshot: Path,
    target_hostname: str,
    reservation_id: str,
    reservation_end_utc: str,
    diagnostic_output: Path,
    attempt_marker: Path,
    result: Path,
    response_body: Path,
) -> tuple[str, dict[str, Any]]:
    manifest = require_exact_keys(
        manifest,
        {
            "schema", "authorization_id", "authorized", "decision",
            "created_unix_ns", "expires_unix_ns", "url", "target",
            "reservation", "sealed_package", "inputs", "paths", "execution",
        },
        "authorization",
    )
    if manifest["schema"] != AUTHORIZATION_SCHEMA:
        raise ValueError("authorization schema mismatch")
    if manifest["authorized"] is not True or manifest["decision"] != "GO":
        raise ValueError("authorization is not active")
    authorization_id = manifest["authorization_id"]
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("authorization_id is missing")
    if not isinstance(manifest["created_unix_ns"], int):
        raise ValueError("authorization creation time is invalid")
    expires_ns = manifest["expires_unix_ns"]
    if not isinstance(expires_ns, int) or time.time_ns() > expires_ns:
        raise ValueError("authorization has expired")

    requested_reservation_end = format_utc(parse_utc(reservation_end_utc))
    reservation_end_ns = int(
        parse_utc(requested_reservation_end).timestamp() * 1_000_000_000
    )
    if time.time_ns() >= reservation_end_ns:
        raise ValueError("reservation has ended")
    if expires_ns > reservation_end_ns:
        raise ValueError("authorization outlives the reservation")
    if manifest["url"] != url:
        raise ValueError("authorization URL mismatch")
    if manifest["target"] != {"hostname": target_hostname}:
        raise ValueError("authorization target hostname mismatch")

    snapshot = require_absolute(reservation_snapshot, "reservation snapshot")
    expected_reservation = {
        "id": reservation_id,
        "end_utc": requested_reservation_end,
        "snapshot_path": str(snapshot),
        "snapshot_sha256": sha256_file(snapshot),
    }
    if manifest["reservation"] != expected_reservation:
        raise ValueError("authorization reservation mismatch")

    package = require_absolute(sealed_package, "sealed package")
    requested_root = validate_sha256(
        sealed_package_root_sha256, "requested sealed package root"
    )
    expected_package = {"path": str(package), "root_sha256": requested_root}
    if manifest["sealed_package"] != expected_package:
        raise ValueError("authorization sealed package mismatch")

    resolved = {
        "body": require_absolute(body, "request body"),
        "adapter": require_absolute(adapter, "adapter"),
        "worker": require_absolute(worker, "worker"),
        "deployment_manifest": require_absolute(
            deployment_manifest, "deployment manifest"
        ),
        "diagnostic_output": require_absolute(
            diagnostic_output, "diagnostic output"
        ),
        "attempt_marker": require_absolute(attempt_marker, "attempt marker"),
        "result": require_absolute(result, "result"),
        "response_body": require_absolute(response_body, "response body"),
    }
    inputs = require_exact_keys(
        manifest["inputs"],
        {
            "body", "body_sha256", "adapter", "adapter_sha256", "worker",
            "worker_sha256", "deployment_manifest",
            "deployment_manifest_sha256",
        },
        "authorization inputs",
    )
    expected_inputs = {
        "body": str(resolved["body"]),
        "body_sha256": validate_sha256(body_sha256, "in-memory body hash"),
        "adapter": str(resolved["adapter"]),
        "adapter_sha256": sha256_file(resolved["adapter"]),
        "worker": str(resolved["worker"]),
        "worker_sha256": sha256_file(resolved["worker"]),
        "deployment_manifest": str(resolved["deployment_manifest"]),
        "deployment_manifest_sha256": sha256_file(
            resolved["deployment_manifest"]
        ),
    }
    if inputs != expected_inputs:
        raise ValueError("authorization input path or hash mismatch")

    expected_paths = {
        "diagnostic_output": str(resolved["diagnostic_output"]),
        "attempt_marker": str(resolved["attempt_marker"]),
        "result": str(resolved["result"]),
        "response_body": str(resolved["response_body"]),
    }
    paths = require_exact_keys(manifest["paths"], set(expected_paths), "paths")
    if paths != expected_paths:
        raise ValueError("authorization output paths mismatch")
    expected_execution = {
        "health_gets": 1,
        "worker_requests": 1,
        "http_posts": 1,
        "retries": 0,
        "redirects": 0,
        "profiled": False,
    }
    execution = require_exact_keys(
        manifest["execution"], set(expected_execution), "execution"
    )
    if execution != expected_execution:
        raise ValueError("authorization execution limits mismatch")

    deployment = load_and_validate_deployment(
        resolved["deployment_manifest"], package, requested_root, url,
        resolved["body"], resolved["adapter"], resolved["worker"], snapshot,
        target_hostname, reservation_id, reservation_end_utc,
        resolved["diagnostic_output"], resolved["attempt_marker"],
        resolved["result"], resolved["response_body"]
    )
    return authorization_id, deployment
