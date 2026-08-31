#!/usr/bin/python3 -I
"""Deterministic sealing and bound-input verification helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_manifest import canonical_json, load_manifest  # noqa: E402


EXCLUDED_ROOT_FILES = {"ROOT_SHA256", "SHA256SUMS"}
EXCLUDED_TOP_LEVEL = {"build", ".git"}
BOUND_INPUTS_SCHEMA = "loom-q16384-first-attention-launch-abi-bound-inputs-v1"
EXPECTED_ARCHIVED_ROOT = (
    "41a2cc18b4a5273fdeaa1ddd77b707bf5cf26356c74fd1659faab8338cae3da1"
)
EXPECTED_BUILD_HSA_SHA256 = (
    "d41abc620d2f228995f809b55c7e4183f513501cc070416316479f5eeaa58253"
)
EXPECTED_BUILD_ROCPROFILER_SHA256 = (
    "f127c7d39c0bac3a619e433dce4e4ad73519cb71a2f7c9e09eab42b628e2c24d"
)
MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _excluded(relative: Path) -> bool:
    return (
        relative.as_posix() in EXCLUDED_ROOT_FILES
        or (relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL)
        or "__pycache__" in relative.parts
        or relative.suffix in {".pyc", ".pyo"}
    )


def collect_files(root: Path, *, exclude_build: bool = True) -> list[Path]:
    root_status = root.lstat()
    require(stat.S_ISDIR(root_status.st_mode), "seal root is not a directory")
    require(not root.is_symlink(), "seal root may not be a symlink")
    require(root_status.st_uid == os.geteuid(), "seal root owner changed")
    require(root_status.st_mode & 0o022 == 0, "seal root is group/world writable")
    result: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            path = directory_path / name
            mode = path.lstat().st_mode
            require(stat.S_ISDIR(mode), f"non-directory tree entry: {path}")
            require(not stat.S_ISLNK(mode), f"symlink is forbidden: {path}")
            require(name != "__pycache__", f"bytecode cache is forbidden: {path}")
            require(path.stat().st_uid == os.geteuid(), f"directory owner changed: {path}")
            require(mode & 0o022 == 0, f"directory is group/world writable: {path}")
        for name in file_names:
            path = directory_path / name
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), f"symlink is forbidden: {path}")
            require(stat.S_ISREG(mode), f"special file is forbidden: {path}")
            require(path.suffix not in {".pyc", ".pyo"},
                    f"bytecode file is forbidden: {path}")
            require(path.stat().st_nlink == 1, f"hardlink is forbidden: {path}")
            require(path.stat().st_uid == os.geteuid(), f"file owner changed: {path}")
            require(mode & 0o022 == 0, f"file is group/world writable: {path}")
            relative = path.relative_to(root)
            if relative.as_posix() in EXCLUDED_ROOT_FILES:
                continue
            if exclude_build and _excluded(relative):
                continue
            result.append(relative)
    return sorted(result, key=lambda item: item.as_posix())


def manifest_bytes(root: Path, files: Iterable[Path]) -> bytes:
    lines = [f"{sha256_file(root / path)}  {path.as_posix()}" for path in files]
    require(lines, "refusing to seal an empty tree")
    return ("\n".join(lines) + "\n").encode("ascii")


def tree_digest(root: Path) -> tuple[str, int]:
    files = collect_files(root, exclude_build=False)
    manifest = manifest_bytes(root, files)
    return hashlib.sha256(manifest).hexdigest(), len(files)


def parse_manifest(raw: bytes, *, require_sorted: bool = True) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("SHA256SUMS is not ASCII") from exc
    require(text.endswith("\n"), "SHA256SUMS lacks its final newline")
    lines = text.splitlines()
    require(lines, "SHA256SUMS is empty")
    entries: dict[str, str] = {}
    previous = ""
    for line_number, line in enumerate(lines, start=1):
        match = MANIFEST_LINE.fullmatch(line)
        require(match is not None, f"malformed SHA256SUMS line {line_number}")
        digest, relative = match.groups()
        path = PurePosixPath(relative)
        require(
            not path.is_absolute()
            and path.as_posix() == relative
            and all(part not in ("", ".", "..") for part in path.parts),
            f"noncanonical SHA256SUMS path on line {line_number}",
        )
        require(relative not in entries, f"duplicate SHA256SUMS path: {relative}")
        if require_sorted:
            require(previous < relative or not previous,
                    "SHA256SUMS entries are not sorted")
        entries[relative] = digest
        previous = relative
    return entries


def _read_regular_single_link(path: Path) -> bytes:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode), f"seal metadata is not regular: {path}")
    require(not stat.S_ISLNK(mode), f"seal metadata is a symlink: {path}")
    require(path.stat().st_nlink == 1, f"seal metadata is hardlinked: {path}")
    return path.read_bytes()


def verify_tree(
    root: Path, *, exclude_build: bool = True, require_sorted: bool = True
) -> tuple[str, int]:
    manifest = _read_regular_single_link(root / "SHA256SUMS")
    entries = parse_manifest(manifest, require_sorted=require_sorted)
    actual = {path.as_posix() for path in collect_files(root, exclude_build=exclude_build)}
    require(
        set(entries) == actual,
        f"sealed file set mismatch: missing={sorted(set(entries) - actual)} "
        f"extra={sorted(actual - set(entries))}",
    )
    for relative, expected in entries.items():
        require(sha256_file(root / relative) == expected, f"hash mismatch for {relative}")
    expected_root = hashlib.sha256(manifest).hexdigest()
    require(
        _read_regular_single_link(root / "ROOT_SHA256") ==
        f"{expected_root}  SHA256SUMS\n".encode("ascii"),
        "ROOT_SHA256 is malformed or differs from SHA256SUMS",
    )
    return expected_root, len(entries)


def _load_bound_inputs(root: Path) -> dict[str, Any]:
    raw = (root / "BOUND_INPUTS.json").read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("bound inputs are invalid JSON") from exc
    require(raw == canonical_json(payload), "bound inputs are not canonical JSON")
    require(isinstance(payload, dict) and set(payload) == {
        "archived_input", "build_toolchain", "code_object", "schema",
        "target_runtime",
    }, "bound input fields changed")
    require(payload["schema"] == BOUND_INPUTS_SCHEMA, "bound input schema changed")
    return payload


def verify_bound_inputs(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = load_manifest(root)
    for name, item in manifest["artifacts"].items():
        path = root / item["path"]
        require(path.is_file() and not path.is_symlink(), f"missing artifact: {name}")
        require(path.stat().st_size == item["size_bytes"], f"artifact size changed: {name}")
        require(sha256_file(path) == item["sha256"], f"artifact hash changed: {name}")

    bound = _load_bound_inputs(root)
    require(bound["archived_input"] == {
        "role": "build-toolchain-and-kernel-source",
        "root_sha256": EXPECTED_ARCHIVED_ROOT,
    }, "archived input binding changed")
    require(bound["code_object"] == manifest["artifacts"]["q16384_kernel"],
            "code object binding differs from package manifest")
    require(bound["target_runtime"] == manifest["runtime"],
            "target runtime binding differs from package manifest")

    toolchain = bound["build_toolchain"]
    require(isinstance(toolchain, dict) and set(toolchain) == {
        "hsa_headers", "hsa_runtime", "rocprofiler_register"
    }, "build toolchain binding changed")
    headers = toolchain["hsa_headers"]
    require(isinstance(headers, dict) and set(headers) == {
        "file_count", "path", "tree_sha256"
    }, "HSA header binding changed")
    header_path = root / headers["path"]
    header_digest, count = tree_digest(header_path)
    require(header_digest == headers["tree_sha256"] and count == headers["file_count"],
            "HSA header tree changed")
    expected_build_libraries = {
        "hsa_runtime": (
            "toolchain/lib/libhsa-runtime64.so.1", EXPECTED_BUILD_HSA_SHA256
        ),
        "rocprofiler_register": (
            "toolchain/lib/librocprofiler-register.so.0",
            EXPECTED_BUILD_ROCPROFILER_SHA256,
        ),
    }
    for name, (relative, expected_digest) in expected_build_libraries.items():
        item = toolchain[name]
        require(isinstance(item, dict) and set(item) == {
            "path", "sha256", "size_bytes"
        }, f"{name} build binding changed")
        path = root / relative
        require(item["path"] == relative and path.is_file() and not path.is_symlink(),
                f"{name} build input is missing")
        require(item["sha256"] == expected_digest == sha256_file(path),
                f"{name} build input hash changed")
        require(item["size_bytes"] == path.stat().st_size,
                f"{name} build input size changed")
    return {
        "artifact_count": len(manifest["artifacts"]),
        "build_header_count": count,
        "build_header_root_sha256": header_digest,
        "pass": True,
        "schema": "loom-q16384-first-attention-launch-abi-input-verification-v1",
    }


def verify_package(root: Path) -> tuple[str, int]:
    root = root.resolve(strict=True)
    digest, count = verify_tree(root)
    verify_bound_inputs(root)
    return digest, count


def seal_package(root: Path) -> tuple[str, int]:
    root = root.resolve(strict=True)
    require(not (root / "SHA256SUMS").exists() and not (root / "ROOT_SHA256").exists(),
            "package is already sealed")
    files = collect_files(root)
    manifest = manifest_bytes(root, files)
    digest = hashlib.sha256(manifest).hexdigest()
    (root / "SHA256SUMS").write_bytes(manifest)
    (root / "ROOT_SHA256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="ascii"
    )
    return digest, len(files)
