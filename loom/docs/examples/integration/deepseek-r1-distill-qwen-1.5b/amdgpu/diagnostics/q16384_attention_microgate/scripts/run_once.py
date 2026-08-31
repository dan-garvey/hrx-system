#!/usr/bin/python3 -I
"""Consume one authorization and execute the sealed launch-ABI runner once."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_health import (  # noqa: E402
    canonical_json,
    capture_kernel_log,
    capture_snapshot,
    compare_amd_health,
    validate_reservation_completion,
)
from package_manifest import load_manifest  # noqa: E402
from seal_lib import verify_package  # noqa: E402
from validate_authorization import (  # noqa: E402
    read_canonical_json,
    require_safe_absolute_path,
    validate,
)
from verify_evidence import validate_success_evidence  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PACKAGE_ROOT / "bin" / "q16384_attention_microgate"
CAMPAIGN_LOCK = Path("/dev/shm/deepseek-campaign.gfx950.lock")
RECEIPT_NAME = "q16384-first-attention-one-run.receipt"
ATTEMPT_NAME = "Q16384_FIRST_ATTENTION_LAUNCH_ABI_ATTEMPTED"
TIMEOUT = "/usr/bin/timeout"
RUNTIME_TIMEOUT_SECONDS = 600
TERMINATION_GRACE_SECONDS = 30
MINIMUM_POSTFLIGHT_RESERVE_SECONDS = 420
INITIAL_MINIMUM_REMAINING_SECONDS = 1200
PRELAUNCH_MINIMUM_REMAINING_SECONDS = 1050
OUTER_SUPERVISOR_TIMEOUT_SECONDS = (
    RUNTIME_TIMEOUT_SECONDS + TERMINATION_GRACE_SECONDS + 5
)
TERMINATION_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


class TerminationRequested(RuntimeError):
    """Convert external termination into normal fail-closed cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"launcher received signal {signum}")
        self.signum = signum


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _write_exclusive(path: Path, data: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(data):
            amount = os.write(descriptor, data[offset:])
            require(amount > 0, f"short write while creating {path}")
            offset += amount
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"created path is not regular: {path}")
        require(metadata.st_nlink == 1, f"created path is hardlinked: {path}")
        require(metadata.st_uid == os.geteuid(), f"created path owner changed: {path}")
        require(stat.S_IMODE(metadata.st_mode) == mode, f"created path mode changed: {path}")
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def receipt_bytes(
    validation: dict[str, Any], authorization_path: Path, result_path: Path
) -> bytes:
    fields = (
        ("schema", "loom-q16384-first-attention-launch-abi-consumption-v1"),
        ("authorization_id", validation["authorization_id"]),
        ("authorization_sha256", validation["authorization_sha256"]),
        ("authorization_path", str(authorization_path)),
        ("microgate_root_sha256", validation["microgate_root_sha256"]),
        ("authorized_cases", "q16384-first-attention"),
        ("authorized_case_count", "1"),
        ("reservation_id", validation["reservation_id"]),
        ("consumption_receipt_path", validation["consumption_receipt_path"]),
        ("result_path", str(result_path)),
        ("reservation_start_epoch", str(validation["reservation_start_epoch"])),
        ("reservation_end_epoch", str(validation["reservation_end_epoch"])),
        ("authorization_issued_epoch", str(validation["authorization_issued_epoch"])),
        ("authorization_expires_epoch", str(validation["authorization_expires_epoch"])),
        ("host", validation["host"]),
        ("target", validation["target"]),
        ("maximum_gpu_invocations", "1"),
        ("invocation_index", "1"),
        ("invocation_count", "1"),
        ("consumed_epoch", str(validation["validated_epoch"])),
    )
    for key, value in fields:
        require("\n" not in value and "\r" not in value and "=" not in value,
                f"unsafe receipt field: {key}")
    encoded = "".join(f"{key}={value}\n" for key, value in fields).encode("ascii")
    require(len(encoded) <= 4096, "authorization receipt exceeds runner limit")
    return encoded


def create_receipt(
    receipt_path: Path,
    validation: dict[str, Any],
    authorization_path: Path,
    result_path: Path,
) -> None:
    consumed_path = Path(f"{receipt_path}.consumed")
    require(not _lexists(receipt_path), "ready receipt already exists")
    require(not _lexists(consumed_path), "authorization was already consumed")
    _write_exclusive(
        receipt_path,
        receipt_bytes(validation, authorization_path, result_path),
        0o400,
    )
    _fsync_directory(receipt_path.parent)
    require(not _lexists(consumed_path), "consumed receipt appeared during creation")


def retire_unclaimed_receipt(receipt_path: Path) -> bool:
    consumed_path = Path(f"{receipt_path}.consumed")
    if not _lexists(receipt_path):
        return False
    require(not _lexists(consumed_path), "both ready and consumed receipts exist")
    os.rename(receipt_path, consumed_path)
    _fsync_directory(receipt_path.parent)
    return True


def seal_evidence(root: Path) -> str:
    root_status = root.lstat()
    require(stat.S_ISDIR(root_status.st_mode) and not root.is_symlink(),
            "evidence root is not a physical directory")
    require(root_status.st_uid == os.geteuid(), "evidence root owner changed")
    require(root_status.st_mode & 0o077 == 0, "evidence root is not mode-private")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), f"evidence symlink is forbidden: {path}")
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode),
                f"evidence special file is forbidden: {path}")
        require(path.stat().st_uid == os.geteuid(), f"evidence owner changed: {path}")
        require(mode & 0o077 == 0, f"evidence entry is not mode-private: {path}")
        if stat.S_ISREG(mode):
            require(path.stat().st_nlink == 1, f"evidence hardlink is forbidden: {path}")
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SHA256"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    require(files, "refusing to seal empty evidence")
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest = ("\n".join(lines) + "\n").encode("ascii")
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _write_exclusive(root / "SHA256SUMS", manifest, 0o600)
    digest = hashlib.sha256(manifest).hexdigest()
    _write_exclusive(
        root / "ROOT_SHA256", f"{digest}  SHA256SUMS\n".encode("ascii"), 0o600
    )
    _fsync_directory(root)
    return digest


def _copy_private(source: Path, destination: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                f"unsafe evidence source: {source}")
        data = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            data.extend(block)
    finally:
        os.close(descriptor)
    _write_exclusive(destination, bytes(data), 0o600)


def _invoke_runner_once(
    command: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    supervised = [
        TIMEOUT,
        "--signal=TERM",
        f"--kill-after={TERMINATION_GRACE_SECONDS}s",
        f"{RUNTIME_TIMEOUT_SECONDS}s",
        *command,
    ]
    started_epoch = int(time.time())
    process = subprocess.Popen(
        supervised,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    outer_timeout = False
    try:
        stdout, stderr = process.communicate(timeout=OUTER_SUPERVISOR_TIMEOUT_SECONDS)
        returncode = process.returncode
    except subprocess.TimeoutExpired as exc:
        outer_timeout = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            stdout = exc.output or b""
            stderr = exc.stderr or b""
        returncode = 124
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
        raise
    finished_epoch = int(time.time())
    require(isinstance(stdout, bytes) and isinstance(stderr, bytes),
            "runner output was not captured as bytes")
    return {
        "finished_epoch": finished_epoch,
        "outer_timeout": outer_timeout,
        "returncode": returncode,
        "started_epoch": started_epoch,
        "stderr": stderr,
        "stdout": stdout,
        "supervised_command": supervised,
        "timed_out": outer_timeout or returncode in {124, 137},
    }


def _capture_postflight(
    evidence: Path,
    runner_started_epoch: int | None,
    reservation_id: str,
    target_id: str,
    target_name: str,
    operator_email: str,
) -> list[str]:
    errors: list[str] = []
    try:
        capture_snapshot(evidence / "postflight", "postflight")
    except BaseException as exc:
        errors.append(f"postflight snapshot: {type(exc).__name__}: {exc}")
    try:
        comparison = compare_amd_health(
            (evidence / "preflight" / "amd-smi-metric.json").read_bytes(),
            (evidence / "postflight" / "amd-smi-metric.json").read_bytes(),
        )
        _write_exclusive(
            evidence / "pre-post-health-comparison.json",
            canonical_json(comparison),
            0o600,
        )
    except BaseException as exc:
        errors.append(f"pre/post health comparison: {type(exc).__name__}: {exc}")
    if runner_started_epoch is not None:
        try:
            capture_kernel_log(evidence, runner_started_epoch)
        except BaseException as exc:
            errors.append(f"kernel log validation: {type(exc).__name__}: {exc}")
    try:
        reservation = read_canonical_json(
            evidence / "admission-reservation.json", "reservation evidence"
        )
        completion = validate_reservation_completion(
            reservation,
            reservation_id,
            target_id,
            target_name,
            operator_email,
        )
        _write_exclusive(
            evidence / "reservation-completion-validation.json",
            canonical_json(completion),
            0o600,
        )
    except BaseException as exc:
        errors.append(f"reservation completion: {type(exc).__name__}: {exc}")
    return errors


def _write_failure(
    evidence: Path,
    *,
    error: str,
    phase: str,
    postflight_errors: list[str],
    runner_invoked: bool,
    runner_returncode: int | None,
    timed_out: bool,
) -> None:
    payload = {
        "error": error,
        "phase": phase,
        "postflight_errors": postflight_errors,
        "runner_invoked": runner_invoked,
        "runner_returncode": runner_returncode,
        "schema": "loom-q16384-first-attention-launch-abi-evidence-fail-v1",
        "status": "fail",
        "timed_out": timed_out,
    }
    _write_exclusive(evidence / "FAIL.json", canonical_json(payload), 0o600)


def _observe_termination(
    termination_state: dict[str, int | bool | None], *, include_pending: bool
) -> int | None:
    signum = termination_state["signum"]
    if signum is None and include_pending:
        pending = signal.sigpending()
        for candidate in TERMINATION_SIGNALS:
            if candidate in pending:
                signum = int(candidate)
                termination_state["signum"] = signum
                break
    return int(signum) if signum is not None else None


def _restore_termination_mask(previous_mask: set[signal.Signals]) -> None:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for termination_signal, previous in previous_handlers.items():
        signal.signal(termination_signal, previous)


def _remove_private_regular(path: Path) -> None:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"evidence verdict is not regular: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"evidence verdict is a symlink: {path}")
    require(metadata.st_nlink == 1, f"evidence verdict is hardlinked: {path}")
    require(metadata.st_uid == os.geteuid(), f"evidence verdict owner changed: {path}")
    path.unlink()


def _replace_sealed_pass_with_failure(
    evidence: Path,
    *,
    signum: int,
    phase: str,
    postflight_errors: list[str],
    runner_invoked: bool,
    runner_returncode: int,
    timed_out: bool,
) -> None:
    for name in ("ROOT_SHA256", "SHA256SUMS", "PASS.json"):
        _remove_private_regular(evidence / name)
    _write_failure(
        evidence,
        error=f"TerminationRequested: launcher received signal {signum}",
        phase=phase,
        postflight_errors=postflight_errors,
        runner_invoked=runner_invoked,
        runner_returncode=runner_returncode,
        timed_out=timed_out,
    )
    seal_evidence(evidence)


def run_once(
    evidence: Path,
    admission: Path,
    authorization: Path,
    state_root: Path,
    *,
    process_exit: Callable[[int], None] | None = os._exit,
) -> int:
    require(
        RUNTIME_TIMEOUT_SECONDS
        + TERMINATION_GRACE_SECONDS
        + MINIMUM_POSTFLIGHT_RESERVE_SECONDS
        == PRELAUNCH_MINIMUM_REMAINING_SECONDS,
        "internal reservation time budget is inconsistent",
    )
    require(os.environ.get("PROCEED_GPU") == "YES", "PROCEED_GPU=YES is required")
    package_root = PACKAGE_ROOT.resolve(strict=True)
    manifest = load_manifest(package_root)
    target = manifest["target"]
    operator = manifest["operator"]
    execution = manifest["execution"]
    reservation_manifest = manifest["reservation"]
    expected_hosts = {target["hostname"], target["fqdn"]}
    expected_user = operator["short_id"]
    expected_home = f"/home/{expected_user}"
    runner_environment = {
        "HOME": expected_home,
        "LANG": "C",
        "LC_ALL": "C",
        "MICROGATE_AUTHORIZATION_CONSUMED": "1",
        "PATH": "/usr/bin:/bin",
        "ROCR_VISIBLE_DEVICES": "0",
    }
    require(os.uname().nodename in expected_hosts, "execution host mismatch")
    account = pwd.getpwuid(os.geteuid())
    require(account.pw_name == expected_user, "execution user mismatch")
    require(getattr(account, "pw_dir", expected_home) == expected_home,
            "execution home directory mismatch")
    require(Path(TIMEOUT).is_file() and os.access(TIMEOUT, os.X_OK),
            "GNU timeout is unavailable")

    require(str(evidence) == execution["evidence_path"],
            "evidence path differs from package manifest")
    require(str(state_root) == execution["state_path"],
            "state path differs from package manifest")
    for path, description, exists in (
        (evidence, "evidence path", False),
        (admission, "admission path", True),
        (authorization, "authorization path", True),
        (state_root, "state path", state_root.exists()),
    ):
        require_safe_absolute_path(path, description, exists=exists)
        require(not _is_within(path, package_root), f"{description} must be outside package")
    require(not evidence.exists(), "evidence path already exists")
    if not state_root.exists():
        os.mkdir(state_root, 0o700)
    state_status = state_root.lstat()
    require(stat.S_ISDIR(state_status.st_mode), "state path is not a directory")
    require(state_status.st_uid == os.geteuid(), "state path owner changed")
    require(state_status.st_mode & 0o077 == 0, "state path is not mode-private")
    require(
        not _is_within(state_root, admission)
        and not _is_within(admission, state_root),
        "state and admission directories must be disjoint",
    )
    require(
        not _is_within(evidence, admission)
        and not _is_within(admission, evidence)
        and not _is_within(evidence, state_root)
        and not _is_within(state_root, evidence),
        "evidence, admission, and state directories must be disjoint",
    )
    require(not _is_within(authorization, admission),
            "authorization may not be inside admission")

    receipt_path = state_root / RECEIPT_NAME
    consumed_path = Path(f"{receipt_path}.consumed")
    attempt_path = state_root / ATTEMPT_NAME
    result_path = evidence / "RESULT.json"
    mutable = {authorization, receipt_path, consumed_path, attempt_path, result_path}
    require(len(mutable) == 5, "live mutable paths collide")
    require(not _lexists(receipt_path), "ready receipt already exists")
    require(not _lexists(consumed_path), "authorization was already consumed")
    require(not _lexists(attempt_path), "the one permitted runner attempt was already made")

    lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(CAMPAIGN_LOCK, lock_flags, 0o600)
    os.fchmod(lock_descriptor, 0o600)
    lock_status = os.fstat(lock_descriptor)
    require(
        stat.S_ISREG(lock_status.st_mode)
        and lock_status.st_nlink == 1
        and lock_status.st_uid == os.geteuid()
        and stat.S_IMODE(lock_status.st_mode) == 0o600,
        "campaign lock metadata is unsafe",
    )

    previous_handlers: dict[int, Any] = {}
    termination_state: dict[str, int | bool | None] = {
        "cleanup": False,
        "signum": None,
    }

    def request_termination(signum: int, _frame: object) -> None:
        if termination_state["signum"] is not None:
            return
        termination_state["signum"] = signum
        if termination_state["cleanup"] is False:
            raise TerminationRequested(signum)

    for termination_signal in TERMINATION_SIGNALS:
        previous_handlers[termination_signal] = signal.signal(
            termination_signal, request_termination
        )

    phase = "lock"
    runner_invoked = False
    preflight_started = False
    retired_by_launcher = False
    outcome: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    pending_exception: BaseException | None = None
    postflight_errors: list[str] = []
    final_returncode = 1
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase = "package-verification"
        root_digest, closure_count = verify_package(package_root)
        os.mkdir(evidence, 0o700)
        _fsync_directory(evidence.parent)
        try:
            phase = "initial-validation"
            initial = validate(
                admission,
                authorization,
                root_digest,
                receipt_path,
                result_path,
                os.uname().nodename,
                minimum_remaining_seconds=INITIAL_MINIMUM_REMAINING_SECONDS,
                package_root=package_root,
            )
            _write_exclusive(
                evidence / "initial-validation.json", canonical_json(initial), 0o600
            )
            for name in ("captured-utc.txt", "jobs.json", "reservation.json", "whoami.json"):
                _copy_private(admission / name, evidence / f"admission-{name}")
            _copy_private(authorization, evidence / "authorization.json")
            _write_exclusive(
                evidence / "package-verification.json",
                canonical_json({
                    "closure_file_count": closure_count,
                    "microgate_root_sha256": root_digest,
                    "pass": True,
                    "schema": "loom-q16384-first-attention-launch-abi-package-verification-v1",
                }),
                0o600,
            )

            phase = "preflight"
            preflight_started = True
            capture_snapshot(evidence / "preflight", "preflight")
            phase = "prelaunch-quiescence"
            capture_snapshot(evidence / "prelaunch", "prelaunch")

            phase = "prelaunch-validation"
            second_root, second_count = verify_package(package_root)
            require((second_root, second_count) == (root_digest, closure_count),
                    "sealed package changed before launch")
            validation = validate(
                admission,
                authorization,
                root_digest,
                receipt_path,
                result_path,
                os.uname().nodename,
                minimum_remaining_seconds=PRELAUNCH_MINIMUM_REMAINING_SECONDS,
                package_root=package_root,
            )
            _write_exclusive(
                evidence / "prelaunch-validation.json", canonical_json(validation), 0o600
            )
            require(not result_path.exists(), "result path appeared before launch")
            require(not _lexists(receipt_path) and not _lexists(consumed_path),
                    "authorization receipt collision before launch")

            phase = "consume-authorization"
            attempt = {
                "authorization_id": validation["authorization_id"],
                "invocation_count": 1,
                "microgate_root_sha256": root_digest,
                "result_path": str(result_path),
                "schema": "loom-q16384-first-attention-launch-abi-attempt-v1",
                "started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            _write_exclusive(attempt_path, canonical_json(attempt), 0o400)
            _fsync_directory(state_root)
            _copy_private(attempt_path, evidence / "attempt.json")
            create_receipt(receipt_path, validation, authorization, result_path)

            command = [
                str(RUNNER),
                "--run",
                str(package_root),
                str(result_path),
                str(receipt_path),
                str(authorization),
            ]
            phase = "gpu-execution"
            runner_invoked = True
            outcome = _invoke_runner_once(command, runner_environment)
            retired_by_launcher = retire_unclaimed_receipt(receipt_path)
            _fsync_directory(state_root)
            _write_exclusive(evidence / "runner.stdout.txt", outcome["stdout"], 0o600)
            _write_exclusive(evidence / "runner.stderr.txt", outcome["stderr"], 0o600)
            if _lexists(consumed_path):
                _copy_private(consumed_path, evidence / "authorization-consumed.txt")
            final_returncode = int(outcome["returncode"])
            phase = "runner-complete"
        except BaseException as exc:
            pending_exception = exc
        finally:
            termination_state["cleanup"] = True
            if _lexists(receipt_path) and not _lexists(consumed_path):
                try:
                    retired_by_launcher = retire_unclaimed_receipt(receipt_path)
                except BaseException as exc:
                    postflight_errors.append(
                        f"receipt retirement: {type(exc).__name__}: {exc}"
                    )
            if evidence.exists() and _lexists(consumed_path) and not (
                evidence / "authorization-consumed.txt"
            ).exists():
                try:
                    _copy_private(consumed_path, evidence / "authorization-consumed.txt")
                except BaseException as exc:
                    postflight_errors.append(
                        f"receipt evidence: {type(exc).__name__}: {exc}"
                    )

            if preflight_started:
                phase = "postflight"
                postflight_errors.extend(
                    _capture_postflight(
                        evidence,
                        int(outcome["started_epoch"]) if outcome is not None else None,
                        reservation_manifest["id"],
                        target["id"],
                        target["hostname"],
                        operator["email"],
                    )
                )

            postflight_pass = outcome is not None and not postflight_errors
            if termination_state["signum"] is not None and pending_exception is None:
                pending_exception = TerminationRequested(
                    int(termination_state["signum"])
                )
            if outcome is not None and validation is not None:
                run_record = {
                    "authorization_id": validation["authorization_id"],
                    "command": command,
                    "environment": runner_environment,
                    "evidence_path": str(evidence),
                    "invocation_count": 1,
                    "kill_after_seconds": TERMINATION_GRACE_SECONDS,
                    "microgate_root_sha256": root_digest,
                    "package_root": str(package_root),
                    "postflight_pass": postflight_pass,
                    "receipt_claimed_or_retired": _lexists(consumed_path),
                    "receipt_retired_by_launcher": retired_by_launcher,
                    "returncode": outcome["returncode"],
                    "runner_finished_epoch": outcome["finished_epoch"],
                    "runner_started_epoch": outcome["started_epoch"],
                    "runtime_timeout_seconds": RUNTIME_TIMEOUT_SECONDS,
                    "schema": "loom-q16384-first-attention-launch-abi-run-record-v1",
                    "supervised_command": outcome["supervised_command"],
                    "timed_out": outcome["timed_out"],
                }
                try:
                    _write_exclusive(
                        evidence / "run-record.json", canonical_json(run_record), 0o600
                    )
                except BaseException as exc:
                    if pending_exception is None:
                        pending_exception = exc

            candidate_success = (
                pending_exception is None
                and outcome is not None
                and outcome["returncode"] == 0
                and outcome["timed_out"] is False
                and retired_by_launcher is False
                and postflight_pass
                and result_path.is_file()
            )
            if evidence.exists():
                if candidate_success and validation is not None:
                    previous_mask = signal.pthread_sigmask(
                        signal.SIG_BLOCK, TERMINATION_SIGNALS
                    )
                    try:
                        pass_record = {
                            "authorization_id": validation["authorization_id"],
                            "completed_epoch": int(time.time()),
                            "invocation_count": 1,
                            "microgate_root_sha256": root_digest,
                            "postflight_pass": True,
                            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                            "runner_returncode": 0,
                            "schema": "loom-q16384-first-attention-launch-abi-evidence-pass-v1",
                            "status": "pass",
                        }
                        _write_exclusive(
                            evidence / "PASS.json", canonical_json(pass_record), 0o600
                        )
                        phase = "semantic-evidence-validation"
                        semantic_failure: BaseException | None = None
                        try:
                            validate_success_evidence(evidence, verify_outer_seal=False)
                        except BaseException as exc:
                            semantic_failure = exc
                        signum = _observe_termination(
                            termination_state, include_pending=True
                        )
                        if signum is not None:
                            semantic_failure = TerminationRequested(signum)
                        if semantic_failure is not None:
                            _remove_private_regular(evidence / "PASS.json")
                            candidate_success = False
                            final_returncode = 90
                            _write_failure(
                                evidence,
                                error=(
                                    "semantic evidence validation: "
                                    f"{type(semantic_failure).__name__}: {semantic_failure}"
                                ),
                                phase=phase,
                                postflight_errors=postflight_errors,
                                runner_invoked=runner_invoked,
                                runner_returncode=int(outcome["returncode"]),
                                timed_out=bool(outcome["timed_out"]),
                            )
                        phase = "evidence-sealing"
                        seal_evidence(evidence)
                        signum = _observe_termination(
                            termination_state, include_pending=True
                        )
                        if candidate_success and signum is not None:
                            _replace_sealed_pass_with_failure(
                                evidence,
                                signum=signum,
                                phase=phase,
                                postflight_errors=postflight_errors,
                                runner_invoked=runner_invoked,
                                runner_returncode=int(outcome["returncode"]),
                                timed_out=bool(outcome["timed_out"]),
                            )
                            candidate_success = False
                            final_returncode = 90
                    finally:
                        _restore_termination_mask(previous_mask)

                    signum = _observe_termination(
                        termination_state, include_pending=False
                    )
                    if candidate_success and signum is not None:
                        phase = "final-return"
                        _replace_sealed_pass_with_failure(
                            evidence,
                            signum=signum,
                            phase=phase,
                            postflight_errors=postflight_errors,
                            runner_invoked=runner_invoked,
                            runner_returncode=int(outcome["returncode"]),
                            timed_out=bool(outcome["timed_out"]),
                        )
                        candidate_success = False
                        final_returncode = 90
                else:
                    if outcome is not None and outcome["returncode"] == 0:
                        final_returncode = 90
                    error = (
                        f"{type(pending_exception).__name__}: {pending_exception}"
                        if pending_exception is not None
                        else f"runner/postflight failure (returncode={final_returncode})"
                    )
                    _write_failure(
                        evidence,
                        error=error,
                        phase=phase,
                        postflight_errors=postflight_errors,
                        runner_invoked=runner_invoked,
                        runner_returncode=(
                            int(outcome["returncode"]) if outcome is not None else None
                        ),
                        timed_out=bool(outcome["timed_out"]) if outcome is not None else False,
                    )
                if not (evidence / "ROOT_SHA256").exists():
                    seal_evidence(evidence)
                if candidate_success:
                    final_returncode = 0
    finally:
        os.close(lock_descriptor)
        preserve_handlers = (
            process_exit is not None
            and final_returncode == 0
            and pending_exception is None
        )
        if not preserve_handlers:
            _restore_signal_handlers(previous_handlers)

    signum = _observe_termination(termination_state, include_pending=False)
    if final_returncode == 0 and signum is not None:
        _replace_sealed_pass_with_failure(
            evidence,
            signum=signum,
            phase="final-return",
            postflight_errors=postflight_errors,
            runner_invoked=runner_invoked,
            runner_returncode=int(outcome["returncode"]) if outcome is not None else 0,
            timed_out=bool(outcome["timed_out"]) if outcome is not None else False,
        )
        final_returncode = 90
    if preserve_handlers and final_returncode != 0:
        _restore_signal_handlers(previous_handlers)
        preserve_handlers = False
    if pending_exception is not None:
        raise pending_exception
    if process_exit is not None and final_returncode == 0:
        try:
            termination_state["cleanup"] = False
            signum = _observe_termination(termination_state, include_pending=False)
            if signum is not None:
                raise TerminationRequested(signum)
            process_exit(0)
        except TerminationRequested as exc:
            termination_state["cleanup"] = True
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, TERMINATION_SIGNALS
            )
            try:
                _replace_sealed_pass_with_failure(
                    evidence,
                    signum=exc.signum,
                    phase="process-exit",
                    postflight_errors=postflight_errors,
                    runner_invoked=runner_invoked,
                    runner_returncode=(
                        int(outcome["returncode"]) if outcome is not None else 0
                    ),
                    timed_out=(
                        bool(outcome["timed_out"]) if outcome is not None else False
                    ),
                )
                final_returncode = 90
                process_exit(final_returncode)
            finally:
                _restore_termination_mask(previous_mask)
                _restore_signal_handlers(previous_handlers)
                preserve_handlers = False
            return final_returncode

        termination_state["cleanup"] = True
        _restore_signal_handlers(previous_handlers)
        preserve_handlers = False
    return final_returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("admission", type=Path)
    parser.add_argument("authorization", type=Path)
    parser.add_argument("state_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_once(
        args.evidence,
        args.admission,
        args.authorization,
        args.state_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
