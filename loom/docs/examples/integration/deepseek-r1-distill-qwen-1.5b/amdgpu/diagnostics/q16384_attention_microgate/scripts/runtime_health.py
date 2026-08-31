#!/usr/bin/python3 -I
"""Capture and validate the live host state around the one-shot GPU run."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Callable


SUDO = "/usr/bin/sudo"
LSOF = "/usr/bin/lsof"
AMD_SMI = "/opt/rocm/bin/amd-smi"
JOURNALCTL = "/usr/bin/journalctl"
KFD_ROOT = Path("/sys/class/kfd/kfd/proc")
EXPECTED_PERFORMANCE_LEVEL = "AMDSMI_DEV_PERF_LEVEL_AUTO"
EXPECTED_MONITOR_NAMESPACE = "moby"
EXPECTED_MONITOR_ID = "82a9e1f48b702e1c1b642cd9f9d45f8f9de33afea52edfbd8b90325f4cf479b1"
EXPECTED_MONITOR_TASK = f"{EXPECTED_MONITOR_NAMESPACE}/{EXPECTED_MONITOR_ID}"
EXPECTED_MONITOR_SNAPSHOT = {
    "binds": ["/:/host:ro,rslave"],
    "config_image": "quay.io/prometheus/node-exporter:latest",
    "device_cgroup_rules": None,
    "device_requests": None,
    "devices": [],
    "id": EXPECTED_MONITOR_ID,
    "image_id": "sha256:47509d7f7c15a729686d9d5eccec66abd4b2495d9ae01f637f63e6375fe93f8b",
    "name": "/node-exporter.service",
    "privileged": False,
    "restart_count": 0,
    "running": True,
    "started_at": "2026-05-21T05:38:34.604915237Z",
    "status": "running",
}
ECC_TOTAL_FIELDS = (
    "total_correctable_count",
    "total_uncorrectable_count",
    "total_deferred_count",
)
ECC_BLOCKS = ("GFX", "UMC", "SDMA", "MMHUB")
ECC_COUNTER_FIELDS = ("correctable_count", "uncorrectable_count", "deferred_count")
MINIMUM_FREE_VRAM_MB = 32_768
REQUIRED_CONSECUTIVE_CLEAN_POLLS = 2
MAXIMUM_QUIESCENCE_POLLS = 30
CONTROL_TIMEOUT_SECONDS = 60
SNAPSHOT_TIMEOUT_SECONDS = 180
CONTROL_ENVIRONMENT = {
    "HOME": "/home/dagarvey",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
GPU_ERROR_PATTERN = re.compile(
    rb"amdgpu_job_timedout|gpu reset|gpu fault|page fault|ring .*timeout|"
    rb"watchdog|ras.*(?:error|uncorrectable)|xgmi.*error|"
    rb"kfd.*(?:error|fault|timeout)",
    re.IGNORECASE,
)
SNAPSHOT_FILES = {
    "allowed-monitor-inspect.json",
    "allowed-monitor-inspect.stderr.txt",
    "allowed-monitor-validation.json",
    "amd-smi-metric.json",
    "amd-smi-metric.stderr.txt",
    "amd-smi-process.json",
    "amd-smi-process.stderr.txt",
    "containerd-tasks.txt",
    "health-validation.json",
    "quiescence.json",
}
KERNEL_FILES = {
    "kernel-errors-during-run.txt",
    "kernel-log-capture.json",
    "kernel-log-during-run.stderr.txt",
    "kernel-log-during-run.txt",
    "kernel-log-window.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, description: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {description}") from exc


def read_canonical_json(path: Path, description: str) -> Any:
    raw = path.read_bytes()
    value = parse_json(raw, description)
    require(raw == canonical_json(value), f"{description} is not canonical JSON")
    return value


def write_exclusive(path: Path, data: bytes, mode: int = 0o600) -> None:
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


def _run_control(
    argv: list[str],
    *,
    allowed_returncodes: set[int] | None = None,
    timeout_seconds: float = CONTROL_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    allowed = allowed_returncodes or {0}
    require(timeout_seconds > 0, "control command has no remaining time budget")
    try:
        completed = subprocess.run(
            argv,
            env=CONTROL_ENVIRONMENT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"control command timed out: {argv!r}") from exc
    require(
        completed.returncode in allowed,
        f"control command failed ({completed.returncode}): {argv!r}: "
        f"{completed.stderr.decode('utf-8', errors='replace').strip()}",
    )
    return completed


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    require(remaining > 0, "host snapshot exceeded its total time budget")
    return min(float(CONTROL_TIMEOUT_SECONDS), remaining)


def _require_live_tools(deadline: float) -> list[str]:
    for path in (SUDO, LSOF, AMD_SMI, JOURNALCTL):
        require(Path(path).is_file() and os.access(path, os.X_OK), f"required tool absent: {path}")
    require(KFD_ROOT.is_dir(), "KFD process registry is unavailable")
    device_paths = [Path("/dev/kfd"), *sorted(Path("/dev/dri").glob("renderD*"))]
    require(len(device_paths) >= 2, "GPU device nodes are unavailable")
    for path in device_paths:
        metadata = path.stat()
        require(stat.S_ISCHR(metadata.st_mode), f"GPU device is not a character node: {path}")
    _run_control(
        [SUDO, "-n", "/usr/bin/true"], timeout_seconds=_remaining_timeout(deadline)
    )
    return [str(path) for path in device_paths]


def _parse_lines(raw: bytes, description: str) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{description} is not UTF-8") from exc
    values = [line.strip() for line in text.splitlines() if line.strip()]
    require(all("/" not in value and not any(ch.isspace() for ch in value) for value in values),
            f"{description} contains an unsafe identifier")
    return values


def _kfd_pids() -> list[int]:
    values = []
    for entry in KFD_ROOT.iterdir():
        if entry.name.isdigit() and entry.is_dir() and not entry.is_symlink():
            values.append(int(entry.name))
    return sorted(set(values))


def _device_user_pids(device_paths: list[str], deadline: float) -> list[int]:
    completed = _run_control(
        [SUDO, "-n", LSOF, "-t", *device_paths],
        allowed_returncodes={0, 1},
        timeout_seconds=_remaining_timeout(deadline),
    )
    if completed.returncode == 1:
        require(not completed.stdout and not completed.stderr,
                "lsof failed rather than reporting no device users")
        return []
    values = _parse_lines(completed.stdout, "lsof PID output")
    require(all(value.isdigit() for value in values), "lsof returned a non-PID")
    return sorted({int(value) for value in values})


def _containerd_tasks(deadline: float) -> list[str]:
    namespaces_result = _run_control(
        [SUDO, "-n", "ctr", "namespaces", "list", "-q"],
        timeout_seconds=_remaining_timeout(deadline),
    )
    namespaces = sorted(set(_parse_lines(namespaces_result.stdout, "containerd namespace output")))
    tasks: list[str] = []
    for namespace in namespaces:
        task_result = _run_control(
            [SUDO, "-n", "ctr", "--namespace", namespace, "tasks", "list", "-q"],
            timeout_seconds=_remaining_timeout(deadline),
        )
        for task in _parse_lines(task_result.stdout, "containerd task output"):
            tasks.append(f"{namespace}/{task}")
    return sorted(tasks)


def _validate_task_set(tasks: list[str]) -> list[str]:
    require(tasks == sorted(tasks), "containerd task list is not sorted")
    require(len(tasks) == len(set(tasks)), "containerd task list contains duplicates")
    require(tasks.count(EXPECTED_MONITOR_TASK) == 1,
            "allowed monitor task is missing or duplicated")
    return [task for task in tasks if task != EXPECTED_MONITOR_TASK]


def validate_monitor(raw: bytes) -> dict[str, Any]:
    rows = parse_json(raw, "monitor inspection JSON")
    require(isinstance(rows, list) and len(rows) == 1, "expected one monitor container")
    row = rows[0]
    require(isinstance(row, dict), "monitor inspection row is invalid")
    host = row.get("HostConfig", {})
    state = row.get("State", {})
    config = row.get("Config", {})
    require(isinstance(host, dict), "monitor HostConfig is invalid")
    require(isinstance(state, dict), "monitor State is invalid")
    require(isinstance(config, dict), "monitor Config is invalid")
    snapshot = {
        "binds": host.get("Binds"),
        "config_image": config.get("Image"),
        "device_cgroup_rules": host.get("DeviceCgroupRules"),
        "device_requests": host.get("DeviceRequests"),
        "devices": host.get("Devices"),
        "id": row.get("Id"),
        "image_id": row.get("Image"),
        "name": row.get("Name"),
        "privileged": host.get("Privileged"),
        "restart_count": row.get("RestartCount"),
        "running": state.get("Running"),
        "started_at": state.get("StartedAt"),
        "status": state.get("Status"),
    }
    require(snapshot == EXPECTED_MONITOR_SNAPSHOT,
            f"allowed monitor configuration changed: {snapshot!r}")
    return snapshot


def _numeric(value: Any, description: str) -> int | float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{description} is not finite numeric data",
    )
    return value


def _gpu_metrics(raw: bytes) -> dict[str, Any]:
    payload = parse_json(raw, "AMD SMI metric JSON")
    require(isinstance(payload, dict), "AMD SMI metric output is not an object")
    rows = payload.get("gpu_data")
    require(isinstance(rows, list) and len(rows) == 1, "expected one GPU metric row")
    gpu = rows[0]
    require(isinstance(gpu, dict) and gpu.get("gpu") == 0, "expected GPU index 0")
    return gpu


def _ecc_state(gpu: dict[str, Any]) -> dict[str, Any]:
    ecc = gpu.get("ecc")
    blocks_payload = gpu.get("ecc_blocks")
    require(isinstance(ecc, dict), "GPU ECC totals are absent")
    require(isinstance(blocks_payload, dict), "GPU ECC blocks are absent")
    totals = {
        name: _numeric(ecc.get(name), f"ECC total {name}")
        for name in ECC_TOTAL_FIELDS
    }
    blocks: dict[str, dict[str, int | float]] = {}
    for name in ECC_BLOCKS:
        counters = blocks_payload.get(name)
        require(isinstance(counters, dict), f"ECC block {name} is invalid")
        require(set(counters) == set(ECC_COUNTER_FIELDS),
                f"ECC block {name} counters changed")
        blocks[name] = {
            counter: _numeric(counters[counter], f"ECC block {name} {counter}")
            for counter in ECC_COUNTER_FIELDS
        }
    return {"blocks": blocks, "totals": totals}


def validate_amd_snapshot(
    process_raw: bytes,
    metric_raw: bytes,
    minimum_free_vram_mb: int = MINIMUM_FREE_VRAM_MB,
) -> dict[str, Any]:
    processes = parse_json(process_raw, "AMD SMI process JSON")
    require(isinstance(processes, list) and len(processes) == 1,
            "expected one GPU process row")
    require(isinstance(processes[0], dict) and processes[0].get("gpu") == 0,
            "expected GPU index 0 in process snapshot")
    process_rows = processes[0].get("process_list", [])
    require(
        isinstance(process_rows, list)
        and len(process_rows) == 1
        and isinstance(process_rows[0], dict)
        and process_rows[0].get("process_info") == "No running processes detected",
        "AMD SMI reports a running GPU process",
    )

    gpu = _gpu_metrics(metric_raw)
    require(gpu.get("perf_level") == EXPECTED_PERFORMANCE_LEVEL,
            "GPU performance level is not auto")
    usage = gpu.get("usage")
    memory = gpu.get("mem_usage")
    require(isinstance(usage, dict) and isinstance(memory, dict),
            "GPU usage or memory metrics are absent")
    gfx = usage.get("gfx_activity")
    umc = usage.get("umc_activity")
    free = memory.get("free_vram")
    require(isinstance(gfx, dict) and gfx.get("unit") == "%", "GFX activity unit changed")
    require(isinstance(umc, dict) and umc.get("unit") == "%", "UMC activity unit changed")
    require(isinstance(free, dict) and free.get("unit") == "MB", "free VRAM unit changed")
    gfx_activity = _numeric(gfx.get("value"), "GFX activity")
    umc_activity = _numeric(umc.get("value"), "UMC activity")
    free_vram_mb = _numeric(free.get("value"), "free VRAM")
    require(gfx_activity == 0, "GFX activity is nonzero")
    require(umc_activity == 0, "UMC activity is nonzero")
    require(type(minimum_free_vram_mb) is int and minimum_free_vram_mb >= 0,
            "minimum free VRAM must be a non-negative integer")
    require(free_vram_mb >= minimum_free_vram_mb, "insufficient free VRAM")
    ecc = _ecc_state(gpu)
    require(all(value == 0 for value in ecc["totals"].values()),
            "GPU ECC total is nonzero")
    require(
        all(value == 0 for counters in ecc["blocks"].values() for value in counters.values()),
        "GPU ECC block count is nonzero",
    )
    return {
        "ecc": ecc,
        "free_vram_mb": free_vram_mb,
        "gfx_activity_percent": gfx_activity,
        "gpu_index": 0,
        "gpu_processes": 0,
        "minimum_free_vram_mb": minimum_free_vram_mb,
        "pass": True,
        "performance_level": gpu["perf_level"],
        "schema": "loom-aiter-attention-gpu-health-v1",
        "umc_activity_percent": umc_activity,
    }


def compare_amd_health(pre_metric_raw: bytes, post_metric_raw: bytes) -> dict[str, Any]:
    pre_gpu = _gpu_metrics(pre_metric_raw)
    post_gpu = _gpu_metrics(post_metric_raw)
    pre_level = pre_gpu.get("perf_level")
    post_level = post_gpu.get("perf_level")
    require(pre_level == post_level, "GPU performance level changed during the run")
    require(post_level == EXPECTED_PERFORMANCE_LEVEL, "GPU performance level is not auto")
    pre = _ecc_state(pre_gpu)
    post = _ecc_state(post_gpu)
    require(pre == post, "GPU ECC state changed during the run")
    require(all(value == 0 for value in post["totals"].values()),
            "GPU ECC total is nonzero")
    require(
        all(value == 0 for counters in post["blocks"].values() for value in counters.values()),
        "GPU ECC block count is nonzero",
    )
    return {
        "ecc": post,
        "pass": True,
        "performance_level": post_level,
        "schema": "loom-aiter-attention-gpu-health-comparison-v1",
        "unchanged": True,
    }


def _canonical_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_quiescence(payload: Any, expected_stage: str) -> dict[str, Any]:
    require(isinstance(payload, dict), "quiescence record is not an object")
    require(set(payload) == {
        "pass", "polls", "required_consecutive_clean", "schema",
        "snapshot_timeout_seconds", "stage"
    }, "quiescence fields changed")
    require(payload["schema"] == "loom-aiter-attention-host-quiescence-v1",
            "quiescence schema changed")
    require(payload["stage"] == expected_stage, "quiescence stage changed")
    require(payload["pass"] is True, "quiescence did not pass")
    require(payload["required_consecutive_clean"] == REQUIRED_CONSECUTIVE_CLEAN_POLLS,
            "quiescence consecutive-poll requirement changed")
    require(payload["snapshot_timeout_seconds"] == SNAPSHOT_TIMEOUT_SECONDS,
            "host snapshot timeout changed")
    polls = payload["polls"]
    require(isinstance(polls, list) and REQUIRED_CONSECUTIVE_CLEAN_POLLS <= len(polls)
            <= MAXIMUM_QUIESCENCE_POLLS, "quiescence poll count is invalid")
    consecutive = 0
    for index, poll in enumerate(polls, start=1):
        require(isinstance(poll, dict), "quiescence poll is not an object")
        require(set(poll) == {
            "captured_utc", "clean", "consecutive_clean", "containerd_tasks",
            "device_user_pids", "kfd_pids", "poll", "unexpected_containerd_tasks"
        }, "quiescence poll fields changed")
        require(poll["poll"] == index, "quiescence poll numbering changed")
        captured = poll["captured_utc"]
        require(isinstance(captured, str) and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", captured
        ) is not None, "quiescence timestamp is not canonical UTC")
        for field in ("kfd_pids", "device_user_pids"):
            values = poll[field]
            require(isinstance(values, list) and values == sorted(set(values))
                    and all(type(value) is int and value > 0 for value in values),
                    f"quiescence {field} is invalid")
        tasks = poll["containerd_tasks"]
        unexpected = poll["unexpected_containerd_tasks"]
        require(isinstance(tasks, list) and all(isinstance(value, str) for value in tasks),
                "quiescence container task list is invalid")
        calculated_unexpected = _validate_task_set(tasks)
        require(unexpected == calculated_unexpected,
                "quiescence unexpected task list is inconsistent")
        clean = not poll["kfd_pids"] and not poll["device_user_pids"] and not unexpected
        require(poll["clean"] is clean, "quiescence clean flag is inconsistent")
        consecutive = consecutive + 1 if clean else 0
        require(poll["consecutive_clean"] == consecutive,
                "quiescence consecutive clean count is inconsistent")
    require(consecutive >= REQUIRED_CONSECUTIVE_CLEAN_POLLS,
            "quiescence did not end with two clean polls")
    return payload


def capture_snapshot(
    root: Path,
    stage: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    require(stage in {"preflight", "prelaunch", "postflight"}, "invalid snapshot stage")
    os.mkdir(root, 0o700)
    deadline = time.monotonic() + SNAPSHOT_TIMEOUT_SECONDS
    device_paths = _require_live_tools(deadline)
    polls: list[dict[str, Any]] = []
    consecutive = 0
    for poll_number in range(1, MAXIMUM_QUIESCENCE_POLLS + 1):
        kfd = _kfd_pids()
        users = _device_user_pids(device_paths, deadline)
        tasks = _containerd_tasks(deadline)
        unexpected = _validate_task_set(tasks)
        clean = not kfd and not users and not unexpected
        consecutive = consecutive + 1 if clean else 0
        polls.append({
            "captured_utc": _canonical_utc_now(),
            "clean": clean,
            "consecutive_clean": consecutive,
            "containerd_tasks": tasks,
            "device_user_pids": users,
            "kfd_pids": kfd,
            "poll": poll_number,
            "unexpected_containerd_tasks": unexpected,
        })
        if consecutive >= REQUIRED_CONSECUTIVE_CLEAN_POLLS:
            break
        require(deadline - time.monotonic() > 1.0,
                f"{stage} snapshot lacks time for another quiescence poll")
        sleep(1.0)
    quiescence = {
        "pass": consecutive >= REQUIRED_CONSECUTIVE_CLEAN_POLLS,
        "polls": polls,
        "required_consecutive_clean": REQUIRED_CONSECUTIVE_CLEAN_POLLS,
        "schema": "loom-aiter-attention-host-quiescence-v1",
        "snapshot_timeout_seconds": SNAPSHOT_TIMEOUT_SECONDS,
        "stage": stage,
    }
    write_exclusive(root / "quiescence.json", canonical_json(quiescence))
    require(quiescence["pass"] is True, f"{stage} GPU quiescence timed out")

    final_tasks = _containerd_tasks(deadline)
    require(not _validate_task_set(final_tasks), f"{stage} has unexpected containerd tasks")
    write_exclusive(
        root / "containerd-tasks.txt",
        ("\n".join(final_tasks) + "\n").encode("ascii"),
    )

    monitor = _run_control(
        [SUDO, "-n", "docker", "inspect", EXPECTED_MONITOR_ID],
        timeout_seconds=_remaining_timeout(deadline),
    )
    write_exclusive(root / "allowed-monitor-inspect.json", monitor.stdout)
    write_exclusive(root / "allowed-monitor-inspect.stderr.txt", monitor.stderr)
    monitor_validation = validate_monitor(monitor.stdout)
    write_exclusive(
        root / "allowed-monitor-validation.json", canonical_json(monitor_validation)
    )

    processes = _run_control(
        [AMD_SMI, "process", "--json"], timeout_seconds=_remaining_timeout(deadline)
    )
    metrics = _run_control(
        [AMD_SMI, "metric", "--json"], timeout_seconds=_remaining_timeout(deadline)
    )
    write_exclusive(root / "amd-smi-process.json", processes.stdout)
    write_exclusive(root / "amd-smi-process.stderr.txt", processes.stderr)
    write_exclusive(root / "amd-smi-metric.json", metrics.stdout)
    write_exclusive(root / "amd-smi-metric.stderr.txt", metrics.stderr)
    health = validate_amd_snapshot(processes.stdout, metrics.stdout)
    write_exclusive(root / "health-validation.json", canonical_json(health))
    return {"health": health, "monitor": monitor_validation, "quiescence": quiescence}


def validate_snapshot_directory(root: Path, expected_stage: str) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"missing {expected_stage} snapshot")
    files = {entry.name for entry in root.iterdir()}
    require(files == SNAPSHOT_FILES,
            f"{expected_stage} snapshot file set changed: {sorted(files)}")
    quiescence = read_canonical_json(root / "quiescence.json", f"{expected_stage} quiescence")
    validate_quiescence(quiescence, expected_stage)
    task_raw = (root / "containerd-tasks.txt").read_bytes()
    require(task_raw == f"{EXPECTED_MONITOR_TASK}\n".encode("ascii"),
            f"{expected_stage} containerd task set changed")
    monitor_raw = (root / "allowed-monitor-inspect.json").read_bytes()
    monitor = validate_monitor(monitor_raw)
    recorded_monitor = read_canonical_json(
        root / "allowed-monitor-validation.json", f"{expected_stage} monitor validation"
    )
    require(recorded_monitor == monitor, f"{expected_stage} monitor validation differs")
    process_raw = (root / "amd-smi-process.json").read_bytes()
    metric_raw = (root / "amd-smi-metric.json").read_bytes()
    health = validate_amd_snapshot(process_raw, metric_raw)
    recorded_health = read_canonical_json(
        root / "health-validation.json", f"{expected_stage} health validation"
    )
    require(recorded_health == health, f"{expected_stage} health validation differs")
    return {"health": health, "monitor": monitor, "quiescence": quiescence}


def parse_utc(value: Any, description: str) -> dt.datetime:
    require(isinstance(value, str) and re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
    ) is not None, f"{description} is not canonical UTC")
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_reservation_completion(
    reservation: Any,
    reservation_id: str,
    target_id: str,
    target_name: str,
    operator_email: str,
    *,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    require(isinstance(reservation, dict), "reservation completion input is invalid")
    require(reservation.get("id") == reservation_id,
            "reservation ID changed at completion")
    require(reservation.get("target") == {
        "id": target_id, "name": target_name
    }, "reservation target changed at completion")
    require(reservation.get("batch_opt_out") is True,
            "reservation is not exclusive at completion")
    members = reservation.get("member_emails")
    require(isinstance(members, list) and operator_email in members,
            "operator is not a reservation member at completion")
    start = parse_utc(reservation.get("date_start"), "reservation start")
    end = parse_utc(reservation.get("date_end"), "reservation end")
    current_epoch = int(time.time()) if now_epoch is None else now_epoch
    require(type(current_epoch) is int and current_epoch >= 0,
            "completion epoch is invalid")
    current = dt.datetime.fromtimestamp(current_epoch, tz=dt.timezone.utc)
    remaining = int(end.timestamp()) - current_epoch
    require(start <= current < end, "reservation is not active at gate completion")
    return {
        "active_at_gate_completion": True,
        "pass": True,
        "reservation_end_epoch": int(end.timestamp()),
        "reservation_id": reservation_id,
        "reservation_remaining_seconds": remaining,
        "schema": "loom-aiter-attention-reservation-completion-v1",
        "validated_epoch": current_epoch,
        "validated_utc": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def capture_kernel_log(root: Path, start_epoch: int) -> dict[str, Any]:
    require(type(start_epoch) is int and start_epoch >= 0, "kernel log start epoch is invalid")
    end_epoch = int(time.time()) + 1
    require(end_epoch >= start_epoch, "kernel log clock moved backwards")
    command = [
        SUDO,
        "-n",
        JOURNALCTL,
        "-k",
        "--since",
        f"@{start_epoch}",
        "--until",
        f"@{end_epoch}",
        "--no-pager",
    ]
    completed = _run_control(command)
    matches = [line for line in completed.stdout.splitlines() if GPU_ERROR_PATTERN.search(line)]
    error_bytes = b"\n".join(matches) + (b"\n" if matches else b"")
    window = {
        "end_epoch": end_epoch,
        "schema": "loom-aiter-attention-kernel-log-window-v1",
        "start_epoch": start_epoch,
    }
    capture = {
        "command": command,
        "returncode": completed.returncode,
        "schema": "loom-aiter-attention-kernel-log-capture-v1",
        "timeout_seconds": CONTROL_TIMEOUT_SECONDS,
        "timed_out": False,
    }
    write_exclusive(root / "kernel-log-window.json", canonical_json(window))
    write_exclusive(root / "kernel-log-capture.json", canonical_json(capture))
    write_exclusive(root / "kernel-log-during-run.txt", completed.stdout)
    write_exclusive(root / "kernel-log-during-run.stderr.txt", completed.stderr)
    write_exclusive(root / "kernel-errors-during-run.txt", error_bytes)
    require(not matches, "kernel log contains a GPU error during the run")
    return {"capture": capture, "window": window}


def validate_kernel_evidence(root: Path, run_record: dict[str, Any]) -> dict[str, Any]:
    window = read_canonical_json(root / "kernel-log-window.json", "kernel log window")
    capture = read_canonical_json(root / "kernel-log-capture.json", "kernel log capture")
    require(set(window) == {"end_epoch", "schema", "start_epoch"},
            "kernel log window fields changed")
    require(window["schema"] == "loom-aiter-attention-kernel-log-window-v1",
            "kernel log window schema changed")
    start = window["start_epoch"]
    end = window["end_epoch"]
    require(type(start) is int and type(end) is int and start <= end,
            "kernel log window is invalid")
    require(start <= run_record["runner_started_epoch"]
            <= run_record["runner_finished_epoch"] < end,
            "kernel log window does not cover the runner")
    expected_command = [
        SUDO, "-n", JOURNALCTL, "-k", "--since", f"@{start}",
        "--until", f"@{end}", "--no-pager"
    ]
    require(capture == {
        "command": expected_command,
        "returncode": 0,
        "schema": "loom-aiter-attention-kernel-log-capture-v1",
        "timeout_seconds": CONTROL_TIMEOUT_SECONDS,
        "timed_out": False,
    }, "kernel log capture record changed")
    log_raw = (root / "kernel-log-during-run.txt").read_bytes()
    matches = [line for line in log_raw.splitlines() if GPU_ERROR_PATTERN.search(line)]
    expected_errors = b"\n".join(matches) + (b"\n" if matches else b"")
    require((root / "kernel-errors-during-run.txt").read_bytes() == expected_errors,
            "kernel error extraction differs from the captured log")
    require(not matches, "kernel log contains a GPU error during the run")
    return {"capture": capture, "window": window}
