#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ONE_POST = Path(sys.argv[1]).resolve()
ANALYZER = Path(sys.argv[2]).resolve()
AUTHORIZE = Path(sys.argv[3]).resolve()
sys.argv = [sys.argv[0]]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER_MODULE = load_module("first_attention_analyzer", ANALYZER)
AUTHORIZATION_MODULE = load_module(
    "first_attention_authorization", AUTHORIZE.parent / "authorization.py"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def clean_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("ROCPROF", "ROCPROFILER"))
        and name not in {"ROCP_TOOL_LIBRARIES", "HSA_TOOLS_LIB", "LD_PRELOAD"}
    }


def seal_package(package: Path) -> str:
    members = sorted(
        path for path in package.rglob("*")
        if path.is_file() and path.name not in {"ROOT_SHA256", "SHA256SUMS"}
    )
    lines = [
        f"{sha256_file(path)}  ./{path.relative_to(package).as_posix()}"
        for path in members
    ]
    sums = ("\n".join(lines) + "\n").encode("ascii")
    (package / "SHA256SUMS").write_bytes(sums)
    root = sha256_bytes(sums)
    (package / "ROOT_SHA256").write_text(root + "\n", encoding="ascii")
    return root


class IdentityHandler(BaseHTTPRequestHandler):
    health_count = 0
    post_count = 0
    worker_pid = 0
    worker_sha256 = ""
    health_headers: dict[str, str] = {}
    post_headers: dict[str, str] = {}

    @classmethod
    def reset(cls) -> None:
        cls.health_count = 0
        cls.post_count = 0
        cls.worker_pid = 0
        cls.worker_sha256 = ""
        cls.health_headers = {}
        cls.post_headers = {}

    def _send(self, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(200)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        type(self).health_count += 1
        if self.path != "/health":
            self.send_error(404)
            return
        payload = {
            "status": "ok",
            "inference_backend": "loom-raw-hsa",
            "busy": False,
            "worker_pid": type(self).worker_pid,
            "backend": {
                "name": "loom-raw-hsa",
                "worker_sha256": type(self).worker_sha256,
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
            },
        }
        self._send(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            type(self).health_headers,
        )

    def do_POST(self) -> None:  # noqa: N802
        type(self).post_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._send(b"data: [DONE]\n\n", type(self).post_headers)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class RuntimeFixture:
    def __init__(self, root: Path, *, worker_profiler: bool = False):
        self.root = root
        self.package = root / "sealed-package"
        self.evidence = root / "evidence"
        (self.package / "bin").mkdir(parents=True)
        (self.package / "deployment").mkdir()
        (self.package / "inputs").mkdir()
        self.evidence.mkdir(mode=0o700)
        self.adapter = self.package / "adapter.py"
        self.worker = self.package / "bin/deepseek_r1_server"
        self.body = self.package / "inputs/request.json"
        self.snapshot = self.package / "deployment/reservation.json"
        self.manifest = self.package / "DEPLOYMENT.json"
        self.provenance = self.package / "PROVENANCE.json"
        self.diagnostic = self.evidence / "first_attention.json"
        self.marker = self.evidence / "attempt.json"
        self.result = self.evidence / "result.json"
        self.response = self.evidence / "response.txt"
        self.authorization = self.evidence / "authorization.json"
        self.hostname = socket.gethostname().strip().rstrip(".").lower()
        self.reservation_id = "host-test-reservation"
        self.reservation_end = "2999-01-01T00:00:00Z"

        self.adapter.write_text("#!/usr/bin/env python3\n", encoding="ascii")
        shutil.copyfile("/bin/sleep", self.worker)
        self.worker.chmod(0o755)
        self.body.write_bytes(
            b'{"model":"deepseek-r1-distill-qwen-1.5b","prompt_token_ids":[1],'
            b'"max_tokens":1,"stream":true,"temperature":0}\n'
        )
        self.snapshot.write_bytes(b'{"captured":"independent-test-snapshot"}\n')
        self.provenance.write_bytes(b'{"source":"canonical-loom"}\n')

        worker_environment = clean_environment()
        worker_environment.update(
            {
                "DEEPSEEK_Q16K_AITER": "1",
                "DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC": str(self.diagnostic),
            }
        )
        if worker_profiler:
            worker_environment["ROCP_TOOL_LIBRARIES"] = "/tmp/forbidden.so"
        self.worker_process = subprocess.Popen(
            [str(self.worker), "300"], env=worker_environment
        )

        IdentityHandler.reset()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), IdentityHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.post_url = (
            f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        )
        self.health_url = f"http://127.0.0.1:{self.server.server_port}/health"
        self.worker_sha256 = sha256_file(self.worker)
        IdentityHandler.worker_pid = self.worker_process.pid
        IdentityHandler.worker_sha256 = self.worker_sha256
        identity_headers = {
            "X-Inference-Backend": "loom-raw-hsa",
            "X-Loom-Worker-SHA256": self.worker_sha256,
            "X-Loom-Runtime-HSACO-SHA256": "1" * 64,
            "X-Loom-Checkpoint-SHA256": "2" * 64,
            "X-Loom-Package-Root-SHA256": "3" * 64,
            "X-Loom-Package-Manifest-SHA256": "4" * 64,
            "X-Loom-Q16K-Asset-Manifest-SHA256": "5" * 64,
            "X-Loom-Q16K-Dense-SHA256": "6" * 64,
            "X-Loom-Q16K-Shadow-Pack-SHA256": "7" * 64,
            "X-Loom-Q16K-AITER-SHA256": "8" * 64,
        }
        IdentityHandler.health_headers = {
            "Content-Type": "application/json",
            **identity_headers,
        }
        IdentityHandler.post_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "close",
            **identity_headers,
        }
        backend_contract = {
            "name": "loom-raw-hsa",
            "worker_sha256": self.worker_sha256,
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
        deployment = {
            "schema": "loom.q16k.first_attention.deployment.v1",
            "status": "READY_FOR_AUTHORIZATION",
            "target": {"hostname": self.hostname},
            "reservation": {
                "id": self.reservation_id,
                "end_utc": self.reservation_end,
                "snapshot": {
                    "path": str(self.snapshot),
                    "sha256": sha256_file(self.snapshot),
                },
            },
            "endpoints": {
                "health_url": self.health_url,
                "post_url": self.post_url,
            },
            "llama_cpp": {
                "role": "cpu-only-http-utility",
                "url": "http://127.0.0.1:8082",
                "source": {
                    "path": (
                        "/home/dagarvey/codex-work/"
                        "llama.cpp-hrx-loom-reorg-9a1e95c-gJmqWw"
                    ),
                    "commit": "9a1e95c124ce47f0d1425f0d068d8eb85f671f26",
                    "clean": True,
                },
                "server": {
                    "path": (
                        "/home/dagarvey/codex-work/"
                        "llama.cpp-hrx-loom-reorg-9a1e95c-gJmqWw/"
                        "build-cpu/bin/llama-server"
                    ),
                    "sha256": (
                        "7c9795c29858a2a3aaee7d8ca5789137731a753e011f054cc"
                        "84c4af6d9006acd"
                    ),
                },
                "model": {
                    "path": (
                        "/home/dagarvey/codex-work/"
                        "llama.cpp-hrx-loom-reorg-9a1e95c-gJmqWw/models/"
                        "DeepSeek-R1-Distill-Qwen-1.5B-F16.gguf"
                    ),
                    "sha256": (
                        "0f1621d5a06bc1d80f3c43616f5baaf36c0e72a833c70190"
                        "8c9fbee08f6cde5d"
                    ),
                },
                "cpu_only_flags": [
                    "--device", "none", "--n-gpu-layers", "0",
                    "--no-kv-offload", "--no-op-offload", "--no-warmup",
                    "--no-webui", "--no-slots", "--ctx-size", "512",
                    "--parallel", "1", "--host", "127.0.0.1", "--port",
                    "8082",
                ],
            },
            "artifacts": {
                "adapter": {
                    "path": str(self.adapter),
                    "sha256": sha256_file(self.adapter),
                },
                "worker": {
                    "path": str(self.worker),
                    "sha256": self.worker_sha256,
                },
                "request": {
                    "path": str(self.body),
                    "sha256": sha256_file(self.body),
                },
            },
            "environment": {
                "required": {
                    "DEEPSEEK_Q16K_AITER": "1",
                    "DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC": str(
                        self.diagnostic
                    ),
                },
                "forbidden_names": [
                    "HSA_TOOLS_LIB",
                    "LD_PRELOAD",
                    "ROCP_TOOL_LIBRARIES",
                ],
                "forbidden_prefixes": ["ROCPROF", "ROCPROFILER"],
            },
            "outputs": {
                "diagnostic": str(self.diagnostic),
                "attempt": str(self.marker),
                "result": str(self.result),
                "response_body": str(self.response),
            },
            "health_contract": {
                "status": 200,
                "headers": dict(IdentityHandler.health_headers),
                "body": {
                    "status": "ok",
                    "inference_backend": "loom-raw-hsa",
                    "busy": False,
                    "backend": backend_contract,
                },
            },
            "post_contract": {
                "status": 200,
                "headers": dict(IdentityHandler.post_headers),
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
        self.manifest.write_text(
            json.dumps(deployment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.deployment = deployment
        self.package_root = seal_package(self.package)

    def write_deployment(self, deployment: dict[str, object]) -> None:
        self.manifest.write_text(
            json.dumps(deployment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.package_root = seal_package(self.package)

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.worker_process.terminate()
        self.worker_process.wait(timeout=10)

    def authorize_command(self) -> list[str]:
        return [
            sys.executable,
            str(AUTHORIZE),
            "--url", self.post_url,
            "--body", str(self.body),
            "--adapter", str(self.adapter),
            "--worker", str(self.worker),
            "--sealed-package", str(self.package),
            "--sealed-package-root-sha256", self.package_root,
            "--deployment-manifest", str(self.manifest),
            "--reservation-snapshot", str(self.snapshot),
            "--target-hostname", self.hostname,
            "--reservation-id", self.reservation_id,
            "--reservation-end-utc", self.reservation_end,
            "--diagnostic-output", str(self.diagnostic),
            "--attempt-marker", str(self.marker),
            "--result", str(self.result),
            "--response-body", str(self.response),
            "--output", str(self.authorization),
            "--authorization-id", "host-test-authorization",
        ]

    def post_command(self) -> list[str]:
        return [
            sys.executable,
            str(ONE_POST),
            "--url", self.post_url,
            "--body", str(self.body),
            "--adapter", str(self.adapter),
            "--worker", str(self.worker),
            "--sealed-package", str(self.package),
            "--sealed-package-root-sha256", self.package_root,
            "--deployment-manifest", str(self.manifest),
            "--reservation-snapshot", str(self.snapshot),
            "--target-hostname", self.hostname,
            "--reservation-id", self.reservation_id,
            "--reservation-end-utc", self.reservation_end,
            "--diagnostic-output", str(self.diagnostic),
            "--authorization-file", str(self.authorization),
            "--attempt-marker", str(self.marker),
            "--result", str(self.result),
            "--response-body", str(self.response),
            "--timeout", "10",
        ]

    def authorize(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.authorize_command(), env=clean_environment(), check=False,
            capture_output=True, text=True
        )

    def post(self, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.post_command(), env=clean_environment() if env is None else env,
            check=False, capture_output=True, text=True
        )


class FirstAttentionToolsTest(unittest.TestCase):
    def fixture(self, root: Path, **kwargs: object) -> RuntimeFixture:
        fixture = RuntimeFixture(root, **kwargs)
        self.addCleanup(fixture.close)
        return fixture

    def test_one_post_is_sealed_attested_and_marker_blocks_second(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            authorize = fixture.authorize()
            self.assertEqual(authorize.returncode, 0, authorize.stderr)
            first = fixture.post()
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(IdentityHandler.health_count, 1)
            self.assertEqual(IdentityHandler.post_count, 1)
            result = json.loads(fixture.result.read_text())
            self.assertEqual(result["health_get_count"], 1)
            self.assertEqual(result["post_count"], 1)
            self.assertEqual(result["body_sha256"], sha256_file(fixture.body))
            self.assertEqual(result["worker"]["pid"], fixture.worker_process.pid)
            self.assertEqual(result["worker"]["exe_sha256"], fixture.worker_sha256)
            fixture.result.unlink()
            fixture.response.unlink()
            second = fixture.post()
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(IdentityHandler.health_count, 1)
            self.assertEqual(IdentityHandler.post_count, 1)

    def test_tampered_sealed_member_blocks_all_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            fixture.provenance.write_text("tampered\n", encoding="ascii")
            result = fixture.post()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sealed package member hash mismatch", result.stderr)
            self.assertEqual(IdentityHandler.health_count, 0)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_llama_cpp_url_and_flags_mutations_block_authorization(self) -> None:
        mutations = (
            ("url", lambda value: value.__setitem__("url", "http://127.0.0.1:8083")),
            (
                "flags",
                lambda value: value["cpu_only_flags"].__setitem__(-1, "8083"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory))
                deployment = json.loads(json.dumps(fixture.deployment))
                mutate(deployment["llama_cpp"])
                fixture.write_deployment(deployment)
                result = fixture.authorize()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("llama.cpp CPU utility contract mismatch", result.stderr)
                self.assertEqual(IdentityHandler.health_count, 0)
                self.assertEqual(IdentityHandler.post_count, 0)

    def test_expired_authorization_blocks_all_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            authorization = json.loads(fixture.authorization.read_text())
            authorization["expires_unix_ns"] = 0
            fixture.authorization.write_text(json.dumps(authorization), encoding="utf-8")
            result = fixture.post()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("authorization has expired", result.stderr)
            self.assertEqual(IdentityHandler.health_count, 0)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_wrong_hostname_blocks_all_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            command = fixture.post_command()
            index = command.index("--target-hostname") + 1
            command[index] = "definitely-not-this-host"
            result = subprocess.run(
                command, env=clean_environment(), check=False,
                capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("actual socket hostname", result.stderr)
            self.assertEqual(IdentityHandler.health_count, 0)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_client_profiler_environment_blocks_all_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            environment = clean_environment()
            environment["HSA_TOOLS_LIB"] = "/tmp/forbidden.so"
            result = fixture.post(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("forbidden profiler/injection", result.stderr)
            self.assertEqual(IdentityHandler.health_count, 0)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_worker_profiler_environment_blocks_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), worker_profiler=True)
            self.assertEqual(fixture.authorize().returncode, 0)
            result = fixture.post()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("worker PID", result.stderr)
            self.assertIn("ROCP_TOOL_LIBRARIES", result.stderr)
            self.assertEqual(IdentityHandler.health_count, 1)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_wrong_health_worker_pid_blocks_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            IdentityHandler.worker_pid = os.getpid()
            result = fixture.post()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("executable is", result.stderr)
            self.assertEqual(IdentityHandler.health_count, 1)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_wrong_post_identity_header_is_recorded_after_single_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            IdentityHandler.post_headers["X-Loom-Worker-SHA256"] = "b" * 64
            result = fixture.post()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(IdentityHandler.health_count, 1)
            self.assertEqual(IdentityHandler.post_count, 1)
            result_data = json.loads(fixture.result.read_text())
            self.assertIn("POST response header", result_data["error"])
            self.assertTrue(fixture.marker.is_file())

    def test_preexisting_output_blocks_all_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            self.assertEqual(fixture.authorize().returncode, 0)
            fixture.result.write_text("do not replace\n", encoding="ascii")
            result = fixture.post()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(fixture.result.read_text(), "do not replace\n")
            self.assertEqual(IdentityHandler.health_count, 0)
            self.assertEqual(IdentityHandler.post_count, 0)

    def test_python_publication_never_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "exclusive.bin"
            AUTHORIZATION_MODULE.publish_bytes_exclusive(path, b"first")
            with self.assertRaises(FileExistsError):
                AUTHORIZATION_MODULE.publish_bytes_exclusive(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_analyzer_accepts_exact_contract_and_rejects_reserved_byte(self) -> None:
        raw = bytearray(424)
        pointers = {
            0: 0x1000, 8: 0x2000, 16: 0x3000, 24: 0x4000,
            72: 0x7000, 80: 0x8000, 88: 0x7008, 152: 0x7010,
        }
        for offset, value in pointers.items():
            struct.pack_into("<Q", raw, offset, value)
        for offset, value in {
            40: -1, 44: -1, 48: 128, 52: 128, 56: 12, 60: 6,
            64: 16768, 68: 1, 100: 1536, 104: 128, 108: 128,
            112: 1536, 116: 128, 120: 128, 124: 128, 128: 128,
            132: -1, 136: 0, 140: 0, 144: 2, 160: 256, 164: 256,
        }.items():
            struct.pack_into("<i", raw, offset, value)
        struct.pack_into("<I", raw, 96, 0x3E0293EE)
        for offset, value in ((168, 12), (172, 1), (176, 128)):
            struct.pack_into("<I", raw, offset, value)
        for offset, value in ((180, 256), (182, 1), (184, 1), (232, 3)):
            struct.pack_into("<H", raw, offset, value)
        softmax = struct.unpack_from("<f", raw, 96)[0]
        record = {
            "schema": "loom.q16k.first_attention.v1",
            "complete": True,
            "request": {"id": 1, "ordinal": 1, "finished": True,
                        "succeeded": False},
            "capture": {"position_base": 384, "query_count": 16384,
                        "layer": 0},
            "ordering": {"host_metadata": 1,
                         "gpu_metadata_copy_complete": 2,
                         "attention_packet_published": 3,
                         "request_finished": 4},
            "metadata": {"gpu_address": "0x0000000000007000",
                         "host_i32": [0, 16768, 1, 0, 0, 16384],
                         "gpu_i32": [0, 16768, 1, 0, 0, 16384],
                         "host_gpu_match": True},
            "kernel": {
                "name": ANALYZER_MODULE.ATTENTION_KERNEL_NAME,
                "object": "0x0000000000009000",
                "kernarg_segment_size": 424,
                "kernarg_segment_alignment": 16,
                "group_segment_size": 26112,
                "private_segment_size": 0,
            },
            "geometry": {"grid": [3072, 1, 128],
                         "workgroup": [256, 1, 1]},
            "packet": {
                "full_header": "0x00031502", "dimensions": 3,
                "grid": [3072, 1, 128], "workgroup": [256, 1, 1],
                "private_segment_size": 0, "group_segment_size": 26112,
                "kernel_object": "0x0000000000009000",
                "kernarg_address": "0x000000000000a000",
                "completion_signal": "0x0000000000000000",
                "packet_id": 2, "kernarg_slot": 2, "doorbell_written": 0,
            },
            "kernarg": {
                "size": 424, "source_matches_ring": True, "hex": raw.hex(),
                "decoded": {
                    "q": "0x0000000000001000",
                    "k": "0x0000000000002000",
                    "v": "0x0000000000003000",
                    "output": "0x0000000000004000",
                    "sequence_sentinels": [-1, -1],
                    "key_end": 16768, "page_size": 1,
                    "kv_indptr": "0x0000000000007000",
                    "kv_page_indices": "0x0000000000008000",
                    "kv_last_page_lens": "0x0000000000007008",
                    "softmax_scale_log2": softmax,
                    "window_left": -1, "window_right": 0, "mask_type": 2,
                    "cu_seqlens_q": "0x0000000000007010",
                    "hidden_block_counts": [12, 1, 128],
                    "group_sizes": [256, 1, 1], "hidden_dimensions": 3,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            accepted = subprocess.run(
                [sys.executable, str(ANALYZER), str(path)], check=False,
                capture_output=True, text=True
            )
            self.assertEqual(accepted.returncode, 0,
                             accepted.stdout + accepted.stderr)
            raw[188] = 1
            record["kernarg"]["hex"] = raw.hex()
            path.write_text(json.dumps(record), encoding="utf-8")
            rejected = subprocess.run(
                [sys.executable, str(ANALYZER), str(path)], check=False,
                capture_output=True, text=True
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("fixed kernarg ABI bytes mismatch", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
