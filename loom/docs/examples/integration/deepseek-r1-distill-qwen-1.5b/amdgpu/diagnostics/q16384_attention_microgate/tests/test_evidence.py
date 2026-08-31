#!/usr/bin/env python3
"""Host-only semantic tests for the launch-ABI result payload."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import verify_evidence as verifier  # noqa: E402
from support import write_manifest  # noqa: E402


class EvidenceResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q16384_evidence_")
        self.root = Path(self.temporary.name).resolve()
        self.evidence = self.root / "evidence"
        self.evidence.mkdir(mode=0o700)
        self.package = self.root / "package"
        self.manifest = write_manifest(
            self.package, self.evidence, self.root / "state"
        )
        self.authorization = {
            "authorization_id": "q16384-review-20260831-001",
            "root_digest": "7" * 64,
        }
        self.result = self._result_payload()
        self._write_result()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _observation(self) -> dict[str, object]:
        return {
            "guard_mismatches": 0,
            "max_absolute_error": 0.01,
            "max_tolerance_ratio": 0.25,
            "nonfinite_values": 0,
            "poison_values_remaining": 0,
            "reference_violations": 0,
            "sampled_elements_compared": 8 * 12 * 128,
            "sha256": "a" * 64,
        }

    def _result_payload(self) -> dict[str, object]:
        observation = self._observation()
        hashes = {
            "k": "1" * 64,
            "metadata": "2" * 64,
            "page_indices": "3" * 64,
            "q": "4" * 64,
            "v": "5" * 64,
        }
        runtime = self.manifest["runtime"]
        return {
            "authorization_id": self.authorization["authorization_id"],
            "comparison": {
                "byte_equal": True,
                "bytes_compared": 16384 * 12 * 128 * 2,
                "first_mismatch_byte": None,
                "first_mismatch_dimension": None,
                "first_mismatch_element": None,
                "first_mismatch_head": None,
                "first_mismatch_row": None,
                "mismatch_bytes": 0,
            },
            "dispatch": {
                "hip_completed_ns": 400,
                "hip_device_synchronizes": 1,
                "hip_started_ns": 300,
                "q512_code_object_loaded": False,
                "q512_dispatches": 0,
                "raw_completed_ns": 200,
                "raw_completion_waits": 1,
                "raw_doorbell_written": 1,
                "raw_full_header": "0x00031502",
                "raw_hsa_first": True,
                "raw_kernarg_slot": 0,
                "raw_packet_id": 0,
                "raw_started_ns": 100,
                "warmup_dispatches": 0,
            },
            "geometry": {
                "group_segment_bytes": 26112,
                "head_dimension": 128,
                "hip_explicit_argument_bytes": 168,
                "hip_grid_blocks": [12, 1, 128],
                "kernarg_segment_bytes": 424,
                "key_end": 16768,
                "kv_heads": 2,
                "metadata_i32": [0, 16768, 1, 0, 0, 16384],
                "position_base": 384,
                "private_segment_bytes": 0,
                "query_count": 16384,
                "query_heads": 12,
                "raw_grid_workitems": [3072, 1, 128],
                "workgroup": [256, 1, 1],
            },
            "gpu_execution_performed": True,
            "inputs": {
                "guard_mismatches_after_hip": 0,
                "guard_mismatches_after_raw": 0,
                "post_hip": copy.deepcopy(hashes),
                "post_raw": copy.deepcopy(hashes),
                "pre_raw": copy.deepcopy(hashes),
                "unchanged_after_hip": True,
                "unchanged_after_raw": True,
            },
            "microgate_root_sha256": self.authorization["root_digest"],
            "outputs": {
                "hip": copy.deepcopy(observation),
                "hip_before_launch": {
                    "expected_poison_values": 16384 * 12 * 128,
                    "guard_mismatches": 0,
                    "poison_values_present": 16384 * 12 * 128,
                },
                "raw_hsa": copy.deepcopy(observation),
                "raw_hsa_after_hip": copy.deepcopy(observation),
                "raw_hsa_unchanged_after_hip": True,
            },
            "queue_contract": {
                "agent": {
                    "max_query_completed": True,
                    "max_query_status": 0,
                    "max_size": 131072,
                    "min_query_completed": True,
                    "min_query_status": 0,
                    "min_size": 64,
                    "type": 0,
                    "type_query_completed": True,
                    "type_query_status": 0,
                },
                "contract_satisfied": True,
                "create": {"attempted": True, "status": 0},
                "expected_size": 64,
                "predicates": {
                    "base_nonnull": True,
                    "contract_satisfied": True,
                    "kernel_dispatch_supported": True,
                    "pointer_nonnull": True,
                    "size_matches_expected": True,
                    "size_power_of_two": True,
                    "size_within_agent_bounds": True,
                    "type_ordinary": True,
                },
                "request": {"size": 64, "type": 1},
                "returned": {
                    "base_address": "0x0000000000002000",
                    "doorbell_signal_handle": "0x0000000000003000",
                    "features": 1,
                    "id": 7,
                    "pointer": "0x0000000000001000",
                    "size": 64,
                    "type": 0,
                },
            },
            "runtime": {
                "cpu_agent": "AMD CPU",
                "gpu_agent": "AMD Instinct MI355X",
                "hip_runtime_sha256": runtime["hip_runtime"]["sha256"],
                "hsa_runtime_sha256": runtime["hsa_runtime"]["sha256"],
                "isa": "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-",
                "rocprofiler_register_sha256": runtime["rocprofiler_register"]["sha256"],
                "visible_gpu_count": 1,
            },
            "schema": verifier.RESULT_SCHEMA,
            "scope": {
                "attention_only": True,
                "case": "q16384-first-attention",
                "dependency_ordering_resolved": False,
                "launch_abi_cleared": True,
                "live_tensor_semantics_resolved": False,
                "pack_semantics_resolved": False,
            },
            "status": "pass",
        }

    def _write_result(self) -> None:
        (self.evidence / "RESULT.json").write_text(
            json.dumps(self.result, indent=2) + "\n", encoding="ascii"
        )

    def _validate(self) -> dict[str, object]:
        return verifier._validate_result(
            self.evidence, self.authorization, self.manifest
        )

    def test_valid_result_and_success_file_contract(self) -> None:
        self.assertEqual(self._validate()["status"], "pass")
        self.assertEqual(len(verifier.EXPECTED_SUCCESS_FILES), 52)

    def test_future_hip_output_must_remain_poisoned_after_raw(self) -> None:
        before = self.result["outputs"]["hip_before_launch"]
        before["poison_values_present"] -= 1
        self._write_result()
        with self.assertRaisesRegex(RuntimeError, "reserved HIP output"):
            self._validate()

    def test_raw_output_must_be_unchanged_after_hip(self) -> None:
        self.result["outputs"]["raw_hsa_after_hip"]["sha256"] = "b" * 64
        self._write_result()
        with self.assertRaisesRegex(RuntimeError, "changed after HIP"):
            self._validate()

    def test_launch_counts_geometry_and_scope_are_strict(self) -> None:
        mutations = (
            (self.result["dispatch"], "raw_completion_waits", 2),
            (self.result["dispatch"], "hip_device_synchronizes", 2),
            (self.result["geometry"], "hip_explicit_argument_bytes", 424),
            (self.result["scope"], "pack_semantics_resolved", True),
        )
        for mapping, key, value in mutations:
            with self.subTest(key=key):
                original = mapping[key]
                mapping[key] = value
                self._write_result()
                with self.assertRaises(RuntimeError):
                    self._validate()
                mapping[key] = original

    def test_queue_contract_accepts_multi_and_rejects_bad_fields(self) -> None:
        self.assertEqual(self.result["queue_contract"]["returned"]["type"], 0)
        self.assertEqual(self._validate()["status"], "pass")
        mutations = (
            ("size", 128, "queue size"),
            ("features", 0, "kernel dispatch"),
            ("type", 2, "queue type"),
        )
        returned = self.result["queue_contract"]["returned"]
        for key, value, message in mutations:
            with self.subTest(key=key):
                original = returned[key]
                returned[key] = value
                self._write_result()
                with self.assertRaisesRegex(RuntimeError, message):
                    self._validate()
                returned[key] = original

    def test_input_mutation_and_byte_mismatch_are_rejected(self) -> None:
        self.result["inputs"]["post_hip"]["q"] = "b" * 64
        self._write_result()
        with self.assertRaisesRegex(RuntimeError, "immutable input"):
            self._validate()
        self.result = self._result_payload()
        self.result["comparison"]["byte_equal"] = False
        self._write_result()
        with self.assertRaisesRegex(RuntimeError, "comparison changed"):
            self._validate()

    def test_type_equivalent_contract_mutations_are_rejected(self) -> None:
        mutations = (
            (self.result["geometry"], "private_segment_bytes", False,
             "geometry changed"),
            (self.result["inputs"], "guard_mismatches_after_raw", False,
             "input guard mismatch"),
            (self.result["outputs"]["hip_before_launch"], "guard_mismatches", False,
             "reserved HIP output"),
            (self.result["outputs"]["raw_hsa"], "sampled_elements_compared",
             float(8 * 12 * 128), "sampled element count"),
            (self.result["comparison"], "byte_equal", 1, "comparison changed"),
            (self.result["comparison"], "mismatch_bytes", False,
             "comparison changed"),
        )
        for mapping, key, value, message in mutations:
            with self.subTest(key=key, value=value):
                original = mapping[key]
                mapping[key] = value
                self._write_result()
                with self.assertRaisesRegex(RuntimeError, message):
                    self._validate()
                mapping[key] = original

    def test_pass_record_rejects_bool_integer_aliases(self) -> None:
        payload = {
            "authorization_id": self.authorization["authorization_id"],
            "completed_epoch": 300,
            "invocation_count": 1,
            "microgate_root_sha256": self.authorization["root_digest"],
            "postflight_pass": True,
            "result_sha256": hashlib.sha256(
                (self.evidence / "RESULT.json").read_bytes()
            ).hexdigest(),
            "runner_returncode": 0,
            "schema": "loom-q16384-first-attention-launch-abi-evidence-pass-v1",
            "status": "pass",
        }
        for field, value in (("invocation_count", True),
                             ("runner_returncode", False)):
            with self.subTest(field=field):
                mutated = copy.deepcopy(payload)
                mutated[field] = value
                with self.assertRaisesRegex(RuntimeError, "PASS invocation result"):
                    verifier._validate_pass_record(
                        mutated,
                        self.evidence,
                        self.authorization,
                        {"runner_finished_epoch": 200},
                        {"validated_epoch": 250},
                    )


if __name__ == "__main__":
    unittest.main()
