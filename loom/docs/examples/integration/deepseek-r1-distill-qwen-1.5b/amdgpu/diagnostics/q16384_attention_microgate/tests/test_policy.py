#!/usr/bin/env python3
"""Static policy checks for the sealed single-case launch-ABI probe."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_manifest  # noqa: E402


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "src/q16384_attention_microgate.c").read_text(
            encoding="utf-8"
        )
        cls.makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        cls.contract = json.loads((ROOT / "CONTRACT.json").read_text(encoding="ascii"))

    def test_single_raw_then_hip_launch_contract_is_literal(self) -> None:
        self.assertEqual(self.source.count("hsa_signal_wait_scacquire("), 1)
        self.assertEqual(self.source.count("api.device_synchronize()"), 1)
        self.assertEqual(self.source.count("q16k_aiter_dispatch_raw("), 1)
        self.assertEqual(self.source.count("api.module_launch_kernel("), 1)
        body = self.source[self.source.index("static bool execute_launch_parity"):]
        self.assertLess(body.index("dispatch_raw(resources"), body.index("dispatch_hip(resources"))
        self.assertIn('dlopen(HIP_RUNTIME_PATH, RTLD_NOW | RTLD_LOCAL)', self.source)
        self.assertIn("kHipExplicitArgumentBytes = 168", self.source)
        self.assertIn("function, 12u, 1u, 128u, 256u, 1u, 1u,", self.source)
        self.assertIn("0u, NULL, NULL, launch_config", self.source)

    def test_canonical_builder_dispatcher_and_first_attention_geometry_are_used(self) -> None:
        self.assertIn("q16k_build_attention_kernarg(", self.source)
        self.assertIn("q16k_aiter_dispatch_raw(", self.source)
        for literal in (
            "kPositionBase == 384",
            "kQueryCount == 16384",
            "kKeyEnd == 16768",
            "0x00031502",
            '\\"metadata_i32\\":[0,16768,1,0,0,16384]',
        ):
            self.assertIn(literal, self.source)
        self.assertNotIn("aiter_prefill_q512", self.source)

    def test_cross_output_isolation_is_part_of_the_pass_gate(self) -> None:
        for literal in (
            "raw HSA modified the reserved HIP output region",
            "HIP modified the completed raw-HSA output region",
            "hip_poison_values_after_raw == output_elements",
            "raw_output_unchanged_after_hip",
            "raw_hsa_after_hip",
        ):
            self.assertIn(literal, self.source)

    def test_queue_contract_accepts_only_valid_ordinary_queues(self) -> None:
        for literal in (
            "HSA_AGENT_INFO_QUEUE_MIN_SIZE",
            "HSA_AGENT_INFO_QUEUE_MAX_SIZE",
            "HSA_AGENT_INFO_QUEUE_TYPE",
            "HSA_QUEUE_TYPE_SINGLE",
            "HSA_QUEUE_TYPE_MULTI",
            "HSA_QUEUE_TYPE_COOPERATIVE",
            "size_matches_expected",
            "kernel_dispatch_supported",
            "HSA returned a null queue base address",
        ):
            self.assertIn(literal, self.source)
        queue = self.contract["runtime"]["queue"]
        self.assertEqual(queue["requested_size"], 64)
        self.assertEqual(queue["requested_type"], "SINGLE")
        self.assertEqual(queue["returned_types_accepted"], ["SINGLE", "MULTI"])
        self.assertTrue(queue["returned_base_address_required"])
        self.assertTrue(queue["kernel_dispatch_feature_required"])

    def test_hsa_initialization_is_behind_receipt_validation_and_claim(self) -> None:
        run_live = self.source.index("static int run_live")
        verify = self.source.index("verify_consumption_receipt(", run_live)
        claim = self.source.index("claim_consumption_receipt(", verify)
        initialize = self.source.index("initialize_runtime(", claim)
        self.assertLess(verify, claim)
        self.assertLess(claim, initialize)
        self.assertEqual(len(re.findall(r"\bhsa_init\s*\(\s*\)", self.source)), 1)

    def test_manifest_template_is_single_case_and_has_no_q512_artifact(self) -> None:
        manifest = package_manifest.load_manifest(
            ROOT, require_artifact_hashes=False, allow_template=True
        )
        self.assertEqual(manifest["execution"]["authorized_cases"],
                         ["q16384-first-attention"])
        self.assertEqual(manifest["execution"]["maximum_gpu_invocations"], 1)
        self.assertEqual(manifest["execution"]["maximum_retries"], 0)
        self.assertFalse(any(
            "q512" in item["path"].lower()
            for item in manifest["artifacts"].values()
        ))

    def test_readme_uses_the_staged_deployment_paths(self) -> None:
        manifest = package_manifest.load_manifest(
            ROOT, require_artifact_hashes=False, allow_template=True
        )
        readme = (ROOT / "README.md").read_text(encoding="ascii")
        if manifest["status"] == package_manifest.TEMPLATE_STATUS:
            self.assertIn("DEPLOYMENT_ID", readme)
        else:
            self.assertNotIn("DEPLOYMENT_ID", readme)
            self.assertIn(manifest["execution"]["evidence_path"], readme)
            self.assertIn(manifest["execution"]["state_path"], readme)

    def test_build_uses_live_rocm_runpath_and_dlopen_for_hip(self) -> None:
        self.assertIn("-rpath,/opt/rocm/lib", self.makefile)
        self.assertIn("-ldl", self.makefile)
        self.assertNotIn("-lamdhip64", self.makefile)
        self.assertIn("LD_LIBRARY_PATH=$(CURDIR)/toolchain/lib", self.makefile)

    def test_contract_and_authorization_template_do_not_overclaim(self) -> None:
        self.assertEqual(self.contract["schema"],
                         "loom-q16384-first-attention-launch-abi-contract-v1")
        self.assertEqual(self.contract["scope"]["unresolved"], [
            "pack-semantics", "dependency-ordering", "live-tensor-semantics"
        ])
        self.assertTrue(self.contract["acceptance"][
            "hip_output_remains_poisoned_after_raw_hsa"
        ])
        self.assertTrue(self.contract["acceptance"][
            "raw_hsa_output_unchanged_after_hip"
        ])
        raw = (ROOT / "AUTHORIZATION_TEMPLATE.json").read_bytes()
        template = json.loads(raw)
        canonical = (
            json.dumps(template, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        self.assertEqual(raw, canonical)
        self.assertEqual(template["authorized_cases"], ["q16384-first-attention"])
        self.assertEqual(template["authorized_case_count"], 1)
        self.assertNotEqual(template["decision"], "AUTHORIZE_ONE_GPU_INVOCATION")
        self.assertFalse(template["pack_and_attention_contract_satisfied"])

    def test_admission_capture_is_read_only_and_manifest_bound(self) -> None:
        capture = (ROOT / "scripts/capture_admission.sh").read_text(encoding="utf-8")
        self.assertIn("package_manifest.py", capture)
        self.assertIn("--state IN_PROGRESS", capture)
        self.assertIn("list jobs-v2", capture)
        self.assertNotRegex(capture, r"\b(?:create|edit|delete)\s+reservation\b")


if __name__ == "__main__":
    unittest.main()
