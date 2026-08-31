# q16384 First-Attention Launch-ABI Microgate

This package answers one narrow question: does the production q16384 AITER
first-attention code object produce byte-identical output when launched first by
the canonical raw-HSA path and then by HIP with only its 168-byte explicit
argument object?

A PASS clears only that launch-ABI question. It does not establish pack
semantics, dependency ordering, or correctness with live model tensors.

## Fixed Probe

- Case: `q16384-first-attention`
- Geometry: `position_base=384`, `query_count=16384`, `key_end=16768`
- Metadata i32: `[0,16768,1,0,0,16384]`
- Raw HSA: `3072x1x128` workitems, `256x1x1` workgroup, 26112 bytes LDS,
  424-byte kernarg, AQL header `0x00031502`
- HIP: `12x1x128` blocks, `256x1x1` workgroup, zero dynamic shared memory,
  168 explicit argument bytes
- Ordering: raw HSA first, then HIP; no warmup and no q512 code object
- Synchronization: exactly one `hsa_signal_wait_scacquire` and one
  `hipDeviceSynchronize`
- Queue: request 64-entry `SINGLE`; accept an ordinary returned `SINGLE` or
  `MULTI` queue only when its non-null ring has the exact HSA-clamped,
  power-of-two size within the agent bounds and supports kernel dispatch

Q, K, V, metadata, and page indices are identical and immutable across both
launches. The raw and HIP outputs are distinct guarded allocations. After raw
HSA, the future HIP output must still contain only its poison value. After HIP,
the raw output is copied and hashed again before the two current output images
are compared byte for byte.

The result records the agent queue minimum, maximum, and type; the requested
size and type; the queue-create status; and the returned queue pointer, ring
base, size, type, feature mask, doorbell handle, and ID. Queue validation
failures identify the rejected field and remain pre-dispatch failures.

## Construction

The canonical source tree intentionally contains only templates. Construct the
runnable package under `/home/dan/codex-work`; the staging script copies only
the listed source files, the canonical `q16k_aiter_integration.{c,h}`, the one
q16384 HSACO, and the build-only HSA toolchain inputs.

```bash
python3 -I scripts/stage_package.py \
  /home/dan/codex-work/DEPLOYMENT_ID \
  DEPLOYMENT_ID \
  --seal
```

Construction runs `make check` and `scripts/offline_review.py` without
initializing HSA or executing GPU work, then creates the package seal exactly
once. `PACKAGE_MANIFEST.json`, `BOUND_INPUTS.json`, and
`OFFLINE_REVIEW.json` in the staged directory contain the resulting hashes and
review record.

The archived HSA and rocprofiler libraries under `toolchain/` are build-only.
Host tests select them with `LD_LIBRARY_PATH`. The runner has a live RUNPATH of
`/opt/rocm/lib`, loads HIP by absolute path after the raw dispatch, and verifies
the fixed target HSA, HIP, and rocprofiler hashes before `hsa_init()`.

## Authorization And Run

The sealed package is not authorization. A reviewer must create a separate,
mode-private canonical JSON grant from `AUTHORIZATION_TEMPLATE.json`, replacing
the decision, authorization ID, package root digest, issue/expiry times, and no
other contract fields. Mutable authorization, admission, state, and evidence
paths remain outside the sealed package.

On the bound target, capture fresh read-only Conductor admission:

```bash
bash scripts/capture_admission.sh /absolute/private/admission-directory
```

Run the launcher once using the exact evidence and state paths in
`PACKAGE_MANIFEST.json`:

```bash
PROCEED_GPU=YES python3 -I scripts/run_once.py \
  /home/dagarvey/codex-work/DEPLOYMENT_ID-evidence \
  /absolute/private/admission-directory \
  /absolute/private/authorization.json \
  /home/dagarvey/codex-work/DEPLOYMENT_ID-state
```

The launcher validates the sealed package twice, reserves `RESULT.json`, and
creates a one-use receipt. The C runner atomically renames that receipt before
its sole explicit `hsa_init()`. There is one supervised runner process, no
retry, mandatory preflight/prelaunch/postflight host health capture, and a
sealed evidence directory on both success and failure.

Verify successful evidence offline with:

```bash
python3 -I scripts/verify_evidence.py \
  /home/dagarvey/codex-work/DEPLOYMENT_ID-evidence
```

The reservation bound by this package is
`06a94f67-275d-78f6-8000-3680fb507fd5`, active from
`2026-08-31T04:00:00Z` through `2026-08-31T20:00:00Z` on
`asrock-gbs6a-wb13c.png-odc.dcgpu`. An expired reservation requires a newly
staged and sealed package; do not edit a sealed manifest.
