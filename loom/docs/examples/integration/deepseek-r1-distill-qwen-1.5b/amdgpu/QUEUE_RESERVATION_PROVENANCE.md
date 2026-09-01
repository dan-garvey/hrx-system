# q16K queue reservation provenance

Generated: 2026-09-01T01:50:44Z

## Authoritative source

- Repository: `/home/dan/hrx-system-deepseek-r1-q16k-dense`
- Branch: `codex/deepseek-r1-q16k-dense`
- Base commit: `a71eba3ab0ce320c0476712802332de124c9b88b`
- Loom-owned worker: `loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu/deepseek_r1.c`

The original queue repair changed these pre-existing tracked files:

- `loom/binding/c/src/status.c`
- `loom/binding/c/test/status_test.cc`

The integration directory is the Loom-owned source set. The expanded canonical
seal is rooted at the repository root and contains 21 selected integration
source members plus the two tracked status files above. Its `SHA256SUMS` paths
are repository-relative, so verify it from the repository root with:

```text
sha256sum -c \
  loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu/SHA256SUMS
```

The integrated worker was imported byte-for-byte from the sealed lineage
snapshot at
`/home/dan/codex-work/deepseek-q16k-dense-rocr-v3-reflection-fix-20260831T045118Z/src/deepseek_r1_server.c`.
Its SHA-256 before this fix was
`a1268e037d23366d9695cb5aec9a203a4845f060e32a22216797cc2c4e0e2652`.
The current Loom-owned worker SHA-256 after the queue fix, first-attention
diagnostic integration, q16K system-fence correction, and full-model
synchronization diagnostic is
`b0231a6e2c8867789563de6c30f99361f86df36f576b8645fe562ec2b5362dbd`.

The historical worker at
`/home/dan/hrx-system-deepseek-r1/loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu/deepseek_r1.c`
has SHA-256
`caee9f8c972cabbfee7989ec175cd0b88f8f7f52f3fde69329b01c5d5fb04a27`.
It predates the q16K dense integration and does not contain
`dispatch_external_kernel`, so it is not the source base for this repair.

The companion files had these hashes at import:

- `deepseek_q16k_schedule.h`: `33d20c794e38c364630301c04af531590fd7585d9b0df58ff41e132a1077e57d`
- `q16k_aiter_integration.c`: `178abc5dfdc12353d0e0e0f97b8145c6b9d8bd751208fb9be6ab024c0c0164e3`
- `q16k_aiter_integration.h`: `3f6672dc4b35c3e13b4b809cbf314b5c9fb828800a1fdec93ffafa15e9993ddb`
- `q16k_dense_asm_contract.h`: `4c8d9914fe29031598d92f49a2cc970b50736b5a2525d87338c87ec34e238897`

The schedule and dense ABI headers retain those imported hashes. The current
canonical integration files have these hashes:

- `q16k_aiter_integration.c`: `e2212d8fe2a9110ec238c39e0a879f5b3cd0876b79c9af319f7a6d54ccf4fef3`
- `q16k_aiter_integration.h`: `5d18e4503a3d4ec4bf37728a9f5c1d2470a6e8283ee2425cc0c07424f1dee4a4`
- `BUILD.bazel`: `aa2c68c1b639ccaa4cdc7f8306db67dfde76d9de6b19e6c1224b4e587d2f8829`

## Failure and repair

The sealed diagnostic reported a deferred six-packet batch at
`write=63`, `read=57`, and effective queue limit `6`. The old internal and
external dispatch paths opened deferred state first and tested capacity for
only one packet, returning `RESOURCE_EXHAUSTED` instead of waiting for the full
six-slot reservation.

Both `dispatch_kernel_internal` and `dispatch_external_kernel` now use one
Loom-owned reservation helper. A new deferred batch polls the live queue read
index until every packet in the configured batch fits. An already-open batch
uses its cached reservation one packet at a time. Capacity arithmetic uses
unsigned `write - read`, checks `depth < limit` before `limit - depth`, rejects
a batch larger than the effective queue limit before mutating kernarg or
deferred state, and resets an open batch if its reserved capacity is
inconsistent.

## First-attention observer

The same canonical Loom integration adds an optional post-publication observer
for the first q16K attention dispatch. Normal serving installs a null observer
and passes no dispatch-result output, preserving the neutral path. Diagnostic
mode arms for one request, captures layer 0 host/GPU metadata, kernarg bytes,
launch geometry, and the published AQL packet, then disables the observer and
writes one record after the request. The output path must be absolute and must
not already exist.

The observer implementation and tools live only under
`diagnostics/first_attention/`. Their individual hashes are included in this
directory's `SHA256SUMS` seal.

## q16K AQL fence scope

Only `q16k_queue_publish_dispatch_packet` uses system-scope acquire and release
fences, producing the low dispatch header `0x1502` and the three-dimensional
`full_header` value `0x00031502`. The legacy `make_dispatch_header` helper and
all non-q16K dispatch paths retain agent-scope fences. The first-attention
diagnostic mock, analyzer, tests, and documentation enforce the q16K value.

## Full-model synchronization diagnostic

The opt-in full-model diagnostic serializes each q16K pack and attention stage,
captures the first layer's dispatch and allocation contracts, and snapshots
metadata immediately before and after the first attention launch. Its v4
record also retains the complete pre-attention output and reports host-side
byte, BF16 element, query-row, head, and query-tile differences after the
post-attention copy. Normal serving leaves the diagnostic unarmed.

The implementation and host-only coverage live under
`diagnostics/full_model_sync/`. All three source files are members of the
repository-root canonical seal.

## Status message ownership repair

`loomc_status_allocate` promises that message bytes are copied. The previous
implementation called the borrowing `iree_status_allocate` API, so callers
that released a temporary message could later format freed storage. The Loom C
binding now calls `iree_status_allocate_copy`, copying both the source-file and
message views into status-owned storage. `status_test.cc` constructs both views
on the heap, overwrites and frees them immediately after allocation, then
checks the retained source location and message.

## Host-only validation

The following commands passed from the repository root:

```text
bazel test --nocache_test_results --test_output=errors \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:deepseek_r1_host_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_first_attention_diagnostic_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_full_model_sync_diagnostic_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_first_attention_tools_test
bazel test --config=asan --nocache_test_results --test_output=errors \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:deepseek_r1_host_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_first_attention_diagnostic_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_first_attention_tools_test
bazel test --nocache_test_results --test_output=errors \
  //loom/binding/c/test:status_test
bazel test --config=asan --nocache_test_results --test_output=errors \
  //loom/binding/c/test:status_test
```

The host mock tests exercise:

- internal dispatch waiting from queue occupancy `6/6` until a six-slot batch
  reservation is available;
- external dispatch committing six packets with one write-index store to `6`
  and doorbell packet `5`;
- a two-packet forced flush followed by another complete six-packet batch;
- rejection of `doorbell_batch_size=6` with effective depth limit `5` without
  changing the kernarg cursor, deferred state, or queue;
- unsigned wrap arithmetic where `write=2` and `read=UINT64_MAX-2` has depth
  `5`, leaving room for one packet but not two under limit `6`.

A direct `--host-self-test` run under `strace` opened no `/dev/kfd`, `/dev/dri`,
HSA, HIP, DRM, or ROCm path. `readelf -d` also found no HSA, HIP, or DRM dynamic
dependency. No GPU or HSA execution was performed.

The final normal Bazel worker has SHA-256
`b813f84c85009a65f27538d133093374e0e87ac5f98554a8acfe3eb576314c3c`
and ELF build ID `d4315bb261cb7896df55f6182fb7769c268a335b`.

## Ownership boundary

This is Loom worker/runtime queue correctness and diagnostic work, not an
external compute-kernel optimization. No llama.cpp, SGLang, AITER checkout,
standalone ASM source, or Hyperloom source was modified. The sealed lineage
snapshots used by completed experiments remain unchanged; new deployment
packages are generated from these canonical Loom-owned bytes. Existing AITER
code objects and ASM artifacts remain read-only external dependencies.
