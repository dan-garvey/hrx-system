# q16K First-Attention Diagnostic

This Loom-owned diagnostic captures the first live AITER attention dispatch of
one q16K `131071+1` serving request without rocprof or a PMC profiler.

The record contains:

- request ID and ordinal, `position_base`, `query_count`, and layer;
- all six host metadata `int32_t` values;
- the same 24 bytes copied back from GPU memory after the existing H-to-D copy;
- all 424 launched kernarg bytes from the CPU-visible kernarg ring;
- independently decoded critical kernarg fields;
- reflected kernel object and segment sizes;
- the final published 64-byte AQL packet fields, including header and geometry;
- event ordering from host metadata through request completion.

The dispatch observer only copies memory into the resident diagnostic record.
It performs no file I/O in the dispatch window. The record is written after the
request returns, including when inference fails. The requested GPU metadata
copy-back is the only additional synchronization before the attention launch.

## Build And Host Tests

```bash
bazel build \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:deepseek_r1

bazel test --test_output=errors \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:deepseek_r1_host_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_first_attention_diagnostic_test \
  //loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu:q16k_first_attention_tools_test
```

These tests are host-only. They do not initialize HSA or run a GPU kernel.

## Worker Contract

Launch the migrated worker in q16K serving mode with a new output path:

```bash
export DEEPSEEK_Q16K_AITER=1
export DEEPSEEK_Q16K_AITER_ASSET_ROOT=/absolute/path/to/q16k-assets
export DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC=/absolute/evidence/first_attention.json

bazel-bin/loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu/deepseek_r1 \
  --serve-stdio-long-context-131072-tiled-q512 \
  /absolute/path/to/kernels.loom \
  /absolute/path/to/prepared-model
```

Diagnostic mode refuses to overwrite an existing record. It arms on the first
generation request, writes one record, emits the normal completion or error,
and exits the stdio server. A second worker inference request cannot run.

## Authorized One-POST Client

Prepare the exact HTTP JSON body and choose paths that do not exist. Bind them,
the URL, sealed adapter and worker, and captured reservation snapshot into a
short-lived authorization. Every path passed to these tools must be absolute.

```bash
DIR=loom/docs/examples/integration/deepseek-r1-distill-qwen-1.5b/amdgpu/diagnostics/first_attention
EVIDENCE=/absolute/evidence/run-001
SEALED_PACKAGE=/absolute/deployment/final-sealed-package
WORKER="$SEALED_PACKAGE/bin/deepseek_r1_server"
ADAPTER="$SEALED_PACKAGE/adapter.py"
REQUEST="$SEALED_PACKAGE/inputs/request.json"
PACKAGE_ROOT=<64-lowercase-hex-root>
DEPLOYMENT_MANIFEST="$SEALED_PACKAGE/DEPLOYMENT.json"
RESERVATION_SNAPSHOT="$SEALED_PACKAGE/deployment/reservation.json"
TARGET_HOST=asrock-gbs6a-wb13c
RESERVATION_ID=<active-conductor-reservation-id>
RESERVATION_END_UTC=<exact-conductor-end-time>

python3 "$DIR/authorize_once.py" \
  --url http://127.0.0.1:8081/v1/chat/completions \
  --body "$REQUEST" \
  --adapter "$ADAPTER" \
  --worker "$WORKER" \
  --sealed-package "$SEALED_PACKAGE" \
  --sealed-package-root-sha256 "$PACKAGE_ROOT" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --reservation-snapshot "$RESERVATION_SNAPSHOT" \
  --target-hostname "$TARGET_HOST" \
  --reservation-id "$RESERVATION_ID" \
  --reservation-end-utc "$RESERVATION_END_UTC" \
  --diagnostic-output "$EVIDENCE/first_attention.json" \
  --attempt-marker "$EVIDENCE/attempt.json" \
  --result "$EVIDENCE/http_result.json" \
  --response-body "$EVIDENCE/http_response.txt" \
  --output "$EVIDENCE/authorization.json"
```

After the HTTP adapter reports ready, issue the request:

```bash
python3 "$DIR/one_post.py" \
  --url http://127.0.0.1:8081/v1/chat/completions \
  --body "$REQUEST" \
  --adapter "$ADAPTER" \
  --worker "$WORKER" \
  --sealed-package "$SEALED_PACKAGE" \
  --sealed-package-root-sha256 "$PACKAGE_ROOT" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --reservation-snapshot "$RESERVATION_SNAPSHOT" \
  --target-hostname "$TARGET_HOST" \
  --reservation-id "$RESERVATION_ID" \
  --reservation-end-utc "$RESERVATION_END_UTC" \
  --diagnostic-output "$EVIDENCE/first_attention.json" \
  --authorization-file "$EVIDENCE/authorization.json" \
  --attempt-marker "$EVIDENCE/attempt.json" \
  --result "$EVIDENCE/http_result.json" \
  --response-body "$EVIDENCE/http_response.txt"
```

Do not create this authorization until the final sealed package, deployed
worker, deployment manifest, target URL, and active reservation have been
independently verified. The authorization expires no later than the exact
reservation end time. Both authorization and execution re-hash the request,
adapter, worker, deployment manifest, reservation snapshot, and sealed package.
The package verifier requires `ROOT_SHA256` to be the SHA-256 of the exact
`SHA256SUMS` bytes and validates every sorted, relative, non-symlink member.
`DEPLOYMENT.json`, the request, adapter, worker, and reservation snapshot must
all be validated members of that seal.

The deployment manifest also binds the llama.cpp CPU utility independently of
the Loom worker: loopback URL `http://127.0.0.1:8082`, source commit
`9a1e95c124ce47f0d1425f0d068d8eb85f671f26`, exact server and GGUF paths and
SHA-256 digests, and the ordered argument vector after `--model`. The required
flags disable GPU selection, GPU layers, KV and operation offload, warmup,
Web UI, and slots; they pin context size `512`, parallelism `1`, host
`127.0.0.1`, and port `8082`. Authorization fails if any field, nested key, or
flag position differs.

The client first compares `socket.gethostname()` with the authorized node and
makes one `GET /health`. It checks the deployment-pinned response headers and
JSON identity, then uses the reported worker PID to verify `/proc/<pid>/exe`,
the running executable hash, the exact diagnostic environment, and the absence
of `ROCPROF*`, `ROCPROFILER*`, `ROCP_TOOL_LIBRARIES`, `HSA_TOOLS_LIB`, and
`LD_PRELOAD`. It then creates the exclusive attempt marker and makes one
`HTTPConnection.request("POST", ...)` call, with no retry or redirect path. The
POST response must repeat the deployment-pinned identity headers. The request
digest is computed from the exact in-memory bytes passed to `HTTPConnection`.

Authorization, attempt, response, result, and C diagnostic records use
same-directory temporary files plus hard-link publication. Publication fails
if the destination appeared concurrently; no result path is ever replaced.

Validate the capture:

```bash
python3 "$DIR/analyze_first_attention.py" \
  "$EVIDENCE/first_attention.json"
```

For the first suffix chunk, the default analyzer contract is:

- position `384`, query count `16384`, layer `0`;
- metadata `[0, 16768, 1, 0, 0, 16384]` on host and after GPU copy-back;
- 424 kernarg bytes with metadata pointers at GPU base offsets `0`, `8`, `16`;
- geometry/grid `[3072, 1, 128]`, workgroup `[256, 1, 1]`;
- AQL `full_header == 0x00031502`, dimensions `3`, and nonzero kernel object.

## Failure Localization

| Observation | Localized failure |
| --- | --- |
| Host metadata wrong | chunk planning or metadata construction |
| Host correct, GPU copy-back wrong | H-to-D copy, address, or visibility |
| Kernarg metadata pointers wrong | attention request or ABI construction |
| Source/ring kernargs differ | kernarg-ring copy or overwrite |
| Geometry correct, packet `grid_z=1` or dimensions `1` | packet construction/publication |
| Packet and kernarg correct, output wrong | attention kernel semantics or input data |
| Capture incomplete after a failed request | failure occurred before the missing event |

All source changes for this diagnostic live under the canonical Loom tree.
llama.cpp, SGLang, the AITER checkout, and standalone external kernel sources
remain read-only, hash-pinned runtime or comparison inputs.
