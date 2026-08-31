#ifndef DEEPSEEK_Q16K_SCHEDULE_H_
#define DEEPSEEK_Q16K_SCHEDULE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPSEEK_Q16K_LEGACY_PREFIX_ROWS 384u
#define DEEPSEEK_Q16K_PROMPT_ROWS 131071u
#define DEEPSEEK_Q16K_SUFFIX_ROWS 130687u
#define DEEPSEEK_Q16K_CHUNK_CAPACITY 16384u
#define DEEPSEEK_Q16K_CHUNK_COUNT 8u
#define DEEPSEEK_Q16K_TAIL_ROWS 15999u
#define DEEPSEEK_Q16K_LAYER_COUNT 28u
#define DEEPSEEK_Q16K_OPERATIONS_PER_LAYER 8u
#define DEEPSEEK_Q16K_DENSE_OPERATIONS_PER_LAYER 4u
#define DEEPSEEK_Q16K_FULL_MICROBENCHMARK_OPERATIONS_PER_LAYER 7u
#define DEEPSEEK_Q16K_DENSE_REFERENCE_DISPATCHES UINT64_C(896)
#define DEEPSEEK_Q16K_FULL_MICROBENCHMARK_DISPATCHES UINT64_C(1576)
#define DEEPSEEK_Q16K_COMPLETE_OPERATION_CALLBACKS 1805u
#define DEEPSEEK_Q16K_QUEUE_PACKET_CAPACITY 65536u
#define DEEPSEEK_Q16K_QUEUE_DEPTH_LIMIT 6u
#define DEEPSEEK_Q16K_DOORBELL_BATCH 6u
#define DEEPSEEK_Q16K_KERNARG_CAPACITY 65536u
#define DEEPSEEK_Q16K_KERNARG_STRIDE ((size_t)256u)
#define DEEPSEEK_Q16K_KERNARG_ARENA_BYTES ((size_t)16777216u)
#define DEEPSEEK_Q16K_AITER_KERNARG_STRIDE ((size_t)512u)
#define DEEPSEEK_Q16K_AITER_KERNARG_ARENA_BYTES ((size_t)33554432u)
#define DEEPSEEK_Q16K_PROTOCOL_TOKEN_CHUNK_ROWS 256u
#define DEEPSEEK_Q16K_PROTOCOL_TOKEN_CHUNKS 512u
#define DEEPSEEK_Q16K_DENSE_GEMM_TILE_ROWS 256u
#define DEEPSEEK_Q16K_DENSE_GEMM_TILE_COLUMNS 256u
#define DEEPSEEK_Q16K_DENSE_ASM_PACKETS_PER_HOOK 2u
#define DEEPSEEK_Q16K_DENSE_ASM_PACKET_DISPATCHES \
  (DEEPSEEK_Q16K_DENSE_REFERENCE_DISPATCHES *                      \
   DEEPSEEK_Q16K_DENSE_ASM_PACKETS_PER_HOOK)
#define DEEPSEEK_Q16K_NO_CHUNK UINT32_MAX
#define DEEPSEEK_Q16K_NO_LAYER UINT32_MAX
#define DEEPSEEK_Q16K_PRODUCTION_REFERENCE_TOKEN 382
#define DEEPSEEK_Q16K_PRODUCTION_REFERENCE_DISPATCHES UINT64_C(65362)
#define DEEPSEEK_Q16K_PRODUCTION_REFERENCE_HOST_WAITS UINT64_C(778)

typedef struct deepseek_q16k_chunk_t {
  uint32_t position_base;
  uint32_t row_count;
  bool final;
} deepseek_q16k_chunk_t;

typedef enum deepseek_q16k_operation_e {
  DEEPSEEK_Q16K_OPERATION_LEGACY_PREFIX = 0,
  DEEPSEEK_Q16K_OPERATION_MIGRATE_LEGACY_KV,
  DEEPSEEK_Q16K_OPERATION_EMBEDDING,
  DEEPSEEK_Q16K_OPERATION_INPUT_NORM,
  DEEPSEEK_Q16K_OPERATION_QKV_PROJECTION,
  DEEPSEEK_Q16K_OPERATION_ROPE_CACHE_WRITE,
  DEEPSEEK_Q16K_OPERATION_ATTENTION,
  DEEPSEEK_Q16K_OPERATION_OUTPUT_PROJECTION,
  DEEPSEEK_Q16K_OPERATION_POST_ATTENTION_NORM,
  DEEPSEEK_Q16K_OPERATION_GATE_UP,
  DEEPSEEK_Q16K_OPERATION_DOWN_PROJECTION,
  DEEPSEEK_Q16K_OPERATION_FINAL_ROW_LOGITS,
  DEEPSEEK_Q16K_OPERATION_GREEDY_ARGMAX,
  DEEPSEEK_Q16K_OPERATION_TOKEN_READBACK,
  DEEPSEEK_Q16K_OPERATION_COUNT,
} deepseek_q16k_operation_t;

typedef enum deepseek_q16k_dense_gemm_kind_e {
  DEEPSEEK_Q16K_DENSE_GEMM_QKV_PROJECTION = 0,
  DEEPSEEK_Q16K_DENSE_GEMM_OUTPUT_PROJECTION,
  DEEPSEEK_Q16K_DENSE_GEMM_GATE_UP,
  DEEPSEEK_Q16K_DENSE_GEMM_DOWN_PROJECTION,
} deepseek_q16k_dense_gemm_kind_t;

typedef enum deepseek_q16k_status_code_e {
  DEEPSEEK_Q16K_STATUS_OK = 0,
  DEEPSEEK_Q16K_STATUS_INVALID_ARGUMENT,
  DEEPSEEK_Q16K_STATUS_INVALID_REQUEST,
  DEEPSEEK_Q16K_STATUS_INVALID_STATIC_CONTRACT,
  DEEPSEEK_Q16K_STATUS_INVALID_QUEUE_CONTRACT,
  DEEPSEEK_Q16K_STATUS_MISSING_CALLBACK,
  DEEPSEEK_Q16K_STATUS_CALLBACK_FAILED,
  DEEPSEEK_Q16K_STATUS_OBSERVATION_MISMATCH,
  DEEPSEEK_Q16K_STATUS_DISPATCH_OVERFLOW,
  DEEPSEEK_Q16K_STATUS_KERNARG_OVERFLOW,
  DEEPSEEK_Q16K_STATUS_HOST_WAIT_OVERFLOW,
  DEEPSEEK_Q16K_STATUS_DOORBELL_OVERFLOW,
} deepseek_q16k_status_code_t;

typedef struct deepseek_q16k_status_t {
  deepseek_q16k_status_code_t code;
  const char* message;
} deepseek_q16k_status_t;

typedef struct deepseek_q16k_device_pointers_t {
  void* checkpoint;
  void* params;
  void* tokens;
  void* rope_table;
  void* key_cache;
  void* value_cache;
  void* shadow_key_cache;
  void* shadow_value_cache;
  void* page_indices;
  void* attention_metadata;
  void* hidden;
  void* normalized;
  void* raw_query;
  void* raw_key;
  void* raw_value;
  void* rotated_query;
  void* attention;
  void* projection;
  void* post_attention;
  void* gate;
  void* up;
  void* activated;
  void* logits;
  void* result;
  size_t key_cache_layer_stride;
  size_t value_cache_layer_stride;
} deepseek_q16k_device_pointers_t;

typedef struct deepseek_q16k_queue_contract_t {
  void* queue;
  void* kernarg_arena;
  uint32_t queue_packet_capacity;
  uint32_t queue_depth_limit;
  uint32_t doorbell_batch;
  uint32_t kernarg_capacity;
  size_t kernarg_stride;
  size_t kernarg_arena_bytes;
  uint32_t initial_kernarg_cursor;
  bool batched_dispatch;
} deepseek_q16k_queue_contract_t;

typedef struct deepseek_q16k_features_t {
  bool queue_native_attention;
  bool dense_gemm_256x256;
} deepseek_q16k_features_t;

typedef struct deepseek_q16k_invocation_t {
  deepseek_q16k_operation_t operation;
  uint32_t chunk_index;
  uint32_t layer;
  uint32_t position_base;
  uint32_t query_count;
  bool final_chunk;
  bool final_layer;
  uint32_t kernarg_cursor;
  uint32_t dense_tile_rows;
  uint32_t dense_tile_columns;
  const deepseek_q16k_queue_contract_t* queue;
  const deepseek_q16k_device_pointers_t* device;
} deepseek_q16k_invocation_t;

typedef struct deepseek_q16k_callback_counts_t {
  uint64_t dispatch_count;
  uint64_t kernarg_count;
  uint64_t host_wait_count;
} deepseek_q16k_callback_counts_t;

typedef struct deepseek_q16k_hook_counts_t {
  uint64_t dispatch_count;
  uint64_t kernarg_count;
} deepseek_q16k_hook_counts_t;

typedef struct deepseek_q16k_oracle_observation_t {
  bool checked;
  bool matched;
  int32_t expected_token;
} deepseek_q16k_oracle_observation_t;

typedef struct deepseek_q16k_worker_counters_t {
  uint64_t launches;
  uint64_t restarts;
  uint64_t gpu_invocations;
} deepseek_q16k_worker_counters_t;

typedef struct deepseek_q16k_http_counters_t {
  uint64_t full_context_requests;
  uint64_t apply_template_successes;
  uint64_t tokenize_successes;
  uint64_t detokenize_successes;
  uint64_t protocol_token_chunks;
} deepseek_q16k_http_counters_t;

typedef struct deepseek_q16k_serving_observation_t {
  int32_t final_token;
  uint32_t generated_token_count;
  deepseek_q16k_oracle_observation_t oracle;
  deepseek_q16k_worker_counters_t worker;
  deepseek_q16k_http_counters_t http;
} deepseek_q16k_serving_observation_t;

typedef struct deepseek_q16k_schedule_counters_t {
  uint32_t suffix_chunks;
  uint32_t suffix_rows;
  uint32_t layer_invocations;
  uint32_t operation_callback_calls;
  uint32_t attention_hook_calls;
  uint32_t dense_gemm_hook_calls;
  uint64_t dispatch_count;
  uint64_t kernarg_count;
  uint64_t host_wait_count;
  uint64_t doorbell_write_count;
  uint32_t peak_kernarg_cursor;
  uint32_t final_kernarg_cursor;
  uint32_t maximum_dispatch_batch;
} deepseek_q16k_schedule_counters_t;

typedef struct deepseek_q16k_serving_result_t {
  deepseek_q16k_serving_observation_t observation;
  deepseek_q16k_schedule_counters_t schedule;
} deepseek_q16k_serving_result_t;

typedef struct deepseek_q16k_production_reference_t {
  const char* baseline_worker_source_sha256;
  const char* baseline_worker_binary_sha256;
  const char* q512_loom_source_sha256;
  const char* q16k_benchmark_source_sha256;
  const char* q16k_loom_source_sha256;
  const char* q16k_hsaco_sha256;
  double wall_seconds;
  int32_t final_token;
  uint64_t dispatch_count;
  uint64_t host_wait_count;
  deepseek_q16k_worker_counters_t worker;
  deepseek_q16k_http_counters_t http;
} deepseek_q16k_production_reference_t;

typedef bool (*deepseek_q16k_operation_callback_t)(
    void* user_data, const deepseek_q16k_invocation_t* invocation,
    deepseek_q16k_callback_counts_t* counts, deepseek_q16k_status_t* status);
typedef bool (*deepseek_q16k_attention_hook_t)(
    void* user_data, const deepseek_q16k_invocation_t* invocation,
    deepseek_q16k_hook_counts_t* counts, deepseek_q16k_status_t* status);
typedef bool (*deepseek_q16k_dense_gemm_hook_t)(
    void* user_data, deepseek_q16k_dense_gemm_kind_t kind,
    const deepseek_q16k_invocation_t* invocation,
    deepseek_q16k_hook_counts_t* counts, deepseek_q16k_status_t* status);
typedef bool (*deepseek_q16k_observation_callback_t)(
    void* user_data, deepseek_q16k_serving_observation_t* observation,
    deepseek_q16k_status_t* status);

typedef struct deepseek_q16k_callbacks_t {
  void* user_data;
  deepseek_q16k_operation_callback_t operation;
  deepseek_q16k_attention_hook_t attention;
  deepseek_q16k_dense_gemm_hook_t dense_gemm_256x256;
  deepseek_q16k_observation_callback_t observe;
} deepseek_q16k_callbacks_t;

typedef struct deepseek_q16k_request_t {
  uint32_t prompt_token_count;
  uint32_t generated_token_count;
} deepseek_q16k_request_t;

void deepseek_q16k_status_reset(deepseek_q16k_status_t* status);
const char* deepseek_q16k_operation_name(deepseek_q16k_operation_t operation);
const deepseek_q16k_chunk_t* deepseek_q16k_chunk_plan(size_t* count);
const deepseek_q16k_production_reference_t*
deepseek_q16k_production_reference(void);
bool deepseek_q16k_validate_static_contract(deepseek_q16k_status_t* status);
bool deepseek_q16k_validate_queue_contract(
    const deepseek_q16k_queue_contract_t* queue,
    deepseek_q16k_status_t* status);
bool deepseek_q16k_run_schedule(
    const deepseek_q16k_request_t* request,
    const deepseek_q16k_queue_contract_t* queue,
    const deepseek_q16k_device_pointers_t* device,
    const deepseek_q16k_features_t* features,
    const deepseek_q16k_callbacks_t* callbacks,
    deepseek_q16k_serving_result_t* result,
    deepseek_q16k_status_t* status);

#ifdef DEEPSEEK_Q16K_SCHEDULE_IMPLEMENTATION

#include <string.h>

_Static_assert(DEEPSEEK_Q16K_LEGACY_PREFIX_ROWS +
                       DEEPSEEK_Q16K_SUFFIX_ROWS ==
                   DEEPSEEK_Q16K_PROMPT_ROWS,
               "q16K prompt partition changed");
_Static_assert(DEEPSEEK_Q16K_SUFFIX_ROWS ==
                   7u * DEEPSEEK_Q16K_CHUNK_CAPACITY +
                       DEEPSEEK_Q16K_TAIL_ROWS,
               "q16K suffix shape changed");
_Static_assert(DEEPSEEK_Q16K_KERNARG_CAPACITY *
                       DEEPSEEK_Q16K_KERNARG_STRIDE ==
                   DEEPSEEK_Q16K_KERNARG_ARENA_BYTES,
               "q16K fallback kernarg arena changed");
_Static_assert(DEEPSEEK_Q16K_KERNARG_CAPACITY *
                       DEEPSEEK_Q16K_AITER_KERNARG_STRIDE ==
                   DEEPSEEK_Q16K_AITER_KERNARG_ARENA_BYTES,
               "q16K AITER kernarg arena changed");
_Static_assert(DEEPSEEK_Q16K_QUEUE_PACKET_CAPACITY ==
                   DEEPSEEK_Q16K_KERNARG_CAPACITY,
               "q16K queue and kernarg capacities changed");
_Static_assert(DEEPSEEK_Q16K_QUEUE_DEPTH_LIMIT ==
                   DEEPSEEK_Q16K_DOORBELL_BATCH,
               "q16K queue batching changed");
_Static_assert(DEEPSEEK_Q16K_CHUNK_COUNT * DEEPSEEK_Q16K_LAYER_COUNT *
                       DEEPSEEK_Q16K_DENSE_OPERATIONS_PER_LAYER ==
                   DEEPSEEK_Q16K_DENSE_REFERENCE_DISPATCHES,
               "q16K dense benchmark dispatch count changed");
_Static_assert(DEEPSEEK_Q16K_DENSE_ASM_PACKET_DISPATCHES ==
                   UINT64_C(1792),
               "q16K dense ASM packet count changed");
_Static_assert(
    DEEPSEEK_Q16K_CHUNK_COUNT *
            (1u + DEEPSEEK_Q16K_LAYER_COUNT *
                      DEEPSEEK_Q16K_FULL_MICROBENCHMARK_OPERATIONS_PER_LAYER) ==
        DEEPSEEK_Q16K_FULL_MICROBENCHMARK_DISPATCHES,
    "q16K full benchmark dispatch count changed");
_Static_assert(
    2u + DEEPSEEK_Q16K_CHUNK_COUNT *
             (1u + DEEPSEEK_Q16K_LAYER_COUNT *
                       DEEPSEEK_Q16K_OPERATIONS_PER_LAYER) +
            3u ==
        DEEPSEEK_Q16K_COMPLETE_OPERATION_CALLBACKS,
    "q16K complete schedule callback count changed");
_Static_assert((DEEPSEEK_Q16K_PROMPT_ROWS +
                        DEEPSEEK_Q16K_PROTOCOL_TOKEN_CHUNK_ROWS -
                    1u) /
                       DEEPSEEK_Q16K_PROTOCOL_TOKEN_CHUNK_ROWS ==
                   DEEPSEEK_Q16K_PROTOCOL_TOKEN_CHUNKS,
               "q16K protocol chunk count changed");
_Static_assert(DEEPSEEK_Q16K_PRODUCTION_REFERENCE_DISPATCHES <
                   DEEPSEEK_Q16K_KERNARG_CAPACITY,
               "production dispatch evidence no longer fits the ring");
_Static_assert(sizeof(deepseek_q16k_hook_counts_t) ==
                   2u * sizeof(uint64_t),
               "q16K hook results must not contain a host-wait field");

static const deepseek_q16k_chunk_t deepseek_q16k_chunks_[
    DEEPSEEK_Q16K_CHUNK_COUNT] = {
    {384u, 16384u, false},   {16768u, 16384u, false},
    {33152u, 16384u, false}, {49536u, 16384u, false},
    {65920u, 16384u, false}, {82304u, 16384u, false},
    {98688u, 16384u, false}, {115072u, 15999u, true},
};

static const deepseek_q16k_production_reference_t
    deepseek_q16k_production_reference_ = {
        .baseline_worker_source_sha256 =
            "efa2a6919fb1ad2b784dd29bbb9c9ebbf27b8b26afb4a28bdc905b452646fe87",
        .baseline_worker_binary_sha256 =
            "5dd19af4d7c0b2d1e09b972fe3ea9755ccd343bdbed690d14c831322c2077e65",
        .q512_loom_source_sha256 =
            "fc15c7845be33ae7a034cea4cfdc456f914486385c32938e99c3dd507c9c6075",
        .q16k_benchmark_source_sha256 =
            "5bf40637fe7d8d37e4dcfb7c6004ec15eab0111ebc2070e33d0ad0817e9e2732",
        .q16k_loom_source_sha256 =
            "8fc50006c49c079a8cfb7433837f3248f2a637e3651631ecd117fc89456483a7",
        .q16k_hsaco_sha256 =
            "1f7f515f84ebca22089f852201115d7e11cb94518da061aee356be59c9b2cb27",
        .wall_seconds = 42.30555955134332,
        .final_token = DEEPSEEK_Q16K_PRODUCTION_REFERENCE_TOKEN,
        .dispatch_count = DEEPSEEK_Q16K_PRODUCTION_REFERENCE_DISPATCHES,
        .host_wait_count = DEEPSEEK_Q16K_PRODUCTION_REFERENCE_HOST_WAITS,
        .worker = {1u, 0u, 1u},
        .http = {1u, 1u, 1u, 3u, DEEPSEEK_Q16K_PROTOCOL_TOKEN_CHUNKS},
};

typedef struct deepseek_q16k_accounting_t {
  const deepseek_q16k_queue_contract_t* queue;
  deepseek_q16k_schedule_counters_t counters;
  uint32_t kernarg_cursor;
  uint32_t pending_dispatches;
} deepseek_q16k_accounting_t;

static void deepseek_q16k_set_status_(deepseek_q16k_status_t* status,
                                      deepseek_q16k_status_code_t code,
                                      const char* message) {
  if (status == NULL) return;
  status->code = code;
  status->message = message;
}

void deepseek_q16k_status_reset(deepseek_q16k_status_t* status) {
  deepseek_q16k_set_status_(status, DEEPSEEK_Q16K_STATUS_OK, "ok");
}

const char* deepseek_q16k_operation_name(
    deepseek_q16k_operation_t operation) {
  switch (operation) {
    case DEEPSEEK_Q16K_OPERATION_LEGACY_PREFIX:
      return "legacy prefix";
    case DEEPSEEK_Q16K_OPERATION_MIGRATE_LEGACY_KV:
      return "legacy KV migration";
    case DEEPSEEK_Q16K_OPERATION_EMBEDDING:
      return "embedding";
    case DEEPSEEK_Q16K_OPERATION_INPUT_NORM:
      return "input norm";
    case DEEPSEEK_Q16K_OPERATION_QKV_PROJECTION:
      return "QKV projection";
    case DEEPSEEK_Q16K_OPERATION_ROPE_CACHE_WRITE:
      return "RoPE/cache write";
    case DEEPSEEK_Q16K_OPERATION_ATTENTION:
      return "attention";
    case DEEPSEEK_Q16K_OPERATION_OUTPUT_PROJECTION:
      return "output projection";
    case DEEPSEEK_Q16K_OPERATION_POST_ATTENTION_NORM:
      return "post-attention norm";
    case DEEPSEEK_Q16K_OPERATION_GATE_UP:
      return "gate/up projection";
    case DEEPSEEK_Q16K_OPERATION_DOWN_PROJECTION:
      return "down projection";
    case DEEPSEEK_Q16K_OPERATION_FINAL_ROW_LOGITS:
      return "final-row logits";
    case DEEPSEEK_Q16K_OPERATION_GREEDY_ARGMAX:
      return "greedy argmax";
    case DEEPSEEK_Q16K_OPERATION_TOKEN_READBACK:
      return "token readback";
    case DEEPSEEK_Q16K_OPERATION_COUNT:
      break;
  }
  return "unknown";
}

const deepseek_q16k_chunk_t* deepseek_q16k_chunk_plan(size_t* count) {
  if (count != NULL) *count = DEEPSEEK_Q16K_CHUNK_COUNT;
  return deepseek_q16k_chunks_;
}

const deepseek_q16k_production_reference_t*
deepseek_q16k_production_reference(void) {
  return &deepseek_q16k_production_reference_;
}

bool deepseek_q16k_validate_static_contract(deepseek_q16k_status_t* status) {
  deepseek_q16k_status_reset(status);
  uint32_t expected_base = DEEPSEEK_Q16K_LEGACY_PREFIX_ROWS;
  uint32_t visited_rows = 0u;
  for (uint32_t i = 0u; i < DEEPSEEK_Q16K_CHUNK_COUNT; ++i) {
    const deepseek_q16k_chunk_t* chunk = &deepseek_q16k_chunks_[i];
    const bool expected_final = i + 1u == DEEPSEEK_Q16K_CHUNK_COUNT;
    if (chunk->position_base != expected_base || chunk->row_count == 0u ||
        chunk->row_count > DEEPSEEK_Q16K_CHUNK_CAPACITY ||
        chunk->final != expected_final ||
        UINT32_MAX - visited_rows < chunk->row_count ||
        UINT32_MAX - expected_base < chunk->row_count) {
      deepseek_q16k_set_status_(
          status, DEEPSEEK_Q16K_STATUS_INVALID_STATIC_CONTRACT,
          "invalid q16K suffix chunk plan");
      return false;
    }
    expected_base += chunk->row_count;
    visited_rows += chunk->row_count;
  }
  if (visited_rows != DEEPSEEK_Q16K_SUFFIX_ROWS ||
      expected_base != DEEPSEEK_Q16K_PROMPT_ROWS ||
      deepseek_q16k_chunks_[DEEPSEEK_Q16K_CHUNK_COUNT - 1u].row_count !=
          DEEPSEEK_Q16K_TAIL_ROWS) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_INVALID_STATIC_CONTRACT,
        "q16K suffix chunks do not cover the frozen prompt");
    return false;
  }
  return true;
}

bool deepseek_q16k_validate_queue_contract(
    const deepseek_q16k_queue_contract_t* queue,
    deepseek_q16k_status_t* status) {
  deepseek_q16k_status_reset(status);
  const bool fallback_kernarg_geometry =
      queue != NULL &&
      queue->kernarg_stride == DEEPSEEK_Q16K_KERNARG_STRIDE &&
      queue->kernarg_arena_bytes == DEEPSEEK_Q16K_KERNARG_ARENA_BYTES;
  const bool aiter_kernarg_geometry =
      queue != NULL &&
      queue->kernarg_stride == DEEPSEEK_Q16K_AITER_KERNARG_STRIDE &&
      queue->kernarg_arena_bytes ==
          DEEPSEEK_Q16K_AITER_KERNARG_ARENA_BYTES;
  if (queue == NULL || queue->queue == NULL || queue->kernarg_arena == NULL ||
      !queue->batched_dispatch ||
      queue->queue_packet_capacity != DEEPSEEK_Q16K_QUEUE_PACKET_CAPACITY ||
      queue->queue_depth_limit != DEEPSEEK_Q16K_QUEUE_DEPTH_LIMIT ||
      queue->doorbell_batch != DEEPSEEK_Q16K_DOORBELL_BATCH ||
      queue->kernarg_capacity != DEEPSEEK_Q16K_KERNARG_CAPACITY ||
      (!fallback_kernarg_geometry && !aiter_kernarg_geometry) ||
      queue->initial_kernarg_cursor != 0u) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_INVALID_QUEUE_CONTRACT,
        "q16K queue contract does not match the frozen production geometry");
    return false;
  }
  return true;
}

static deepseek_q16k_invocation_t deepseek_q16k_make_invocation_(
    deepseek_q16k_operation_t operation,
    const deepseek_q16k_queue_contract_t* queue,
    const deepseek_q16k_device_pointers_t* device,
    const deepseek_q16k_accounting_t* accounting, uint32_t chunk_index,
    uint32_t layer, uint32_t position_base, uint32_t query_count,
    bool final_chunk, bool final_layer) {
  const deepseek_q16k_invocation_t invocation = {
      .operation = operation,
      .chunk_index = chunk_index,
      .layer = layer,
      .position_base = position_base,
      .query_count = query_count,
      .final_chunk = final_chunk,
      .final_layer = final_layer,
      .kernarg_cursor = accounting->kernarg_cursor,
      .dense_tile_rows = 0u,
      .dense_tile_columns = 0u,
      .queue = queue,
      .device = device,
  };
  return invocation;
}

static bool deepseek_q16k_account_(
    deepseek_q16k_accounting_t* accounting,
    const deepseek_q16k_callback_counts_t* counts,
    deepseek_q16k_status_t* status) {
  deepseek_q16k_schedule_counters_t* counters = &accounting->counters;
  if (counts->dispatch_count > UINT64_MAX - counters->dispatch_count) {
    deepseek_q16k_set_status_(status,
                              DEEPSEEK_Q16K_STATUS_DISPATCH_OVERFLOW,
                              "q16K dispatch count overflow");
    return false;
  }
  if (counts->kernarg_count > UINT64_MAX - counters->kernarg_count) {
    deepseek_q16k_set_status_(status, DEEPSEEK_Q16K_STATUS_KERNARG_OVERFLOW,
                              "q16K kernarg count overflow");
    return false;
  }
  if (counts->host_wait_count > UINT64_MAX - counters->host_wait_count) {
    deepseek_q16k_set_status_(status,
                              DEEPSEEK_Q16K_STATUS_HOST_WAIT_OVERFLOW,
                              "q16K host-wait count overflow");
    return false;
  }
  if (accounting->kernarg_cursor > accounting->queue->kernarg_capacity ||
      counts->kernarg_count >
          (uint64_t)(accounting->queue->kernarg_capacity -
                     accounting->kernarg_cursor)) {
    deepseek_q16k_set_status_(status, DEEPSEEK_Q16K_STATUS_KERNARG_OVERFLOW,
                              "q16K kernarg ring overflow");
    return false;
  }
  if (counts->dispatch_count >
      UINT64_MAX - (uint64_t)accounting->pending_dispatches) {
    deepseek_q16k_set_status_(status,
                              DEEPSEEK_Q16K_STATUS_DISPATCH_OVERFLOW,
                              "q16K pending dispatch count overflow");
    return false;
  }

  const uint64_t combined_dispatches =
      (uint64_t)accounting->pending_dispatches + counts->dispatch_count;
  uint64_t added_doorbells =
      combined_dispatches / accounting->queue->doorbell_batch;
  uint32_t next_pending =
      (uint32_t)(combined_dispatches % accounting->queue->doorbell_batch);
  if (counts->host_wait_count != 0u && next_pending != 0u) {
    if (added_doorbells == UINT64_MAX) {
      deepseek_q16k_set_status_(status,
                                DEEPSEEK_Q16K_STATUS_DOORBELL_OVERFLOW,
                                "q16K doorbell count overflow");
      return false;
    }
    ++added_doorbells;
    next_pending = 0u;
  }
  if (added_doorbells > UINT64_MAX - counters->doorbell_write_count) {
    deepseek_q16k_set_status_(status,
                              DEEPSEEK_Q16K_STATUS_DOORBELL_OVERFLOW,
                              "q16K doorbell count overflow");
    return false;
  }

  counters->dispatch_count += counts->dispatch_count;
  counters->kernarg_count += counts->kernarg_count;
  counters->host_wait_count += counts->host_wait_count;
  counters->doorbell_write_count += added_doorbells;
  accounting->kernarg_cursor += (uint32_t)counts->kernarg_count;
  if (accounting->kernarg_cursor > counters->peak_kernarg_cursor) {
    counters->peak_kernarg_cursor = accounting->kernarg_cursor;
  }
  if (counts->dispatch_count != 0u) {
    const uint32_t batch =
        combined_dispatches >= accounting->queue->doorbell_batch
            ? accounting->queue->doorbell_batch
            : (uint32_t)combined_dispatches;
    if (batch > counters->maximum_dispatch_batch) {
      counters->maximum_dispatch_batch = batch;
    }
  }
  accounting->pending_dispatches = next_pending;
  if (counts->host_wait_count != 0u) accounting->kernarg_cursor = 0u;
  counters->final_kernarg_cursor = accounting->kernarg_cursor;
  return true;
}

static bool deepseek_q16k_finish_accounting_(
    deepseek_q16k_accounting_t* accounting,
    deepseek_q16k_status_t* status) {
  if (accounting->pending_dispatches != 0u) {
    if (accounting->counters.doorbell_write_count == UINT64_MAX) {
      deepseek_q16k_set_status_(status,
                                DEEPSEEK_Q16K_STATUS_DOORBELL_OVERFLOW,
                                "q16K doorbell count overflow");
      return false;
    }
    ++accounting->counters.doorbell_write_count;
    accounting->pending_dispatches = 0u;
  }
  accounting->counters.final_kernarg_cursor = accounting->kernarg_cursor;
  return true;
}

static bool deepseek_q16k_invoke_operation_(
    const deepseek_q16k_callbacks_t* callbacks,
    const deepseek_q16k_invocation_t* invocation,
    deepseek_q16k_accounting_t* accounting,
    deepseek_q16k_status_t* status) {
  deepseek_q16k_callback_counts_t counts = {0};
  if (!callbacks->operation(callbacks->user_data, invocation, &counts,
                            status)) {
    if (status->code == DEEPSEEK_Q16K_STATUS_OK) {
      deepseek_q16k_set_status_(status,
                                DEEPSEEK_Q16K_STATUS_CALLBACK_FAILED,
                                "q16K operation callback failed");
    }
    return false;
  }
  ++accounting->counters.operation_callback_calls;
  return deepseek_q16k_account_(accounting, &counts, status);
}

static bool deepseek_q16k_invoke_attention_(
    const deepseek_q16k_features_t* features,
    const deepseek_q16k_callbacks_t* callbacks,
    const deepseek_q16k_invocation_t* invocation,
    deepseek_q16k_accounting_t* accounting,
    deepseek_q16k_status_t* status) {
  if (!features->queue_native_attention) {
    return deepseek_q16k_invoke_operation_(callbacks, invocation, accounting,
                                           status);
  }
  deepseek_q16k_hook_counts_t hook_counts = {0};
  if (!callbacks->attention(callbacks->user_data, invocation, &hook_counts,
                            status)) {
    if (status->code == DEEPSEEK_Q16K_STATUS_OK) {
      deepseek_q16k_set_status_(status,
                                DEEPSEEK_Q16K_STATUS_CALLBACK_FAILED,
                                "q16K queue-native attention hook failed");
    }
    return false;
  }
  ++accounting->counters.attention_hook_calls;
  const deepseek_q16k_callback_counts_t counts = {
      .dispatch_count = hook_counts.dispatch_count,
      .kernarg_count = hook_counts.kernarg_count,
      .host_wait_count = 0u,
  };
  return deepseek_q16k_account_(accounting, &counts, status);
}

static bool deepseek_q16k_invoke_dense_(
    const deepseek_q16k_features_t* features,
    const deepseek_q16k_callbacks_t* callbacks,
    deepseek_q16k_dense_gemm_kind_t kind,
    const deepseek_q16k_invocation_t* invocation,
    deepseek_q16k_accounting_t* accounting,
    deepseek_q16k_status_t* status) {
  if (!features->dense_gemm_256x256) {
    return deepseek_q16k_invoke_operation_(callbacks, invocation, accounting,
                                           status);
  }
  deepseek_q16k_invocation_t hook_invocation = *invocation;
  hook_invocation.dense_tile_rows = DEEPSEEK_Q16K_DENSE_GEMM_TILE_ROWS;
  hook_invocation.dense_tile_columns =
      DEEPSEEK_Q16K_DENSE_GEMM_TILE_COLUMNS;
  deepseek_q16k_hook_counts_t hook_counts = {0};
  if (!callbacks->dense_gemm_256x256(callbacks->user_data, kind,
                                    &hook_invocation, &hook_counts, status)) {
    if (status->code == DEEPSEEK_Q16K_STATUS_OK) {
      deepseek_q16k_set_status_(status,
                                DEEPSEEK_Q16K_STATUS_CALLBACK_FAILED,
                                "q16K 256x256 dense GEMM hook failed");
    }
    return false;
  }
  ++accounting->counters.dense_gemm_hook_calls;
  const deepseek_q16k_callback_counts_t counts = {
      .dispatch_count = hook_counts.dispatch_count,
      .kernarg_count = hook_counts.kernarg_count,
      .host_wait_count = 0u,
  };
  return deepseek_q16k_account_(accounting, &counts, status);
}

bool deepseek_q16k_run_schedule(
    const deepseek_q16k_request_t* request,
    const deepseek_q16k_queue_contract_t* queue,
    const deepseek_q16k_device_pointers_t* device,
    const deepseek_q16k_features_t* features,
    const deepseek_q16k_callbacks_t* callbacks,
    deepseek_q16k_serving_result_t* result,
    deepseek_q16k_status_t* status) {
  if (status != NULL) deepseek_q16k_status_reset(status);
  if (request == NULL || queue == NULL || device == NULL || features == NULL ||
      callbacks == NULL || result == NULL || status == NULL) {
    deepseek_q16k_set_status_(status,
                              DEEPSEEK_Q16K_STATUS_INVALID_ARGUMENT,
                              "q16K schedule received a null argument");
    return false;
  }
  memset(result, 0, sizeof(*result));
  result->observation.final_token = -1;
  result->observation.oracle.expected_token = -1;
  if (request->prompt_token_count != DEEPSEEK_Q16K_PROMPT_ROWS ||
      request->generated_token_count != 1u) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_INVALID_REQUEST,
        "q16K schedule requires 131071 prompt tokens and one generated token");
    return false;
  }
  if (!deepseek_q16k_validate_static_contract(status) ||
      !deepseek_q16k_validate_queue_contract(queue, status)) {
    return false;
  }
  if (callbacks->operation == NULL || callbacks->observe == NULL) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_MISSING_CALLBACK,
        "q16K schedule requires operation and observation callbacks");
    return false;
  }
  if (features->queue_native_attention && callbacks->attention == NULL) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_MISSING_CALLBACK,
        "q16K queue-native attention is enabled without a hook");
    return false;
  }
  if ((features->queue_native_attention ||
       features->dense_gemm_256x256) &&
      (queue->kernarg_stride != DEEPSEEK_Q16K_AITER_KERNARG_STRIDE ||
       queue->kernarg_arena_bytes !=
           DEEPSEEK_Q16K_AITER_KERNARG_ARENA_BYTES)) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_INVALID_QUEUE_CONTRACT,
        "q16K AITER hooks require a 512-byte kernarg stride");
    return false;
  }
  if (features->dense_gemm_256x256 &&
      callbacks->dense_gemm_256x256 == NULL) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_MISSING_CALLBACK,
        "q16K 256x256 dense GEMMs are enabled without a hook");
    return false;
  }

  deepseek_q16k_accounting_t accounting = {
      .queue = queue,
      .kernarg_cursor = queue->initial_kernarg_cursor,
  };
  accounting.counters.peak_kernarg_cursor = accounting.kernarg_cursor;
  accounting.counters.final_kernarg_cursor = accounting.kernarg_cursor;

  deepseek_q16k_invocation_t invocation = deepseek_q16k_make_invocation_(
      DEEPSEEK_Q16K_OPERATION_LEGACY_PREFIX, queue, device, &accounting,
      DEEPSEEK_Q16K_NO_CHUNK, DEEPSEEK_Q16K_NO_LAYER, 0u,
      DEEPSEEK_Q16K_LEGACY_PREFIX_ROWS, false, false);
  if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                       status)) {
    return false;
  }
  invocation = deepseek_q16k_make_invocation_(
      DEEPSEEK_Q16K_OPERATION_MIGRATE_LEGACY_KV, queue, device, &accounting,
      DEEPSEEK_Q16K_NO_CHUNK, DEEPSEEK_Q16K_NO_LAYER,
      DEEPSEEK_Q16K_LEGACY_PREFIX_ROWS, 0u, false, false);
  if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                       status)) {
    return false;
  }

  for (uint32_t chunk_index = 0u; chunk_index < DEEPSEEK_Q16K_CHUNK_COUNT;
       ++chunk_index) {
    const deepseek_q16k_chunk_t* chunk =
        &deepseek_q16k_chunks_[chunk_index];
    invocation = deepseek_q16k_make_invocation_(
        DEEPSEEK_Q16K_OPERATION_EMBEDDING, queue, device, &accounting,
        chunk_index, DEEPSEEK_Q16K_NO_LAYER, chunk->position_base,
        chunk->row_count, chunk->final, false);
    if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                         status)) {
      return false;
    }
    ++accounting.counters.suffix_chunks;
    accounting.counters.suffix_rows += chunk->row_count;

    for (uint32_t layer = 0u; layer < DEEPSEEK_Q16K_LAYER_COUNT; ++layer) {
      const bool final_layer = layer + 1u == DEEPSEEK_Q16K_LAYER_COUNT;
      invocation = deepseek_q16k_make_invocation_(
          DEEPSEEK_Q16K_OPERATION_INPUT_NORM, queue, device, &accounting,
          chunk_index, layer, chunk->position_base, chunk->row_count,
          chunk->final, final_layer);
      if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                           status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_QKV_PROJECTION;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_dense_(
              features, callbacks, DEEPSEEK_Q16K_DENSE_GEMM_QKV_PROJECTION,
              &invocation, &accounting, status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_ROPE_CACHE_WRITE;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                           status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_ATTENTION;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_attention_(features, callbacks, &invocation,
                                           &accounting, status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_OUTPUT_PROJECTION;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_dense_(
              features, callbacks,
              DEEPSEEK_Q16K_DENSE_GEMM_OUTPUT_PROJECTION, &invocation,
              &accounting, status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_POST_ATTENTION_NORM;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                           status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_GATE_UP;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_dense_(
              features, callbacks, DEEPSEEK_Q16K_DENSE_GEMM_GATE_UP,
              &invocation, &accounting, status)) {
        return false;
      }
      invocation.operation = DEEPSEEK_Q16K_OPERATION_DOWN_PROJECTION;
      invocation.kernarg_cursor = accounting.kernarg_cursor;
      if (!deepseek_q16k_invoke_dense_(
              features, callbacks, DEEPSEEK_Q16K_DENSE_GEMM_DOWN_PROJECTION,
              &invocation, &accounting, status)) {
        return false;
      }
      ++accounting.counters.layer_invocations;
    }
  }

  const deepseek_q16k_operation_t final_operations[] = {
      DEEPSEEK_Q16K_OPERATION_FINAL_ROW_LOGITS,
      DEEPSEEK_Q16K_OPERATION_GREEDY_ARGMAX,
      DEEPSEEK_Q16K_OPERATION_TOKEN_READBACK,
  };
  for (size_t i = 0u; i < sizeof(final_operations) / sizeof(final_operations[0]);
       ++i) {
    invocation = deepseek_q16k_make_invocation_(
        final_operations[i], queue, device, &accounting,
        DEEPSEEK_Q16K_CHUNK_COUNT - 1u, DEEPSEEK_Q16K_NO_LAYER,
        DEEPSEEK_Q16K_PROMPT_ROWS - 1u, 1u, true, false);
    if (!deepseek_q16k_invoke_operation_(callbacks, &invocation, &accounting,
                                         status)) {
      return false;
    }
  }
  if (!deepseek_q16k_finish_accounting_(&accounting, status)) return false;
  if (!callbacks->observe(callbacks->user_data, &result->observation,
                          status)) {
    if (status->code == DEEPSEEK_Q16K_STATUS_OK) {
      deepseek_q16k_set_status_(status,
                                DEEPSEEK_Q16K_STATUS_CALLBACK_FAILED,
                                "q16K observation callback failed");
    }
    return false;
  }
  if (result->observation.generated_token_count !=
      request->generated_token_count) {
    deepseek_q16k_set_status_(
        status, DEEPSEEK_Q16K_STATUS_OBSERVATION_MISMATCH,
        "q16K observation generated-token count mismatch");
    return false;
  }
  result->schedule = accounting.counters;
  return true;
}

#endif  // DEEPSEEK_Q16K_SCHEDULE_IMPLEMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DEEPSEEK_Q16K_SCHEDULE_H_
