#ifndef DEEPSEEK_Q16K_FULL_MODEL_SYNC_DIAGNOSTIC_H_
#define DEEPSEEK_Q16K_FULL_MODEL_SYNC_DIAGNOSTIC_H_

#include <stddef.h>
#include <stdint.h>

#include "../../q16k_aiter_integration.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q16K_FULL_MODEL_SYNC_DIAGNOSTIC_SCHEMA \
  "loom.q16k.full_model_sync.v4"
#define Q16K_FULL_MODEL_SYNC_LAYER_RECORD_CAPACITY \
  (Q16K_AITER_MAX_CHUNK_COUNT * Q16K_AITER_LAYER_COUNT)
#define Q16K_FULL_MODEL_SYNC_EXPECTED_STAGE_WAIT_COUNT \
  (2u * Q16K_FULL_MODEL_SYNC_LAYER_RECORD_CAPACITY)
#define Q16K_FULL_MODEL_SYNC_METADATA_WORD_COUNT 6u
#define Q16K_FULL_MODEL_SYNC_EXPECTED_METADATA_COPY_WAIT_COUNT 2u
#define Q16K_FULL_MODEL_SYNC_EXPECTED_FIRST_ATTENTION_COPY_WAIT_COUNT 2u
#define Q16K_FULL_MODEL_SYNC_EXPECTED_ORDINARY_REQUEST_WAIT_COUNT 43u
#define Q16K_FULL_MODEL_SYNC_EXPECTED_REQUEST_WAIT_COUNT \
  (Q16K_FULL_MODEL_SYNC_EXPECTED_ORDINARY_REQUEST_WAIT_COUNT + \
   Q16K_FULL_MODEL_SYNC_EXPECTED_STAGE_WAIT_COUNT + \
   Q16K_FULL_MODEL_SYNC_EXPECTED_METADATA_COPY_WAIT_COUNT + \
   Q16K_FULL_MODEL_SYNC_EXPECTED_FIRST_ATTENTION_COPY_WAIT_COUNT)
#define Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE 65u
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT \
  ((size_t)Q16K_AITER_CHUNK_CAPACITY * (size_t)Q16K_AITER_HIDDEN_SIZE * \
   sizeof(uint16_t))
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_COUNT \
  Q16K_AITER_CHUNK_CAPACITY
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT \
  (Q16K_AITER_HIDDEN_SIZE / Q16K_AITER_HEAD_DIMENSION)
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_DIMENSION \
  Q16K_AITER_HEAD_DIMENSION
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_ROWS 128u
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_COUNT \
  (Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_COUNT / \
   Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_ROWS)
#define Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_ELEMENT_COUNT \
  ((uint64_t)Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_COUNT * \
   (uint64_t)Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT * \
   (uint64_t)Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_DIMENSION)
#define Q16K_FULL_MODEL_SYNC_RETAINED_PRE_ATTENTION_OFFSET 256u

#if defined(__cplusplus)
static_assert(Q16K_FULL_MODEL_SYNC_EXPECTED_REQUEST_WAIT_COUNT == 495u,
              "q16K full-model diagnostic wait accounting changed");
static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT == 50331648u,
              "q16K first-attention byte count changed");
static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT == 12u,
              "q16K first-attention head count changed");
static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_COUNT == 128u,
              "q16K first-attention query tile count changed");
static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_ELEMENT_COUNT == 25165824u,
              "q16K first-attention element count changed");
static_assert(Q16K_FULL_MODEL_SYNC_RETAINED_PRE_ATTENTION_OFFSET >=
                  Q16K_AITER_METADATA_SIZE,
              "retained pre-attention bytes overlap metadata staging");
#else
_Static_assert(Q16K_FULL_MODEL_SYNC_EXPECTED_REQUEST_WAIT_COUNT == 495u,
               "q16K full-model diagnostic wait accounting changed");
_Static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT == 50331648u,
               "q16K first-attention byte count changed");
_Static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT == 12u,
               "q16K first-attention head count changed");
_Static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_COUNT == 128u,
               "q16K first-attention query tile count changed");
_Static_assert(Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_ELEMENT_COUNT == 25165824u,
               "q16K first-attention element count changed");
_Static_assert(Q16K_FULL_MODEL_SYNC_RETAINED_PRE_ATTENTION_OFFSET >=
                   Q16K_AITER_METADATA_SIZE,
               "retained pre-attention bytes overlap metadata staging");
#endif

typedef struct q16k_full_model_sync_allocation_input_s {
  const void* base;
  size_t size;
} q16k_full_model_sync_allocation_input_t;

typedef struct q16k_full_model_sync_runtime_layout_input_s {
  q16k_full_model_sync_allocation_input_t metadata;
  q16k_full_model_sync_allocation_input_t page_indices;
  q16k_full_model_sync_allocation_input_t scratch;
  q16k_full_model_sync_allocation_input_t loom_key;
  q16k_full_model_sync_allocation_input_t loom_value;
  q16k_full_model_sync_allocation_input_t shadow_key;
  q16k_full_model_sync_allocation_input_t shadow_value;
  q16k_full_model_sync_allocation_input_t rope;
  q16k_full_model_sync_allocation_input_t params;
  q16k_full_model_sync_allocation_input_t dense_workspace;
  q16k_full_model_sync_allocation_input_t kernarg_ring;
} q16k_full_model_sync_runtime_layout_input_t;

typedef struct q16k_full_model_sync_allocation_record_s {
  uint32_t captured;
  uint64_t base;
  uint64_t size;
  uint64_t end;
  uint64_t alignment;
  uint32_t base_mod_16;
  uint32_t base_mod_256;
} q16k_full_model_sync_allocation_record_t;

typedef struct q16k_full_model_sync_runtime_layout_s {
  uint32_t captured;
  q16k_full_model_sync_allocation_record_t metadata;
  q16k_full_model_sync_allocation_record_t page_indices;
  q16k_full_model_sync_allocation_record_t scratch;
  q16k_full_model_sync_allocation_record_t loom_key;
  q16k_full_model_sync_allocation_record_t loom_value;
  q16k_full_model_sync_allocation_record_t shadow_key;
  q16k_full_model_sync_allocation_record_t shadow_value;
  q16k_full_model_sync_allocation_record_t rope;
  q16k_full_model_sync_allocation_record_t params;
  q16k_full_model_sync_allocation_record_t dense_workspace;
  q16k_full_model_sync_allocation_record_t kernarg_ring;
} q16k_full_model_sync_runtime_layout_t;

typedef enum q16k_full_model_sync_metadata_phase_e {
  Q16K_FULL_MODEL_SYNC_METADATA_POST_PACK_PRE_ATTENTION = 0,
  Q16K_FULL_MODEL_SYNC_METADATA_POST_ATTENTION,
} q16k_full_model_sync_metadata_phase_t;

typedef struct q16k_full_model_sync_metadata_snapshot_s {
  uint32_t copy_attempted;
  uint32_t copy_completed;
  uint32_t captured;
  uint32_t expected_match;
  uint32_t position_base;
  uint32_t logical_query_count;
  uint32_t layer;
  size_t byte_count;
  uint64_t copy_begin_ns;
  uint64_t copy_end_ns;
  uint64_t copy_wait_count;
  int32_t words[Q16K_FULL_MODEL_SYNC_METADATA_WORD_COUNT];
  char sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE];
} q16k_full_model_sync_metadata_snapshot_t;

typedef struct q16k_full_model_sync_kernarg_snapshot_s {
  uint32_t captured;
  size_t byte_count;
  char sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE];
  uint8_t bytes[Q16K_AITER_ATTENTION_KERNARG_SIZE];
} q16k_full_model_sync_kernarg_snapshot_t;

typedef struct q16k_full_model_sync_attention_pointers_s {
  uint32_t captured;
  uint32_t match_plan;
  uint64_t query;
  uint64_t key;
  uint64_t value;
  uint64_t output;
  uint64_t kv_indptr;
  uint64_t kv_page_indices;
  uint64_t kv_last_page_lens;
  uint64_t cu_seqlens_q;
} q16k_full_model_sync_attention_pointers_t;

typedef struct q16k_full_model_sync_first_layer_kernargs_s {
  uint32_t captured;
  q16k_full_model_sync_kernarg_snapshot_t pack;
  q16k_full_model_sync_kernarg_snapshot_t attention;
  q16k_full_model_sync_attention_pointers_t attention_pointers;
} q16k_full_model_sync_first_layer_kernargs_t;

typedef enum q16k_full_model_sync_status_e {
  Q16K_FULL_MODEL_SYNC_OK = 0,
  Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT,
  Q16K_FULL_MODEL_SYNC_BAD_STATE,
  Q16K_FULL_MODEL_SYNC_INCOMPLETE,
  Q16K_FULL_MODEL_SYNC_IO_ERROR,
} q16k_full_model_sync_status_t;

typedef enum q16k_full_model_sync_stage_e {
  Q16K_FULL_MODEL_SYNC_STAGE_PACK = 0,
  Q16K_FULL_MODEL_SYNC_STAGE_ATTENTION,
} q16k_full_model_sync_stage_t;

typedef struct q16k_full_model_sync_stage_record_s {
  uint32_t arm_attempted;
  uint32_t armed;
  uint32_t publish_attempted;
  uint32_t published;
  uint32_t wait_attempted;
  uint32_t wait_acquired;
  q16k_aiter_status_t arm_status;
  q16k_aiter_status_t publish_status;
  q16k_aiter_status_t wait_status;
  uint64_t arm_ns;
  uint64_t publish_begin_ns;
  uint64_t publish_end_ns;
  uint64_t wait_begin_ns;
  uint64_t wait_end_ns;
  uint32_t cursor_before_publish;
  uint32_t cursor_after_publish;
  uint32_t cursor_after_wait;
  q16k_aiter_dispatch_result_t dispatch;
} q16k_full_model_sync_stage_record_t;

typedef struct q16k_full_model_sync_layer_record_s {
  uint32_t chunk_ordinal;
  uint32_t position_base;
  uint32_t logical_query_count;
  uint32_t layer;
  q16k_full_model_sync_stage_record_t pack;
  q16k_full_model_sync_stage_record_t attention;
  uint32_t attention_post_wait_completed;
  uint32_t output_enqueued;
} q16k_full_model_sync_layer_record_t;

typedef struct q16k_full_model_sync_first_attention_difference_s {
  uint32_t analyzed;
  uint32_t layout_valid;
  uint32_t query_count;
  uint32_t head_count;
  uint32_t head_dimension;
  uint32_t query_tile_rows;
  uint32_t query_tile_count;
  size_t byte_count;
  uint64_t element_count;
  uint64_t changed_byte_count;
  uint64_t changed_element_count;
  uint64_t first_changed_byte;
  uint64_t last_changed_byte;
  uint64_t first_changed_element;
  uint64_t last_changed_element;
  uint64_t changed_query_row_count;
  uint64_t first_changed_query_row;
  uint64_t last_changed_query_row;
  uint64_t unchanged_prefix_byte_count;
  uint64_t unchanged_suffix_byte_count;
  uint64_t unchanged_prefix_element_count;
  uint64_t unchanged_suffix_element_count;
  uint64_t per_head_changed_element_count
      [Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT];
  uint64_t per_query_tile_changed_element_count
      [Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_COUNT];
} q16k_full_model_sync_first_attention_difference_t;

typedef struct q16k_full_model_sync_diagnostic_s {
  uint32_t armed;
  uint32_t request_finished;
  uint32_t request_succeeded;
  uint64_t request_id;
  uint32_t request_ordinal;
  q16k_full_model_sync_runtime_layout_t runtime_layout;
  q16k_full_model_sync_metadata_snapshot_t post_pack_pre_attention_metadata;
  q16k_full_model_sync_metadata_snapshot_t post_attention_metadata;
  q16k_full_model_sync_first_layer_kernargs_t first_layer_kernargs;
  uint32_t layer_record_count;
  q16k_full_model_sync_layer_record_t
      layers[Q16K_FULL_MODEL_SYNC_LAYER_RECORD_CAPACITY];
  uint32_t first_attention_pre_hash_captured;
  uint32_t first_attention_pre_copy_wait_attempted;
  uint32_t first_attention_pre_copy_wait_completed;
  size_t first_attention_pre_byte_count;
  uint64_t first_attention_pre_copy_begin_ns;
  uint64_t first_attention_pre_copy_end_ns;
  uint64_t first_attention_pre_copy_wait_count;
  char first_attention_pre_sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE];
  uint32_t first_attention_hash_captured;
  uint32_t first_attention_copy_wait_attempted;
  uint32_t first_attention_copy_wait_completed;
  uint32_t first_attention_position_base;
  uint32_t first_attention_query_count;
  uint32_t first_attention_layer;
  size_t first_attention_byte_count;
  uint64_t first_attention_copy_begin_ns;
  uint64_t first_attention_copy_end_ns;
  char first_attention_sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE];
  uint32_t first_attention_byte_equal;
  q16k_full_model_sync_first_attention_difference_t
      first_attention_difference;
  uint32_t final_token_captured;
  int32_t final_token;
  uint64_t dispatch_count_begin;
  uint64_t wait_count_begin;
  uint64_t total_dispatch_count;
  uint64_t total_wait_count;
  uint64_t stage_wait_count;
  uint64_t metadata_copy_wait_count;
  uint64_t first_attention_copy_wait_count;
} q16k_full_model_sync_diagnostic_t;

typedef uint64_t (*q16k_full_model_sync_now_ns_fn)(void* user_data);
typedef q16k_aiter_status_t (*q16k_full_model_sync_arm_fn)(
    void* user_data, q16k_full_model_sync_stage_t stage, uint32_t layer,
    uint64_t completion_signal);
typedef q16k_aiter_status_t (*q16k_full_model_sync_wait_fn)(
    void* user_data, q16k_full_model_sync_stage_t stage, uint32_t layer,
    uint64_t completion_signal);
typedef void (*q16k_full_model_sync_cursor_fn)(
    void* user_data, const q16k_aiter_queue_state_t* queue,
    uint32_t reclaimed);
typedef q16k_aiter_status_t (*q16k_full_model_sync_pre_attention_fn)(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan,
    const void* pack_kernarg, size_t pack_kernarg_size,
    const void* attention_kernarg, size_t attention_kernarg_size);
typedef q16k_aiter_status_t (*q16k_full_model_sync_post_attention_fn)(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan);

typedef struct q16k_full_model_sync_ops_s {
  void* user_data;
  q16k_full_model_sync_now_ns_fn now_ns;
  q16k_full_model_sync_arm_fn arm;
  q16k_full_model_sync_wait_fn wait;
  q16k_full_model_sync_cursor_fn synchronize_cursor;
  q16k_full_model_sync_pre_attention_fn pre_attention_publish;
  q16k_full_model_sync_post_attention_fn post_attention_wait;
} q16k_full_model_sync_ops_t;

const char* q16k_full_model_sync_status_string(
    q16k_full_model_sync_status_t status);

void q16k_full_model_sync_begin(
    q16k_full_model_sync_diagnostic_t* diagnostic, uint64_t request_id,
    uint32_t request_ordinal, uint64_t dispatch_count_begin,
    uint64_t wait_count_begin);

q16k_full_model_sync_status_t q16k_full_model_sync_capture_runtime_layout(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_full_model_sync_runtime_layout_input_t* layout);

q16k_full_model_sync_status_t q16k_full_model_sync_begin_layer(
    q16k_full_model_sync_diagnostic_t* diagnostic, uint32_t position_base,
    uint32_t logical_query_count, uint32_t layer,
    q16k_full_model_sync_layer_record_t** out_record);

q16k_aiter_status_t q16k_full_model_sync_enqueue_layer(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_layer_plan_t* layer_plan, uint64_t completion_signal,
    const q16k_full_model_sync_ops_t* ops,
    q16k_full_model_sync_layer_record_t* record);

q16k_full_model_sync_status_t q16k_full_model_sync_capture_first_attention(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_aiter_layer_plan_t* layer_plan, size_t byte_count,
    uint64_t copy_begin_ns, uint64_t copy_end_ns, uint64_t copy_wait_count,
    const char sha256[65]);

q16k_full_model_sync_status_t
q16k_full_model_sync_analyze_first_attention_difference(
    const void* pre_attention_bytes, const void* post_attention_bytes,
    size_t byte_count, uint32_t query_count, uint32_t head_count,
    uint32_t head_dimension,
    q16k_full_model_sync_first_attention_difference_t* out_difference);

q16k_full_model_sync_status_t
q16k_full_model_sync_capture_first_attention_difference(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_full_model_sync_first_attention_difference_t* difference);

q16k_full_model_sync_status_t
q16k_full_model_sync_capture_first_attention_pre(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_aiter_layer_plan_t* layer_plan, size_t byte_count,
    uint64_t copy_begin_ns, uint64_t copy_end_ns, uint64_t copy_wait_count,
    const char sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE]);

q16k_full_model_sync_status_t q16k_full_model_sync_capture_metadata_snapshot(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    q16k_full_model_sync_metadata_phase_t phase,
    const q16k_aiter_layer_plan_t* layer_plan, size_t byte_count,
    uint64_t copy_begin_ns, uint64_t copy_end_ns, uint64_t copy_wait_count,
    const int32_t words[Q16K_FULL_MODEL_SYNC_METADATA_WORD_COUNT],
    const char sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE]);

q16k_full_model_sync_status_t q16k_full_model_sync_capture_first_layer_kernargs(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_aiter_layer_plan_t* layer_plan, const void* pack_kernarg,
    size_t pack_kernarg_size, const char pack_sha256[65],
    const void* attention_kernarg, size_t attention_kernarg_size,
    const char attention_sha256[65]);

void q16k_full_model_sync_finish(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    uint32_t request_succeeded, uint32_t final_token_captured,
    int32_t final_token, uint64_t dispatch_count_end,
    uint64_t wait_count_end);

int q16k_full_model_sync_is_complete(
    const q16k_full_model_sync_diagnostic_t* diagnostic);

q16k_full_model_sync_status_t q16k_full_model_sync_write_json_atomic(
    const q16k_full_model_sync_diagnostic_t* diagnostic,
    const char* output_path, char* error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif  /* DEEPSEEK_Q16K_FULL_MODEL_SYNC_DIAGNOSTIC_H_ */
