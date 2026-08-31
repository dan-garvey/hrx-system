#ifndef DEEPSEEK_Q16K_AITER_INTEGRATION_H_
#define DEEPSEEK_Q16K_AITER_INTEGRATION_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q16K_AITER_LAYER_COUNT 28u
#define Q16K_AITER_HIDDEN_SIZE 1536u
#define Q16K_AITER_INTERMEDIATE_SIZE 8960u
#define Q16K_AITER_KV_HEAD_COUNT 2u
#define Q16K_AITER_HEAD_DIMENSION 128u
#define Q16K_AITER_KV_SIZE \
  (Q16K_AITER_KV_HEAD_COUNT * Q16K_AITER_HEAD_DIMENSION)
#define Q16K_AITER_VOCAB_SIZE 151936u
#define Q16K_AITER_CONTEXT_CAPACITY 131072u
#define Q16K_AITER_MAX_PROMPT_TOKEN_COUNT 131071u
#define Q16K_AITER_LEGACY_PREFIX_ROWS 384u
#define Q16K_AITER_CHUNK_CAPACITY 16384u
#define Q16K_AITER_MAX_CHUNK_COUNT 8u
#define Q16K_AITER_LOOM_BLOCK_TOKENS 24u
#define Q16K_AITER_LOOM_BLOCK_COUNT 5462u
#define Q16K_AITER_PAGE_INDEX_PADDING 256u
#define Q16K_AITER_KERNARG_ALIGNMENT 16u
#define Q16K_AITER_KERNARG_STRIDE 512u
#define Q16K_AITER_KERNARG_RING_CAPACITY 65536u
#define Q16K_AITER_PACK_KERNARG_SIZE 336u
#define Q16K_AITER_ATTENTION_KERNARG_SIZE 424u
#define Q16K_AITER_LAYER_DISPATCH_COUNT 9u
#define Q16K_AITER_NONFINAL_CHUNK_DISPATCH_COUNT \
  (1u + Q16K_AITER_LAYER_COUNT * Q16K_AITER_LAYER_DISPATCH_COUNT)
#define Q16K_AITER_FINAL_CHUNK_DISPATCH_COUNT \
  (Q16K_AITER_NONFINAL_CHUNK_DISPATCH_COUNT + 4u)
#define Q16K_AITER_METADATA_SIZE 24u
#define Q16K_AITER_METADATA_KV_INDPTR_OFFSET 0u
#define Q16K_AITER_METADATA_LAST_PAGE_LENGTH_OFFSET 8u
#define Q16K_AITER_METADATA_CU_SEQLENS_Q_OFFSET 16u

typedef enum q16k_aiter_status_e {
  Q16K_AITER_STATUS_OK = 0,
  Q16K_AITER_STATUS_INVALID_ARGUMENT,
  Q16K_AITER_STATUS_OUT_OF_RANGE,
  Q16K_AITER_STATUS_BUFFER_TOO_SMALL,
  Q16K_AITER_STATUS_OVERFLOW,
  Q16K_AITER_STATUS_FEATURE_DISABLED,
  Q16K_AITER_STATUS_QUEUE_FULL,
  Q16K_AITER_STATUS_RING_EXHAUSTED,
  Q16K_AITER_STATUS_CALLBACK_FAILED,
} q16k_aiter_status_t;

const char* q16k_aiter_status_string(q16k_aiter_status_t status);

typedef enum q16k_aiter_asset_kind_e {
  Q16K_AITER_ASSET_DENSE = 0,
  Q16K_AITER_ASSET_PACK,
  Q16K_AITER_ASSET_ATTENTION,
  Q16K_AITER_ASSET_COUNT,
} q16k_aiter_asset_kind_t;

typedef struct q16k_aiter_asset_spec_s {
  const char* relative_path;
  const char* sha256;
  size_t byte_size;
} q16k_aiter_asset_spec_t;

const q16k_aiter_asset_spec_t* q16k_aiter_asset_spec(
    q16k_aiter_asset_kind_t kind);

typedef struct q16k_aiter_launch_geometry_s {
  uint32_t grid_size[3];
  uint16_t workgroup_size[3];
} q16k_aiter_launch_geometry_t;

typedef struct q16k_aiter_kernel_spec_s {
  const char* symbol;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_alignment;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
  uint32_t explicit_argument_count;
} q16k_aiter_kernel_spec_t;

typedef enum q16k_dense_kernel_e {
  Q16K_DENSE_PREPARE_ROPE = 0,
  Q16K_DENSE_EMBEDDING,
  Q16K_DENSE_RMS_NORM,
  Q16K_DENSE_QKV,
  Q16K_DENSE_ROPE_CACHE,
  Q16K_DENSE_OUTPUT_PROJECTION,
  Q16K_DENSE_GATE_UP,
  Q16K_DENSE_DOWN_PROJECTION,
  Q16K_DENSE_KERNEL_COUNT,
} q16k_dense_kernel_t;

const q16k_aiter_kernel_spec_t* q16k_dense_kernel_spec(
    q16k_dense_kernel_t kernel);
const q16k_aiter_kernel_spec_t* q16k_pack_kernel_spec(void);
const q16k_aiter_kernel_spec_t* q16k_attention_kernel_spec(void);

q16k_aiter_status_t q16k_dense_launch_geometry(
    q16k_dense_kernel_t kernel, uint32_t logical_query_count,
    q16k_aiter_launch_geometry_t* out_geometry);

q16k_aiter_status_t q16k_build_dense_kernarg(
    q16k_dense_kernel_t kernel, void* const* bindings,
    size_t binding_count, void* kernarg, size_t kernarg_size);

typedef struct q16k_aiter_chunk_s {
  uint32_t position_base;
  uint32_t logical_query_count;
  uint32_t final;
} q16k_aiter_chunk_t;

q16k_aiter_status_t q16k_aiter_plan_chunks(
    uint32_t prompt_token_count, q16k_aiter_chunk_t* chunks,
    size_t chunk_capacity, size_t* out_chunk_count);

typedef enum q16k_aiter_scratch_segment_e {
  Q16K_SCRATCH_HIDDEN = 0,
  Q16K_SCRATCH_POST_ATTENTION,
  Q16K_SCRATCH_NORMALIZED,
  Q16K_SCRATCH_QUERY,
  Q16K_SCRATCH_ROTATED_QUERY,
  Q16K_SCRATCH_KEY,
  Q16K_SCRATCH_ROTATED_KEY,
  Q16K_SCRATCH_VALUE,
  Q16K_SCRATCH_ATTENTION,
  Q16K_SCRATCH_PROJECTION,
  Q16K_SCRATCH_GATE,
  Q16K_SCRATCH_UP,
  Q16K_SCRATCH_ACTIVATED,
  Q16K_SCRATCH_LOGITS,
  Q16K_SCRATCH_ARGMAX_VALUES,
  Q16K_SCRATCH_ARGMAX_IDS,
  Q16K_SCRATCH_SEGMENT_COUNT,
} q16k_aiter_scratch_segment_t;

typedef struct q16k_aiter_memory_region_s {
  size_t offset;
  size_t size;
} q16k_aiter_memory_region_t;

typedef struct q16k_aiter_allocation_plan_s {
  size_t scratch_alignment;
  size_t scratch_bytes;
  q16k_aiter_memory_region_t scratch[Q16K_SCRATCH_SEGMENT_COUNT];
  size_t loom_kv_layer_stride_bytes;
  size_t loom_kv_allocation_bytes;
  size_t shadow_kv_layer_stride_bytes;
  size_t shadow_kv_allocation_bytes;
  size_t shadow_kv_combined_bytes;
  size_t page_index_count;
  size_t page_index_bytes;
  size_t attention_metadata_bytes;
  size_t rope_table_bytes;
  size_t token_bytes;
  size_t kernarg_stride;
  size_t kernarg_ring_bytes;
} q16k_aiter_allocation_plan_t;

q16k_aiter_status_t q16k_aiter_make_allocation_plan(
    q16k_aiter_allocation_plan_t* out_plan);

q16k_aiter_status_t q16k_aiter_layer_offsets(
    uint32_t layer, size_t* out_loom_offset,
    size_t* out_shadow_offset);

typedef struct q16k_aiter_pack_request_s {
  const void* loom_k;
  const void* loom_v;
  void* shadow_k;
  void* shadow_v;
  uint32_t layer_count;
  uint32_t position_base;
  uint32_t logical_query_count;
  uint32_t destination_capacity;
  uint32_t source_block_count;
  uint32_t source_block_base;
} q16k_aiter_pack_request_t;

typedef struct q16k_aiter_pack_plan_s {
  q16k_aiter_pack_request_t request;
  size_t loom_layer_offset;
  size_t shadow_layer_offset;
} q16k_aiter_pack_plan_t;

q16k_aiter_status_t q16k_aiter_plan_prefix_pack(
    const void* loom_k, const void* loom_v, void* shadow_k,
    void* shadow_v, q16k_aiter_pack_plan_t* out_plan);

q16k_aiter_status_t q16k_aiter_plan_layer_pack(
    uint32_t layer, uint32_t position_base,
    uint32_t logical_query_count, const void* loom_k_base,
    const void* loom_v_base, void* shadow_k_base,
    void* shadow_v_base, q16k_aiter_pack_plan_t* out_plan);

q16k_aiter_status_t q16k_build_pack_kernarg(
    const q16k_aiter_pack_request_t* request, void* kernarg,
    size_t kernarg_size, q16k_aiter_launch_geometry_t* out_geometry);

typedef struct q16k_aiter_attention_metadata_s {
  int32_t kv_indptr[2];
  int32_t kv_last_page_lens[1];
  int32_t reserved_12;
  int32_t cu_seqlens_q[2];
} q16k_aiter_attention_metadata_t;

q16k_aiter_status_t q16k_build_attention_metadata(
    uint32_t position_base, uint32_t logical_query_count,
    q16k_aiter_attention_metadata_t* out_metadata);

q16k_aiter_status_t q16k_fill_identity_page_indices(
    int32_t* page_indices, size_t page_index_capacity);

typedef struct q16k_aiter_attention_request_s {
  const void* q;
  const void* k;
  const void* v;
  void* output;
  const int32_t* kv_indptr;
  const int32_t* kv_page_indices;
  const int32_t* kv_last_page_lens;
  const int32_t* cu_seqlens_q;
  uint32_t position_base;
  uint32_t logical_query_count;
} q16k_aiter_attention_request_t;

typedef struct q16k_aiter_layer_plan_s {
  uint32_t layer;
  uint32_t position_base;
  uint32_t logical_query_count;
  size_t loom_layer_offset;
  size_t shadow_layer_offset;
  q16k_aiter_pack_request_t pack;
  q16k_aiter_attention_request_t attention;
} q16k_aiter_layer_plan_t;

q16k_aiter_status_t q16k_aiter_plan_layer(
    uint32_t layer, uint32_t position_base,
    uint32_t logical_query_count, const void* loom_k_base,
    const void* loom_v_base, void* shadow_k_base,
    void* shadow_v_base, const void* rotated_query,
    void* attention_output, const int32_t* kv_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    const int32_t* cu_seqlens_q,
    q16k_aiter_layer_plan_t* out_plan);

q16k_aiter_status_t q16k_build_attention_kernarg(
    const q16k_aiter_attention_request_t* request, void* kernarg,
    size_t kernarg_size, q16k_aiter_launch_geometry_t* out_geometry);

/*
 * HSA-independent representation of the 64-byte AQL kernel dispatch packet.
 * The worker statically verifies this layout against its HSA headers before
 * casting a caller-owned queue packet to this type.
 */
typedef struct q16k_aiter_dispatch_packet_s {
  uint32_t full_header;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint64_t kernarg_address;
  uint64_t reserved2;
  uint64_t completion_signal;
} q16k_aiter_dispatch_packet_t;

typedef uint64_t (*q16k_aiter_queue_load_index_fn)(
    void* user_data, const void* queue);
typedef void (*q16k_aiter_queue_store_index_fn)(
    void* user_data, void* queue, uint64_t value);
typedef void (*q16k_aiter_queue_ring_doorbell_fn)(
    void* user_data, void* queue, uint64_t packet_id);
typedef void (*q16k_aiter_queue_publish_packet_fn)(
    void* user_data, q16k_aiter_dispatch_packet_t* packet,
    uint16_t dimensions);

typedef struct q16k_aiter_queue_ops_s {
  q16k_aiter_queue_load_index_fn load_write_index_relaxed;
  q16k_aiter_queue_load_index_fn load_read_index_scacquire;
  q16k_aiter_queue_store_index_fn store_write_index_screlease;
  q16k_aiter_queue_ring_doorbell_fn ring_doorbell_screlease;
  q16k_aiter_queue_publish_packet_fn publish_dispatch_packet;
} q16k_aiter_queue_ops_t;

typedef struct q16k_aiter_queue_state_s {
  void* queue;
  q16k_aiter_dispatch_packet_t* packets;
  uint32_t queue_size;
  void* kernarg_data;
  size_t kernarg_stride;
  uint32_t kernarg_capacity;
  uint32_t kernarg_cursor;
  uint32_t queue_depth_limit;
  uint32_t doorbell_batch_size;
  uint32_t batch_dispatch;
  uint32_t deferred_queue_open;
  uint64_t deferred_queue_write_index;
  uint64_t deferred_queue_read_index;
  uint32_t deferred_queue_pending_count;
  void* user_data;
  q16k_aiter_queue_ops_t ops;
} q16k_aiter_queue_state_t;

typedef struct q16k_aiter_loaded_kernel_s {
  const char* name;
  uint64_t kernel_object;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_alignment;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
} q16k_aiter_loaded_kernel_t;

typedef struct q16k_aiter_dispatch_result_s {
  uint64_t packet_id;
  uint32_t kernarg_slot;
  uint32_t doorbell_written;
} q16k_aiter_dispatch_result_t;

q16k_aiter_status_t q16k_aiter_dispatch_raw(
    q16k_aiter_queue_state_t* queue,
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_launch_geometry_t* geometry,
    const void* kernarg, size_t kernarg_size,
    uint64_t completion_signal, uint32_t force_flush,
    q16k_aiter_dispatch_result_t* out_result);

q16k_aiter_status_t q16k_aiter_flush_queue(
    q16k_aiter_queue_state_t* queue);

typedef enum q16k_aiter_loom_stage_e {
  Q16K_AITER_LOOM_QKV = 0,
  Q16K_AITER_LOOM_ROPE_CACHE,
  Q16K_AITER_LOOM_OUTPUT,
} q16k_aiter_loom_stage_t;

typedef q16k_aiter_status_t (*q16k_aiter_loom_enqueue_fn)(
    void* user_data, q16k_aiter_loom_stage_t stage,
    uint32_t layer, uint32_t position_base,
    uint32_t logical_query_count, uint64_t completion_signal);

typedef void (*q16k_aiter_attention_observer_fn)(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan,
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_launch_geometry_t* geometry,
    const void* source_kernarg, size_t source_kernarg_size,
    const q16k_aiter_dispatch_packet_t* published_packet,
    const q16k_aiter_dispatch_result_t* dispatch_result);

typedef struct q16k_aiter_layer_hook_s {
  uint32_t enabled;
  void* user_data;
  q16k_aiter_loom_enqueue_fn enqueue_loom;
  void* attention_observer_user_data;
  q16k_aiter_attention_observer_fn observe_attention;
  q16k_aiter_queue_state_t* queue;
  const q16k_aiter_loaded_kernel_t* pack_kernel;
  const q16k_aiter_loaded_kernel_t* attention_kernel;
} q16k_aiter_layer_hook_t;

q16k_aiter_status_t q16k_aiter_enqueue_prefix_pack(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_pack_plan_t* prefix_plan,
    uint64_t completion_signal);

q16k_aiter_status_t q16k_aiter_enqueue_layer(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_layer_plan_t* layer_plan,
    uint64_t output_completion_signal);

#ifdef __cplusplus
}
#endif

#endif  /* DEEPSEEK_Q16K_AITER_INTEGRATION_H_ */
