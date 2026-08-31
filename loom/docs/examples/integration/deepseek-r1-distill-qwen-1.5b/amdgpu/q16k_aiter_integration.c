#include "q16k_aiter_integration.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#define Q16K_SCRATCH_ALIGNMENT 256u
#define Q16K_HIDDEN_SEGMENT_BYTES                                      \
  ((size_t)Q16K_AITER_CHUNK_CAPACITY * Q16K_AITER_HIDDEN_SIZE *       \
   sizeof(uint16_t))
#define Q16K_KV_SEGMENT_BYTES                                          \
  ((size_t)Q16K_AITER_CHUNK_CAPACITY * Q16K_AITER_KV_SIZE *           \
   sizeof(uint16_t))
#define Q16K_MLP_SEGMENT_BYTES                                         \
  ((size_t)Q16K_AITER_CHUNK_CAPACITY * Q16K_AITER_INTERMEDIATE_SIZE * \
   sizeof(uint16_t))
#define Q16K_LOGITS_BYTES                                              \
  ((size_t)16u * Q16K_AITER_VOCAB_SIZE * sizeof(uint16_t))
#define Q16K_ARGMAX_VALUES_BYTES ((size_t)256u * sizeof(uint16_t))
#define Q16K_ARGMAX_IDS_BYTES ((size_t)256u * sizeof(int32_t))
#define Q16K_LOOM_KV_LAYER_STRIDE_BYTES                                \
  ((size_t)Q16K_AITER_LOOM_BLOCK_COUNT * Q16K_AITER_LOOM_BLOCK_TOKENS * \
   Q16K_AITER_KV_SIZE * sizeof(uint16_t))
#define Q16K_SHADOW_KV_LAYER_STRIDE_BYTES                              \
  ((size_t)Q16K_AITER_CONTEXT_CAPACITY * Q16K_AITER_KV_SIZE *         \
   sizeof(uint16_t))
#define Q16K_ROPE_TABLE_BYTES                                          \
  ((size_t)Q16K_AITER_CONTEXT_CAPACITY * 64u * 2u * sizeof(uint16_t))
#define Q16K_TOKEN_BYTES                                               \
  ((size_t)Q16K_AITER_CONTEXT_CAPACITY * sizeof(int32_t))
#define Q16K_SOFTMAX_SCALE_F 0.08838834764831845f
#define Q16K_LOG2E_F 1.4426950408889634074f

_Static_assert(sizeof(void*) == 8u, "q16K ABI requires 64-bit pointers");
_Static_assert(SIZE_MAX > UINT32_MAX, "q16K allocation plan requires size_t64");
_Static_assert(Q16K_HIDDEN_SEGMENT_BYTES == 50331648u,
               "q16K hidden segment changed");
_Static_assert(Q16K_KV_SEGMENT_BYTES == 8388608u,
               "q16K KV segment changed");
_Static_assert(Q16K_MLP_SEGMENT_BYTES == 293601280u,
               "q16K MLP segment changed");
_Static_assert(Q16K_LOOM_KV_LAYER_STRIDE_BYTES == 67117056u,
               "Loom KV layer stride changed");
_Static_assert(Q16K_SHADOW_KV_LAYER_STRIDE_BYTES == 67108864u,
               "shadow KV layer stride changed");
_Static_assert(Q16K_AITER_NONFINAL_CHUNK_DISPATCH_COUNT == 253u &&
                   Q16K_AITER_FINAL_CHUNK_DISPATCH_COUNT == 257u,
               "q16K chunk dispatch accounting changed");
_Static_assert(sizeof(q16k_aiter_attention_metadata_t) ==
                   Q16K_AITER_METADATA_SIZE,
               "attention metadata size changed");
_Static_assert(sizeof(q16k_aiter_dispatch_packet_t) == 64u,
               "AQL dispatch packet size changed");
_Static_assert(offsetof(q16k_aiter_dispatch_packet_t, kernel_object) == 32u,
               "AQL kernel object offset changed");
_Static_assert(offsetof(q16k_aiter_dispatch_packet_t, kernarg_address) == 40u,
               "AQL kernarg offset changed");
_Static_assert(offsetof(q16k_aiter_dispatch_packet_t, completion_signal) ==
                   56u,
               "AQL completion signal offset changed");
_Static_assert(offsetof(q16k_aiter_attention_metadata_t, kv_indptr) ==
                   Q16K_AITER_METADATA_KV_INDPTR_OFFSET,
               "kv_indptr offset changed");
_Static_assert(offsetof(q16k_aiter_attention_metadata_t,
                        kv_last_page_lens) ==
                   Q16K_AITER_METADATA_LAST_PAGE_LENGTH_OFFSET,
               "last-page offset changed");
_Static_assert(offsetof(q16k_aiter_attention_metadata_t, cu_seqlens_q) ==
                   Q16K_AITER_METADATA_CU_SEQLENS_Q_OFFSET,
               "cu_seqlens_q offset changed");

static const q16k_aiter_asset_spec_t kAssetSpecs[Q16K_AITER_ASSET_COUNT] = {
    {
        .relative_path = "artifacts/q16k_dense_gfx950.hsaco",
        .sha256 =
            "1f7f515f84ebca22089f852201115d7e11cb94518da061aee356be59c9b2cb27",
        .byte_size = 26512u,
    },
    {
        .relative_path = "artifacts/loom_shadow_pack_gfx950.hsaco",
        .sha256 =
            "02fbcdee0e2fe3911f050063fc72d1e7e2816979b353f698c25b6d13b49d6376",
        .byte_size = 8456u,
    },
    {
        .relative_path = "artifacts/aiter_prefill_q16384_gfx950.hsaco",
        .sha256 =
            "0e90cc246649934b27c886187dea4d7b960266e921edbb0b9b7d0b38eb5e4edc",
        .byte_size = 30056u,
    },
};

static const q16k_aiter_kernel_spec_t
    kDenseKernelSpecs[Q16K_DENSE_KERNEL_COUNT] = {
        {
            .symbol = "deepseek_prepare_rope_longctx_131072",
            .kernarg_segment_size = 8u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 1u,
        },
        {
            .symbol = "deepseek_embedding_prefill_q16k_longctx_131072",
            .kernarg_segment_size = 32u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 4u,
        },
        {
            .symbol = "deepseek_rms_norm_q16k",
            .kernarg_segment_size = 32u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 16u,
            .private_segment_size = 0u,
            .explicit_argument_count = 4u,
        },
        {
            .symbol = "deepseek_qkv_prefill_q16k_longctx_131072",
            .kernarg_segment_size = 88u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 11u,
        },
        {
            .symbol = "deepseek_rope_cache_prefill_q16k_longctx_131072",
            .kernarg_segment_size = 64u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 8u,
        },
        {
            .symbol = "deepseek_linear_1536_residual_prefill_q16k",
            .kernarg_segment_size = 56u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 7u,
        },
        {
            .symbol = "deepseek_gate_up_swiglu_prefill_q16k",
            .kernarg_segment_size = 56u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 7u,
        },
        {
            .symbol = "deepseek_down_residual_prefill_q16k",
            .kernarg_segment_size = 48u,
            .kernarg_segment_alignment = 16u,
            .group_segment_size = 0u,
            .private_segment_size = 0u,
            .explicit_argument_count = 6u,
        },
};

static const uint16_t kDenseWorkgroupSizes[Q16K_DENSE_KERNEL_COUNT] = {
    256u, 256u, 256u, 64u, 256u, 64u, 64u, 64u,
};

static const char kPackKernelSymbol[] =
    "_ZN12_GLOBAL__N_139loom_vec4_24_to_aiter_page1_bf16_kernelEPKN3c108BFloat16ES3_PS1_S4_mmmmmm";

static const q16k_aiter_kernel_spec_t kPackKernelSpec = {
    .symbol = kPackKernelSymbol,
    .kernarg_segment_size = Q16K_AITER_PACK_KERNARG_SIZE,
    .kernarg_segment_alignment = Q16K_AITER_KERNARG_ALIGNMENT,
    .group_segment_size = 0u,
    .private_segment_size = 0u,
    .explicit_argument_count = 10u,
};

static const char kAttentionKernelSymbol[] =
    "_ZN7ck_tile6kentryILi2ENS_38FmhaBatchPrefillWithPagedKVCacheKernelINS_40BlockFmhaBatchPrefillPipelineQRKSVSAsyncINS_36BlockFmhaBatchPrefillPipelineProblemIDF16bDF16bDF16bffDF16bhfDF16bfDF16bNS_13TileFmhaShapeINS_8sequenceIJLi128ELi128ELi32ELi128ELi32ELi128EEEENS5_IJLi4ELi1ELi1EEEENS5_IJLi32ELi32ELi16EEEES7_S8_Lb1EEELb1ENS_17ComposedAttentionILj0ELb1EEENS_30SimplifiedGenericAttentionMaskILb1EEELb0ELi1ENS_26TileFmhaBatchPrefillTraitsILb1ELb1ELb1ELb1ELb0ELNS_22BlockAttentionBiasEnumE0ELb0ELb0ELb0ELNS_28BlockAttentionQuantScaleEnumE0ELin1ELb0ELb0ELi1ELNS_37BlockAttentionKVCacheMemoryLayoutEnumE1ELNS_36BlockAttentionKVCacheLookupTableEnumE1ELNS_33BlockAttentionKVCacheLoadModeEnumE0EEEEENS_53BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicyEEENS_17Default2DEpilogueINS_24Default2DEpilogueProblemIfDF16bLb1ELb1ELb1EEEvEEEEJNSS_21FmhaFwdGroupModeKargsEEEEvDpT1_";

_Static_assert(sizeof(kPackKernelSymbol) - 1u == 92u,
               "pack kernel symbol changed");
_Static_assert(sizeof(kAttentionKernelSymbol) - 1u == 867u,
               "AITER kernel symbol changed");

static const q16k_aiter_kernel_spec_t kAttentionKernelSpec = {
    .symbol = kAttentionKernelSymbol,
    .kernarg_segment_size = Q16K_AITER_ATTENTION_KERNARG_SIZE,
    .kernarg_segment_alignment = Q16K_AITER_KERNARG_ALIGNMENT,
    .group_segment_size = 26112u,
    .private_segment_size = 0u,
    .explicit_argument_count = 1u,
};

static int checked_add_size(size_t left, size_t right, size_t* output) {
  if (output == NULL || left > SIZE_MAX - right) return 0;
  *output = left + right;
  return 1;
}

static int checked_mul_size(size_t left, size_t right, size_t* output) {
  if (output == NULL || (left != 0u && right > SIZE_MAX / left)) return 0;
  *output = left * right;
  return 1;
}

static int checked_align_size(size_t value, size_t alignment,
                              size_t* output) {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) return 0;
  const size_t mask = alignment - 1u;
  if (value > SIZE_MAX - mask) return 0;
  *output = (value + mask) & ~mask;
  return 1;
}

static int checked_pointer_offset(const void* base, size_t offset,
                                  void** output) {
  if (base == NULL || output == NULL ||
      (uintptr_t)base > UINTPTR_MAX - offset) {
    return 0;
  }
  *output = (void*)((uintptr_t)base + offset);
  return 1;
}

static int kernarg_is_valid(const void* kernarg, size_t kernarg_size,
                            size_t required_size) {
  return kernarg != NULL && kernarg_size >= required_size &&
         ((uintptr_t)kernarg & (Q16K_AITER_KERNARG_ALIGNMENT - 1u)) == 0u;
}

static void store_u16(uint8_t* bytes, size_t offset, uint16_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void store_u32(uint8_t* bytes, size_t offset, uint32_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void store_i32(uint8_t* bytes, size_t offset, int32_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void store_u64(uint8_t* bytes, size_t offset, uint64_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void store_f32(uint8_t* bytes, size_t offset, float value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static q16k_aiter_status_t validate_attention_chunk(
    uint32_t position_base, uint32_t logical_query_count,
    uint32_t* out_key_end) {
  if (logical_query_count == 0u ||
      logical_query_count > Q16K_AITER_CHUNK_CAPACITY ||
      position_base < Q16K_AITER_LEGACY_PREFIX_ROWS) {
    return Q16K_AITER_STATUS_OUT_OF_RANGE;
  }
  if (position_base > UINT32_MAX - logical_query_count) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  const uint32_t key_end = position_base + logical_query_count;
  if (key_end > Q16K_AITER_MAX_PROMPT_TOKEN_COUNT) {
    return Q16K_AITER_STATUS_OUT_OF_RANGE;
  }
  if (out_key_end != NULL) *out_key_end = key_end;
  return Q16K_AITER_STATUS_OK;
}

const char* q16k_aiter_status_string(q16k_aiter_status_t status) {
  switch (status) {
    case Q16K_AITER_STATUS_OK:
      return "ok";
    case Q16K_AITER_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case Q16K_AITER_STATUS_OUT_OF_RANGE:
      return "out of range";
    case Q16K_AITER_STATUS_BUFFER_TOO_SMALL:
      return "buffer too small";
    case Q16K_AITER_STATUS_OVERFLOW:
      return "overflow";
    case Q16K_AITER_STATUS_FEATURE_DISABLED:
      return "feature disabled";
    case Q16K_AITER_STATUS_QUEUE_FULL:
      return "queue full";
    case Q16K_AITER_STATUS_RING_EXHAUSTED:
      return "kernarg ring exhausted";
    case Q16K_AITER_STATUS_CALLBACK_FAILED:
      return "callback failed";
  }
  return "unknown status";
}

const q16k_aiter_asset_spec_t* q16k_aiter_asset_spec(
    q16k_aiter_asset_kind_t kind) {
  if ((unsigned)kind >= Q16K_AITER_ASSET_COUNT) return NULL;
  return &kAssetSpecs[kind];
}

const q16k_aiter_kernel_spec_t* q16k_dense_kernel_spec(
    q16k_dense_kernel_t kernel) {
  if ((unsigned)kernel >= Q16K_DENSE_KERNEL_COUNT) return NULL;
  return &kDenseKernelSpecs[kernel];
}

const q16k_aiter_kernel_spec_t* q16k_pack_kernel_spec(void) {
  return &kPackKernelSpec;
}

const q16k_aiter_kernel_spec_t* q16k_attention_kernel_spec(void) {
  return &kAttentionKernelSpec;
}

q16k_aiter_status_t q16k_dense_launch_geometry(
    q16k_dense_kernel_t kernel, uint32_t logical_query_count,
    q16k_aiter_launch_geometry_t* out_geometry) {
  if ((unsigned)kernel >= Q16K_DENSE_KERNEL_COUNT || out_geometry == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (logical_query_count == 0u ||
      logical_query_count > Q16K_AITER_CHUNK_CAPACITY) {
    return Q16K_AITER_STATUS_OUT_OF_RANGE;
  }

  uint64_t workgroups = 0u;
  switch (kernel) {
    case Q16K_DENSE_PREPARE_ROPE:
      workgroups = 32768u;
      break;
    case Q16K_DENSE_EMBEDDING:
      workgroups =
          ((uint64_t)logical_query_count * 384u + 255u) / 256u;
      break;
    case Q16K_DENSE_RMS_NORM:
    case Q16K_DENSE_ROPE_CACHE:
      workgroups = logical_query_count;
      break;
    case Q16K_DENSE_QKV:
    case Q16K_DENSE_OUTPUT_PROJECTION:
    case Q16K_DENSE_DOWN_PROJECTION:
      workgroups =
          (((uint64_t)logical_query_count + 15u) / 16u) * 96u;
      break;
    case Q16K_DENSE_GATE_UP:
      workgroups =
          (((uint64_t)logical_query_count + 31u) / 32u) * 560u;
      break;
    case Q16K_DENSE_KERNEL_COUNT:
      return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }

  const uint32_t workgroup = kDenseWorkgroupSizes[kernel];
  const uint64_t grid = workgroups * workgroup;
  if (workgroups == 0u || grid > UINT32_MAX) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  memset(out_geometry, 0, sizeof(*out_geometry));
  out_geometry->grid_size[0] = (uint32_t)grid;
  out_geometry->grid_size[1] = 1u;
  out_geometry->grid_size[2] = 1u;
  out_geometry->workgroup_size[0] = (uint16_t)workgroup;
  out_geometry->workgroup_size[1] = 1u;
  out_geometry->workgroup_size[2] = 1u;
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_build_dense_kernarg(
    q16k_dense_kernel_t kernel, void* const* bindings,
    size_t binding_count, void* kernarg, size_t kernarg_size) {
  const q16k_aiter_kernel_spec_t* spec = q16k_dense_kernel_spec(kernel);
  if (spec == NULL || bindings == NULL ||
      binding_count != spec->explicit_argument_count) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (!kernarg_is_valid(kernarg, kernarg_size,
                        spec->kernarg_segment_size)) {
    return kernarg == NULL ||
                   ((uintptr_t)kernarg &
                    (Q16K_AITER_KERNARG_ALIGNMENT - 1u)) != 0u
               ? Q16K_AITER_STATUS_INVALID_ARGUMENT
               : Q16K_AITER_STATUS_BUFFER_TOO_SMALL;
  }
  for (size_t i = 0u; i < binding_count; ++i) {
    if (bindings[i] == NULL) return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }

  uint8_t* bytes = (uint8_t*)kernarg;
  memset(bytes, 0, spec->kernarg_segment_size);
  for (size_t i = 0u; i < binding_count; ++i) {
    store_u64(bytes, i * sizeof(uint64_t),
              (uint64_t)(uintptr_t)bindings[i]);
  }
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_aiter_plan_chunks(
    uint32_t prompt_token_count, q16k_aiter_chunk_t* chunks,
    size_t chunk_capacity, size_t* out_chunk_count) {
  if (out_chunk_count == NULL) return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  *out_chunk_count = 0u;
  if (prompt_token_count < Q16K_AITER_LEGACY_PREFIX_ROWS ||
      prompt_token_count > Q16K_AITER_MAX_PROMPT_TOKEN_COUNT) {
    return Q16K_AITER_STATUS_OUT_OF_RANGE;
  }

  const uint32_t remaining =
      prompt_token_count - Q16K_AITER_LEGACY_PREFIX_ROWS;
  const size_t chunk_count =
      (remaining + Q16K_AITER_CHUNK_CAPACITY - 1u) /
      Q16K_AITER_CHUNK_CAPACITY;
  *out_chunk_count = chunk_count;
  if (chunk_count == 0u) return Q16K_AITER_STATUS_OK;
  if (chunks == NULL || chunk_capacity < chunk_count) {
    return Q16K_AITER_STATUS_BUFFER_TOO_SMALL;
  }

  uint32_t base = Q16K_AITER_LEGACY_PREFIX_ROWS;
  uint32_t rows_left = remaining;
  for (size_t i = 0u; i < chunk_count; ++i) {
    const uint32_t count =
        rows_left > Q16K_AITER_CHUNK_CAPACITY
            ? Q16K_AITER_CHUNK_CAPACITY
            : rows_left;
    chunks[i].position_base = base;
    chunks[i].logical_query_count = count;
    chunks[i].final = i + 1u == chunk_count ? 1u : 0u;
    base += count;
    rows_left -= count;
  }
  return rows_left == 0u && base == prompt_token_count
             ? Q16K_AITER_STATUS_OK
             : Q16K_AITER_STATUS_OVERFLOW;
}

q16k_aiter_status_t q16k_aiter_make_allocation_plan(
    q16k_aiter_allocation_plan_t* out_plan) {
  if (out_plan == NULL) return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  memset(out_plan, 0, sizeof(*out_plan));
  out_plan->scratch_alignment = Q16K_SCRATCH_ALIGNMENT;

  static const size_t kSegmentSizes[Q16K_SCRATCH_SEGMENT_COUNT] = {
      Q16K_HIDDEN_SEGMENT_BYTES, Q16K_HIDDEN_SEGMENT_BYTES,
      Q16K_HIDDEN_SEGMENT_BYTES, Q16K_HIDDEN_SEGMENT_BYTES,
      Q16K_HIDDEN_SEGMENT_BYTES, Q16K_KV_SEGMENT_BYTES,
      Q16K_KV_SEGMENT_BYTES,     Q16K_KV_SEGMENT_BYTES,
      Q16K_HIDDEN_SEGMENT_BYTES, Q16K_HIDDEN_SEGMENT_BYTES,
      Q16K_MLP_SEGMENT_BYTES,    Q16K_MLP_SEGMENT_BYTES,
      Q16K_MLP_SEGMENT_BYTES,    Q16K_LOGITS_BYTES,
      Q16K_ARGMAX_VALUES_BYTES,  Q16K_ARGMAX_IDS_BYTES,
  };
  size_t offset = 0u;
  for (size_t i = 0u; i < Q16K_SCRATCH_SEGMENT_COUNT; ++i) {
    size_t aligned = 0u;
    if (!checked_align_size(offset, Q16K_SCRATCH_ALIGNMENT, &aligned)) {
      return Q16K_AITER_STATUS_OVERFLOW;
    }
    out_plan->scratch[i].offset = aligned;
    out_plan->scratch[i].size = kSegmentSizes[i];
    if (!checked_add_size(aligned, kSegmentSizes[i], &offset)) {
      return Q16K_AITER_STATUS_OVERFLOW;
    }
  }
  out_plan->scratch_bytes = offset;

  out_plan->loom_kv_layer_stride_bytes = Q16K_LOOM_KV_LAYER_STRIDE_BYTES;
  if (!checked_mul_size(Q16K_LOOM_KV_LAYER_STRIDE_BYTES,
                        Q16K_AITER_LAYER_COUNT,
                        &out_plan->loom_kv_allocation_bytes)) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  out_plan->shadow_kv_layer_stride_bytes =
      Q16K_SHADOW_KV_LAYER_STRIDE_BYTES;
  if (!checked_mul_size(Q16K_SHADOW_KV_LAYER_STRIDE_BYTES,
                        Q16K_AITER_LAYER_COUNT,
                        &out_plan->shadow_kv_allocation_bytes) ||
      !checked_mul_size(out_plan->shadow_kv_allocation_bytes, 2u,
                        &out_plan->shadow_kv_combined_bytes) ||
      !checked_add_size(Q16K_AITER_MAX_PROMPT_TOKEN_COUNT,
                        Q16K_AITER_PAGE_INDEX_PADDING,
                        &out_plan->page_index_count) ||
      !checked_mul_size(out_plan->page_index_count, sizeof(int32_t),
                        &out_plan->page_index_bytes) ||
      !checked_mul_size(Q16K_AITER_KERNARG_STRIDE,
                        Q16K_AITER_KERNARG_RING_CAPACITY,
                        &out_plan->kernarg_ring_bytes)) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  out_plan->attention_metadata_bytes = Q16K_AITER_METADATA_SIZE;
  out_plan->rope_table_bytes = Q16K_ROPE_TABLE_BYTES;
  out_plan->token_bytes = Q16K_TOKEN_BYTES;
  out_plan->kernarg_stride = Q16K_AITER_KERNARG_STRIDE;

  if (out_plan->scratch_bytes != 1263154688u ||
      out_plan->loom_kv_allocation_bytes != 1879277568u ||
      out_plan->shadow_kv_allocation_bytes != 1879048192u ||
      out_plan->shadow_kv_combined_bytes != 3758096384u ||
      out_plan->page_index_bytes != 525308u ||
      out_plan->kernarg_ring_bytes != 33554432u) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_aiter_layer_offsets(
    uint32_t layer, size_t* out_loom_offset,
    size_t* out_shadow_offset) {
  if (layer >= Q16K_AITER_LAYER_COUNT || out_loom_offset == NULL ||
      out_shadow_offset == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (!checked_mul_size(layer, Q16K_LOOM_KV_LAYER_STRIDE_BYTES,
                        out_loom_offset) ||
      !checked_mul_size(layer, Q16K_SHADOW_KV_LAYER_STRIDE_BYTES,
                        out_shadow_offset)) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_aiter_plan_prefix_pack(
    const void* loom_k, const void* loom_v, void* shadow_k,
    void* shadow_v, q16k_aiter_pack_plan_t* out_plan) {
  if (loom_k == NULL || loom_v == NULL || shadow_k == NULL ||
      shadow_v == NULL || out_plan == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  memset(out_plan, 0, sizeof(*out_plan));
  out_plan->request = (q16k_aiter_pack_request_t){
      .loom_k = loom_k,
      .loom_v = loom_v,
      .shadow_k = shadow_k,
      .shadow_v = shadow_v,
      .layer_count = Q16K_AITER_LAYER_COUNT,
      .position_base = 0u,
      .logical_query_count = Q16K_AITER_LEGACY_PREFIX_ROWS,
      .destination_capacity = Q16K_AITER_CONTEXT_CAPACITY,
      .source_block_count = Q16K_AITER_LOOM_BLOCK_COUNT,
      .source_block_base = 0u,
  };
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_aiter_plan_layer_pack(
    uint32_t layer, uint32_t position_base,
    uint32_t logical_query_count, const void* loom_k_base,
    const void* loom_v_base, void* shadow_k_base,
    void* shadow_v_base, q16k_aiter_pack_plan_t* out_plan) {
  if (loom_k_base == NULL || loom_v_base == NULL ||
      shadow_k_base == NULL || shadow_v_base == NULL ||
      out_plan == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  uint32_t key_end = 0u;
  q16k_aiter_status_t status = validate_attention_chunk(
      position_base, logical_query_count, &key_end);
  if (status != Q16K_AITER_STATUS_OK) return status;
  (void)key_end;

  size_t loom_offset = 0u;
  size_t shadow_offset = 0u;
  status = q16k_aiter_layer_offsets(layer, &loom_offset, &shadow_offset);
  if (status != Q16K_AITER_STATUS_OK) return status;

  void* loom_k = NULL;
  void* loom_v = NULL;
  void* shadow_k = NULL;
  void* shadow_v = NULL;
  if (!checked_pointer_offset(loom_k_base, loom_offset, &loom_k) ||
      !checked_pointer_offset(loom_v_base, loom_offset, &loom_v) ||
      !checked_pointer_offset(shadow_k_base, shadow_offset, &shadow_k) ||
      !checked_pointer_offset(shadow_v_base, shadow_offset, &shadow_v)) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }

  memset(out_plan, 0, sizeof(*out_plan));
  out_plan->loom_layer_offset = loom_offset;
  out_plan->shadow_layer_offset = shadow_offset;
  out_plan->request = (q16k_aiter_pack_request_t){
      .loom_k = loom_k,
      .loom_v = loom_v,
      .shadow_k = shadow_k,
      .shadow_v = shadow_v,
      .layer_count = 1u,
      .position_base = position_base,
      .logical_query_count = logical_query_count,
      .destination_capacity = Q16K_AITER_CONTEXT_CAPACITY,
      .source_block_count = Q16K_AITER_LOOM_BLOCK_COUNT,
      .source_block_base = 0u,
  };
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_build_pack_kernarg(
    const q16k_aiter_pack_request_t* request, void* kernarg,
    size_t kernarg_size, q16k_aiter_launch_geometry_t* out_geometry) {
  if (request == NULL || request->loom_k == NULL ||
      request->loom_v == NULL || request->shadow_k == NULL ||
      request->shadow_v == NULL || request->layer_count == 0u ||
      request->layer_count > Q16K_AITER_LAYER_COUNT ||
      request->logical_query_count == 0u ||
      request->logical_query_count > Q16K_AITER_CHUNK_CAPACITY ||
      request->destination_capacity != Q16K_AITER_CONTEXT_CAPACITY ||
      request->source_block_count != Q16K_AITER_LOOM_BLOCK_COUNT ||
      request->source_block_base != 0u || out_geometry == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (!kernarg_is_valid(kernarg, kernarg_size,
                        Q16K_AITER_PACK_KERNARG_SIZE)) {
    return kernarg == NULL ||
                   ((uintptr_t)kernarg &
                    (Q16K_AITER_KERNARG_ALIGNMENT - 1u)) != 0u
               ? Q16K_AITER_STATUS_INVALID_ARGUMENT
               : Q16K_AITER_STATUS_BUFFER_TOO_SMALL;
  }
  if (request->position_base >
      UINT32_MAX - request->logical_query_count) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  const uint32_t position_end =
      request->position_base + request->logical_query_count;
  if (position_end > Q16K_AITER_MAX_PROMPT_TOKEN_COUNT) {
    return Q16K_AITER_STATUS_OUT_OF_RANGE;
  }

  uint64_t element_count = request->layer_count;
  if (element_count > UINT64_MAX / request->logical_query_count) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  element_count *= request->logical_query_count;
  if (element_count > UINT64_MAX / Q16K_AITER_KV_SIZE) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  element_count *= Q16K_AITER_KV_SIZE;
  if (element_count > UINT64_MAX - 255u) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }
  const uint64_t grid_x = (element_count + 255u) & ~UINT64_C(255);
  if (grid_x == 0u || grid_x > UINT32_MAX) {
    return Q16K_AITER_STATUS_OVERFLOW;
  }

  uint8_t* bytes = (uint8_t*)kernarg;
  memset(bytes, 0, Q16K_AITER_PACK_KERNARG_SIZE);
  store_u64(bytes, 0u, (uint64_t)(uintptr_t)request->loom_k);
  store_u64(bytes, 8u, (uint64_t)(uintptr_t)request->loom_v);
  store_u64(bytes, 16u, (uint64_t)(uintptr_t)request->shadow_k);
  store_u64(bytes, 24u, (uint64_t)(uintptr_t)request->shadow_v);
  store_u64(bytes, 32u, element_count);
  store_u64(bytes, 40u, request->logical_query_count);
  store_u64(bytes, 48u, request->destination_capacity);
  store_u64(bytes, 56u, request->source_block_count);
  store_u64(bytes, 64u, request->position_base);
  store_u64(bytes, 72u, request->source_block_base);
  store_u32(bytes, 80u, (uint32_t)(grid_x / 256u));
  store_u32(bytes, 84u, 1u);
  store_u32(bytes, 88u, 1u);
  store_u16(bytes, 92u, 256u);
  store_u16(bytes, 94u, 1u);
  store_u16(bytes, 96u, 1u);
  store_u16(bytes, 144u, 1u);

  memset(out_geometry, 0, sizeof(*out_geometry));
  out_geometry->grid_size[0] = (uint32_t)grid_x;
  out_geometry->grid_size[1] = 1u;
  out_geometry->grid_size[2] = 1u;
  out_geometry->workgroup_size[0] = 256u;
  out_geometry->workgroup_size[1] = 1u;
  out_geometry->workgroup_size[2] = 1u;
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_build_attention_metadata(
    uint32_t position_base, uint32_t logical_query_count,
    q16k_aiter_attention_metadata_t* out_metadata) {
  if (out_metadata == NULL) return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  uint32_t key_end = 0u;
  const q16k_aiter_status_t status = validate_attention_chunk(
      position_base, logical_query_count, &key_end);
  if (status != Q16K_AITER_STATUS_OK) return status;
  memset(out_metadata, 0, sizeof(*out_metadata));
  out_metadata->kv_indptr[1] = (int32_t)key_end;
  out_metadata->kv_last_page_lens[0] = 1;
  out_metadata->cu_seqlens_q[1] = (int32_t)logical_query_count;
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_fill_identity_page_indices(
    int32_t* page_indices, size_t page_index_capacity) {
  const size_t required =
      (size_t)Q16K_AITER_MAX_PROMPT_TOKEN_COUNT +
      Q16K_AITER_PAGE_INDEX_PADDING;
  if (page_indices == NULL) return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  if (page_index_capacity < required) {
    return Q16K_AITER_STATUS_BUFFER_TOO_SMALL;
  }
  for (uint32_t i = 0u; i < Q16K_AITER_MAX_PROMPT_TOKEN_COUNT; ++i) {
    page_indices[i] = (int32_t)i;
  }
  memset(page_indices + Q16K_AITER_MAX_PROMPT_TOKEN_COUNT, 0,
         (page_index_capacity - Q16K_AITER_MAX_PROMPT_TOKEN_COUNT) *
             sizeof(*page_indices));
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_build_attention_kernarg(
    const q16k_aiter_attention_request_t* request, void* kernarg,
    size_t kernarg_size, q16k_aiter_launch_geometry_t* out_geometry) {
  if (request == NULL || request->q == NULL || request->k == NULL ||
      request->v == NULL || request->output == NULL ||
      request->kv_indptr == NULL || request->kv_page_indices == NULL ||
      request->kv_last_page_lens == NULL ||
      request->cu_seqlens_q == NULL || out_geometry == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (!kernarg_is_valid(kernarg, kernarg_size,
                        Q16K_AITER_ATTENTION_KERNARG_SIZE)) {
    return kernarg == NULL ||
                   ((uintptr_t)kernarg &
                    (Q16K_AITER_KERNARG_ALIGNMENT - 1u)) != 0u
               ? Q16K_AITER_STATUS_INVALID_ARGUMENT
               : Q16K_AITER_STATUS_BUFFER_TOO_SMALL;
  }
  uint32_t key_end = 0u;
  const q16k_aiter_status_t status = validate_attention_chunk(
      request->position_base, request->logical_query_count, &key_end);
  if (status != Q16K_AITER_STATUS_OK) return status;

  uint8_t* bytes = (uint8_t*)kernarg;
  memset(bytes, 0, Q16K_AITER_ATTENTION_KERNARG_SIZE);
  store_u64(bytes, 0u, (uint64_t)(uintptr_t)request->q);
  store_u64(bytes, 8u, (uint64_t)(uintptr_t)request->k);
  store_u64(bytes, 16u, (uint64_t)(uintptr_t)request->v);
  store_u64(bytes, 24u, (uint64_t)(uintptr_t)request->output);
  store_i32(bytes, 40u, -1);
  store_i32(bytes, 44u, -1);
  store_i32(bytes, 48u, 128);
  store_i32(bytes, 52u, 128);
  store_i32(bytes, 56u, 12);
  store_i32(bytes, 60u, 6);
  store_i32(bytes, 64u, (int32_t)key_end);
  store_i32(bytes, 68u, 1);
  store_u64(bytes, 72u, (uint64_t)(uintptr_t)request->kv_indptr);
  store_u64(bytes, 80u,
            (uint64_t)(uintptr_t)request->kv_page_indices);
  store_u64(bytes, 88u,
            (uint64_t)(uintptr_t)request->kv_last_page_lens);
  store_f32(bytes, 96u, Q16K_SOFTMAX_SCALE_F * Q16K_LOG2E_F);
  store_i32(bytes, 100u, 1536);
  store_i32(bytes, 104u, 128);
  store_i32(bytes, 108u, 128);
  store_i32(bytes, 112u, 1536);
  store_i32(bytes, 116u, 128);
  store_i32(bytes, 120u, 128);
  store_i32(bytes, 124u, 128);
  store_i32(bytes, 128u, 128);
  store_i32(bytes, 132u, -1);
  store_i32(bytes, 136u, 0);
  store_i32(bytes, 140u, 0);
  store_i32(bytes, 144u, 2);
  store_u64(bytes, 152u,
            (uint64_t)(uintptr_t)request->cu_seqlens_q);
  store_i32(bytes, 160u, 256);
  store_i32(bytes, 164u, 256);
  store_u32(bytes, 168u, 12u);
  store_u32(bytes, 172u, 1u);
  store_u32(bytes, 176u, 128u);
  store_u16(bytes, 180u, 256u);
  store_u16(bytes, 182u, 1u);
  store_u16(bytes, 184u, 1u);
  store_u16(bytes, 232u, 3u);

  memset(out_geometry, 0, sizeof(*out_geometry));
  out_geometry->grid_size[0] = 3072u;
  out_geometry->grid_size[1] = 1u;
  out_geometry->grid_size[2] = 128u;
  out_geometry->workgroup_size[0] = 256u;
  out_geometry->workgroup_size[1] = 1u;
  out_geometry->workgroup_size[2] = 1u;
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_aiter_plan_layer(
    uint32_t layer, uint32_t position_base,
    uint32_t logical_query_count, const void* loom_k_base,
    const void* loom_v_base, void* shadow_k_base,
    void* shadow_v_base, const void* rotated_query,
    void* attention_output, const int32_t* kv_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    const int32_t* cu_seqlens_q,
    q16k_aiter_layer_plan_t* out_plan) {
  if (rotated_query == NULL || attention_output == NULL ||
      kv_indptr == NULL || kv_page_indices == NULL ||
      kv_last_page_lens == NULL || cu_seqlens_q == NULL ||
      out_plan == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  q16k_aiter_pack_plan_t pack_plan;
  q16k_aiter_status_t status = q16k_aiter_plan_layer_pack(
      layer, position_base, logical_query_count, loom_k_base,
      loom_v_base, shadow_k_base, shadow_v_base, &pack_plan);
  if (status != Q16K_AITER_STATUS_OK) return status;

  memset(out_plan, 0, sizeof(*out_plan));
  out_plan->layer = layer;
  out_plan->position_base = position_base;
  out_plan->logical_query_count = logical_query_count;
  out_plan->loom_layer_offset = pack_plan.loom_layer_offset;
  out_plan->shadow_layer_offset = pack_plan.shadow_layer_offset;
  out_plan->pack = pack_plan.request;
  out_plan->attention = (q16k_aiter_attention_request_t){
      .q = rotated_query,
      .k = pack_plan.request.shadow_k,
      .v = pack_plan.request.shadow_v,
      .output = attention_output,
      .kv_indptr = kv_indptr,
      .kv_page_indices = kv_page_indices,
      .kv_last_page_lens = kv_last_page_lens,
      .cu_seqlens_q = cu_seqlens_q,
      .position_base = position_base,
      .logical_query_count = logical_query_count,
  };
  return Q16K_AITER_STATUS_OK;
}

static int is_power_of_two_u32(uint32_t value) {
  return value != 0u && (value & (value - 1u)) == 0u;
}

static q16k_aiter_status_t validate_queue_state(
    const q16k_aiter_queue_state_t* queue) {
  if (queue == NULL || queue->queue == NULL || queue->packets == NULL ||
      queue->kernarg_data == NULL || !is_power_of_two_u32(queue->queue_size) ||
      queue->kernarg_stride == 0u || queue->kernarg_capacity == 0u ||
      queue->batch_dispatch == 0u ||
      queue->ops.load_write_index_relaxed == NULL ||
      queue->ops.load_read_index_scacquire == NULL ||
      queue->ops.store_write_index_screlease == NULL ||
      queue->ops.ring_doorbell_screlease == NULL ||
      queue->ops.publish_dispatch_packet == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  const uint32_t depth_limit =
      queue->queue_depth_limit != 0u &&
              queue->queue_depth_limit < queue->queue_size
          ? queue->queue_depth_limit
          : queue->queue_size;
  if (queue->doorbell_batch_size > depth_limit ||
      (queue->doorbell_batch_size == 0u &&
       (queue->deferred_queue_open != 0u ||
        queue->deferred_queue_pending_count != 0u)) ||
      (queue->deferred_queue_open == 0u &&
       queue->deferred_queue_pending_count != 0u) ||
      queue->deferred_queue_pending_count > depth_limit) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  return Q16K_AITER_STATUS_OK;
}

static void reset_queue_batch(q16k_aiter_queue_state_t* queue) {
  queue->deferred_queue_open = 0u;
  queue->deferred_queue_write_index = 0u;
  queue->deferred_queue_read_index = 0u;
  queue->deferred_queue_pending_count = 0u;
}

q16k_aiter_status_t q16k_aiter_flush_queue(
    q16k_aiter_queue_state_t* queue) {
  q16k_aiter_status_t status = validate_queue_state(queue);
  if (status != Q16K_AITER_STATUS_OK) return status;
  if (queue->deferred_queue_open == 0u) return Q16K_AITER_STATUS_OK;
  if (queue->deferred_queue_pending_count == 0u ||
      queue->deferred_queue_write_index == 0u) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  const uint64_t write_index = queue->deferred_queue_write_index;
  queue->ops.store_write_index_screlease(
      queue->user_data, queue->queue, write_index);
  queue->ops.ring_doorbell_screlease(
      queue->user_data, queue->queue, write_index - 1u);
  reset_queue_batch(queue);
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_aiter_dispatch_raw(
    q16k_aiter_queue_state_t* queue,
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_launch_geometry_t* geometry,
    const void* kernarg, size_t kernarg_size,
    uint64_t completion_signal, uint32_t force_flush,
    q16k_aiter_dispatch_result_t* out_result) {
  if (out_result != NULL) {
    *out_result = (q16k_aiter_dispatch_result_t){
        .packet_id = UINT64_MAX,
        .kernarg_slot = UINT32_MAX,
        .doorbell_written = 0u,
    };
  }
  q16k_aiter_status_t status = validate_queue_state(queue);
  if (status != Q16K_AITER_STATUS_OK) return status;
  if (kernel == NULL || geometry == NULL || kernarg == NULL ||
      kernel->name == NULL || kernel->name[0] == '\0' ||
      kernel->kernel_object == 0u || kernel->kernarg_segment_size == 0u ||
      kernel->kernarg_segment_alignment == 0u ||
      !is_power_of_two_u32(kernel->kernarg_segment_alignment) ||
      kernarg_size != kernel->kernarg_segment_size ||
      kernarg_size > queue->kernarg_stride) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  uint16_t dimensions = 1u;
  for (uint32_t i = 0u; i < 3u; ++i) {
    if (geometry->grid_size[i] == 0u ||
        geometry->workgroup_size[i] == 0u ||
        geometry->grid_size[i] < geometry->workgroup_size[i]) {
      return Q16K_AITER_STATUS_INVALID_ARGUMENT;
    }
  }
  if (geometry->grid_size[2] != 1u ||
      geometry->workgroup_size[2] != 1u) {
    dimensions = 3u;
  } else if (geometry->grid_size[1] != 1u ||
             geometry->workgroup_size[1] != 1u) {
    dimensions = 2u;
  }
  if (queue->kernarg_cursor >= queue->kernarg_capacity) {
    return Q16K_AITER_STATUS_RING_EXHAUSTED;
  }

  const uint32_t depth_limit =
      queue->queue_depth_limit != 0u &&
              queue->queue_depth_limit < queue->queue_size
          ? queue->queue_depth_limit
          : queue->queue_size;
  const uint32_t defer_submission = queue->doorbell_batch_size != 0u;
  uint64_t write_index = 0u;
  uint64_t read_index = 0u;
  if (defer_submission != 0u && queue->deferred_queue_open != 0u) {
    write_index = queue->deferred_queue_write_index;
    read_index = queue->deferred_queue_read_index;
  } else {
    write_index = queue->ops.load_write_index_relaxed(
        queue->user_data, queue->queue);
    read_index = queue->ops.load_read_index_scacquire(
        queue->user_data, queue->queue);
  }
  if (write_index - read_index >= depth_limit) {
    return Q16K_AITER_STATUS_QUEUE_FULL;
  }

  const uint32_t kernarg_slot = queue->kernarg_cursor;
  uint8_t* kernarg_address =
      (uint8_t*)queue->kernarg_data +
      (size_t)kernarg_slot * queue->kernarg_stride;
  if (((uintptr_t)kernarg_address &
       (kernel->kernarg_segment_alignment - 1u)) != 0u) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  memset(kernarg_address, 0, queue->kernarg_stride);
  memcpy(kernarg_address, kernarg, kernarg_size);

  q16k_aiter_dispatch_packet_t* packet =
      &queue->packets[write_index & (queue->queue_size - 1u)];
  memset(packet, 0, sizeof(*packet));
  packet->workgroup_size_x = geometry->workgroup_size[0];
  packet->workgroup_size_y = geometry->workgroup_size[1];
  packet->workgroup_size_z = geometry->workgroup_size[2];
  packet->grid_size_x = geometry->grid_size[0];
  packet->grid_size_y = geometry->grid_size[1];
  packet->grid_size_z = geometry->grid_size[2];
  packet->private_segment_size = kernel->private_segment_size;
  packet->group_segment_size = kernel->group_segment_size;
  packet->kernel_object = kernel->kernel_object;
  packet->kernarg_address = (uint64_t)(uintptr_t)kernarg_address;
  packet->completion_signal = completion_signal;
  queue->ops.publish_dispatch_packet(
      queue->user_data, packet, dimensions);
  queue->kernarg_cursor++;

  uint32_t doorbell_written = 0u;
  if (defer_submission != 0u) {
    if (queue->deferred_queue_open == 0u) {
      queue->deferred_queue_open = 1u;
      queue->deferred_queue_read_index = read_index;
    }
    queue->deferred_queue_write_index = write_index + 1u;
    queue->deferred_queue_pending_count++;
    if (force_flush != 0u || completion_signal != 0u ||
        queue->deferred_queue_pending_count >= queue->doorbell_batch_size) {
      queue->ops.store_write_index_screlease(
          queue->user_data, queue->queue, write_index + 1u);
      queue->ops.ring_doorbell_screlease(
          queue->user_data, queue->queue, write_index);
      reset_queue_batch(queue);
      doorbell_written = 1u;
    }
  } else {
    queue->ops.store_write_index_screlease(
        queue->user_data, queue->queue, write_index + 1u);
    queue->ops.ring_doorbell_screlease(
        queue->user_data, queue->queue, write_index);
    doorbell_written = 1u;
  }
  if (out_result != NULL) {
    out_result->packet_id = write_index;
    out_result->kernarg_slot = kernarg_slot;
    out_result->doorbell_written = doorbell_written;
  }
  return Q16K_AITER_STATUS_OK;
}

static int loaded_kernel_matches_spec(
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_kernel_spec_t* spec) {
  return kernel != NULL && spec != NULL && kernel->kernel_object != 0u &&
         kernel->kernarg_segment_size == spec->kernarg_segment_size &&
         kernel->kernarg_segment_alignment ==
             spec->kernarg_segment_alignment &&
         kernel->group_segment_size == spec->group_segment_size &&
         kernel->private_segment_size == spec->private_segment_size;
}

q16k_aiter_status_t q16k_aiter_enqueue_prefix_pack(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_pack_plan_t* prefix_plan,
    uint64_t completion_signal) {
  if (hook == NULL || prefix_plan == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (hook->enabled == 0u) return Q16K_AITER_STATUS_FEATURE_DISABLED;
  if (hook->queue == NULL ||
      !loaded_kernel_matches_spec(hook->pack_kernel,
                                  q16k_pack_kernel_spec()) ||
      prefix_plan->request.layer_count != Q16K_AITER_LAYER_COUNT ||
      prefix_plan->request.position_base != 0u ||
      prefix_plan->request.logical_query_count !=
          Q16K_AITER_LEGACY_PREFIX_ROWS ||
      prefix_plan->loom_layer_offset != 0u ||
      prefix_plan->shadow_layer_offset != 0u) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t kernarg[Q16K_AITER_PACK_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t geometry;
  q16k_aiter_status_t status = q16k_build_pack_kernarg(
      &prefix_plan->request, kernarg, sizeof(kernarg), &geometry);
  if (status != Q16K_AITER_STATUS_OK) return status;
  return q16k_aiter_dispatch_raw(
      hook->queue, hook->pack_kernel, &geometry, kernarg,
      sizeof(kernarg), completion_signal, 0u, NULL);
}

q16k_aiter_status_t q16k_aiter_enqueue_layer(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_layer_plan_t* layer_plan,
    uint64_t output_completion_signal) {
  if (hook == NULL || layer_plan == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  if (hook->enabled == 0u) return Q16K_AITER_STATUS_FEATURE_DISABLED;
  if (hook->enqueue_loom == NULL || hook->queue == NULL ||
      !loaded_kernel_matches_spec(hook->pack_kernel,
                                  q16k_pack_kernel_spec()) ||
      !loaded_kernel_matches_spec(hook->attention_kernel,
                                  q16k_attention_kernel_spec()) ||
      layer_plan->layer >= Q16K_AITER_LAYER_COUNT) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }

  q16k_aiter_status_t status = hook->enqueue_loom(
      hook->user_data, Q16K_AITER_LOOM_QKV, layer_plan->layer,
      layer_plan->position_base, layer_plan->logical_query_count, 0u);
  if (status != Q16K_AITER_STATUS_OK) return Q16K_AITER_STATUS_CALLBACK_FAILED;
  status = hook->enqueue_loom(
      hook->user_data, Q16K_AITER_LOOM_ROPE_CACHE, layer_plan->layer,
      layer_plan->position_base, layer_plan->logical_query_count, 0u);
  if (status != Q16K_AITER_STATUS_OK) return Q16K_AITER_STATUS_CALLBACK_FAILED;

  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t pack_kernarg[Q16K_AITER_PACK_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t pack_geometry;
  status = q16k_build_pack_kernarg(
      &layer_plan->pack, pack_kernarg, sizeof(pack_kernarg),
      &pack_geometry);
  if (status != Q16K_AITER_STATUS_OK) return status;
  status = q16k_aiter_dispatch_raw(
      hook->queue, hook->pack_kernel, &pack_geometry, pack_kernarg,
      sizeof(pack_kernarg), 0u, 0u, NULL);
  if (status != Q16K_AITER_STATUS_OK) return status;

  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t attention_kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t attention_geometry;
  status = q16k_build_attention_kernarg(
      &layer_plan->attention, attention_kernarg,
      sizeof(attention_kernarg), &attention_geometry);
  if (status != Q16K_AITER_STATUS_OK) return status;
  q16k_aiter_dispatch_result_t attention_dispatch;
  status = q16k_aiter_dispatch_raw(
      hook->queue, hook->attention_kernel, &attention_geometry,
      attention_kernarg, sizeof(attention_kernarg), 0u, 0u,
      hook->observe_attention != NULL ? &attention_dispatch : NULL);
  if (status != Q16K_AITER_STATUS_OK) return status;
  if (hook->observe_attention != NULL) {
    const q16k_aiter_dispatch_packet_t* published_packet =
        &hook->queue->packets[
            attention_dispatch.packet_id & (hook->queue->queue_size - 1u)];
    hook->observe_attention(
        hook->attention_observer_user_data, layer_plan,
        hook->attention_kernel, &attention_geometry, attention_kernarg,
        sizeof(attention_kernarg), published_packet, &attention_dispatch);
  }

  status = hook->enqueue_loom(
      hook->user_data, Q16K_AITER_LOOM_OUTPUT, layer_plan->layer,
      layer_plan->position_base, layer_plan->logical_query_count,
      output_completion_signal);
  return status == Q16K_AITER_STATUS_OK
             ? Q16K_AITER_STATUS_OK
             : Q16K_AITER_STATUS_CALLBACK_FAILED;
}
