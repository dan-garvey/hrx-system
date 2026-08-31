#ifndef DEEPSEEK_Q16K_DENSE_ASM_CONTRACT_H_
#define DEEPSEEK_Q16K_DENSE_ASM_CONTRACT_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define Q16K_DENSE_ASM_LAYER_COUNT 28u
#define Q16K_DENSE_ASM_MAX_ROWS 16384u
#define Q16K_DENSE_ASM_BOUNDARY_ROWS 16383u
#define Q16K_DENSE_ASM_TAIL_ROWS 15999u
#define Q16K_DENSE_ASM_HIDDEN_SIZE 1536u
#define Q16K_DENSE_ASM_KV_SIZE 256u
#define Q16K_DENSE_ASM_QKV_SIZE 2048u
#define Q16K_DENSE_ASM_INTERMEDIATE_SIZE 8960u
#define Q16K_DENSE_ASM_GATE_UP_SIZE 17920u
#define Q16K_DENSE_ASM_TILE_ROWS 256u
#define Q16K_DENSE_ASM_TILE_COLUMNS 256u
#define Q16K_DENSE_ASM_KERNARG_BYTES 352u
#define Q16K_DENSE_ASM_METADATA_KERNARG_ALIGNMENT 4u
#define Q16K_DENSE_ASM_DESCRIPTOR_KERNARG_BYTES 0u
#define Q16K_DENSE_ASM_HSA_KERNARG_ALIGNMENT 16u
#define Q16K_DENSE_ASM_KERNARG_STRIDE 512u
#define Q16K_DENSE_ASM_KERNARG_CAPACITY 65536u
#define Q16K_DENSE_ASM_KERNARG_ARENA_BYTES ((size_t)33554432u)
#define Q16K_DENSE_ASM_GROUP_SEGMENT_BYTES 163840u
#define Q16K_DENSE_ASM_WORKGROUP_SIZE 256u
#define Q16K_DENSE_ASM_PACKETS_PER_HOOK 2u
#define Q16K_DENSE_ASM_QKV_SCATTER_KERNARG_BYTES 48u
#define Q16K_DENSE_ASM_RESIDUAL_KERNARG_BYTES 32u
#define Q16K_DENSE_ASM_SWIGLU_KERNARG_BYTES 24u

#define Q16K_DENSE_ASM_QKV_WEIGHT_LAYER_BYTES                         \
  ((size_t)Q16K_DENSE_ASM_QKV_SIZE * Q16K_DENSE_ASM_HIDDEN_SIZE *   \
   sizeof(uint16_t))
#define Q16K_DENSE_ASM_QKV_BIAS_LAYER_BYTES \
  ((size_t)Q16K_DENSE_ASM_QKV_SIZE * sizeof(uint16_t))
#define Q16K_DENSE_ASM_GATE_UP_WEIGHT_LAYER_BYTES                    \
  ((size_t)Q16K_DENSE_ASM_GATE_UP_SIZE *                            \
   Q16K_DENSE_ASM_HIDDEN_SIZE * sizeof(uint16_t))
#define Q16K_DENSE_ASM_PACKED_QKV_OUTPUT_BYTES                       \
  ((size_t)Q16K_DENSE_ASM_MAX_ROWS * Q16K_DENSE_ASM_QKV_SIZE *      \
   sizeof(uint16_t))

typedef struct q16k_dense_asm_pad8_s {
  uint64_t value;
} q16k_dense_asm_pad8_t;

typedef struct q16k_dense_asm_pad12_s {
  uint32_t value[3];
} q16k_dense_asm_pad12_t;

#pragma pack(push, 1)
typedef struct q16k_dense_asm_args_s {
  void* ptr_D;
  q16k_dense_asm_pad8_t p0;
  void* ptr_C;
  q16k_dense_asm_pad8_t p1;
  void* ptr_A;
  q16k_dense_asm_pad8_t p2;
  void* ptr_B;
  q16k_dense_asm_pad8_t p3;
  float alpha;
  q16k_dense_asm_pad12_t p4;
  float beta;
  q16k_dense_asm_pad12_t p5;
  uint32_t stride_D0;
  q16k_dense_asm_pad12_t p6;
  uint32_t stride_D1;
  q16k_dense_asm_pad12_t p7;
  uint32_t stride_C0;
  q16k_dense_asm_pad12_t p8;
  uint32_t stride_C1;
  q16k_dense_asm_pad12_t p9;
  uint32_t stride_A0;
  q16k_dense_asm_pad12_t p10;
  uint32_t stride_A1;
  q16k_dense_asm_pad12_t p11;
  uint32_t stride_B0;
  q16k_dense_asm_pad12_t p12;
  uint32_t stride_B1;
  q16k_dense_asm_pad12_t p13;
  uint32_t M;
  q16k_dense_asm_pad12_t p14;
  uint32_t N;
  q16k_dense_asm_pad12_t p15;
  uint32_t K;
  q16k_dense_asm_pad12_t p16;
  uint32_t splitk;
  q16k_dense_asm_pad12_t p17;
  uint32_t is_out_b16;
  q16k_dense_asm_pad12_t p18;
  void* ptr_Bias;
  q16k_dense_asm_pad8_t p19;
  uint32_t add_bias;
  q16k_dense_asm_pad12_t p20;
  void* ptr_semaphore;
  q16k_dense_asm_pad8_t p21;
} q16k_dense_asm_args_t;
#pragma pack(pop)

typedef struct q16k_dense_asm_geometry_s {
  uint32_t grid_x;
  uint32_t grid_y;
  uint16_t workgroup_x;
  uint8_t dimensions;
} q16k_dense_asm_geometry_t;

typedef struct q16k_dense_asm_layer_offsets_s {
  size_t qkv_weight;
  size_t qkv_bias;
  size_t gate_up_weight;
} q16k_dense_asm_layer_offsets_t;

_Static_assert(sizeof(void*) == 8u,
               "q16K dense ASM requires 64-bit pointers");
_Static_assert(sizeof(q16k_dense_asm_args_t) ==
                   Q16K_DENSE_ASM_KERNARG_BYTES,
               "q16K dense ASM kernarg size drift");
_Static_assert(offsetof(q16k_dense_asm_args_t, ptr_A) == 32u,
               "q16K dense ASM A offset drift");
_Static_assert(offsetof(q16k_dense_asm_args_t, stride_D0) == 96u,
               "q16K dense ASM stride offset drift");
_Static_assert(offsetof(q16k_dense_asm_args_t, M) == 224u,
               "q16K dense ASM M offset drift");
_Static_assert(offsetof(q16k_dense_asm_args_t, ptr_Bias) == 304u,
               "q16K dense ASM bias offset drift");
_Static_assert(offsetof(q16k_dense_asm_args_t, ptr_semaphore) == 336u,
               "q16K dense ASM semaphore offset drift");
_Static_assert(Q16K_DENSE_ASM_KERNARG_STRIDE >=
                   Q16K_DENSE_ASM_KERNARG_BYTES,
               "q16K dense ASM slot is too small");
_Static_assert(Q16K_DENSE_ASM_KERNARG_CAPACITY *
                       Q16K_DENSE_ASM_KERNARG_STRIDE ==
                   Q16K_DENSE_ASM_KERNARG_ARENA_BYTES,
               "q16K dense ASM arena geometry drift");

static inline bool q16k_dense_asm_rows_valid(uint32_t rows) {
  return rows != 0u && rows <= Q16K_DENSE_ASM_MAX_ROWS;
}

static inline bool q16k_dense_asm_geometry(
    uint32_t rows, uint32_t columns,
    q16k_dense_asm_geometry_t* out_geometry) {
  if (!q16k_dense_asm_rows_valid(rows) || columns == 0u ||
      out_geometry == NULL) {
    return false;
  }
  const uint64_t grid_x =
      ((uint64_t)columns + Q16K_DENSE_ASM_TILE_COLUMNS - 1u) /
      Q16K_DENSE_ASM_TILE_COLUMNS * Q16K_DENSE_ASM_WORKGROUP_SIZE;
  const uint64_t grid_y =
      ((uint64_t)rows + Q16K_DENSE_ASM_TILE_ROWS - 1u) /
      Q16K_DENSE_ASM_TILE_ROWS;
  if (grid_x == 0u || grid_x > UINT32_MAX || grid_y == 0u ||
      grid_y > UINT32_MAX) {
    return false;
  }
  *out_geometry = (q16k_dense_asm_geometry_t){
      .grid_x = (uint32_t)grid_x,
      .grid_y = (uint32_t)grid_y,
      .workgroup_x = Q16K_DENSE_ASM_WORKGROUP_SIZE,
      .dimensions = 2u,
  };
  return true;
}

static inline bool q16k_dense_asm_build_gemm_kernarg(
    void* slot, size_t slot_size, void* input, uint32_t input_columns,
    void* weight, void* output, void* valid_dummy, uint32_t rows,
    uint32_t columns, uint32_t reduction) {
  if (slot == NULL || slot_size < Q16K_DENSE_ASM_KERNARG_STRIDE ||
      ((uintptr_t)slot & 15u) != 0u || input == NULL || weight == NULL ||
      output == NULL || valid_dummy == NULL || input_columns == 0u ||
      columns == 0u || reduction == 0u ||
      !q16k_dense_asm_rows_valid(rows) ||
      columns > UINT32_MAX / sizeof(uint16_t) ||
      input_columns > UINT32_MAX / sizeof(uint16_t) ||
      reduction > UINT32_MAX / sizeof(uint16_t)) {
    return false;
  }
  memset(slot, 0, Q16K_DENSE_ASM_KERNARG_STRIDE);
  q16k_dense_asm_args_t args = {0};
  args.ptr_D = output;
  args.ptr_C = valid_dummy;
  args.ptr_A = input;
  args.ptr_B = weight;
  args.alpha = 1.0f;
  args.beta = 0.0f;
  args.stride_D0 = columns * sizeof(uint16_t);
  args.stride_C0 = columns * sizeof(uint16_t);
  args.stride_A0 = input_columns * sizeof(uint16_t);
  args.stride_B0 = reduction * sizeof(uint16_t);
  args.M = rows;
  args.N = columns;
  args.K = reduction;
  args.splitk = 1u;
  args.is_out_b16 = 1u;
  args.ptr_Bias = valid_dummy;
  args.ptr_semaphore = valid_dummy;
  memcpy(slot, &args, sizeof(args));
  return true;
}

static inline bool q16k_dense_asm_build_pointer_kernarg(
    void* slot, size_t slot_size, void* const* bindings,
    uint32_t binding_count, uint32_t expected_bytes) {
  if (slot == NULL || slot_size < Q16K_DENSE_ASM_KERNARG_STRIDE ||
      ((uintptr_t)slot & 15u) != 0u || bindings == NULL ||
      expected_bytes != binding_count * sizeof(uint64_t) ||
      expected_bytes > Q16K_DENSE_ASM_KERNARG_STRIDE) {
    return false;
  }
  memset(slot, 0, Q16K_DENSE_ASM_KERNARG_STRIDE);
  for (uint32_t i = 0u; i < binding_count; ++i) {
    if (bindings[i] == NULL) return false;
    const uint64_t address = (uint64_t)(uintptr_t)bindings[i];
    memcpy((uint8_t*)slot + i * sizeof(address), &address, sizeof(address));
  }
  return true;
}

static inline bool q16k_dense_asm_layer_offsets(
    uint32_t layer, q16k_dense_asm_layer_offsets_t* out_offsets) {
  if (layer >= Q16K_DENSE_ASM_LAYER_COUNT || out_offsets == NULL) {
    return false;
  }
  *out_offsets = (q16k_dense_asm_layer_offsets_t){
      .qkv_weight = (size_t)layer * Q16K_DENSE_ASM_QKV_WEIGHT_LAYER_BYTES,
      .qkv_bias = (size_t)layer * Q16K_DENSE_ASM_QKV_BIAS_LAYER_BYTES,
      .gate_up_weight =
          (size_t)layer * Q16K_DENSE_ASM_GATE_UP_WEIGHT_LAYER_BYTES,
  };
  return true;
}

#endif  // DEEPSEEK_Q16K_DENSE_ASM_CONTRACT_H_
