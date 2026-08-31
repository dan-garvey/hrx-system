#ifndef LOOM_AITER_MICROGATE_REFERENCE_H_
#define LOOM_AITER_MICROGATE_REFERENCE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
  MICROGATE_Q_HEADS = 12,
  MICROGATE_KV_HEADS = 2,
  MICROGATE_GQA_RATIO = 6,
  MICROGATE_HEAD_DIM = 128,
  MICROGATE_POSITION_BASE = 384,
  MICROGATE_QUERY_COUNT = 16384,
  MICROGATE_KEY_END = MICROGATE_POSITION_BASE + MICROGATE_QUERY_COUNT,
  MICROGATE_SHADOW_CAPACITY = 131072,
  MICROGATE_PAGE_PADDING = 256,
  MICROGATE_SAMPLE_COUNT = 8,
};

#define MICROGATE_ABSOLUTE_TOLERANCE 0.0625f
#define MICROGATE_RELATIVE_TOLERANCE 0.015f
#define MICROGATE_SOFTMAX_SCALE 0.08838834764831844055f

typedef enum microgate_pattern_kind_e {
  MICROGATE_PATTERN_K = 0,
  MICROGATE_PATTERN_V = 1,
  MICROGATE_PATTERN_Q = 2,
} microgate_pattern_kind_t;

typedef struct microgate_reference_s {
  uint32_t query_count;
  uint32_t position_base;
  uint32_t rows[MICROGATE_SAMPLE_COUNT];
  float values[MICROGATE_SAMPLE_COUNT][MICROGATE_Q_HEADS]
              [MICROGATE_HEAD_DIM];
  uint64_t zero_control_violations;
  uint64_t full_noncausal_control_violations;
  uint64_t top_left_control_violations;
  double zero_control_max_difference;
  double full_noncausal_control_max_difference;
  double top_left_control_max_difference;
} microgate_reference_t;

uint16_t microgate_float_to_bf16(float value);
float microgate_bf16_to_float(uint16_t value);
uint16_t microgate_pattern_bf16(uint32_t position, uint32_t head,
                                uint32_t dimension,
                                microgate_pattern_kind_t kind,
                                uint32_t causal_probe_base);
void microgate_fill_q(uint16_t* output, uint32_t query_count,
                      uint32_t position_base);
void microgate_fill_kv(uint16_t* key, uint16_t* value,
                       uint32_t causal_probe_base);
bool microgate_build_reference(uint32_t query_count,
                               microgate_reference_t* output);

#endif  /* LOOM_AITER_MICROGATE_REFERENCE_H_ */
