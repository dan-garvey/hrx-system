#include "reference.h"

#include <math.h>
#include <string.h>

enum {
  kPatternModulus = 127,
  kPatternDenominator = 256,
  kValueBaselineDimension = 0,
  kValueCausalProbeDimension = 1,
  kValueTopLeftProbeDimension = 2,
  kValueBaseline = 1,
  kValueCausalProbe = 64,
};

uint16_t microgate_float_to_bf16(float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16u) & 1u;
  bits += 0x7fffu + lsb;
  return (uint16_t)(bits >> 16u);
}

float microgate_bf16_to_float(uint16_t value) {
  uint32_t bits = (uint32_t)value << 16u;
  float result = 0.0f;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t microgate_pattern_bf16(uint32_t position, uint32_t head,
                                uint32_t dimension,
                                microgate_pattern_kind_t kind,
                                uint32_t causal_probe_base) {
  if (kind == MICROGATE_PATTERN_V &&
      dimension == kValueBaselineDimension) {
    return microgate_float_to_bf16((float)kValueBaseline);
  }
  if (kind == MICROGATE_PATTERN_V &&
      dimension == kValueCausalProbeDimension) {
    return microgate_float_to_bf16(
        position >= causal_probe_base ? (float)kValueCausalProbe : 0.0f);
  }
  if (kind == MICROGATE_PATTERN_V &&
      dimension == kValueTopLeftProbeDimension) {
    return microgate_float_to_bf16(
        position >= MICROGATE_QUERY_COUNT ? (float)kValueCausalProbe : 0.0f);
  }
  const uint32_t code =
      (position * 17u + head * 37u + dimension * 43u +
       (uint32_t)kind * 53u) %
      kPatternModulus;
  const float value = (float)((int32_t)code - 63) /
                      (float)kPatternDenominator;
  return microgate_float_to_bf16(value);
}

void microgate_fill_q(uint16_t* output, uint32_t query_count,
                      uint32_t position_base) {
  for (uint32_t row = 0u; row < query_count; ++row) {
    for (uint32_t head = 0u; head < MICROGATE_Q_HEADS; ++head) {
      for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
           ++dimension) {
        const size_t index =
            ((size_t)row * MICROGATE_Q_HEADS + head) *
                MICROGATE_HEAD_DIM +
            dimension;
        output[index] = microgate_pattern_bf16(
            position_base + row, head, dimension, MICROGATE_PATTERN_Q,
            position_base);
      }
    }
  }
}

void microgate_fill_kv(uint16_t* key, uint16_t* value,
                       uint32_t causal_probe_base) {
  for (uint32_t position = 0u; position < MICROGATE_SHADOW_CAPACITY;
       ++position) {
    for (uint32_t head = 0u; head < MICROGATE_KV_HEADS; ++head) {
      for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
           ++dimension) {
        const size_t index =
            ((size_t)position * MICROGATE_KV_HEADS + head) *
                MICROGATE_HEAD_DIM +
            dimension;
        key[index] = microgate_pattern_bf16(
            position, head, dimension, MICROGATE_PATTERN_K,
            causal_probe_base);
        value[index] = microgate_pattern_bf16(
            position, head, dimension, MICROGATE_PATTERN_V,
            causal_probe_base);
      }
    }
  }
}

static uint32_t residue_count(uint32_t length, uint32_t residue) {
  const uint32_t quotient = length / kPatternModulus;
  const uint32_t remainder = length % kPatternModulus;
  return quotient + (residue < remainder ? 1u : 0u);
}

static void reference_vector(uint32_t query_position, uint32_t query_head,
                             uint32_t allowed, uint32_t causal_probe_base,
                             float output[MICROGATE_HEAD_DIM]) {
  const uint32_t kv_head = query_head / MICROGATE_GQA_RATIO;
  float query[MICROGATE_HEAD_DIM];
  float scores[kPatternModulus];
  float maximum = -INFINITY;
  for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
       ++dimension) {
    query[dimension] = microgate_bf16_to_float(microgate_pattern_bf16(
        query_position, query_head, dimension, MICROGATE_PATTERN_Q,
        causal_probe_base));
  }
  for (uint32_t residue = 0u; residue < kPatternModulus; ++residue) {
    float dot = 0.0f;
    for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
         ++dimension) {
      const float key = microgate_bf16_to_float(microgate_pattern_bf16(
          residue, kv_head, dimension, MICROGATE_PATTERN_K,
          causal_probe_base));
      dot += query[dimension] * key;
    }
    scores[residue] = dot * MICROGATE_SOFTMAX_SCALE;
    if (scores[residue] > maximum) maximum = scores[residue];
  }

  double denominator = 0.0;
  double numerators[MICROGATE_HEAD_DIM];
  memset(numerators, 0, sizeof(numerators));
  for (uint32_t residue = 0u; residue < kPatternModulus; ++residue) {
    const uint32_t count = residue_count(allowed, residue);
    if (count == 0u) continue;
    const double weight = exp((double)scores[residue] - (double)maximum);
    denominator += (double)count * weight;
    numerators[kValueBaselineDimension] += (double)count * weight;
    uint32_t step_count = 0u;
    if (allowed > causal_probe_base) {
      step_count = residue_count(allowed, residue) -
                   residue_count(causal_probe_base, residue);
    }
    numerators[kValueCausalProbeDimension] +=
        (double)step_count * weight * (double)kValueCausalProbe;
    uint32_t top_left_count = 0u;
    if (allowed > MICROGATE_QUERY_COUNT) {
      top_left_count = residue_count(allowed, residue) -
                       residue_count(MICROGATE_QUERY_COUNT, residue);
    }
    numerators[kValueTopLeftProbeDimension] +=
        (double)top_left_count * weight * (double)kValueCausalProbe;
    for (uint32_t dimension = 3u; dimension < MICROGATE_HEAD_DIM;
         ++dimension) {
      const float value = microgate_bf16_to_float(microgate_pattern_bf16(
          residue, kv_head, dimension, MICROGATE_PATTERN_V,
          causal_probe_base));
      numerators[dimension] += (double)count * weight * (double)value;
    }
  }
  for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
       ++dimension) {
    output[dimension] = (float)(numerators[dimension] / denominator);
  }
}

static void record_control(const float wrong[MICROGATE_HEAD_DIM],
                           const float reference[MICROGATE_HEAD_DIM],
                           uint64_t* violations, double* max_difference) {
  for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
       ++dimension) {
    const double difference = fabs((double)wrong[dimension] -
                                   (double)reference[dimension]);
    const double tolerance = MICROGATE_ABSOLUTE_TOLERANCE +
                             MICROGATE_RELATIVE_TOLERANCE *
                                 fabs((double)reference[dimension]);
    if (difference > tolerance) ++*violations;
    if (difference > *max_difference) *max_difference = difference;
  }
}

bool microgate_build_reference(uint32_t query_count,
                               microgate_reference_t* output) {
  if (output == NULL || query_count != MICROGATE_QUERY_COUNT) {
    return false;
  }
  memset(output, 0, sizeof(*output));
  output->query_count = query_count;
  output->position_base = MICROGATE_POSITION_BASE;
  const uint32_t candidates[MICROGATE_SAMPLE_COUNT] = {
      0u, 1u, 7u, 31u, query_count / 4u, query_count / 2u,
      query_count - 2u, query_count - 1u,
  };
  memcpy(output->rows, candidates, sizeof(candidates));

  float zero[MICROGATE_HEAD_DIM] = {0.0f};
  for (uint32_t sample = 0u; sample < MICROGATE_SAMPLE_COUNT; ++sample) {
    const uint32_t row = output->rows[sample];
    const uint32_t allowed = output->position_base + row + 1u;
    for (uint32_t head = 0u; head < MICROGATE_Q_HEADS; ++head) {
      float* reference = output->values[sample][head];
      reference_vector(output->position_base + row, head, allowed,
                       output->position_base, reference);
      for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
           ++dimension) {
        if (!isfinite(reference[dimension])) return false;
      }
      record_control(zero, reference, &output->zero_control_violations,
                     &output->zero_control_max_difference);
      if (row == 0u) {
        float noncausal[MICROGATE_HEAD_DIM];
        reference_vector(output->position_base, head,
                         MICROGATE_KEY_END, output->position_base,
                         noncausal);
        record_control(noncausal, reference,
                       &output->full_noncausal_control_violations,
                       &output->full_noncausal_control_max_difference);
      }
      if (row == query_count - 1u) {
        float top_left[MICROGATE_HEAD_DIM];
        reference_vector(MICROGATE_KEY_END - 1u, head, query_count,
                         output->position_base, top_left);
        record_control(top_left, reference,
                       &output->top_left_control_violations,
                       &output->top_left_control_max_difference);
      }
    }
  }
  return output->zero_control_violations != 0u &&
         output->full_noncausal_control_violations != 0u &&
         output->top_left_control_violations != 0u;
}
