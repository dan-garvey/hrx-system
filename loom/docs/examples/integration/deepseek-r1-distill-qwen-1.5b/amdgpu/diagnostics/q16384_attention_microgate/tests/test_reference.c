#define _POSIX_C_SOURCE 200809L

#include "reference.h"
#include "sha256.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

static void reference_digest(const microgate_reference_t* reference,
                             char output[65]) {
  sha256_context_t sha;
  sha256_init(&sha);
  for (size_t sample = 0u; sample < MICROGATE_SAMPLE_COUNT; ++sample) {
    for (size_t head = 0u; head < MICROGATE_Q_HEADS; ++head) {
      for (size_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
           ++dimension) {
        uint32_t bits = 0u;
        uint8_t canonical[4];
        memcpy(&bits, &reference->values[sample][head][dimension],
               sizeof(bits));
        canonical[0] = (uint8_t)(bits >> 24u);
        canonical[1] = (uint8_t)(bits >> 16u);
        canonical[2] = (uint8_t)(bits >> 8u);
        canonical[3] = (uint8_t)bits;
        sha256_update(&sha, canonical, sizeof(canonical));
      }
    }
  }
  uint8_t digest[32];
  sha256_final(&sha, digest);
  sha256_hex(digest, output);
}

static int literal_reference(uint32_t query_position, uint32_t query_head,
                             uint32_t allowed, uint32_t causal_probe_base,
                             float output[MICROGATE_HEAD_DIM]) {
  const uint32_t kv_head = query_head / MICROGATE_GQA_RATIO;
  float query[MICROGATE_HEAD_DIM];
  float* scores = (float*)malloc((size_t)allowed * sizeof(*scores));
  if (scores == NULL) return 1;
  for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
       ++dimension) {
    query[dimension] = microgate_bf16_to_float(microgate_pattern_bf16(
        query_position, query_head, dimension, MICROGATE_PATTERN_Q,
        causal_probe_base));
  }
  float maximum = -INFINITY;
  for (uint32_t position = 0u; position < allowed; ++position) {
    float dot = 0.0f;
    for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
         ++dimension) {
      const float key = microgate_bf16_to_float(microgate_pattern_bf16(
          position, kv_head, dimension, MICROGATE_PATTERN_K,
          causal_probe_base));
      dot += query[dimension] * key;
    }
    scores[position] = dot * MICROGATE_SOFTMAX_SCALE;
    if (scores[position] > maximum) maximum = scores[position];
  }
  double denominator = 0.0;
  double numerators[MICROGATE_HEAD_DIM] = {0.0};
  for (uint32_t position = 0u; position < allowed; ++position) {
    const double weight = exp((double)scores[position] - (double)maximum);
    denominator += weight;
    for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
         ++dimension) {
      const float value = microgate_bf16_to_float(microgate_pattern_bf16(
          position, kv_head, dimension, MICROGATE_PATTERN_V,
          causal_probe_base));
      numerators[dimension] += weight * (double)value;
    }
  }
  free(scores);
  for (uint32_t dimension = 0u; dimension < MICROGATE_HEAD_DIM;
       ++dimension) {
    output[dimension] = (float)(numerators[dimension] / denominator);
  }
  return 0;
}

static int cross_check_literal(const microgate_reference_t* reduced) {
  const uint32_t samples[] = {0u, MICROGATE_SAMPLE_COUNT - 1u};
  const uint32_t heads[] = {0u, MICROGATE_Q_HEADS - 1u};
  for (size_t sample_index = 0u;
       sample_index < sizeof(samples) / sizeof(samples[0]); ++sample_index) {
    const uint32_t sample = samples[sample_index];
    const uint32_t row = reduced->rows[sample];
    const uint32_t allowed = reduced->position_base + row + 1u;
    for (size_t head_index = 0u;
         head_index < sizeof(heads) / sizeof(heads[0]); ++head_index) {
      const uint32_t head = heads[head_index];
      float literal[MICROGATE_HEAD_DIM];
      CHECK(literal_reference(reduced->position_base + row, head, allowed,
                              reduced->position_base, literal) == 0);
      CHECK(memcmp(literal, reduced->values[sample][head],
                   sizeof(literal)) == 0);
    }
  }
  return 0;
}

static int test_sha256(void) {
  static const uint8_t kEmptyDigest[32] = {
      0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
      0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
      0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
      0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
  };
  static const uint8_t kAbcDigest[32] = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
      0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
      0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
      0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
  };
  static const uint8_t kMultiBlockDigest[32] = {
      0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8,
      0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39,
      0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67,
      0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1,
  };
  static const char kMultiBlockMessage[] =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  static const char kMillionADigest[] =
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
  sha256_context_t sha;
  uint8_t digest[32];

  sha256_init(&sha);
  sha256_final(&sha, digest);
  CHECK(memcmp(digest, kEmptyDigest, sizeof(digest)) == 0);

  sha256_init(&sha);
  sha256_update(&sha, "abc", 3u);
  sha256_final(&sha, digest);
  CHECK(memcmp(digest, kAbcDigest, sizeof(digest)) == 0);

  sha256_init(&sha);
  sha256_update(&sha, kMultiBlockMessage, strlen(kMultiBlockMessage));
  sha256_final(&sha, digest);
  CHECK(memcmp(digest, kMultiBlockDigest, sizeof(digest)) == 0);

  uint8_t block[1000];
  memset(block, 'a', sizeof(block));
  sha256_init(&sha);
  for (size_t i = 0u; i < 1000u; ++i) {
    sha256_update(&sha, block, sizeof(block));
  }
  sha256_final(&sha, digest);
  char digest_text[65];
  sha256_hex(digest, digest_text);
  CHECK(strcmp(digest_text, kMillionADigest) == 0);

  char path[] = "/tmp/loom-attention-sha256-XXXXXX";
  const int descriptor = mkstemp(path);
  CHECK(descriptor >= 0);
  FILE* stream = fdopen(descriptor, "wb");
  CHECK(stream != NULL);
  for (size_t i = 0u; i < 1000u; ++i) {
    CHECK(fwrite(block, 1u, sizeof(block), stream) == sizeof(block));
  }
  CHECK(fclose(stream) == 0);
  CHECK(sha256_file(path, digest_text) == 0);
  CHECK(unlink(path) == 0);
  CHECK(strcmp(digest_text, kMillionADigest) == 0);
  CHECK(sha256_file(path, digest_text) == -1);
  return 0;
}

int main(void) {
  static const char kQ16384ReferenceDigest[] =
      "c1ded1184833d9be8b2952cf38d4f672a30c84c476a1c3c775791e21bad03c0a";
  CHECK(test_sha256() == 0);

  CHECK(microgate_bf16_to_float(microgate_float_to_bf16(1.0f)) == 1.0f);
  CHECK(microgate_bf16_to_float(microgate_float_to_bf16(64.0f)) == 64.0f);
  CHECK(microgate_pattern_bf16(5u, 1u, 7u, MICROGATE_PATTERN_K, 0u) ==
        microgate_pattern_bf16(132u, 1u, 7u, MICROGATE_PATTERN_K, 0u));

  microgate_reference_t q16384;
  CHECK(!microgate_build_reference(512u, &q16384));
  CHECK(!microgate_build_reference(16384u, NULL));
  CHECK(microgate_build_reference(MICROGATE_QUERY_COUNT, &q16384));
  CHECK(q16384.position_base == 384u);
  CHECK(q16384.rows[4] == 4096u && q16384.rows[7] == 16383u);
  CHECK(q16384.zero_control_violations == 216u);
  CHECK(q16384.full_noncausal_control_violations == 24u);
  CHECK(q16384.top_left_control_violations == 12u);
  char q16384_digest[65];
  reference_digest(&q16384, q16384_digest);
  CHECK(strcmp(q16384_digest, kQ16384ReferenceDigest) == 0);
  CHECK(cross_check_literal(&q16384) == 0);
  CHECK(microgate_bf16_to_float(microgate_pattern_bf16(
            383u, 0u, 1u, MICROGATE_PATTERN_V, 384u)) == 0.0f);
  CHECK(microgate_bf16_to_float(microgate_pattern_bf16(
            384u, 0u, 1u, MICROGATE_PATTERN_V, 384u)) == 64.0f);
  CHECK(microgate_bf16_to_float(microgate_pattern_bf16(
            16383u, 0u, 2u, MICROGATE_PATTERN_V, 384u)) == 0.0f);
  CHECK(microgate_bf16_to_float(microgate_pattern_bf16(
            16384u, 0u, 2u, MICROGATE_PATTERN_V, 384u)) == 64.0f);
  for (size_t sample = 0u; sample < MICROGATE_SAMPLE_COUNT; ++sample) {
    for (size_t head = 0u; head < MICROGATE_Q_HEADS; ++head) {
      CHECK(fabsf(q16384.values[sample][head][0] - 1.0f) < 1.0e-5f);
    }
  }
  puts("reference tests: PASS");
  return 0;
}
