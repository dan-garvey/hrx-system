#ifndef LOOM_AITER_MICROGATE_SHA256_H_
#define LOOM_AITER_MICROGATE_SHA256_H_

#include <stddef.h>
#include <stdint.h>

typedef struct sha256_context_s {
  uint32_t state[8];
  uint64_t bit_count;
  uint8_t block[64];
  size_t block_size;
} sha256_context_t;

void sha256_init(sha256_context_t* context);
void sha256_update(sha256_context_t* context, const void* data, size_t size);
void sha256_final(sha256_context_t* context, uint8_t digest[32]);
void sha256_hex(const uint8_t digest[32], char output[65]);
int sha256_file(const char* path, char output[65]);

#endif  /* LOOM_AITER_MICROGATE_SHA256_H_ */
