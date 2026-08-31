#include "sha256.h"

#include <stdio.h>
#include <string.h>

static const uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t rotate_right(uint32_t value, unsigned count) {
  return (value >> count) | (value << (32u - count));
}

static uint32_t load_be32(const uint8_t* input) {
  return ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) |
         ((uint32_t)input[2] << 8) | (uint32_t)input[3];
}

static void store_be32(uint8_t* output, uint32_t value) {
  output[0] = (uint8_t)(value >> 24);
  output[1] = (uint8_t)(value >> 16);
  output[2] = (uint8_t)(value >> 8);
  output[3] = (uint8_t)value;
}

static void transform(sha256_context_t* context, const uint8_t block[64]) {
  uint32_t words[64];
  for (size_t i = 0; i < 16u; ++i) words[i] = load_be32(block + 4u * i);
  for (size_t i = 16u; i < 64u; ++i) {
    const uint32_t s0 = rotate_right(words[i - 15u], 7u) ^
                        rotate_right(words[i - 15u], 18u) ^
                        (words[i - 15u] >> 3u);
    const uint32_t s1 = rotate_right(words[i - 2u], 17u) ^
                        rotate_right(words[i - 2u], 19u) ^
                        (words[i - 2u] >> 10u);
    words[i] = words[i - 16u] + s0 + words[i - 7u] + s1;
  }

  uint32_t a = context->state[0];
  uint32_t b = context->state[1];
  uint32_t c = context->state[2];
  uint32_t d = context->state[3];
  uint32_t e = context->state[4];
  uint32_t f = context->state[5];
  uint32_t g = context->state[6];
  uint32_t h = context->state[7];
  for (size_t i = 0; i < 64u; ++i) {
    const uint32_t sum1 = rotate_right(e, 6u) ^ rotate_right(e, 11u) ^
                          rotate_right(e, 25u);
    const uint32_t choose = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + sum1 + choose + kRoundConstants[i] + words[i];
    const uint32_t sum0 = rotate_right(a, 2u) ^ rotate_right(a, 13u) ^
                          rotate_right(a, 22u);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  context->state[0] += a;
  context->state[1] += b;
  context->state[2] += c;
  context->state[3] += d;
  context->state[4] += e;
  context->state[5] += f;
  context->state[6] += g;
  context->state[7] += h;
}

void sha256_init(sha256_context_t* context) {
  static const uint32_t initial[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  memcpy(context->state, initial, sizeof(initial));
  context->bit_count = 0u;
  context->block_size = 0u;
}

void sha256_update(sha256_context_t* context, const void* data, size_t size) {
  const uint8_t* input = (const uint8_t*)data;
  context->bit_count += (uint64_t)size * 8u;
  while (size != 0u) {
    size_t available = sizeof(context->block) - context->block_size;
    size_t count = size < available ? size : available;
    memcpy(context->block + context->block_size, input, count);
    context->block_size += count;
    input += count;
    size -= count;
    if (context->block_size == sizeof(context->block)) {
      transform(context, context->block);
      context->block_size = 0u;
    }
  }
}

void sha256_final(sha256_context_t* context, uint8_t digest[32]) {
  context->block[context->block_size++] = 0x80u;
  if (context->block_size > 56u) {
    memset(context->block + context->block_size, 0,
           sizeof(context->block) - context->block_size);
    transform(context, context->block);
    context->block_size = 0u;
  }
  memset(context->block + context->block_size, 0, 56u - context->block_size);
  for (size_t i = 0; i < 8u; ++i) {
    context->block[63u - i] = (uint8_t)(context->bit_count >> (8u * i));
  }
  transform(context, context->block);
  for (size_t i = 0; i < 8u; ++i) store_be32(digest + 4u * i, context->state[i]);
  memset(context, 0, sizeof(*context));
}

void sha256_hex(const uint8_t digest[32], char output[65]) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < 32u; ++i) {
    output[2u * i] = hex[digest[i] >> 4];
    output[2u * i + 1u] = hex[digest[i] & 15u];
  }
  output[64] = '\0';
}

int sha256_file(const char* path, char output[65]) {
  FILE* stream = fopen(path, "rb");
  if (stream == NULL) return -1;
  sha256_context_t context;
  sha256_init(&context);
  uint8_t buffer[65536];
  while (!feof(stream)) {
    size_t count = fread(buffer, 1u, sizeof(buffer), stream);
    if (count != 0u) sha256_update(&context, buffer, count);
    if (ferror(stream)) {
      fclose(stream);
      return -1;
    }
  }
  if (fclose(stream) != 0) return -1;
  uint8_t digest[32];
  sha256_final(&context, digest);
  sha256_hex(digest, output);
  return 0;
}
