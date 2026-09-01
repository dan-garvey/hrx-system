#include "q16k_first_attention_diagnostic.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct queue_mock_s {
  uint64_t write_index;
  uint64_t read_index;
  uint64_t doorbell;
  uint32_t loom_stage_count;
  q16k_aiter_dispatch_packet_t* packets;
  uint32_t queue_size;
} queue_mock_t;

static uint64_t mock_load_write(void* user_data, const void* queue) {
  (void)queue;
  return ((queue_mock_t*)user_data)->write_index;
}

static uint64_t mock_load_read(void* user_data, const void* queue) {
  (void)queue;
  return ((queue_mock_t*)user_data)->read_index;
}

static void mock_store_write(void* user_data, void* queue, uint64_t value) {
  (void)queue;
  ((queue_mock_t*)user_data)->write_index = value;
}

static void mock_ring_doorbell(void* user_data, void* queue,
                               uint64_t packet_id) {
  (void)queue;
  queue_mock_t* mock = (queue_mock_t*)user_data;
  mock->doorbell = packet_id;
  memset(&mock->packets[packet_id & (mock->queue_size - 1u)], 0,
         sizeof(mock->packets[0]));
}

static void mock_publish(void* user_data,
                         q16k_aiter_dispatch_packet_t* packet,
                         uint32_t full_header) {
  (void)user_data;
  packet->full_header = full_header;
}

static q16k_aiter_status_t mock_enqueue_loom(
    void* user_data, q16k_aiter_loom_stage_t stage, uint32_t layer,
    uint32_t position_base, uint32_t logical_query_count,
    uint64_t completion_signal) {
  (void)stage;
  (void)completion_signal;
  queue_mock_t* mock = (queue_mock_t*)user_data;
  if (layer != 0u || position_base != Q16K_AITER_LEGACY_PREFIX_ROWS ||
      logical_query_count != Q16K_AITER_CHUNK_CAPACITY) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  mock->loom_stage_count++;
  return Q16K_AITER_STATUS_OK;
}

static q16k_aiter_loaded_kernel_t loaded_kernel(
    const q16k_aiter_kernel_spec_t* spec, uint64_t object) {
  return (q16k_aiter_loaded_kernel_t){
      .name = spec->symbol,
      .kernel_object = object,
      .kernarg_segment_size = spec->kernarg_segment_size,
      .kernarg_segment_alignment = spec->kernarg_segment_alignment,
      .group_segment_size = spec->group_segment_size,
      .private_segment_size = spec->private_segment_size,
  };
}

static uint64_t load_u64(const uint8_t* bytes, size_t offset) {
  uint64_t value = 0u;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

static int32_t load_i32(const uint8_t* bytes, size_t offset) {
  int32_t value = 0;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__,     \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  _Static_assert(sizeof(q16k_aiter_attention_metadata_t) == 24u,
                 "attention metadata must contain six int32 values");

  q16k_aiter_attention_metadata_t metadata;
  CHECK(q16k_build_attention_metadata(
            Q16K_AITER_LEGACY_PREFIX_ROWS, Q16K_AITER_CHUNK_CAPACITY,
            &metadata) == Q16K_AITER_STATUS_OK);

  q16k_first_attention_diagnostic_t diagnostic;
  q16k_first_attention_diagnostic_begin(&diagnostic, UINT64_C(77), 1u);
  CHECK(q16k_first_attention_diagnostic_capture_host_metadata(
            &diagnostic, Q16K_AITER_LEGACY_PREFIX_ROWS,
            Q16K_AITER_CHUNK_CAPACITY, UINT64_C(0x40000000),
            &metadata) == Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK);
  CHECK(q16k_first_attention_diagnostic_capture_gpu_metadata(
            &diagnostic, &metadata) ==
        Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK);

  q16k_aiter_layer_plan_t layer_plan;
  CHECK(q16k_aiter_plan_layer(
            0u, Q16K_AITER_LEGACY_PREFIX_ROWS,
            Q16K_AITER_CHUNK_CAPACITY,
            (const void*)(uintptr_t)UINT64_C(0x10000000),
            (const void*)(uintptr_t)UINT64_C(0x20000000),
            (void*)(uintptr_t)UINT64_C(0x30000000),
            (void*)(uintptr_t)UINT64_C(0x40000000),
            (const void*)(uintptr_t)UINT64_C(0x50000000),
            (void*)(uintptr_t)UINT64_C(0x60000000),
            (const int32_t*)(uintptr_t)UINT64_C(0x70000000),
            (const int32_t*)(uintptr_t)UINT64_C(0x71000000),
            (const int32_t*)(uintptr_t)UINT64_C(0x70000008),
            (const int32_t*)(uintptr_t)UINT64_C(0x70000010),
            &layer_plan) == Q16K_AITER_STATUS_OK);

  queue_mock_t mock = {0};
  q16k_aiter_dispatch_packet_t packets[8] = {0};
  mock.packets = packets;
  mock.queue_size = 8u;
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t kernarg_ring[4u * Q16K_AITER_KERNARG_STRIDE] = {0};
  int queue_identity = 0;
  q16k_aiter_queue_state_t queue = {
      .queue = &queue_identity,
      .packets = packets,
      .queue_size = 8u,
      .kernarg_data = kernarg_ring,
      .kernarg_stride = Q16K_AITER_KERNARG_STRIDE,
      .kernarg_capacity = 4u,
      .queue_depth_limit = 6u,
      .batch_dispatch = 1u,
      .user_data = &mock,
      .ops = {
          .load_write_index_relaxed = mock_load_write,
          .load_read_index_scacquire = mock_load_read,
          .store_write_index_screlease = mock_store_write,
          .ring_doorbell_screlease = mock_ring_doorbell,
          .publish_dispatch_packet = mock_publish,
      },
  };
  const q16k_aiter_loaded_kernel_t pack =
      loaded_kernel(q16k_pack_kernel_spec(), UINT64_C(0xabc000));
  const q16k_aiter_loaded_kernel_t attention =
      loaded_kernel(q16k_attention_kernel_spec(), UINT64_C(0xdef000));
  CHECK(strlen(attention.name) < Q16K_FIRST_ATTENTION_KERNEL_NAME_CAPACITY);
  const q16k_aiter_layer_hook_t hook = {
      .enabled = 1u,
      .user_data = &mock,
      .enqueue_loom = mock_enqueue_loom,
      .attention_observer_user_data = &diagnostic,
      .observe_attention = q16k_first_attention_diagnostic_observe_attention,
      .queue = &queue,
      .pack_kernel = &pack,
      .attention_kernel = &attention,
  };

  CHECK(q16k_aiter_enqueue_layer(&hook, &layer_plan, 0u) ==
        Q16K_AITER_STATUS_OK);
  CHECK(mock.loom_stage_count == 3u);
  CHECK(diagnostic.attention_dispatch_captured == 1u);
  CHECK(diagnostic.layer == 0u);
  CHECK(diagnostic.geometry.grid_size[0] == 3072u);
  CHECK(diagnostic.geometry.grid_size[1] == 1u);
  CHECK(diagnostic.geometry.grid_size[2] == 128u);
  CHECK(diagnostic.packet.full_header == UINT32_C(0x00031502));
  CHECK(diagnostic.packet.kernel_object == UINT64_C(0xdef000));
  CHECK(diagnostic.packet.grid_size_z == 128u);
  CHECK(packets[1].full_header == 0u);
  CHECK(packets[1].kernel_object == 0u);
  CHECK(diagnostic.kernarg_size == Q16K_AITER_ATTENTION_KERNARG_SIZE);
  CHECK(diagnostic.source_kernarg_matches_ring == 1u);
  CHECK(load_i32(diagnostic.kernarg, 40u) == -1);
  CHECK(load_i32(diagnostic.kernarg, 44u) == -1);
  CHECK(load_i32(diagnostic.kernarg, 64u) == 16768);
  CHECK(load_u64(diagnostic.kernarg, 72u) == UINT64_C(0x70000000));
  CHECK(load_u64(diagnostic.kernarg, 88u) == UINT64_C(0x70000008));
  CHECK(load_u64(diagnostic.kernarg, 152u) == UINT64_C(0x70000010));

  q16k_first_attention_diagnostic_finish_request(&diagnostic, 1u);
  CHECK(q16k_first_attention_diagnostic_is_complete(&diagnostic));
  CHECK(diagnostic.host_metadata_sequence == 1u);
  CHECK(diagnostic.gpu_metadata_sequence == 2u);
  CHECK(diagnostic.attention_dispatch_sequence == 3u);
  CHECK(diagnostic.request_finished_sequence == 4u);

  const char* test_tmpdir = getenv("TEST_TMPDIR");
  char current_directory[1024];
  if (test_tmpdir == NULL || test_tmpdir[0] == '\0') {
    CHECK(getcwd(current_directory, sizeof(current_directory)) != NULL);
    test_tmpdir = current_directory;
  }
  char path[1024];
  CHECK(snprintf(path, sizeof(path), "%s/first_attention.json", test_tmpdir) >
        0);
  remove(path);
  char error[256];
  CHECK(q16k_first_attention_diagnostic_write_json_atomic(
            &diagnostic, path, error, sizeof(error)) ==
        Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK);
  FILE* file = fopen(path, "rb");
  CHECK(file != NULL);
  CHECK(fseek(file, 0, SEEK_END) == 0);
  const long length = ftell(file);
  CHECK(length > 0 && length < 65536);
  CHECK(fseek(file, 0, SEEK_SET) == 0);
  char* json = (char*)malloc((size_t)length + 1u);
  CHECK(json != NULL);
  CHECK(fread(json, 1u, (size_t)length, file) == (size_t)length);
  json[length] = '\0';
  CHECK(fclose(file) == 0);
  CHECK(strstr(json, "\"complete\": true") != NULL);
  CHECK(strstr(json, "\"full_header\": \"0x00031502\"") != NULL);
  CHECK(strstr(json, "\"key_end\": 16768") != NULL);
  CHECK(strstr(json, "\"size\": 424") != NULL);
  CHECK(q16k_first_attention_diagnostic_write_json_atomic(
            &diagnostic, path, error, sizeof(error)) ==
        Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR);
  CHECK(strstr(error, "without replacement") != NULL);
  CHECK(q16k_first_attention_diagnostic_write_json_atomic(
            &diagnostic, "relative-first-attention.json", error,
            sizeof(error)) == Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT);
  free(json);
  CHECK(remove(path) == 0);

  puts("q16k first-attention diagnostic test: ok");
  return 0;
}
