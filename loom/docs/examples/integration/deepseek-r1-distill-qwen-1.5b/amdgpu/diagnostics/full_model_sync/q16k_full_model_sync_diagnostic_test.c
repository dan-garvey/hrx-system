#include "q16k_full_model_sync_diagnostic.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef enum event_e {
  EVENT_QKV = 1,
  EVENT_ROPE,
  EVENT_ARM_PACK,
  EVENT_PUBLISH_PACK,
  EVENT_STORE_PACK,
  EVENT_DOORBELL_PACK,
  EVENT_CURSOR_LIVE_PACK,
  EVENT_WAIT_PACK,
  EVENT_CURSOR_RECLAIM_PACK,
  EVENT_METADATA_POST_PACK,
  EVENT_PRE_ATTENTION_COPY,
  EVENT_ARM_ATTENTION,
  EVENT_PUBLISH_ATTENTION,
  EVENT_STORE_ATTENTION,
  EVENT_DOORBELL_ATTENTION,
  EVENT_CURSOR_LIVE_ATTENTION,
  EVENT_WAIT_ATTENTION,
  EVENT_CURSOR_RECLAIM_ATTENTION,
  EVENT_METADATA_POST_ATTENTION,
  EVENT_POST_ATTENTION,
  EVENT_OUTPUT,
} event_t;

#define MAX_EVENTS 64u

typedef struct mock_s {
  uint64_t write_index;
  uint64_t read_index;
  uint64_t signal_value;
  uint64_t now_ns;
  uint32_t global_cursor;
  uint32_t store_count;
  uint32_t doorbell_count;
  uint32_t publish_count;
  uint32_t arm_count;
  uint32_t wait_count;
  uint32_t post_attention_count;
  uint32_t copy_wait_count;
  uint32_t metadata_copy_count;
  size_t pre_attention_copied_bytes;
  size_t post_attention_copied_bytes;
  q16k_full_model_sync_stage_t current_stage;
  q16k_full_model_sync_stage_t fail_arm_stage;
  q16k_full_model_sync_stage_t fail_wait_stage;
  uint32_t fail_arm;
  uint32_t fail_wait;
  uint32_t fail_pre_attention;
  uint32_t fail_post_attention;
  uint32_t complete_during_publish;
  uint32_t invalidate_ring_on_doorbell;
  q16k_aiter_dispatch_packet_t* packets;
  uint32_t queue_size;
  uint32_t valid;
  event_t events[MAX_EVENTS];
  uint32_t event_count;
} mock_t;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__,     \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static void push_event(mock_t* mock, event_t event) {
  if (mock->event_count >= MAX_EVENTS) {
    mock->valid = 0u;
    return;
  }
  mock->events[mock->event_count++] = event;
}

static event_t stage_event(q16k_full_model_sync_stage_t stage,
                           event_t pack, event_t attention) {
  return stage == Q16K_FULL_MODEL_SYNC_STAGE_PACK ? pack : attention;
}

static uint64_t mock_load_write(void* user_data, const void* queue) {
  (void)queue;
  return ((mock_t*)user_data)->write_index;
}

static uint64_t mock_load_read(void* user_data, const void* queue) {
  (void)queue;
  return ((mock_t*)user_data)->read_index;
}

static void mock_store_write(void* user_data, void* queue, uint64_t value) {
  (void)queue;
  mock_t* mock = (mock_t*)user_data;
  mock->write_index = value;
  mock->store_count++;
  push_event(mock, stage_event(mock->current_stage, EVENT_STORE_PACK,
                               EVENT_STORE_ATTENTION));
}

static void mock_ring_doorbell(void* user_data, void* queue,
                               uint64_t packet_id) {
  (void)queue;
  mock_t* mock = (mock_t*)user_data;
  if (packet_id + 1u != mock->write_index) mock->valid = 0u;
  mock->doorbell_count++;
  push_event(mock, stage_event(mock->current_stage, EVENT_DOORBELL_PACK,
                               EVENT_DOORBELL_ATTENTION));
  if (mock->invalidate_ring_on_doorbell != 0u) {
    memset(&mock->packets[packet_id & (mock->queue_size - 1u)], 0,
           sizeof(mock->packets[0]));
  }
}

static void mock_publish(void* user_data,
                         q16k_aiter_dispatch_packet_t* packet,
                         uint32_t full_header) {
  mock_t* mock = (mock_t*)user_data;
  const uint32_t dimensions = (full_header >> 16) & UINT32_C(0x3);
  if (mock->signal_value != 1u || packet->completion_signal == 0u ||
      dimensions == 0u || dimensions > 3u ||
      (full_header & UINT32_C(0xffff)) != UINT32_C(0x1502)) {
    mock->valid = 0u;
  }
  packet->full_header = full_header;
  mock->publish_count++;
  push_event(mock, stage_event(mock->current_stage, EVENT_PUBLISH_PACK,
                               EVENT_PUBLISH_ATTENTION));
  if (mock->complete_during_publish != 0u) mock->signal_value = 0u;
}

static q16k_aiter_status_t mock_enqueue_loom(
    void* user_data, q16k_aiter_loom_stage_t stage, uint32_t layer,
    uint32_t position_base, uint32_t logical_query_count,
    uint64_t completion_signal) {
  mock_t* mock = (mock_t*)user_data;
  if (layer != 0u || position_base != Q16K_AITER_LEGACY_PREFIX_ROWS ||
      logical_query_count != Q16K_AITER_CHUNK_CAPACITY ||
      completion_signal != 0u) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }
  switch (stage) {
    case Q16K_AITER_LOOM_QKV:
      push_event(mock, EVENT_QKV);
      break;
    case Q16K_AITER_LOOM_ROPE_CACHE:
      push_event(mock, EVENT_ROPE);
      break;
    case Q16K_AITER_LOOM_OUTPUT:
      push_event(mock, EVENT_OUTPUT);
      break;
  }
  return Q16K_AITER_STATUS_OK;
}

static uint64_t mock_now_ns(void* user_data) {
  mock_t* mock = (mock_t*)user_data;
  return ++mock->now_ns;
}

static q16k_aiter_status_t mock_arm(
    void* user_data, q16k_full_model_sync_stage_t stage, uint32_t layer,
    uint64_t completion_signal) {
  mock_t* mock = (mock_t*)user_data;
  if (layer != 0u || completion_signal == 0u || mock->signal_value != 0u) {
    mock->valid = 0u;
  }
  mock->current_stage = stage;
  mock->arm_count++;
  push_event(mock, stage_event(stage, EVENT_ARM_PACK, EVENT_ARM_ATTENTION));
  if (mock->fail_arm != 0u && stage == mock->fail_arm_stage) {
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }
  mock->signal_value = 1u;
  return Q16K_AITER_STATUS_OK;
}

static q16k_aiter_status_t mock_wait(
    void* user_data, q16k_full_model_sync_stage_t stage, uint32_t layer,
    uint64_t completion_signal) {
  mock_t* mock = (mock_t*)user_data;
  if (layer != 0u || completion_signal == 0u ||
      (mock->signal_value != 1u && mock->signal_value != 0u)) {
    mock->valid = 0u;
  }
  mock->wait_count++;
  push_event(mock, stage_event(stage, EVENT_WAIT_PACK, EVENT_WAIT_ATTENTION));
  if (mock->fail_wait != 0u && stage == mock->fail_wait_stage) {
    mock->signal_value = 0u;
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }
  mock->signal_value = 0u;
  mock->read_index = mock->write_index;
  return Q16K_AITER_STATUS_OK;
}

static void mock_synchronize_cursor(
    void* user_data, const q16k_aiter_queue_state_t* queue,
    uint32_t reclaimed) {
  mock_t* mock = (mock_t*)user_data;
  mock->global_cursor = queue->kernarg_cursor;
  push_event(
      mock,
      stage_event(mock->current_stage,
                  reclaimed != 0u ? EVENT_CURSOR_RECLAIM_PACK
                                  : EVENT_CURSOR_LIVE_PACK,
                  reclaimed != 0u ? EVENT_CURSOR_RECLAIM_ATTENTION
                                  : EVENT_CURSOR_LIVE_ATTENTION));
  if ((reclaimed != 0u) != (queue->kernarg_cursor == 0u)) {
    mock->valid = 0u;
  }
}

static q16k_aiter_status_t mock_pre_attention(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan,
    const void* pack_kernarg, size_t pack_kernarg_size,
    const void* attention_kernarg, size_t attention_kernarg_size) {
  mock_t* mock = (mock_t*)user_data;
  if (layer_plan->layer != 0u || mock->signal_value != 0u ||
      mock->global_cursor != 0u || pack_kernarg == NULL ||
      pack_kernarg_size != Q16K_AITER_PACK_KERNARG_SIZE ||
      attention_kernarg == NULL ||
      attention_kernarg_size != Q16K_AITER_ATTENTION_KERNARG_SIZE) {
    mock->valid = 0u;
  }
  mock->metadata_copy_count++;
  mock->copy_wait_count += 2u;
  mock->pre_attention_copied_bytes =
      Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT;
  push_event(mock, EVENT_METADATA_POST_PACK);
  push_event(mock, EVENT_PRE_ATTENTION_COPY);
  return mock->fail_pre_attention != 0u
             ? Q16K_AITER_STATUS_CALLBACK_FAILED
             : Q16K_AITER_STATUS_OK;
}

static q16k_aiter_status_t mock_post_attention(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan) {
  mock_t* mock = (mock_t*)user_data;
  if (layer_plan->layer != 0u || mock->signal_value != 0u ||
      mock->global_cursor != 0u) {
    mock->valid = 0u;
  }
  mock->post_attention_count++;
  mock->metadata_copy_count++;
  mock->copy_wait_count += 2u;
  mock->post_attention_copied_bytes =
      (size_t)layer_plan->logical_query_count * Q16K_AITER_HIDDEN_SIZE *
      sizeof(uint16_t);
  push_event(mock, EVENT_METADATA_POST_ATTENTION);
  push_event(mock, EVENT_POST_ATTENTION);
  if (mock->fail_post_attention != 0u) {
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }
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

typedef struct fixture_s {
  mock_t mock;
  q16k_aiter_dispatch_packet_t packets[8];
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t kernargs[8u * Q16K_AITER_KERNARG_STRIDE];
  int queue_identity;
  q16k_aiter_queue_state_t queue;
  q16k_aiter_loaded_kernel_t pack;
  q16k_aiter_loaded_kernel_t attention;
  q16k_aiter_layer_hook_t hook;
  q16k_full_model_sync_ops_t ops;
  q16k_aiter_layer_plan_t plan;
  q16k_full_model_sync_layer_record_t record;
} fixture_t;

static int initialize_fixture(fixture_t* fixture) {
  memset(fixture, 0, sizeof(*fixture));
  fixture->mock.valid = 1u;
  fixture->mock.packets = fixture->packets;
  fixture->mock.queue_size = 8u;
  fixture->mock.fail_arm_stage = Q16K_FULL_MODEL_SYNC_STAGE_ATTENTION;
  fixture->mock.fail_wait_stage = Q16K_FULL_MODEL_SYNC_STAGE_ATTENTION;
  fixture->queue = (q16k_aiter_queue_state_t){
      .queue = &fixture->queue_identity,
      .packets = fixture->packets,
      .queue_size = 8u,
      .kernarg_data = fixture->kernargs,
      .kernarg_stride = Q16K_AITER_KERNARG_STRIDE,
      .kernarg_capacity = 8u,
      .queue_depth_limit = 6u,
      .doorbell_batch_size = 6u,
      .batch_dispatch = 1u,
      .user_data = &fixture->mock,
      .ops = {
          .load_write_index_relaxed = mock_load_write,
          .load_read_index_scacquire = mock_load_read,
          .store_write_index_screlease = mock_store_write,
          .ring_doorbell_screlease = mock_ring_doorbell,
          .publish_dispatch_packet = mock_publish,
      },
  };
  fixture->pack =
      loaded_kernel(q16k_pack_kernel_spec(), UINT64_C(0xabc000));
  fixture->attention =
      loaded_kernel(q16k_attention_kernel_spec(), UINT64_C(0xdef000));
  fixture->hook = (q16k_aiter_layer_hook_t){
      .enabled = 1u,
      .user_data = &fixture->mock,
      .enqueue_loom = mock_enqueue_loom,
      .queue = &fixture->queue,
      .pack_kernel = &fixture->pack,
      .attention_kernel = &fixture->attention,
  };
  fixture->ops = (q16k_full_model_sync_ops_t){
      .user_data = &fixture->mock,
      .now_ns = mock_now_ns,
      .arm = mock_arm,
      .wait = mock_wait,
      .synchronize_cursor = mock_synchronize_cursor,
      .pre_attention_publish = mock_pre_attention,
      .post_attention_wait = mock_post_attention,
  };
  if (q16k_aiter_plan_layer(
          0u, Q16K_AITER_LEGACY_PREFIX_ROWS, Q16K_AITER_CHUNK_CAPACITY,
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
          &fixture->plan) != Q16K_AITER_STATUS_OK) {
    return 0;
  }
  fixture->record.chunk_ordinal = 0u;
  fixture->record.position_base = Q16K_AITER_LEGACY_PREFIX_ROWS;
  fixture->record.logical_query_count = Q16K_AITER_CHUNK_CAPACITY;
  fixture->record.layer = 0u;
  return 1;
}

static int expect_events(const mock_t* mock, const event_t* expected,
                         size_t expected_count) {
  if (mock->event_count != expected_count) return 0;
  return memcmp(mock->events, expected, expected_count * sizeof(expected[0])) ==
         0;
}

static int test_success_order_cursor_lifetime_and_consumed_ring_race(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.queue.kernarg_cursor = 2u;
  fixture.mock.global_cursor = 2u;
  fixture.mock.complete_during_publish = 1u;
  fixture.mock.invalidate_ring_on_doorbell = 1u;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_OK);
  const event_t expected[] = {
      EVENT_QKV,
      EVENT_ROPE,
      EVENT_ARM_PACK,
      EVENT_PUBLISH_PACK,
      EVENT_STORE_PACK,
      EVENT_DOORBELL_PACK,
      EVENT_CURSOR_LIVE_PACK,
      EVENT_WAIT_PACK,
      EVENT_CURSOR_RECLAIM_PACK,
      EVENT_METADATA_POST_PACK,
      EVENT_PRE_ATTENTION_COPY,
      EVENT_ARM_ATTENTION,
      EVENT_PUBLISH_ATTENTION,
      EVENT_STORE_ATTENTION,
      EVENT_DOORBELL_ATTENTION,
      EVENT_CURSOR_LIVE_ATTENTION,
      EVENT_WAIT_ATTENTION,
      EVENT_CURSOR_RECLAIM_ATTENTION,
      EVENT_METADATA_POST_ATTENTION,
      EVENT_POST_ATTENTION,
      EVENT_OUTPUT,
  };
  CHECK(expect_events(&fixture.mock, expected,
                      sizeof(expected) / sizeof(expected[0])));
  CHECK(fixture.mock.valid);
  CHECK(fixture.mock.arm_count == 2u);
  CHECK(fixture.mock.publish_count == 2u);
  CHECK(fixture.mock.store_count == 2u);
  CHECK(fixture.mock.doorbell_count == 2u);
  CHECK(fixture.mock.wait_count == 2u);
  CHECK(fixture.mock.copy_wait_count == 4u);
  CHECK(fixture.mock.metadata_copy_count == 2u);
  CHECK(fixture.mock.pre_attention_copied_bytes == UINT64_C(50331648));
  CHECK(fixture.mock.post_attention_copied_bytes == UINT64_C(50331648));
  CHECK(fixture.queue.kernarg_cursor == 0u);
  CHECK(fixture.mock.global_cursor == 0u);
  CHECK(fixture.packets[0].full_header == 0u);
  CHECK(fixture.packets[1].full_header == 0u);
  CHECK(fixture.record.pack.dispatch.packet_snapshot.full_header ==
        UINT32_C(0x00011502));
  CHECK(fixture.record.attention.dispatch.packet_snapshot.full_header ==
        UINT32_C(0x00031502));
  CHECK(fixture.record.pack.dispatch.packet_snapshot.completion_signal ==
        UINT64_C(0x5151));
  CHECK(fixture.record.attention.dispatch.packet_snapshot.completion_signal ==
        UINT64_C(0x5151));
  CHECK(fixture.record.pack.dispatch.kernarg_slot == 2u);
  CHECK(fixture.record.attention.dispatch.kernarg_slot == 0u);
  CHECK(fixture.record.pack.wait_acquired == 1u);
  CHECK(fixture.record.attention.wait_acquired == 1u);
  CHECK(fixture.record.output_enqueued == 1u);
  return 0;
}

static int test_pre_attention_failure_stops_attention_publish(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.fail_pre_attention = 1u;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_CALLBACK_FAILED);
  CHECK(fixture.mock.wait_count == 1u);
  CHECK(fixture.mock.publish_count == 1u);
  CHECK(fixture.mock.metadata_copy_count == 1u);
  CHECK(fixture.record.pack.wait_acquired == 1u);
  CHECK(fixture.record.attention.arm_attempted == 0u);
  CHECK(fixture.record.attention.publish_attempted == 0u);
  CHECK(fixture.record.output_enqueued == 0u);
  return 0;
}

static int test_pack_wait_failure_stops_layer(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.fail_wait = 1u;
  fixture.mock.fail_wait_stage = Q16K_FULL_MODEL_SYNC_STAGE_PACK;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_CALLBACK_FAILED);
  CHECK(fixture.mock.wait_count == 1u);
  CHECK(fixture.mock.publish_count == 1u);
  CHECK(fixture.mock.post_attention_count == 0u);
  CHECK(fixture.queue.kernarg_cursor == 1u);
  CHECK(fixture.mock.global_cursor == 1u);
  CHECK(fixture.record.pack.wait_acquired == 0u);
  CHECK(fixture.record.attention.publish_attempted == 0u);
  CHECK(fixture.record.output_enqueued == 0u);
  return 0;
}

static int test_attention_wait_failure_stops_output(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.fail_wait = 1u;
  fixture.mock.fail_wait_stage = Q16K_FULL_MODEL_SYNC_STAGE_ATTENTION;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_CALLBACK_FAILED);
  CHECK(fixture.mock.wait_count == 2u);
  CHECK(fixture.mock.publish_count == 2u);
  CHECK(fixture.mock.post_attention_count == 0u);
  CHECK(fixture.queue.kernarg_cursor == 1u);
  CHECK(fixture.mock.global_cursor == 1u);
  CHECK(fixture.record.pack.wait_acquired == 1u);
  CHECK(fixture.record.attention.wait_acquired == 0u);
  CHECK(fixture.record.output_enqueued == 0u);
  return 0;
}

static int test_post_attention_failure_stops_output(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.fail_post_attention = 1u;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_CALLBACK_FAILED);
  CHECK(fixture.mock.wait_count == 2u);
  CHECK(fixture.mock.publish_count == 2u);
  CHECK(fixture.mock.post_attention_count == 1u);
  CHECK(fixture.mock.copy_wait_count == 4u);
  CHECK(fixture.mock.metadata_copy_count == 2u);
  CHECK(fixture.mock.pre_attention_copied_bytes == UINT64_C(50331648));
  CHECK(fixture.mock.post_attention_copied_bytes == UINT64_C(50331648));
  CHECK(fixture.queue.kernarg_cursor == 0u);
  CHECK(fixture.mock.global_cursor == 0u);
  CHECK(fixture.record.pack.wait_acquired == 1u);
  CHECK(fixture.record.attention.wait_acquired == 1u);
  CHECK(fixture.record.attention_post_wait_completed == 0u);
  CHECK(fixture.record.output_enqueued == 0u);
  return 0;
}

static int test_failed_reservation_never_waits(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.write_index = 6u;
  fixture.mock.read_index = 0u;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_QUEUE_FULL);
  CHECK(fixture.mock.arm_count == 1u);
  CHECK(fixture.mock.publish_count == 0u);
  CHECK(fixture.mock.store_count == 0u);
  CHECK(fixture.mock.doorbell_count == 0u);
  CHECK(fixture.mock.wait_count == 0u);
  CHECK(fixture.queue.kernarg_cursor == 0u);
  CHECK(fixture.mock.global_cursor == 0u);
  return 0;
}

static int test_arm_failure_propagates_without_publish(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.fail_arm = 1u;
  fixture.mock.fail_arm_stage = Q16K_FULL_MODEL_SYNC_STAGE_ATTENTION;
  CHECK(q16k_full_model_sync_enqueue_layer(
            &fixture.hook, &fixture.plan, UINT64_C(0x5151), &fixture.ops,
            &fixture.record) == Q16K_AITER_STATUS_CALLBACK_FAILED);
  CHECK(fixture.record.pack.wait_acquired == 1u);
  CHECK(fixture.record.attention.arm_attempted == 1u);
  CHECK(fixture.record.attention.publish_attempted == 0u);
  CHECK(fixture.mock.wait_count == 1u);
  CHECK(fixture.mock.publish_count == 1u);
  CHECK(fixture.record.output_enqueued == 0u);
  return 0;
}

static int test_disabled_path_preserves_original_behavior(void) {
  fixture_t fixture;
  CHECK(initialize_fixture(&fixture));
  fixture.mock.signal_value = 1u;
  CHECK(q16k_aiter_enqueue_layer(&fixture.hook, &fixture.plan, 0u) ==
        Q16K_AITER_STATUS_OK);
  CHECK(fixture.mock.arm_count == 0u);
  CHECK(fixture.mock.wait_count == 0u);
  CHECK(fixture.mock.post_attention_count == 0u);
  CHECK(fixture.queue.kernarg_cursor == 2u);
  CHECK(fixture.mock.publish_count == 2u);
  CHECK(fixture.mock.store_count == 0u);
  CHECK(fixture.mock.doorbell_count == 0u);
  CHECK(fixture.packets[0].completion_signal == 0u);
  CHECK(fixture.packets[1].completion_signal == 0u);
  return 0;
}

static void complete_stage(q16k_full_model_sync_stage_record_t* stage,
                           uint32_t slot, uint64_t timestamp,
                           uint32_t full_header) {
  memset(stage, 0, sizeof(*stage));
  stage->arm_attempted = 1u;
  stage->armed = 1u;
  stage->publish_attempted = 1u;
  stage->published = 1u;
  stage->wait_attempted = 1u;
  stage->wait_acquired = 1u;
  stage->arm_status = Q16K_AITER_STATUS_OK;
  stage->publish_status = Q16K_AITER_STATUS_OK;
  stage->wait_status = Q16K_AITER_STATUS_OK;
  stage->arm_ns = timestamp;
  stage->publish_begin_ns = timestamp + 1u;
  stage->publish_end_ns = timestamp + 2u;
  stage->wait_begin_ns = timestamp + 3u;
  stage->wait_end_ns = timestamp + 4u;
  stage->cursor_before_publish = slot;
  stage->cursor_after_publish = slot + 1u;
  stage->cursor_after_wait = 0u;
  stage->dispatch.packet_snapshot.full_header = full_header;
  stage->dispatch.packet_snapshot.kernel_object = UINT64_C(0x12340000);
  stage->dispatch.packet_snapshot.kernarg_address =
      UINT64_C(0x90000000) + (uint64_t)slot * Q16K_AITER_KERNARG_STRIDE;
  stage->dispatch.packet_snapshot.completion_signal = UINT64_C(0x5151);
  stage->dispatch.packet_id = timestamp;
  stage->dispatch.kernarg_slot = slot;
  stage->dispatch.doorbell_written = 1u;
}

static q16k_full_model_sync_status_t analyze_first_attention(
    const uint8_t* pre, const uint8_t* post,
    q16k_full_model_sync_first_attention_difference_t* difference) {
  return q16k_full_model_sync_analyze_first_attention_difference(
      pre, post, Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT,
      Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_COUNT,
      Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT,
      Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_DIMENSION, difference);
}

static int test_first_attention_difference_none(void) {
  const size_t byte_count = Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT;
  uint8_t* bytes = (uint8_t*)calloc(1u, byte_count);
  CHECK(bytes != NULL);
  q16k_full_model_sync_first_attention_difference_t difference;
  CHECK(analyze_first_attention(bytes, bytes, &difference) ==
        Q16K_FULL_MODEL_SYNC_OK);
  CHECK(difference.analyzed == 1u);
  CHECK(difference.layout_valid == 1u);
  CHECK(difference.query_count == 16384u);
  CHECK(difference.head_count == 12u);
  CHECK(difference.head_dimension == 128u);
  CHECK(difference.query_tile_rows == 128u);
  CHECK(difference.query_tile_count == 128u);
  CHECK(difference.byte_count == UINT64_C(50331648));
  CHECK(difference.element_count == UINT64_C(25165824));
  CHECK(difference.changed_byte_count == 0u);
  CHECK(difference.changed_element_count == 0u);
  CHECK(difference.changed_query_row_count == 0u);
  CHECK(difference.first_changed_byte == UINT64_MAX);
  CHECK(difference.last_changed_byte == UINT64_MAX);
  CHECK(difference.first_changed_element == UINT64_MAX);
  CHECK(difference.last_changed_element == UINT64_MAX);
  CHECK(difference.first_changed_query_row == UINT64_MAX);
  CHECK(difference.last_changed_query_row == UINT64_MAX);
  CHECK(difference.unchanged_prefix_byte_count == byte_count);
  CHECK(difference.unchanged_suffix_byte_count == byte_count);
  CHECK(difference.unchanged_prefix_element_count ==
        difference.element_count);
  CHECK(difference.unchanged_suffix_element_count ==
        difference.element_count);
  for (uint32_t i = 0u; i < difference.head_count; ++i) {
    CHECK(difference.per_head_changed_element_count[i] == 0u);
  }
  for (uint32_t i = 0u; i < difference.query_tile_count; ++i) {
    CHECK(difference.per_query_tile_changed_element_count[i] == 0u);
  }
  CHECK(q16k_full_model_sync_analyze_first_attention_difference(
            bytes, bytes, byte_count, 16384u, 11u, 128u, &difference) ==
        Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT);
  free(bytes);
  return 0;
}

static int test_first_attention_difference_prefix(void) {
  const size_t byte_count = Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT;
  uint8_t* pre = (uint8_t*)calloc(1u, byte_count);
  uint8_t* post = (uint8_t*)calloc(1u, byte_count);
  CHECK(pre != NULL && post != NULL);
  memset(post, 0x5a, 5u);
  q16k_full_model_sync_first_attention_difference_t difference;
  CHECK(analyze_first_attention(pre, post, &difference) ==
        Q16K_FULL_MODEL_SYNC_OK);
  CHECK(difference.changed_byte_count == 5u);
  CHECK(difference.changed_element_count == 3u);
  CHECK(difference.first_changed_byte == 0u);
  CHECK(difference.last_changed_byte == 4u);
  CHECK(difference.first_changed_element == 0u);
  CHECK(difference.last_changed_element == 2u);
  CHECK(difference.changed_query_row_count == 1u);
  CHECK(difference.first_changed_query_row == 0u);
  CHECK(difference.last_changed_query_row == 0u);
  CHECK(difference.unchanged_prefix_byte_count == 0u);
  CHECK(difference.unchanged_suffix_byte_count == byte_count - 5u);
  CHECK(difference.unchanged_prefix_element_count == 0u);
  CHECK(difference.unchanged_suffix_element_count ==
        difference.element_count - 3u);
  CHECK(difference.per_head_changed_element_count[0] == 3u);
  CHECK(difference.per_query_tile_changed_element_count[0] == 3u);
  free(post);
  free(pre);
  return 0;
}

static uint64_t first_attention_element_index(uint32_t query_row,
                                              uint32_t head,
                                              uint32_t dimension) {
  return ((uint64_t)query_row *
              Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT +
          head) *
             Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_DIMENSION +
         dimension;
}

static int test_first_attention_difference_sparse(void) {
  const size_t byte_count = Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT;
  uint8_t* pre = (uint8_t*)calloc(1u, byte_count);
  uint8_t* post = (uint8_t*)calloc(1u, byte_count);
  CHECK(pre != NULL && post != NULL);
  const uint64_t first = first_attention_element_index(1u, 2u, 3u);
  const uint64_t middle = first_attention_element_index(129u, 11u, 127u);
  const uint64_t last = first_attention_element_index(16383u, 0u, 64u);
  post[first * 2u] = 1u;
  post[middle * 2u] = 2u;
  post[middle * 2u + 1u] = 3u;
  post[last * 2u + 1u] = 4u;
  q16k_full_model_sync_first_attention_difference_t difference;
  CHECK(analyze_first_attention(pre, post, &difference) ==
        Q16K_FULL_MODEL_SYNC_OK);
  CHECK(difference.changed_byte_count == 4u);
  CHECK(difference.changed_element_count == 3u);
  CHECK(difference.first_changed_byte == first * 2u);
  CHECK(difference.last_changed_byte == last * 2u + 1u);
  CHECK(difference.first_changed_element == first);
  CHECK(difference.last_changed_element == last);
  CHECK(difference.changed_query_row_count == 3u);
  CHECK(difference.first_changed_query_row == 1u);
  CHECK(difference.last_changed_query_row == 16383u);
  CHECK(difference.unchanged_prefix_byte_count == first * 2u);
  CHECK(difference.unchanged_suffix_byte_count ==
        byte_count - (last * 2u + 1u) - 1u);
  CHECK(difference.unchanged_prefix_element_count == first);
  CHECK(difference.unchanged_suffix_element_count ==
        difference.element_count - last - 1u);
  CHECK(difference.per_head_changed_element_count[0] == 1u);
  CHECK(difference.per_head_changed_element_count[2] == 1u);
  CHECK(difference.per_head_changed_element_count[11] == 1u);
  CHECK(difference.per_query_tile_changed_element_count[0] == 1u);
  CHECK(difference.per_query_tile_changed_element_count[1] == 1u);
  CHECK(difference.per_query_tile_changed_element_count[127] == 1u);
  free(post);
  free(pre);
  return 0;
}

static int test_first_attention_difference_full(void) {
  const size_t byte_count = Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT;
  uint8_t* pre = (uint8_t*)calloc(1u, byte_count);
  uint8_t* post = (uint8_t*)malloc(byte_count);
  CHECK(pre != NULL && post != NULL);
  memset(post, 0xff, byte_count);
  q16k_full_model_sync_first_attention_difference_t difference;
  CHECK(analyze_first_attention(pre, post, &difference) ==
        Q16K_FULL_MODEL_SYNC_OK);
  CHECK(difference.changed_byte_count == byte_count);
  CHECK(difference.changed_element_count == difference.element_count);
  CHECK(difference.first_changed_byte == 0u);
  CHECK(difference.last_changed_byte == byte_count - 1u);
  CHECK(difference.first_changed_element == 0u);
  CHECK(difference.last_changed_element == difference.element_count - 1u);
  CHECK(difference.changed_query_row_count == 16384u);
  CHECK(difference.first_changed_query_row == 0u);
  CHECK(difference.last_changed_query_row == 16383u);
  CHECK(difference.unchanged_prefix_byte_count == 0u);
  CHECK(difference.unchanged_suffix_byte_count == 0u);
  CHECK(difference.unchanged_prefix_element_count == 0u);
  CHECK(difference.unchanged_suffix_element_count == 0u);
  for (uint32_t i = 0u; i < difference.head_count; ++i) {
    CHECK(difference.per_head_changed_element_count[i] == UINT64_C(2097152));
  }
  for (uint32_t i = 0u; i < difference.query_tile_count; ++i) {
    CHECK(difference.per_query_tile_changed_element_count[i] ==
          UINT64_C(196608));
  }
  free(post);
  free(pre);
  return 0;
}

static int test_complete_record_and_atomic_json(void) {
  q16k_full_model_sync_diagnostic_t* diagnostic =
      (q16k_full_model_sync_diagnostic_t*)calloc(1, sizeof(*diagnostic));
  CHECK(diagnostic != NULL);
  q16k_full_model_sync_begin(diagnostic, UINT64_C(77), 1u, 100u, 200u);
  const q16k_full_model_sync_runtime_layout_input_t layout = {
      .metadata = {(const void*)(uintptr_t)UINT64_C(0x70000000), 24u},
      .page_indices = {(const void*)(uintptr_t)UINT64_C(0x71000000), 525308u},
      .scratch = {(const void*)(uintptr_t)UINT64_C(0x80000000), 1263154688u},
      .loom_key = {(const void*)(uintptr_t)UINT64_C(0x100000000), 1879277568u},
      .loom_value = {(const void*)(uintptr_t)UINT64_C(0x200000000), 1879277568u},
      .shadow_key = {(const void*)(uintptr_t)UINT64_C(0x300000000), 1879048192u},
      .shadow_value = {(const void*)(uintptr_t)UINT64_C(0x400000000), 1879048192u},
      .rope = {(const void*)(uintptr_t)UINT64_C(0x500000000), 33554432u},
      .params = {(const void*)(uintptr_t)UINT64_C(0x510000000), 16u},
      .dense_workspace = {(const void*)(uintptr_t)UINT64_C(0x600000000),
                          587202560u},
      .kernarg_ring = {(const void*)(uintptr_t)UINT64_C(0x90000000),
                       33554432u},
  };
  CHECK(q16k_full_model_sync_capture_runtime_layout(diagnostic, &layout) ==
        Q16K_FULL_MODEL_SYNC_OK);
  for (uint32_t chunk = 0u; chunk < Q16K_AITER_MAX_CHUNK_COUNT; ++chunk) {
    const uint32_t base = Q16K_AITER_LEGACY_PREFIX_ROWS +
                          chunk * Q16K_AITER_CHUNK_CAPACITY;
    const uint32_t count = chunk + 1u == Q16K_AITER_MAX_CHUNK_COUNT
                               ? Q16K_AITER_MAX_PROMPT_TOKEN_COUNT - base
                               : Q16K_AITER_CHUNK_CAPACITY;
    for (uint32_t layer = 0u; layer < Q16K_AITER_LAYER_COUNT; ++layer) {
      q16k_full_model_sync_layer_record_t* record = NULL;
      CHECK(q16k_full_model_sync_begin_layer(diagnostic, base, count, layer,
                                              &record) ==
            Q16K_FULL_MODEL_SYNC_OK);
      CHECK(record != NULL);
      complete_stage(&record->pack, 3u, 10u + diagnostic->layer_record_count,
                     UINT32_C(0x00011502));
      complete_stage(&record->attention, 0u,
                     20u + diagnostic->layer_record_count,
                     UINT32_C(0x00031502));
      record->attention_post_wait_completed = 1u;
      record->output_enqueued = 1u;
    }
  }
  const char hash[] =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  q16k_aiter_layer_plan_t first_plan = {.layer = 0u,
                                       .position_base =
                                           Q16K_AITER_LEGACY_PREFIX_ROWS,
                                       .logical_query_count =
                                           Q16K_AITER_CHUNK_CAPACITY};
  first_plan.attention = (q16k_aiter_attention_request_t){
      .q = (const void*)(uintptr_t)UINT64_C(0x50000000),
      .k = (const void*)(uintptr_t)UINT64_C(0x30000000),
      .v = (const void*)(uintptr_t)UINT64_C(0x40000000),
      .output = (void*)(uintptr_t)UINT64_C(0x60000000),
      .kv_indptr = (const int32_t*)(uintptr_t)UINT64_C(0x70000000),
      .kv_page_indices = (const int32_t*)(uintptr_t)UINT64_C(0x71000000),
      .kv_last_page_lens =
          (const int32_t*)(uintptr_t)UINT64_C(0x70000008),
      .cu_seqlens_q = (const int32_t*)(uintptr_t)UINT64_C(0x70000010),
      .position_base = Q16K_AITER_LEGACY_PREFIX_ROWS,
      .logical_query_count = Q16K_AITER_CHUNK_CAPACITY,
  };
  first_plan.pack = (q16k_aiter_pack_request_t){
      .loom_k = (const void*)(uintptr_t)UINT64_C(0x10000000),
      .loom_v = (const void*)(uintptr_t)UINT64_C(0x20000000),
      .shadow_k = (void*)(uintptr_t)UINT64_C(0x30000000),
      .shadow_v = (void*)(uintptr_t)UINT64_C(0x40000000),
      .layer_count = 1u,
      .position_base = Q16K_AITER_LEGACY_PREFIX_ROWS,
      .logical_query_count = Q16K_AITER_CHUNK_CAPACITY,
      .destination_capacity = Q16K_AITER_CONTEXT_CAPACITY,
      .source_block_count = Q16K_AITER_LOOM_BLOCK_COUNT,
      .source_block_base = 0u,
  };
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t pack_kernarg[Q16K_AITER_PACK_KERNARG_SIZE];
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t attention_kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t geometry;
  CHECK(q16k_build_pack_kernarg(&first_plan.pack, pack_kernarg,
                                sizeof(pack_kernarg), &geometry) ==
        Q16K_AITER_STATUS_OK);
  CHECK(q16k_build_attention_kernarg(
            &first_plan.attention, attention_kernarg,
            sizeof(attention_kernarg), &geometry) == Q16K_AITER_STATUS_OK);
  CHECK(q16k_full_model_sync_capture_first_layer_kernargs(
            diagnostic, &first_plan, pack_kernarg, sizeof(pack_kernarg), hash,
            attention_kernarg, sizeof(attention_kernarg), hash) ==
        Q16K_FULL_MODEL_SYNC_OK);
  const int32_t metadata_words[Q16K_FULL_MODEL_SYNC_METADATA_WORD_COUNT] = {
      0, 16768, 1, 0, 0, 16384,
  };
  CHECK(q16k_full_model_sync_capture_metadata_snapshot(
            diagnostic,
            Q16K_FULL_MODEL_SYNC_METADATA_POST_PACK_PRE_ATTENTION, &first_plan,
            Q16K_AITER_METADATA_SIZE, 16u, 17u, 1u, metadata_words, hash) ==
        Q16K_FULL_MODEL_SYNC_OK);
  CHECK(q16k_full_model_sync_capture_first_attention_pre(
            diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT, 18u, 19u, 1u,
            hash) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(q16k_full_model_sync_capture_metadata_snapshot(
            diagnostic, Q16K_FULL_MODEL_SYNC_METADATA_POST_ATTENTION,
            &first_plan, Q16K_AITER_METADATA_SIZE, 250u, 251u, 1u,
            metadata_words, hash) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(q16k_full_model_sync_capture_first_attention(
            diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT, 1000u, 1001u,
            1u, hash) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(diagnostic->first_attention_byte_equal == 1u);
  uint8_t* unchanged = (uint8_t*)calloc(
      1u, Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT);
  CHECK(unchanged != NULL);
  q16k_full_model_sync_first_attention_difference_t difference;
  CHECK(analyze_first_attention(unchanged, unchanged, &difference) ==
        Q16K_FULL_MODEL_SYNC_OK);
  CHECK(q16k_full_model_sync_capture_first_attention_difference(
            diagnostic, &difference) == Q16K_FULL_MODEL_SYNC_OK);
  free(unchanged);
  q16k_full_model_sync_finish(diagnostic, 1u, 1u, 382, 3615u, 695u);
  CHECK(q16k_full_model_sync_is_complete(diagnostic));
  CHECK(diagnostic->total_dispatch_count == 3515u);
  CHECK(diagnostic->total_wait_count == 495u);
  CHECK(diagnostic->stage_wait_count == 448u);
  CHECK(diagnostic->metadata_copy_wait_count == 2u);
  CHECK(diagnostic->first_attention_pre_copy_wait_count == 1u);
  CHECK(diagnostic->first_attention_copy_wait_count == 1u);
  diagnostic->first_attention_pre_copy_end_ns =
      diagnostic->layers[0].attention.arm_ns + 1u;
  CHECK(!q16k_full_model_sync_is_complete(diagnostic));
  diagnostic->first_attention_pre_copy_end_ns = 19u;
  diagnostic->first_attention_copy_begin_ns =
      diagnostic->layers[0].attention.wait_end_ns - 1u;
  CHECK(!q16k_full_model_sync_is_complete(diagnostic));
  diagnostic->first_attention_copy_begin_ns = 1000u;
  diagnostic->total_wait_count = 1227u;
  CHECK(!q16k_full_model_sync_is_complete(diagnostic));
  diagnostic->total_wait_count = 1229u;
  CHECK(!q16k_full_model_sync_is_complete(diagnostic));
  diagnostic->total_wait_count = 495u;
  diagnostic->first_attention_difference.head_count = 11u;
  CHECK(!q16k_full_model_sync_is_complete(diagnostic));
  diagnostic->first_attention_difference.head_count = 12u;
  CHECK(q16k_full_model_sync_is_complete(diagnostic));

  const char* tmpdir = getenv("TEST_TMPDIR");
  char cwd[1024];
  if (tmpdir == NULL || tmpdir[0] == '\0') {
    CHECK(getcwd(cwd, sizeof(cwd)) != NULL);
    tmpdir = cwd;
  }
  char path[1200];
  CHECK(snprintf(path, sizeof(path), "%s/full_model_sync.json", tmpdir) > 0);
  remove(path);
  char error[256];
  CHECK(q16k_full_model_sync_write_json_atomic(
            diagnostic, path, error, sizeof(error)) ==
        Q16K_FULL_MODEL_SYNC_OK);
  FILE* file = fopen(path, "rb");
  CHECK(file != NULL);
  CHECK(fseek(file, 0, SEEK_END) == 0);
  const long length = ftell(file);
  CHECK(length > 0 && length < 1048576);
  CHECK(fseek(file, 0, SEEK_SET) == 0);
  char* json = (char*)malloc((size_t)length + 1u);
  CHECK(json != NULL);
  CHECK(fread(json, 1u, (size_t)length, file) == (size_t)length);
  json[length] = '\0';
  CHECK(fclose(file) == 0);
  CHECK(strstr(json, "\"complete\": true") != NULL);
  CHECK(strstr(json, "\"dispatches\": 3515") != NULL);
  CHECK(strstr(json, "\"waits\": 495") != NULL);
  CHECK(strstr(json, "\"stage_waits\": 448") != NULL);
  CHECK(strstr(json, "\"metadata_copy_waits\": 2") != NULL);
  CHECK(strstr(json, "\"first_attention_copy_waits\": 2") != NULL);
  CHECK(strstr(json, "\"pre_byte_count\": 50331648") != NULL);
  CHECK(strstr(json, "\"byte_equal\": true") != NULL);
  CHECK(strstr(json, "\"layout_valid\": true") != NULL);
  CHECK(strstr(json, "\"shape\": [16384, 12, 128]") != NULL);
  CHECK(strstr(json, "\"changed_bytes\": 0") != NULL);
  CHECK(strstr(json, "\"changed_elements\": 0") != NULL);
  CHECK(strstr(json, "\"first_changed_byte\": null") != NULL);
  CHECK(strstr(json, "\"changed_query_rows\": 0") != NULL);
  CHECK(strstr(json, "\"unchanged_prefix_bytes\": 50331648") != NULL);
  CHECK(strstr(json, "\"per_head_changed_elements\": [0, 0, 0") != NULL);
  CHECK(strstr(json, "\"expected_match\": true") != NULL);
  CHECK(strstr(json, "\"snapshot_hex\":") != NULL);
  CHECK(strstr(json, "\"kv_indptr\": \"0x0000000070000000\"") != NULL);
  CHECK(strstr(json, "\"full_header\": \"0x00031502\"") != NULL);
  CHECK(strstr(json, "\"final_token\": 382") != NULL);
  CHECK(q16k_full_model_sync_write_json_atomic(
            diagnostic, path, error, sizeof(error)) ==
        Q16K_FULL_MODEL_SYNC_IO_ERROR);
  CHECK(strstr(error, "without replacement") != NULL);
  free(json);
  CHECK(remove(path) == 0);
  free(diagnostic);
  return 0;
}

static int test_first_attention_hash_equality(void) {
  const q16k_aiter_layer_plan_t first_plan = {
      .layer = 0u,
      .position_base = Q16K_AITER_LEGACY_PREFIX_ROWS,
      .logical_query_count = Q16K_AITER_CHUNK_CAPACITY,
  };
  const char hash_a[] =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  const char hash_b[] =
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  q16k_full_model_sync_diagnostic_t diagnostic;

  q16k_full_model_sync_begin(&diagnostic, UINT64_C(1), 1u, 0u, 0u);
  CHECK(q16k_full_model_sync_capture_first_attention_pre(
            &diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT, 1u, 2u, 1u,
            hash_a) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(q16k_full_model_sync_capture_first_attention(
            &diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT, 3u, 4u, 1u,
            hash_a) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(diagnostic.first_attention_byte_equal == 1u);

  q16k_full_model_sync_begin(&diagnostic, UINT64_C(2), 1u, 0u, 0u);
  CHECK(q16k_full_model_sync_capture_first_attention_pre(
            &diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT, 1u, 2u, 1u,
            hash_a) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(q16k_full_model_sync_capture_first_attention(
            &diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT, 3u, 4u, 1u,
            hash_b) == Q16K_FULL_MODEL_SYNC_OK);
  CHECK(diagnostic.first_attention_byte_equal == 0u);
  CHECK(q16k_full_model_sync_capture_first_attention_pre(
            &diagnostic, &first_plan,
            Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT - 1u, 5u, 6u,
            1u, hash_a) == Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT);
  return 0;
}

int main(void) {
  CHECK(test_success_order_cursor_lifetime_and_consumed_ring_race() == 0);
  CHECK(test_pack_wait_failure_stops_layer() == 0);
  CHECK(test_pre_attention_failure_stops_attention_publish() == 0);
  CHECK(test_attention_wait_failure_stops_output() == 0);
  CHECK(test_post_attention_failure_stops_output() == 0);
  CHECK(test_failed_reservation_never_waits() == 0);
  CHECK(test_arm_failure_propagates_without_publish() == 0);
  CHECK(test_disabled_path_preserves_original_behavior() == 0);
  CHECK(test_first_attention_hash_equality() == 0);
  CHECK(test_first_attention_difference_none() == 0);
  CHECK(test_first_attention_difference_prefix() == 0);
  CHECK(test_first_attention_difference_sparse() == 0);
  CHECK(test_first_attention_difference_full() == 0);
  CHECK(test_complete_record_and_atomic_json() == 0);
  puts("q16k full-model synchronization diagnostic test: ok");
  return 0;
}
