#define _POSIX_C_SOURCE 200809L

#include "q16k_full_model_sync_diagnostic.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define Q16K_FULL_MODEL_SYNC_SYSTEM_HEADER UINT32_C(0x00001502)

static const int32_t kExpectedFirstMetadata[
    Q16K_FULL_MODEL_SYNC_METADATA_WORD_COUNT] = {
    0, 16768, 1, 0, 0, 16384,
};

static int is_first_layer(const q16k_aiter_layer_plan_t* layer_plan) {
  return layer_plan != NULL && layer_plan->layer == 0u &&
         layer_plan->position_base == Q16K_AITER_LEGACY_PREFIX_ROWS &&
         layer_plan->logical_query_count == Q16K_AITER_CHUNK_CAPACITY;
}

static int sha256_is_valid(const char* sha256) {
  if (sha256 == NULL || strlen(sha256) != 64u) return 0;
  for (size_t i = 0u; i < 64u; ++i) {
    if (!((sha256[i] >= '0' && sha256[i] <= '9') ||
          (sha256[i] >= 'a' && sha256[i] <= 'f'))) {
      return 0;
    }
  }
  return 1;
}

static uint64_t load_u64(const uint8_t* bytes, size_t offset) {
  uint64_t value = 0u;
  for (size_t i = 0u; i < sizeof(value); ++i) {
    value |= (uint64_t)bytes[offset + i] << (8u * i);
  }
  return value;
}

static int capture_allocation(
    const q16k_full_model_sync_allocation_input_t* input,
    q16k_full_model_sync_allocation_record_t* record) {
  if (input == NULL || record == NULL || input->base == NULL ||
      input->size == 0u) {
    return 0;
  }
  const uint64_t base = (uint64_t)(uintptr_t)input->base;
  const uint64_t size = (uint64_t)input->size;
  if (base > UINT64_MAX - size) return 0;
  *record = (q16k_full_model_sync_allocation_record_t){
      .captured = 1u,
      .base = base,
      .size = size,
      .end = base + size,
      .alignment = base & (~base + 1u),
      .base_mod_16 = (uint32_t)(base & UINT64_C(15)),
      .base_mod_256 = (uint32_t)(base & UINT64_C(255)),
  };
  return record->alignment != 0u;
}

static void initialize_stage_record(
    q16k_full_model_sync_stage_record_t* record) {
  memset(record, 0, sizeof(*record));
  record->arm_status = Q16K_AITER_STATUS_CALLBACK_FAILED;
  record->publish_status = Q16K_AITER_STATUS_CALLBACK_FAILED;
  record->wait_status = Q16K_AITER_STATUS_CALLBACK_FAILED;
  record->dispatch.packet_id = UINT64_MAX;
  record->dispatch.kernarg_slot = UINT32_MAX;
  record->cursor_before_publish = UINT32_MAX;
  record->cursor_after_publish = UINT32_MAX;
  record->cursor_after_wait = UINT32_MAX;
}

static int loaded_kernel_matches(
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_kernel_spec_t* spec) {
  return kernel != NULL && spec != NULL && kernel->name != NULL &&
         kernel->name[0] != '\0' && kernel->kernel_object != 0u &&
         kernel->kernarg_segment_size == spec->kernarg_segment_size &&
         kernel->kernarg_segment_alignment ==
             spec->kernarg_segment_alignment &&
         kernel->group_segment_size == spec->group_segment_size &&
         kernel->private_segment_size == spec->private_segment_size;
}

static q16k_aiter_status_t dispatch_and_wait(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_layer_plan_t* layer_plan,
    q16k_full_model_sync_stage_t stage, uint64_t completion_signal,
    const q16k_full_model_sync_ops_t* ops, const void* kernarg,
    size_t kernarg_size, const q16k_aiter_launch_geometry_t* geometry,
    const q16k_aiter_loaded_kernel_t* kernel,
    q16k_full_model_sync_stage_record_t* record) {
  record->arm_attempted = 1u;
  record->arm_ns = ops->now_ns(ops->user_data);
  record->arm_status =
      ops->arm(ops->user_data, stage, layer_plan->layer, completion_signal);
  if (record->arm_status != Q16K_AITER_STATUS_OK) {
    return record->arm_status;
  }
  record->armed = 1u;

  record->publish_attempted = 1u;
  record->publish_begin_ns = ops->now_ns(ops->user_data);
  record->cursor_before_publish = hook->queue->kernarg_cursor;
  record->publish_status = q16k_aiter_dispatch_raw(
      hook->queue, kernel, geometry, kernarg, kernarg_size,
      completion_signal, 0u, &record->dispatch);
  record->publish_end_ns = ops->now_ns(ops->user_data);
  record->cursor_after_publish = hook->queue->kernarg_cursor;
  ops->synchronize_cursor(ops->user_data, hook->queue, 0u);
  if (record->publish_status != Q16K_AITER_STATUS_OK) {
    return record->publish_status;
  }

  record->published = 1u;
  if (completion_signal == 0u ||
      record->dispatch.packet_snapshot.completion_signal !=
          completion_signal ||
      record->dispatch.doorbell_written != 1u) {
    record->publish_status = Q16K_AITER_STATUS_CALLBACK_FAILED;
    return record->publish_status;
  }

  record->wait_attempted = 1u;
  record->wait_begin_ns = ops->now_ns(ops->user_data);
  record->wait_status =
      ops->wait(ops->user_data, stage, layer_plan->layer,
                completion_signal);
  record->wait_end_ns = ops->now_ns(ops->user_data);
  if (record->wait_status != Q16K_AITER_STATUS_OK) {
    return record->wait_status;
  }

  record->wait_acquired = 1u;
  hook->queue->kernarg_cursor = 0u;
  record->cursor_after_wait = 0u;
  ops->synchronize_cursor(ops->user_data, hook->queue, 1u);
  return Q16K_AITER_STATUS_OK;
}

q16k_aiter_status_t q16k_full_model_sync_enqueue_layer(
    const q16k_aiter_layer_hook_t* hook,
    const q16k_aiter_layer_plan_t* layer_plan, uint64_t completion_signal,
    const q16k_full_model_sync_ops_t* ops,
    q16k_full_model_sync_layer_record_t* record) {
  if (hook == NULL || layer_plan == NULL || ops == NULL || record == NULL ||
      hook->enabled == 0u || hook->enqueue_loom == NULL ||
      hook->queue == NULL || hook->queue->queue_size == 0u ||
      (hook->queue->queue_size & (hook->queue->queue_size - 1u)) != 0u ||
      !loaded_kernel_matches(hook->pack_kernel, q16k_pack_kernel_spec()) ||
      !loaded_kernel_matches(hook->attention_kernel,
                             q16k_attention_kernel_spec()) ||
      layer_plan->layer >= Q16K_AITER_LAYER_COUNT ||
      layer_plan->position_base < Q16K_AITER_LEGACY_PREFIX_ROWS ||
      layer_plan->logical_query_count == 0u ||
      layer_plan->logical_query_count > Q16K_AITER_CHUNK_CAPACITY ||
      layer_plan->position_base >
          Q16K_AITER_MAX_PROMPT_TOKEN_COUNT -
              layer_plan->logical_query_count ||
      completion_signal == 0u || ops->now_ns == NULL || ops->arm == NULL ||
      ops->wait == NULL || ops->synchronize_cursor == NULL ||
      ops->pre_attention_publish == NULL ||
      ops->post_attention_wait == NULL) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }

  const uint32_t expected_chunk =
      (layer_plan->position_base - Q16K_AITER_LEGACY_PREFIX_ROWS) /
      Q16K_AITER_CHUNK_CAPACITY;
  if (record->chunk_ordinal != expected_chunk ||
      record->position_base != layer_plan->position_base ||
      record->logical_query_count != layer_plan->logical_query_count ||
      record->layer != layer_plan->layer) {
    return Q16K_AITER_STATUS_INVALID_ARGUMENT;
  }

  q16k_aiter_status_t status = hook->enqueue_loom(
      hook->user_data, Q16K_AITER_LOOM_QKV, layer_plan->layer,
      layer_plan->position_base, layer_plan->logical_query_count, 0u);
  if (status != Q16K_AITER_STATUS_OK) {
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }
  status = hook->enqueue_loom(
      hook->user_data, Q16K_AITER_LOOM_ROPE_CACHE, layer_plan->layer,
      layer_plan->position_base, layer_plan->logical_query_count, 0u);
  if (status != Q16K_AITER_STATUS_OK) {
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }

  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t pack_kernarg[Q16K_AITER_PACK_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t pack_geometry;
  status = q16k_build_pack_kernarg(
      &layer_plan->pack, pack_kernarg, sizeof(pack_kernarg), &pack_geometry);
  if (status != Q16K_AITER_STATUS_OK) return status;
  status = dispatch_and_wait(
      hook, layer_plan, Q16K_FULL_MODEL_SYNC_STAGE_PACK, completion_signal,
      ops, pack_kernarg, sizeof(pack_kernarg), &pack_geometry,
      hook->pack_kernel, &record->pack);
  if (status != Q16K_AITER_STATUS_OK) return status;

  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t attention_kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t attention_geometry;
  status = q16k_build_attention_kernarg(
      &layer_plan->attention, attention_kernarg, sizeof(attention_kernarg),
      &attention_geometry);
  if (status != Q16K_AITER_STATUS_OK) return status;
  if (is_first_layer(layer_plan)) {
    status = ops->pre_attention_publish(
        ops->user_data, layer_plan, pack_kernarg, sizeof(pack_kernarg),
        attention_kernarg, sizeof(attention_kernarg));
    if (status != Q16K_AITER_STATUS_OK) {
      return Q16K_AITER_STATUS_CALLBACK_FAILED;
    }
  }
  status = dispatch_and_wait(
      hook, layer_plan, Q16K_FULL_MODEL_SYNC_STAGE_ATTENTION,
      completion_signal, ops, attention_kernarg, sizeof(attention_kernarg),
      &attention_geometry, hook->attention_kernel, &record->attention);
  if (status != Q16K_AITER_STATUS_OK) return status;

  if (hook->observe_attention != NULL) {
    hook->observe_attention(
        hook->attention_observer_user_data, layer_plan,
        hook->attention_kernel, &attention_geometry, attention_kernarg,
        sizeof(attention_kernarg),
        &record->attention.dispatch.packet_snapshot,
        &record->attention.dispatch);
  }
  status = ops->post_attention_wait(ops->user_data, layer_plan);
  if (status != Q16K_AITER_STATUS_OK) {
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }
  record->attention_post_wait_completed = 1u;

  status = hook->enqueue_loom(
      hook->user_data, Q16K_AITER_LOOM_OUTPUT, layer_plan->layer,
      layer_plan->position_base, layer_plan->logical_query_count, 0u);
  if (status != Q16K_AITER_STATUS_OK) {
    return Q16K_AITER_STATUS_CALLBACK_FAILED;
  }
  record->output_enqueued = 1u;
  return Q16K_AITER_STATUS_OK;
}

const char* q16k_full_model_sync_status_string(
    q16k_full_model_sync_status_t status) {
  switch (status) {
    case Q16K_FULL_MODEL_SYNC_OK:
      return "ok";
    case Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT:
      return "invalid argument";
    case Q16K_FULL_MODEL_SYNC_BAD_STATE:
      return "bad state";
    case Q16K_FULL_MODEL_SYNC_INCOMPLETE:
      return "incomplete";
    case Q16K_FULL_MODEL_SYNC_IO_ERROR:
      return "I/O error";
  }
  return "unknown";
}

void q16k_full_model_sync_begin(
    q16k_full_model_sync_diagnostic_t* diagnostic, uint64_t request_id,
    uint32_t request_ordinal, uint64_t dispatch_count_begin,
    uint64_t wait_count_begin) {
  if (diagnostic == NULL) return;
  memset(diagnostic, 0, sizeof(*diagnostic));
  diagnostic->armed = 1u;
  diagnostic->request_id = request_id;
  diagnostic->request_ordinal = request_ordinal;
  diagnostic->dispatch_count_begin = dispatch_count_begin;
  diagnostic->wait_count_begin = wait_count_begin;
  diagnostic->first_attention_layer = UINT32_MAX;
  diagnostic->final_token = -1;
}

q16k_full_model_sync_status_t q16k_full_model_sync_capture_runtime_layout(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_full_model_sync_runtime_layout_input_t* layout) {
  if (diagnostic == NULL || layout == NULL) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      diagnostic->runtime_layout.captured != 0u) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }

  q16k_full_model_sync_runtime_layout_t captured = {0};
  if (!capture_allocation(&layout->metadata, &captured.metadata) ||
      !capture_allocation(&layout->page_indices, &captured.page_indices) ||
      !capture_allocation(&layout->scratch, &captured.scratch) ||
      !capture_allocation(&layout->loom_key, &captured.loom_key) ||
      !capture_allocation(&layout->loom_value, &captured.loom_value) ||
      !capture_allocation(&layout->shadow_key, &captured.shadow_key) ||
      !capture_allocation(&layout->shadow_value, &captured.shadow_value) ||
      !capture_allocation(&layout->rope, &captured.rope) ||
      !capture_allocation(&layout->params, &captured.params) ||
      !capture_allocation(&layout->dense_workspace,
                          &captured.dense_workspace) ||
      !capture_allocation(&layout->kernarg_ring, &captured.kernarg_ring)) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  captured.captured = 1u;
  diagnostic->runtime_layout = captured;
  return Q16K_FULL_MODEL_SYNC_OK;
}

q16k_full_model_sync_status_t q16k_full_model_sync_begin_layer(
    q16k_full_model_sync_diagnostic_t* diagnostic, uint32_t position_base,
    uint32_t logical_query_count, uint32_t layer,
    q16k_full_model_sync_layer_record_t** out_record) {
  if (diagnostic == NULL || out_record == NULL || logical_query_count == 0u ||
      logical_query_count > Q16K_AITER_CHUNK_CAPACITY ||
      position_base < Q16K_AITER_LEGACY_PREFIX_ROWS ||
      position_base > Q16K_AITER_MAX_PROMPT_TOKEN_COUNT - logical_query_count ||
      layer >= Q16K_AITER_LAYER_COUNT) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  *out_record = NULL;
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      diagnostic->layer_record_count >=
          Q16K_FULL_MODEL_SYNC_LAYER_RECORD_CAPACITY) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  const uint32_t index = diagnostic->layer_record_count;
  const uint32_t chunk_ordinal =
      (position_base - Q16K_AITER_LEGACY_PREFIX_ROWS) /
      Q16K_AITER_CHUNK_CAPACITY;
  if (index != chunk_ordinal * Q16K_AITER_LAYER_COUNT + layer) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  q16k_full_model_sync_layer_record_t* record = &diagnostic->layers[index];
  memset(record, 0, sizeof(*record));
  record->chunk_ordinal = chunk_ordinal;
  record->position_base = position_base;
  record->logical_query_count = logical_query_count;
  record->layer = layer;
  initialize_stage_record(&record->pack);
  initialize_stage_record(&record->attention);
  diagnostic->layer_record_count++;
  *out_record = record;
  return Q16K_FULL_MODEL_SYNC_OK;
}

q16k_full_model_sync_status_t
q16k_full_model_sync_capture_first_attention_pre(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_aiter_layer_plan_t* layer_plan, size_t byte_count,
    uint64_t copy_begin_ns, uint64_t copy_end_ns, uint64_t copy_wait_count,
    const char sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE]) {
  if (diagnostic == NULL || layer_plan == NULL || sha256 == NULL ||
      !sha256_is_valid(sha256) ||
      byte_count != Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT ||
      copy_begin_ns == 0u || copy_end_ns < copy_begin_ns ||
      copy_wait_count != 1u) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      diagnostic->first_attention_pre_hash_captured != 0u ||
      !is_first_layer(layer_plan)) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  diagnostic->first_attention_pre_copy_wait_attempted = 1u;
  diagnostic->first_attention_pre_copy_wait_completed = 1u;
  diagnostic->first_attention_position_base = layer_plan->position_base;
  diagnostic->first_attention_query_count = layer_plan->logical_query_count;
  diagnostic->first_attention_layer = layer_plan->layer;
  diagnostic->first_attention_pre_byte_count = byte_count;
  diagnostic->first_attention_pre_copy_begin_ns = copy_begin_ns;
  diagnostic->first_attention_pre_copy_end_ns = copy_end_ns;
  diagnostic->first_attention_pre_copy_wait_count = copy_wait_count;
  memcpy(diagnostic->first_attention_pre_sha256, sha256,
         Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE);
  diagnostic->first_attention_pre_hash_captured = 1u;
  return Q16K_FULL_MODEL_SYNC_OK;
}

q16k_full_model_sync_status_t q16k_full_model_sync_capture_first_attention(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_aiter_layer_plan_t* layer_plan, size_t byte_count,
    uint64_t copy_begin_ns, uint64_t copy_end_ns, uint64_t copy_wait_count,
    const char sha256[65]) {
  if (diagnostic == NULL || layer_plan == NULL || sha256 == NULL ||
      !sha256_is_valid(sha256) ||
      byte_count != Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT ||
      copy_begin_ns == 0u || copy_end_ns < copy_begin_ns ||
      copy_wait_count != 1u) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      diagnostic->first_attention_pre_hash_captured == 0u ||
      diagnostic->first_attention_hash_captured != 0u ||
      !is_first_layer(layer_plan) ||
      diagnostic->first_attention_position_base != layer_plan->position_base ||
      diagnostic->first_attention_query_count !=
          layer_plan->logical_query_count ||
      diagnostic->first_attention_layer != layer_plan->layer) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  diagnostic->first_attention_copy_wait_attempted = 1u;
  diagnostic->first_attention_copy_wait_completed = 1u;
  diagnostic->first_attention_byte_count = byte_count;
  diagnostic->first_attention_copy_begin_ns = copy_begin_ns;
  diagnostic->first_attention_copy_end_ns = copy_end_ns;
  memcpy(diagnostic->first_attention_sha256, sha256,
         Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE);
  diagnostic->first_attention_copy_wait_count = copy_wait_count;
  diagnostic->first_attention_byte_equal =
      memcmp(diagnostic->first_attention_pre_sha256, sha256,
             Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE) == 0;
  diagnostic->first_attention_hash_captured = 1u;
  return Q16K_FULL_MODEL_SYNC_OK;
}

static int first_attention_difference_is_valid(
    const q16k_full_model_sync_first_attention_difference_t* difference) {
  if (difference == NULL || difference->analyzed == 0u ||
      difference->layout_valid == 0u ||
      difference->query_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_COUNT ||
      difference->head_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT ||
      difference->head_dimension !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_DIMENSION ||
      difference->query_tile_rows !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_ROWS ||
      difference->query_tile_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_COUNT ||
      difference->byte_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT ||
      difference->element_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_ELEMENT_COUNT ||
      difference->changed_byte_count > difference->byte_count ||
      difference->changed_element_count > difference->element_count ||
      difference->changed_byte_count < difference->changed_element_count ||
      difference->changed_byte_count >
          2u * difference->changed_element_count ||
      difference->changed_query_row_count > difference->query_count ||
      difference->changed_query_row_count >
          difference->changed_element_count) {
    return 0;
  }

  uint64_t per_head_total = 0u;
  for (uint32_t head = 0u; head < difference->head_count; ++head) {
    if (difference->per_head_changed_element_count[head] >
        (uint64_t)difference->query_count * difference->head_dimension) {
      return 0;
    }
    per_head_total += difference->per_head_changed_element_count[head];
  }
  uint64_t per_tile_total = 0u;
  for (uint32_t tile = 0u; tile < difference->query_tile_count; ++tile) {
    if (difference->per_query_tile_changed_element_count[tile] >
        (uint64_t)difference->query_tile_rows * difference->head_count *
            difference->head_dimension) {
      return 0;
    }
    per_tile_total += difference->per_query_tile_changed_element_count[tile];
  }
  if (per_head_total != difference->changed_element_count ||
      per_tile_total != difference->changed_element_count) {
    return 0;
  }

  if (difference->changed_byte_count == 0u) {
    if (difference->changed_element_count != 0u ||
        difference->changed_query_row_count != 0u ||
        difference->first_changed_byte != UINT64_MAX ||
        difference->last_changed_byte != UINT64_MAX ||
        difference->first_changed_element != UINT64_MAX ||
        difference->last_changed_element != UINT64_MAX ||
        difference->first_changed_query_row != UINT64_MAX ||
        difference->last_changed_query_row != UINT64_MAX ||
        difference->unchanged_prefix_byte_count != difference->byte_count ||
        difference->unchanged_suffix_byte_count != difference->byte_count ||
        difference->unchanged_prefix_element_count !=
            difference->element_count ||
        difference->unchanged_suffix_element_count !=
            difference->element_count) {
      return 0;
    }
    return 1;
  }

  const uint64_t elements_per_row =
      (uint64_t)difference->head_count * difference->head_dimension;
  return difference->changed_element_count != 0u &&
         difference->changed_query_row_count != 0u &&
         difference->first_changed_byte < difference->byte_count &&
         difference->last_changed_byte < difference->byte_count &&
         difference->first_changed_byte <= difference->last_changed_byte &&
         difference->first_changed_element < difference->element_count &&
         difference->last_changed_element < difference->element_count &&
         difference->first_changed_element <=
             difference->last_changed_element &&
         difference->first_changed_byte / sizeof(uint16_t) ==
             difference->first_changed_element &&
         difference->last_changed_byte / sizeof(uint16_t) ==
             difference->last_changed_element &&
         difference->first_changed_query_row < difference->query_count &&
         difference->last_changed_query_row < difference->query_count &&
         difference->first_changed_query_row <=
             difference->last_changed_query_row &&
         difference->first_changed_query_row ==
             difference->first_changed_element / elements_per_row &&
         difference->last_changed_query_row ==
             difference->last_changed_element / elements_per_row &&
         difference->unchanged_prefix_byte_count ==
             difference->first_changed_byte &&
         difference->unchanged_suffix_byte_count ==
             difference->byte_count - difference->last_changed_byte - 1u &&
         difference->unchanged_prefix_element_count ==
             difference->first_changed_element &&
         difference->unchanged_suffix_element_count ==
             difference->element_count - difference->last_changed_element - 1u;
}

q16k_full_model_sync_status_t
q16k_full_model_sync_analyze_first_attention_difference(
    const void* pre_attention_bytes, const void* post_attention_bytes,
    size_t byte_count, uint32_t query_count, uint32_t head_count,
    uint32_t head_dimension,
    q16k_full_model_sync_first_attention_difference_t* out_difference) {
  if (pre_attention_bytes == NULL || post_attention_bytes == NULL ||
      out_difference == NULL ||
      byte_count != Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT ||
      query_count != Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_COUNT ||
      head_count != Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_COUNT ||
      head_dimension !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_HEAD_DIMENSION) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }

  q16k_full_model_sync_first_attention_difference_t difference = {
      .analyzed = 1u,
      .layout_valid = 1u,
      .query_count = query_count,
      .head_count = head_count,
      .head_dimension = head_dimension,
      .query_tile_rows =
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_ROWS,
      .query_tile_count =
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_QUERY_TILE_COUNT,
      .byte_count = byte_count,
      .element_count = Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_ELEMENT_COUNT,
      .first_changed_byte = UINT64_MAX,
      .last_changed_byte = UINT64_MAX,
      .first_changed_element = UINT64_MAX,
      .last_changed_element = UINT64_MAX,
      .first_changed_query_row = UINT64_MAX,
      .last_changed_query_row = UINT64_MAX,
  };
  const uint8_t* pre = (const uint8_t*)pre_attention_bytes;
  const uint8_t* post = (const uint8_t*)post_attention_bytes;
  const uint64_t elements_per_row =
      (uint64_t)head_count * head_dimension;
  uint64_t last_counted_query_row = UINT64_MAX;

  for (uint64_t element = 0u; element < difference.element_count; ++element) {
    const uint64_t byte = element * sizeof(uint16_t);
    const int low_changed = pre[byte] != post[byte];
    const int high_changed = pre[byte + 1u] != post[byte + 1u];
    if (!low_changed && !high_changed) continue;

    if (low_changed) {
      if (difference.first_changed_byte == UINT64_MAX) {
        difference.first_changed_byte = byte;
      }
      difference.last_changed_byte = byte;
      difference.changed_byte_count++;
    }
    if (high_changed) {
      if (difference.first_changed_byte == UINT64_MAX) {
        difference.first_changed_byte = byte + 1u;
      }
      difference.last_changed_byte = byte + 1u;
      difference.changed_byte_count++;
    }

    if (difference.first_changed_element == UINT64_MAX) {
      difference.first_changed_element = element;
    }
    difference.last_changed_element = element;
    difference.changed_element_count++;

    const uint64_t query_row = element / elements_per_row;
    const uint64_t element_in_row = element % elements_per_row;
    const uint32_t head = (uint32_t)(element_in_row / head_dimension);
    const uint32_t query_tile =
        (uint32_t)(query_row / difference.query_tile_rows);
    difference.per_head_changed_element_count[head]++;
    difference.per_query_tile_changed_element_count[query_tile]++;
    if (query_row != last_counted_query_row) {
      if (difference.first_changed_query_row == UINT64_MAX) {
        difference.first_changed_query_row = query_row;
      }
      difference.last_changed_query_row = query_row;
      difference.changed_query_row_count++;
      last_counted_query_row = query_row;
    }
  }

  if (difference.changed_byte_count == 0u) {
    difference.unchanged_prefix_byte_count = byte_count;
    difference.unchanged_suffix_byte_count = byte_count;
    difference.unchanged_prefix_element_count = difference.element_count;
    difference.unchanged_suffix_element_count = difference.element_count;
  } else {
    difference.unchanged_prefix_byte_count = difference.first_changed_byte;
    difference.unchanged_suffix_byte_count =
        byte_count - difference.last_changed_byte - 1u;
    difference.unchanged_prefix_element_count =
        difference.first_changed_element;
    difference.unchanged_suffix_element_count =
        difference.element_count - difference.last_changed_element - 1u;
  }
  if (!first_attention_difference_is_valid(&difference)) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  *out_difference = difference;
  return Q16K_FULL_MODEL_SYNC_OK;
}

q16k_full_model_sync_status_t
q16k_full_model_sync_capture_first_attention_difference(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_full_model_sync_first_attention_difference_t* difference) {
  if (diagnostic == NULL ||
      !first_attention_difference_is_valid(difference)) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      diagnostic->first_attention_hash_captured == 0u ||
      diagnostic->first_attention_difference.analyzed != 0u ||
      diagnostic->first_attention_byte_equal !=
          (difference->changed_byte_count == 0u)) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  diagnostic->first_attention_difference = *difference;
  return Q16K_FULL_MODEL_SYNC_OK;
}

q16k_full_model_sync_status_t q16k_full_model_sync_capture_metadata_snapshot(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    q16k_full_model_sync_metadata_phase_t phase,
    const q16k_aiter_layer_plan_t* layer_plan, size_t byte_count,
    uint64_t copy_begin_ns, uint64_t copy_end_ns, uint64_t copy_wait_count,
    const int32_t words[Q16K_FULL_MODEL_SYNC_METADATA_WORD_COUNT],
    const char sha256[Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE]) {
  if (diagnostic == NULL || layer_plan == NULL || words == NULL ||
      !sha256_is_valid(sha256) || byte_count != Q16K_AITER_METADATA_SIZE ||
      copy_begin_ns == 0u || copy_end_ns < copy_begin_ns ||
      copy_wait_count != 1u ||
      (phase != Q16K_FULL_MODEL_SYNC_METADATA_POST_PACK_PRE_ATTENTION &&
       phase != Q16K_FULL_MODEL_SYNC_METADATA_POST_ATTENTION)) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      !is_first_layer(layer_plan)) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }
  q16k_full_model_sync_metadata_snapshot_t* snapshot =
      phase == Q16K_FULL_MODEL_SYNC_METADATA_POST_PACK_PRE_ATTENTION
          ? &diagnostic->post_pack_pre_attention_metadata
          : &diagnostic->post_attention_metadata;
  if (snapshot->captured != 0u) return Q16K_FULL_MODEL_SYNC_BAD_STATE;

  snapshot->copy_attempted = 1u;
  snapshot->copy_completed = 1u;
  snapshot->position_base = layer_plan->position_base;
  snapshot->logical_query_count = layer_plan->logical_query_count;
  snapshot->layer = layer_plan->layer;
  snapshot->byte_count = byte_count;
  snapshot->copy_begin_ns = copy_begin_ns;
  snapshot->copy_end_ns = copy_end_ns;
  snapshot->copy_wait_count = copy_wait_count;
  memcpy(snapshot->words, words, sizeof(snapshot->words));
  snapshot->expected_match =
      memcmp(words, kExpectedFirstMetadata, sizeof(snapshot->words)) == 0;
  memcpy(snapshot->sha256, sha256, Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE);
  snapshot->captured = 1u;
  diagnostic->metadata_copy_wait_count += copy_wait_count;
  return Q16K_FULL_MODEL_SYNC_OK;
}

q16k_full_model_sync_status_t q16k_full_model_sync_capture_first_layer_kernargs(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    const q16k_aiter_layer_plan_t* layer_plan, const void* pack_kernarg,
    size_t pack_kernarg_size, const char pack_sha256[65],
    const void* attention_kernarg, size_t attention_kernarg_size,
    const char attention_sha256[65]) {
  if (diagnostic == NULL || layer_plan == NULL || pack_kernarg == NULL ||
      attention_kernarg == NULL || !sha256_is_valid(pack_sha256) ||
      !sha256_is_valid(attention_sha256) ||
      pack_kernarg_size != Q16K_AITER_PACK_KERNARG_SIZE ||
      attention_kernarg_size != Q16K_AITER_ATTENTION_KERNARG_SIZE) {
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->request_finished != 0u ||
      diagnostic->first_layer_kernargs.captured != 0u ||
      !is_first_layer(layer_plan)) {
    return Q16K_FULL_MODEL_SYNC_BAD_STATE;
  }

  q16k_full_model_sync_first_layer_kernargs_t* captured =
      &diagnostic->first_layer_kernargs;
  captured->pack.captured = 1u;
  captured->pack.byte_count = pack_kernarg_size;
  memcpy(captured->pack.sha256, pack_sha256,
         Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE);
  memcpy(captured->pack.bytes, pack_kernarg, pack_kernarg_size);
  captured->attention.captured = 1u;
  captured->attention.byte_count = attention_kernarg_size;
  memcpy(captured->attention.sha256, attention_sha256,
         Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE);
  memcpy(captured->attention.bytes, attention_kernarg,
         attention_kernarg_size);

  const uint8_t* bytes = (const uint8_t*)attention_kernarg;
  q16k_full_model_sync_attention_pointers_t* pointers =
      &captured->attention_pointers;
  pointers->query = load_u64(bytes, 0u);
  pointers->key = load_u64(bytes, 8u);
  pointers->value = load_u64(bytes, 16u);
  pointers->output = load_u64(bytes, 24u);
  pointers->kv_indptr = load_u64(bytes, 72u);
  pointers->kv_page_indices = load_u64(bytes, 80u);
  pointers->kv_last_page_lens = load_u64(bytes, 88u);
  pointers->cu_seqlens_q = load_u64(bytes, 152u);
  pointers->match_plan =
      pointers->query == (uint64_t)(uintptr_t)layer_plan->attention.q &&
      pointers->key == (uint64_t)(uintptr_t)layer_plan->attention.k &&
      pointers->value == (uint64_t)(uintptr_t)layer_plan->attention.v &&
      pointers->output ==
          (uint64_t)(uintptr_t)layer_plan->attention.output &&
      pointers->kv_indptr ==
          (uint64_t)(uintptr_t)layer_plan->attention.kv_indptr &&
      pointers->kv_page_indices ==
          (uint64_t)(uintptr_t)layer_plan->attention.kv_page_indices &&
      pointers->kv_last_page_lens ==
          (uint64_t)(uintptr_t)layer_plan->attention.kv_last_page_lens &&
      pointers->cu_seqlens_q ==
          (uint64_t)(uintptr_t)layer_plan->attention.cu_seqlens_q;
  pointers->captured = 1u;
  captured->captured = 1u;
  return Q16K_FULL_MODEL_SYNC_OK;
}

void q16k_full_model_sync_finish(
    q16k_full_model_sync_diagnostic_t* diagnostic,
    uint32_t request_succeeded, uint32_t final_token_captured,
    int32_t final_token, uint64_t dispatch_count_end,
    uint64_t wait_count_end) {
  if (diagnostic == NULL || diagnostic->armed == 0u ||
      diagnostic->request_finished != 0u) {
    return;
  }
  diagnostic->request_succeeded = request_succeeded != 0u ? 1u : 0u;
  diagnostic->final_token_captured = final_token_captured != 0u ? 1u : 0u;
  diagnostic->final_token = final_token;
  diagnostic->total_dispatch_count =
      dispatch_count_end >= diagnostic->dispatch_count_begin
          ? dispatch_count_end - diagnostic->dispatch_count_begin
          : 0u;
  diagnostic->total_wait_count =
      wait_count_end >= diagnostic->wait_count_begin
          ? wait_count_end - diagnostic->wait_count_begin
          : 0u;
  diagnostic->stage_wait_count = 0u;
  for (uint32_t i = 0u; i < diagnostic->layer_record_count; ++i) {
    diagnostic->stage_wait_count +=
        diagnostic->layers[i].pack.wait_acquired != 0u ? 1u : 0u;
    diagnostic->stage_wait_count +=
        diagnostic->layers[i].attention.wait_acquired != 0u ? 1u : 0u;
  }
  diagnostic->request_finished = 1u;
}

static int stage_record_is_complete(
    const q16k_full_model_sync_stage_record_t* stage) {
  return stage->arm_attempted != 0u && stage->armed != 0u &&
         stage->publish_attempted != 0u && stage->published != 0u &&
         stage->wait_attempted != 0u && stage->wait_acquired != 0u &&
         stage->arm_status == Q16K_AITER_STATUS_OK &&
         stage->publish_status == Q16K_AITER_STATUS_OK &&
         stage->wait_status == Q16K_AITER_STATUS_OK && stage->arm_ns != 0u &&
         stage->arm_ns <= stage->publish_begin_ns &&
         stage->publish_begin_ns <= stage->publish_end_ns &&
         stage->publish_end_ns <= stage->wait_begin_ns &&
         stage->wait_begin_ns <= stage->wait_end_ns &&
         stage->cursor_before_publish != UINT32_MAX &&
         stage->cursor_after_publish == stage->cursor_before_publish + 1u &&
         stage->cursor_after_wait == 0u &&
         (stage->dispatch.packet_snapshot.full_header &
          UINT32_C(0x0000ffff)) ==
             Q16K_FULL_MODEL_SYNC_SYSTEM_HEADER &&
         ((stage->dispatch.packet_snapshot.full_header >> 16) &
          UINT32_C(0x3)) != 0u &&
         stage->dispatch.packet_snapshot.kernel_object != 0u &&
         stage->dispatch.packet_snapshot.kernarg_address != 0u &&
         stage->dispatch.packet_snapshot.completion_signal != 0u &&
         stage->dispatch.packet_id != UINT64_MAX &&
         stage->dispatch.kernarg_slot == stage->cursor_before_publish &&
         stage->dispatch.doorbell_written == 1u;
}

static int allocation_record_is_complete(
    const q16k_full_model_sync_allocation_record_t* allocation) {
  return allocation->captured != 0u && allocation->base != 0u &&
         allocation->size != 0u &&
         allocation->base <= UINT64_MAX - allocation->size &&
         allocation->end == allocation->base + allocation->size &&
         allocation->alignment != 0u &&
         allocation->alignment ==
             (allocation->base & (~allocation->base + 1u)) &&
         allocation->base_mod_16 == (allocation->base & UINT64_C(15)) &&
         allocation->base_mod_256 == (allocation->base & UINT64_C(255));
}

static int runtime_layout_is_complete(
    const q16k_full_model_sync_runtime_layout_t* layout) {
  return layout->captured != 0u &&
         allocation_record_is_complete(&layout->metadata) &&
         layout->metadata.size == Q16K_AITER_METADATA_SIZE &&
         allocation_record_is_complete(&layout->page_indices) &&
         allocation_record_is_complete(&layout->scratch) &&
         allocation_record_is_complete(&layout->loom_key) &&
         allocation_record_is_complete(&layout->loom_value) &&
         allocation_record_is_complete(&layout->shadow_key) &&
         allocation_record_is_complete(&layout->shadow_value) &&
         allocation_record_is_complete(&layout->rope) &&
         allocation_record_is_complete(&layout->params) &&
         allocation_record_is_complete(&layout->dense_workspace) &&
         allocation_record_is_complete(&layout->kernarg_ring);
}

static int metadata_snapshot_is_complete(
    const q16k_full_model_sync_metadata_snapshot_t* snapshot) {
  return snapshot->copy_attempted != 0u &&
         snapshot->copy_completed != 0u && snapshot->captured != 0u &&
         snapshot->expected_match != 0u &&
         snapshot->position_base == Q16K_AITER_LEGACY_PREFIX_ROWS &&
         snapshot->logical_query_count == Q16K_AITER_CHUNK_CAPACITY &&
         snapshot->layer == 0u &&
         snapshot->byte_count == Q16K_AITER_METADATA_SIZE &&
         snapshot->copy_begin_ns != 0u &&
         snapshot->copy_end_ns >= snapshot->copy_begin_ns &&
         snapshot->copy_wait_count == 1u &&
         sha256_is_valid(snapshot->sha256) &&
         memcmp(snapshot->words, kExpectedFirstMetadata,
                sizeof(snapshot->words)) == 0;
}

static int first_layer_kernargs_are_complete(
    const q16k_full_model_sync_first_layer_kernargs_t* kernargs) {
  return kernargs->captured != 0u && kernargs->pack.captured != 0u &&
         kernargs->pack.byte_count == Q16K_AITER_PACK_KERNARG_SIZE &&
         sha256_is_valid(kernargs->pack.sha256) &&
         kernargs->attention.captured != 0u &&
         kernargs->attention.byte_count == Q16K_AITER_ATTENTION_KERNARG_SIZE &&
         sha256_is_valid(kernargs->attention.sha256) &&
         kernargs->attention_pointers.captured != 0u &&
         kernargs->attention_pointers.match_plan != 0u;
}

int q16k_full_model_sync_is_complete(
    const q16k_full_model_sync_diagnostic_t* diagnostic) {
  if (diagnostic == NULL || diagnostic->armed == 0u ||
      diagnostic->request_finished == 0u ||
      diagnostic->request_succeeded == 0u ||
      !runtime_layout_is_complete(&diagnostic->runtime_layout) ||
      !metadata_snapshot_is_complete(
          &diagnostic->post_pack_pre_attention_metadata) ||
      !metadata_snapshot_is_complete(&diagnostic->post_attention_metadata) ||
      !first_layer_kernargs_are_complete(&diagnostic->first_layer_kernargs) ||
      diagnostic->layer_record_count !=
          Q16K_FULL_MODEL_SYNC_LAYER_RECORD_CAPACITY ||
      diagnostic->first_attention_pre_hash_captured == 0u ||
      diagnostic->first_attention_pre_copy_wait_attempted == 0u ||
      diagnostic->first_attention_pre_copy_wait_completed == 0u ||
      diagnostic->first_attention_pre_byte_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT ||
      diagnostic->first_attention_pre_copy_begin_ns == 0u ||
      diagnostic->first_attention_pre_copy_end_ns <
          diagnostic->first_attention_pre_copy_begin_ns ||
      diagnostic->first_attention_pre_copy_wait_count != 1u ||
      !sha256_is_valid(diagnostic->first_attention_pre_sha256) ||
      diagnostic->first_attention_hash_captured == 0u ||
      diagnostic->first_attention_copy_wait_attempted == 0u ||
      diagnostic->first_attention_copy_wait_completed == 0u ||
      diagnostic->first_attention_position_base !=
          Q16K_AITER_LEGACY_PREFIX_ROWS ||
      diagnostic->first_attention_query_count != Q16K_AITER_CHUNK_CAPACITY ||
      diagnostic->first_attention_layer != 0u ||
      diagnostic->first_attention_byte_count !=
          Q16K_FULL_MODEL_SYNC_FIRST_ATTENTION_BYTE_COUNT ||
      diagnostic->first_attention_copy_begin_ns == 0u ||
      diagnostic->first_attention_copy_end_ns <
          diagnostic->first_attention_copy_begin_ns ||
      diagnostic->first_attention_copy_wait_count != 1u ||
      !sha256_is_valid(diagnostic->first_attention_sha256) ||
      diagnostic->first_attention_byte_equal !=
          (memcmp(diagnostic->first_attention_pre_sha256,
                  diagnostic->first_attention_sha256,
                  Q16K_FULL_MODEL_SYNC_SHA256_STRING_SIZE) == 0) ||
      !first_attention_difference_is_valid(
          &diagnostic->first_attention_difference) ||
      diagnostic->first_attention_byte_equal !=
          (diagnostic->first_attention_difference.changed_byte_count == 0u) ||
      diagnostic->final_token_captured == 0u ||
      diagnostic->final_token < 0 ||
      diagnostic->final_token >= (int32_t)Q16K_AITER_VOCAB_SIZE ||
      diagnostic->stage_wait_count !=
          Q16K_FULL_MODEL_SYNC_EXPECTED_STAGE_WAIT_COUNT ||
      diagnostic->metadata_copy_wait_count !=
          Q16K_FULL_MODEL_SYNC_EXPECTED_METADATA_COPY_WAIT_COUNT ||
      diagnostic->first_attention_pre_copy_wait_count +
              diagnostic->first_attention_copy_wait_count !=
          Q16K_FULL_MODEL_SYNC_EXPECTED_FIRST_ATTENTION_COPY_WAIT_COUNT ||
      diagnostic->total_dispatch_count == 0u ||
      diagnostic->total_wait_count !=
          Q16K_FULL_MODEL_SYNC_EXPECTED_REQUEST_WAIT_COUNT) {
    return 0;
  }

  const q16k_full_model_sync_layer_record_t* first = &diagnostic->layers[0];
  if (first->pack.wait_end_ns >
          diagnostic->post_pack_pre_attention_metadata.copy_begin_ns ||
      diagnostic->post_pack_pre_attention_metadata.copy_end_ns >
          diagnostic->first_attention_pre_copy_begin_ns ||
      diagnostic->first_attention_pre_copy_end_ns >
          first->attention.arm_ns ||
      first->attention.wait_end_ns >
          diagnostic->post_attention_metadata.copy_begin_ns ||
      diagnostic->post_attention_metadata.copy_end_ns >
          diagnostic->first_attention_copy_begin_ns) {
    return 0;
  }

  for (uint32_t i = 0u; i < diagnostic->layer_record_count; ++i) {
    const q16k_full_model_sync_layer_record_t* record =
        &diagnostic->layers[i];
    const uint32_t expected_chunk = i / Q16K_AITER_LAYER_COUNT;
    const uint32_t expected_layer = i % Q16K_AITER_LAYER_COUNT;
    const uint32_t expected_base =
        Q16K_AITER_LEGACY_PREFIX_ROWS +
        expected_chunk * Q16K_AITER_CHUNK_CAPACITY;
    const uint32_t expected_count =
        expected_chunk + 1u == Q16K_AITER_MAX_CHUNK_COUNT
            ? Q16K_AITER_MAX_PROMPT_TOKEN_COUNT - expected_base
            : Q16K_AITER_CHUNK_CAPACITY;
    if (record->chunk_ordinal != expected_chunk ||
        record->position_base != expected_base ||
        record->logical_query_count != expected_count ||
        record->layer != expected_layer ||
        !stage_record_is_complete(&record->pack) ||
        !stage_record_is_complete(&record->attention) ||
        record->attention_post_wait_completed == 0u ||
        record->output_enqueued == 0u) {
      return 0;
    }
  }
  return 1;
}

static void set_error(char* error_message, size_t error_message_capacity,
                      const char* message) {
  if (error_message == NULL || error_message_capacity == 0u) return;
  snprintf(error_message, error_message_capacity, "%s", message);
}

static int write_allocation(
    FILE* file, const char* name,
    const q16k_full_model_sync_allocation_record_t* allocation,
    int trailing_comma) {
  return fprintf(
             file,
             "    \"%s\": {\"base\": \"0x%016" PRIx64
             "\", \"size\": %" PRIu64 ", \"end\": \"0x%016" PRIx64
             "\", \"alignment\": %" PRIu64
             ", \"base_mod_16\": %" PRIu32
             ", \"base_mod_256\": %" PRIu32 "}%s\n",
             name, allocation->base, allocation->size, allocation->end,
             allocation->alignment, allocation->base_mod_16,
             allocation->base_mod_256, trailing_comma ? "," : "") >= 0;
}

static int write_runtime_layout(
    FILE* file, const q16k_full_model_sync_runtime_layout_t* layout) {
  return fputs("  \"runtime_layout\": {\n", file) != EOF &&
         write_allocation(file, "metadata", &layout->metadata, 1) &&
         write_allocation(file, "page_indices", &layout->page_indices, 1) &&
         write_allocation(file, "scratch", &layout->scratch, 1) &&
         write_allocation(file, "loom_key", &layout->loom_key, 1) &&
         write_allocation(file, "loom_value", &layout->loom_value, 1) &&
         write_allocation(file, "shadow_key", &layout->shadow_key, 1) &&
         write_allocation(file, "shadow_value", &layout->shadow_value, 1) &&
         write_allocation(file, "rope", &layout->rope, 1) &&
         write_allocation(file, "params", &layout->params, 1) &&
         write_allocation(file, "dense_workspace", &layout->dense_workspace,
                          1) &&
         write_allocation(file, "kernarg_ring", &layout->kernarg_ring, 0) &&
         fputs("  },\n", file) != EOF;
}

static int write_metadata_snapshot(
    FILE* file, const char* name,
    const q16k_full_model_sync_metadata_snapshot_t* snapshot,
    int trailing_comma) {
  return fprintf(
             file,
             "    \"%s\": {\"position_base\": %" PRIu32
             ", \"query_count\": %" PRIu32 ", \"layer\": %" PRIu32
             ", \"byte_count\": %zu, \"copy_begin_ns\": %" PRIu64
             ", \"copy_end_ns\": %" PRIu64
             ", \"copy_waits\": %" PRIu64
             ", \"sha256\": \"%s\", \"words\": [%" PRId32 ", %" PRId32
             ", %" PRId32 ", %" PRId32 ", %" PRId32 ", %" PRId32
             "], \"expected_match\": %s}%s\n",
             name, snapshot->position_base, snapshot->logical_query_count,
             snapshot->layer, snapshot->byte_count, snapshot->copy_begin_ns,
             snapshot->copy_end_ns, snapshot->copy_wait_count,
             snapshot->sha256, snapshot->words[0], snapshot->words[1],
             snapshot->words[2], snapshot->words[3], snapshot->words[4],
             snapshot->words[5],
             snapshot->expected_match ? "true" : "false",
             trailing_comma ? "," : "") >= 0;
}

static int write_hex_bytes(FILE* file, const uint8_t* bytes, size_t count) {
  static const char kHex[] = "0123456789abcdef";
  for (size_t i = 0u; i < count; ++i) {
    const char encoded[2] = {kHex[bytes[i] >> 4], kHex[bytes[i] & 15u]};
    if (fwrite(encoded, 1u, sizeof(encoded), file) != sizeof(encoded)) return 0;
  }
  return 1;
}

static int write_kernarg_snapshot(
    FILE* file, const char* name,
    const q16k_full_model_sync_kernarg_snapshot_t* snapshot) {
  if (fprintf(file,
              "    \"%s\": {\"byte_count\": %zu, \"sha256\": \"%s\", "
              "\"snapshot_hex\": \"",
              name, snapshot->byte_count, snapshot->sha256) < 0 ||
      !write_hex_bytes(file, snapshot->bytes, snapshot->byte_count)) {
    return 0;
  }
  return fputs("\"},\n", file) != EOF;
}

static int write_first_layer_kernargs(
    FILE* file,
    const q16k_full_model_sync_first_layer_kernargs_t* kernargs) {
  const q16k_full_model_sync_attention_pointers_t* pointers =
      &kernargs->attention_pointers;
  return fputs("  \"first_layer_kernargs\": {\n", file) != EOF &&
         write_kernarg_snapshot(file, "pack", &kernargs->pack) &&
         write_kernarg_snapshot(file, "attention", &kernargs->attention) &&
         fprintf(
             file,
             "    \"attention_pointers\": {\"query\": \"0x%016" PRIx64
             "\", \"key\": \"0x%016" PRIx64
             "\", \"value\": \"0x%016" PRIx64
             "\", \"output\": \"0x%016" PRIx64
             "\", \"kv_indptr\": \"0x%016" PRIx64
             "\", \"kv_page_indices\": \"0x%016" PRIx64
             "\", \"kv_last_page_lens\": \"0x%016" PRIx64
             "\", \"cu_seqlens_q\": \"0x%016" PRIx64
             "\", \"match_plan\": %s}\n  },\n",
             pointers->query, pointers->key, pointers->value,
             pointers->output, pointers->kv_indptr,
             pointers->kv_page_indices, pointers->kv_last_page_lens,
             pointers->cu_seqlens_q,
             pointers->match_plan ? "true" : "false") >= 0;
}

static int write_nullable_u64(FILE* file, uint64_t value) {
  return value == UINT64_MAX ? fputs("null", file) != EOF
                             : fprintf(file, "%" PRIu64, value) >= 0;
}

static int write_u64_array(FILE* file, const uint64_t* values,
                           uint32_t count) {
  if (fputc('[', file) == EOF) return 0;
  for (uint32_t i = 0u; i < count; ++i) {
    if ((i != 0u && fputs(", ", file) == EOF) ||
        fprintf(file, "%" PRIu64, values[i]) < 0) {
      return 0;
    }
  }
  return fputc(']', file) != EOF;
}

static int write_first_attention_difference(
    FILE* file,
    const q16k_full_model_sync_first_attention_difference_t* difference) {
  if (fprintf(file,
              "    \"difference\": {\"layout_valid\": %s"
              ", \"shape\": [%" PRIu32 ", %" PRIu32 ", %" PRIu32 "]"
              ", \"element_type\": \"bf16\", \"element_byte_count\": 2"
              ", \"query_tile_rows\": %" PRIu32
              ", \"query_tile_count\": %" PRIu32
              ", \"byte_count\": %zu, \"element_count\": %" PRIu64
              ", \"changed_bytes\": %" PRIu64
              ", \"changed_elements\": %" PRIu64
              ", \"first_changed_byte\": ",
              difference->layout_valid ? "true" : "false",
              difference->query_count, difference->head_count,
              difference->head_dimension, difference->query_tile_rows,
              difference->query_tile_count, difference->byte_count,
              difference->element_count, difference->changed_byte_count,
              difference->changed_element_count) < 0 ||
      !write_nullable_u64(file, difference->first_changed_byte) ||
      fputs(", \"last_changed_byte\": ", file) == EOF ||
      !write_nullable_u64(file, difference->last_changed_byte) ||
      fputs(", \"first_changed_element\": ", file) == EOF ||
      !write_nullable_u64(file, difference->first_changed_element) ||
      fputs(", \"last_changed_element\": ", file) == EOF ||
      !write_nullable_u64(file, difference->last_changed_element) ||
      fprintf(file,
              ", \"changed_query_rows\": %" PRIu64
              ", \"first_changed_query_row\": ",
              difference->changed_query_row_count) < 0 ||
      !write_nullable_u64(file, difference->first_changed_query_row) ||
      fputs(", \"last_changed_query_row\": ", file) == EOF ||
      !write_nullable_u64(file, difference->last_changed_query_row) ||
      fprintf(file,
              ", \"unchanged_prefix_bytes\": %" PRIu64
              ", \"unchanged_suffix_bytes\": %" PRIu64
              ", \"unchanged_prefix_elements\": %" PRIu64
              ", \"unchanged_suffix_elements\": %" PRIu64
              ", \"per_head_changed_elements\": ",
              difference->unchanged_prefix_byte_count,
              difference->unchanged_suffix_byte_count,
              difference->unchanged_prefix_element_count,
              difference->unchanged_suffix_element_count) < 0 ||
      !write_u64_array(file, difference->per_head_changed_element_count,
                       difference->head_count) ||
      fputs(", \"per_query_tile_changed_elements\": ", file) == EOF ||
      !write_u64_array(file,
                       difference->per_query_tile_changed_element_count,
                       difference->query_tile_count)) {
    return 0;
  }
  return fputs("}\n", file) != EOF;
}

static int write_stage(FILE* file, const char* name,
                       const q16k_full_model_sync_stage_record_t* stage) {
  return fprintf(
             file,
             "      \"%s\": {\n"
             "        \"arm\": {\"attempted\": %s, \"succeeded\": %s, "
             "\"timestamp_ns\": %" PRIu64 ", \"status\": \"%s\"},\n"
             "        \"publish\": {\"attempted\": %s, \"succeeded\": "
             "%s, \"begin_ns\": %" PRIu64 ", \"end_ns\": %" PRIu64
             ", \"status\": \"%s\"},\n"
             "        \"wait\": {\"attempted\": %s, \"acquired\": %s, "
             "\"begin_ns\": %" PRIu64 ", \"end_ns\": %" PRIu64
             ", \"status\": \"%s\"},\n"
             "        \"cursor\": {\"before_publish\": %" PRIu32
             ", \"after_publish\": %" PRIu32
             ", \"after_wait\": %" PRIu32 "},\n"
             "        \"packet\": {\"id\": %" PRIu64
             ", \"full_header\": \"0x%08" PRIx32
             "\", \"kernarg_slot\": %" PRIu32
             ", \"kernel_object\": \"0x%016" PRIx64
             "\", \"kernarg_address\": \"0x%016" PRIx64
             "\", \"group_segment_size\": %" PRIu32
             ", \"private_segment_size\": %" PRIu32
             ", \"completion_signal\": \"0x%016" PRIx64
             "\", \"doorbell_written\": %" PRIu32 "}\n"
             "      }",
             name, stage->arm_attempted ? "true" : "false",
             stage->armed ? "true" : "false", stage->arm_ns,
             q16k_aiter_status_string(stage->arm_status),
             stage->publish_attempted ? "true" : "false",
             stage->published ? "true" : "false", stage->publish_begin_ns,
             stage->publish_end_ns,
             q16k_aiter_status_string(stage->publish_status),
             stage->wait_attempted ? "true" : "false",
             stage->wait_acquired ? "true" : "false", stage->wait_begin_ns,
             stage->wait_end_ns,
             q16k_aiter_status_string(stage->wait_status),
             stage->cursor_before_publish, stage->cursor_after_publish,
             stage->cursor_after_wait, stage->dispatch.packet_id,
             stage->dispatch.packet_snapshot.full_header,
             stage->dispatch.kernarg_slot,
             stage->dispatch.packet_snapshot.kernel_object,
             stage->dispatch.packet_snapshot.kernarg_address,
             stage->dispatch.packet_snapshot.group_segment_size,
             stage->dispatch.packet_snapshot.private_segment_size,
             stage->dispatch.packet_snapshot.completion_signal,
             stage->dispatch.doorbell_written) >= 0;
}

static int write_json(FILE* file,
                      const q16k_full_model_sync_diagnostic_t* diagnostic) {
  if (fprintf(
          file,
          "{\n  \"schema\": \"%s\",\n"
          "  \"complete\": %s,\n"
          "  \"request\": {\"id\": %" PRIu64
          ", \"ordinal\": %" PRIu32
          ", \"finished\": %s, \"succeeded\": %s},\n",
          Q16K_FULL_MODEL_SYNC_DIAGNOSTIC_SCHEMA,
          q16k_full_model_sync_is_complete(diagnostic) ? "true" : "false",
          diagnostic->request_id, diagnostic->request_ordinal,
          diagnostic->request_finished ? "true" : "false",
          diagnostic->request_succeeded ? "true" : "false") < 0 ||
      !write_runtime_layout(file, &diagnostic->runtime_layout) ||
      fputs("  \"metadata_lifetime\": {\n"
            "    \"expected_words\": [0, 16768, 1, 0, 0, 16384],\n",
            file) == EOF ||
      !write_metadata_snapshot(
          file, "post_pack_pre_attention",
          &diagnostic->post_pack_pre_attention_metadata, 1) ||
      !write_metadata_snapshot(file, "post_attention",
                               &diagnostic->post_attention_metadata, 0) ||
      fputs("  },\n", file) == EOF ||
      !write_first_layer_kernargs(file, &diagnostic->first_layer_kernargs) ||
      fprintf(
          file,
          "  \"first_attention\": {\"position_base\": %" PRIu32
          ", \"query_count\": %" PRIu32 ", \"layer\": %" PRIu32
          ", \"pre_copy_attempted\": %s, \"pre_copy_completed\": %s"
          ", \"pre_byte_count\": %zu, \"pre_copy_begin_ns\": %" PRIu64
          ", \"pre_copy_end_ns\": %" PRIu64
          ", \"pre_copy_waits\": %" PRIu64
          ", \"pre_sha256\": \"%s\""
          ", \"post_copy_attempted\": %s, \"post_copy_completed\": %s"
          ", \"byte_count\": %zu, \"copy_begin_ns\": %" PRIu64
          ", \"copy_end_ns\": %" PRIu64
          ", \"copy_waits\": %" PRIu64 ", \"sha256\": \"%s\""
          ", \"byte_equal\": %s,\n",
          diagnostic->first_attention_position_base,
          diagnostic->first_attention_query_count,
          diagnostic->first_attention_layer,
          diagnostic->first_attention_pre_copy_wait_attempted ? "true"
                                                              : "false",
          diagnostic->first_attention_pre_copy_wait_completed ? "true"
                                                              : "false",
          diagnostic->first_attention_pre_byte_count,
          diagnostic->first_attention_pre_copy_begin_ns,
          diagnostic->first_attention_pre_copy_end_ns,
          diagnostic->first_attention_pre_copy_wait_count,
          diagnostic->first_attention_pre_sha256,
          diagnostic->first_attention_copy_wait_attempted ? "true" : "false",
          diagnostic->first_attention_copy_wait_completed ? "true" : "false",
          diagnostic->first_attention_byte_count,
          diagnostic->first_attention_copy_begin_ns,
          diagnostic->first_attention_copy_end_ns,
          diagnostic->first_attention_copy_wait_count,
          diagnostic->first_attention_sha256,
          diagnostic->first_attention_byte_equal ? "true" : "false") < 0 ||
      !write_first_attention_difference(
          file, &diagnostic->first_attention_difference) ||
      fputs("  },\n", file) == EOF ||
      fprintf(
          file,
          "  \"final_token\": %" PRId32 ",\n"
          "  \"totals\": {\"dispatches\": %" PRIu64
          ", \"waits\": %" PRIu64 ", \"stage_waits\": %" PRIu64
          ", \"metadata_copy_waits\": %" PRIu64
          ", \"first_attention_copy_waits\": %" PRIu64 "},\n"
          "  \"layers\": [\n",
          diagnostic->final_token,
          diagnostic->total_dispatch_count, diagnostic->total_wait_count,
          diagnostic->stage_wait_count,
          diagnostic->metadata_copy_wait_count,
          diagnostic->first_attention_pre_copy_wait_count +
              diagnostic->first_attention_copy_wait_count) < 0) {
    return 0;
  }
  for (uint32_t i = 0u; i < diagnostic->layer_record_count; ++i) {
    const q16k_full_model_sync_layer_record_t* record =
        &diagnostic->layers[i];
    if (fprintf(file,
                "%s    {\"chunk_ordinal\": %" PRIu32
                ", \"position_base\": %" PRIu32
                ", \"query_count\": %" PRIu32 ", \"layer\": %" PRIu32
                ",\n",
                i == 0u ? "" : ",\n", record->chunk_ordinal,
                record->position_base, record->logical_query_count,
                record->layer) < 0 ||
        !write_stage(file, "pack", &record->pack) ||
        fputs(",\n", file) == EOF ||
        !write_stage(file, "attention", &record->attention) ||
        fprintf(file,
                ",\n      \"attention_post_wait_completed\": %s,\n"
                "      \"output_enqueued\": %s\n    }",
                record->attention_post_wait_completed ? "true" : "false",
                record->output_enqueued ? "true" : "false") < 0) {
      return 0;
    }
  }
  return fputs("\n  ]\n}\n", file) != EOF;
}

q16k_full_model_sync_status_t q16k_full_model_sync_write_json_atomic(
    const q16k_full_model_sync_diagnostic_t* diagnostic,
    const char* output_path, char* error_message,
    size_t error_message_capacity) {
  if (diagnostic == NULL || output_path == NULL || output_path[0] != '/') {
    set_error(error_message, error_message_capacity,
              "diagnostic and an absolute output path are required");
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }

  const size_t path_length = strlen(output_path);
  if (path_length > SIZE_MAX - 64u) {
    set_error(error_message, error_message_capacity,
              "diagnostic output path is too long");
    return Q16K_FULL_MODEL_SYNC_INVALID_ARGUMENT;
  }
  char* temporary_path = (char*)malloc(path_length + 64u);
  if (temporary_path == NULL) {
    set_error(error_message, error_message_capacity,
              "allocating temporary output path failed");
    return Q16K_FULL_MODEL_SYNC_IO_ERROR;
  }

  int descriptor = -1;
  for (unsigned attempt = 0u; attempt < 128u; ++attempt) {
    const int length = snprintf(temporary_path, path_length + 64u,
                                "%s.tmp.%ld.%u", output_path,
                                (long)getpid(), attempt);
    if (length < 0 || (size_t)length >= path_length + 64u) {
      free(temporary_path);
      set_error(error_message, error_message_capacity,
                "formatting temporary output path failed");
      return Q16K_FULL_MODEL_SYNC_IO_ERROR;
    }
    descriptor = open(temporary_path,
                      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                      S_IRUSR | S_IWUSR);
    if (descriptor >= 0 || errno != EEXIST) break;
  }
  if (descriptor < 0) {
    char message[256];
    snprintf(message, sizeof(message), "opening diagnostic output failed: %s",
             strerror(errno));
    set_error(error_message, error_message_capacity, message);
    free(temporary_path);
    return Q16K_FULL_MODEL_SYNC_IO_ERROR;
  }
  FILE* file = fdopen(descriptor, "wb");
  if (file == NULL) {
    const int saved_errno = errno;
    close(descriptor);
    unlink(temporary_path);
    char message[256];
    snprintf(message, sizeof(message), "opening diagnostic stream failed: %s",
             strerror(saved_errno));
    set_error(error_message, error_message_capacity, message);
    free(temporary_path);
    return Q16K_FULL_MODEL_SYNC_IO_ERROR;
  }

  int ok = write_json(file, diagnostic);
  if (ok && fflush(file) != 0) ok = 0;
  if (ok && fsync(fileno(file)) != 0) ok = 0;
  if (fclose(file) != 0) ok = 0;
  if (!ok) {
    unlink(temporary_path);
    set_error(error_message, error_message_capacity,
              "writing diagnostic JSON failed");
    free(temporary_path);
    return Q16K_FULL_MODEL_SYNC_IO_ERROR;
  }
  if (link(temporary_path, output_path) != 0) {
    char message[256];
    snprintf(message, sizeof(message),
             "publishing diagnostic without replacement failed: %s",
             strerror(errno));
    unlink(temporary_path);
    set_error(error_message, error_message_capacity, message);
    free(temporary_path);
    return Q16K_FULL_MODEL_SYNC_IO_ERROR;
  }
  unlink(temporary_path);
  free(temporary_path);
  if (error_message != NULL && error_message_capacity != 0u) {
    error_message[0] = '\0';
  }
  return q16k_full_model_sync_is_complete(diagnostic)
             ? Q16K_FULL_MODEL_SYNC_OK
             : Q16K_FULL_MODEL_SYNC_INCOMPLETE;
}
