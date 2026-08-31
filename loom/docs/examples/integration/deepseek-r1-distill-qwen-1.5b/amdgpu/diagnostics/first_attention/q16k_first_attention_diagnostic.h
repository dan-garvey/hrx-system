#ifndef DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC_H_
#define DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC_H_

#include <stddef.h>
#include <stdint.h>

#include "../../q16k_aiter_integration.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q16K_FIRST_ATTENTION_DIAGNOSTIC_SCHEMA \
  "loom.q16k.first_attention.v1"
#define Q16K_FIRST_ATTENTION_KERNEL_NAME_CAPACITY 1024u

typedef enum q16k_first_attention_diagnostic_status_e {
  Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK = 0,
  Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT,
  Q16K_FIRST_ATTENTION_DIAGNOSTIC_BAD_STATE,
  Q16K_FIRST_ATTENTION_DIAGNOSTIC_INCOMPLETE,
  Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR,
} q16k_first_attention_diagnostic_status_t;

typedef struct q16k_first_attention_diagnostic_s {
  uint32_t armed;
  uint32_t host_metadata_captured;
  uint32_t gpu_metadata_captured;
  uint32_t attention_dispatch_captured;
  uint32_t request_finished;
  uint32_t request_succeeded;
  uint32_t next_sequence;
  uint32_t host_metadata_sequence;
  uint32_t gpu_metadata_sequence;
  uint32_t attention_dispatch_sequence;
  uint32_t request_finished_sequence;
  uint64_t request_id;
  uint32_t request_ordinal;
  uint32_t layer;
  uint32_t position_base;
  uint32_t logical_query_count;
  uint64_t gpu_metadata_address;
  q16k_aiter_attention_metadata_t host_metadata;
  q16k_aiter_attention_metadata_t gpu_metadata;
  char kernel_name[Q16K_FIRST_ATTENTION_KERNEL_NAME_CAPACITY];
  q16k_aiter_loaded_kernel_t kernel;
  q16k_aiter_launch_geometry_t geometry;
  q16k_aiter_dispatch_packet_t packet;
  q16k_aiter_dispatch_result_t dispatch_result;
  uint8_t kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  size_t kernarg_size;
  uint32_t source_kernarg_matches_ring;
} q16k_first_attention_diagnostic_t;

const char* q16k_first_attention_diagnostic_status_string(
    q16k_first_attention_diagnostic_status_t status);

void q16k_first_attention_diagnostic_begin(
    q16k_first_attention_diagnostic_t* diagnostic, uint64_t request_id,
    uint32_t request_ordinal);

q16k_first_attention_diagnostic_status_t
q16k_first_attention_diagnostic_capture_host_metadata(
    q16k_first_attention_diagnostic_t* diagnostic, uint32_t position_base,
    uint32_t logical_query_count, uint64_t gpu_metadata_address,
    const q16k_aiter_attention_metadata_t* metadata);

q16k_first_attention_diagnostic_status_t
q16k_first_attention_diagnostic_capture_gpu_metadata(
    q16k_first_attention_diagnostic_t* diagnostic,
    const q16k_aiter_attention_metadata_t* metadata);

void q16k_first_attention_diagnostic_observe_attention(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan,
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_launch_geometry_t* geometry,
    const void* source_kernarg, size_t source_kernarg_size,
    const q16k_aiter_dispatch_packet_t* published_packet,
    const q16k_aiter_dispatch_result_t* dispatch_result);

void q16k_first_attention_diagnostic_finish_request(
    q16k_first_attention_diagnostic_t* diagnostic,
    uint32_t request_succeeded);

int q16k_first_attention_diagnostic_is_complete(
    const q16k_first_attention_diagnostic_t* diagnostic);

q16k_first_attention_diagnostic_status_t
q16k_first_attention_diagnostic_write_json_atomic(
    const q16k_first_attention_diagnostic_t* diagnostic,
    const char* output_path, char* error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif  /* DEEPSEEK_Q16K_FIRST_ATTENTION_DIAGNOSTIC_H_ */
