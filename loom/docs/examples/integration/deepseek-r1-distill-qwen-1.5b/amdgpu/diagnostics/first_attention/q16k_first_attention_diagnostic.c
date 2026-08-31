#define _POSIX_C_SOURCE 200809L

#include "q16k_first_attention_diagnostic.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t next_event_sequence(
    q16k_first_attention_diagnostic_t* diagnostic) {
  diagnostic->next_sequence++;
  return diagnostic->next_sequence;
}

static uint64_t load_u64(const uint8_t* bytes, size_t offset) {
  uint64_t value = 0u;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

static uint32_t load_u32(const uint8_t* bytes, size_t offset) {
  uint32_t value = 0u;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

static int32_t load_i32(const uint8_t* bytes, size_t offset) {
  int32_t value = 0;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

static uint16_t load_u16(const uint8_t* bytes, size_t offset) {
  uint16_t value = 0u;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

static float load_f32(const uint8_t* bytes, size_t offset) {
  float value = 0.0f;
  memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

static void set_error(char* error_message, size_t error_message_capacity,
                      const char* message) {
  if (error_message == NULL || error_message_capacity == 0u) return;
  snprintf(error_message, error_message_capacity, "%s", message);
}

const char* q16k_first_attention_diagnostic_status_string(
    q16k_first_attention_diagnostic_status_t status) {
  switch (status) {
    case Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK:
      return "ok";
    case Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT:
      return "invalid argument";
    case Q16K_FIRST_ATTENTION_DIAGNOSTIC_BAD_STATE:
      return "bad state";
    case Q16K_FIRST_ATTENTION_DIAGNOSTIC_INCOMPLETE:
      return "incomplete";
    case Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR:
      return "I/O error";
  }
  return "unknown";
}

void q16k_first_attention_diagnostic_begin(
    q16k_first_attention_diagnostic_t* diagnostic, uint64_t request_id,
    uint32_t request_ordinal) {
  if (diagnostic == NULL) return;
  memset(diagnostic, 0, sizeof(*diagnostic));
  diagnostic->armed = 1u;
  diagnostic->request_id = request_id;
  diagnostic->request_ordinal = request_ordinal;
  diagnostic->layer = UINT32_MAX;
  diagnostic->dispatch_result.packet_id = UINT64_MAX;
  diagnostic->dispatch_result.kernarg_slot = UINT32_MAX;
}

q16k_first_attention_diagnostic_status_t
q16k_first_attention_diagnostic_capture_host_metadata(
    q16k_first_attention_diagnostic_t* diagnostic, uint32_t position_base,
    uint32_t logical_query_count, uint64_t gpu_metadata_address,
    const q16k_aiter_attention_metadata_t* metadata) {
  if (diagnostic == NULL || metadata == NULL || position_base == 0u ||
      logical_query_count == 0u || gpu_metadata_address == 0u) {
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u || diagnostic->host_metadata_captured != 0u ||
      diagnostic->attention_dispatch_captured != 0u) {
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_BAD_STATE;
  }
  diagnostic->position_base = position_base;
  diagnostic->logical_query_count = logical_query_count;
  diagnostic->gpu_metadata_address = gpu_metadata_address;
  diagnostic->host_metadata = *metadata;
  diagnostic->host_metadata_captured = 1u;
  diagnostic->host_metadata_sequence = next_event_sequence(diagnostic);
  return Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK;
}

q16k_first_attention_diagnostic_status_t
q16k_first_attention_diagnostic_capture_gpu_metadata(
    q16k_first_attention_diagnostic_t* diagnostic,
    const q16k_aiter_attention_metadata_t* metadata) {
  if (diagnostic == NULL || metadata == NULL) {
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT;
  }
  if (diagnostic->armed == 0u ||
      diagnostic->host_metadata_captured == 0u ||
      diagnostic->gpu_metadata_captured != 0u ||
      diagnostic->attention_dispatch_captured != 0u) {
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_BAD_STATE;
  }
  diagnostic->gpu_metadata = *metadata;
  diagnostic->gpu_metadata_captured = 1u;
  diagnostic->gpu_metadata_sequence = next_event_sequence(diagnostic);
  return Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK;
}

void q16k_first_attention_diagnostic_observe_attention(
    void* user_data, const q16k_aiter_layer_plan_t* layer_plan,
    const q16k_aiter_loaded_kernel_t* kernel,
    const q16k_aiter_launch_geometry_t* geometry,
    const void* source_kernarg, size_t source_kernarg_size,
    const q16k_aiter_dispatch_packet_t* published_packet,
    const q16k_aiter_dispatch_result_t* dispatch_result) {
  q16k_first_attention_diagnostic_t* diagnostic =
      (q16k_first_attention_diagnostic_t*)user_data;
  if (diagnostic == NULL || diagnostic->armed == 0u ||
      diagnostic->host_metadata_captured == 0u ||
      diagnostic->gpu_metadata_captured == 0u ||
      diagnostic->attention_dispatch_captured != 0u || layer_plan == NULL ||
      kernel == NULL || geometry == NULL || source_kernarg == NULL ||
      source_kernarg_size != Q16K_AITER_ATTENTION_KERNARG_SIZE ||
      published_packet == NULL || dispatch_result == NULL ||
      published_packet->kernarg_address == 0u || kernel->name == NULL ||
      strlen(kernel->name) >= sizeof(diagnostic->kernel_name) ||
      layer_plan->layer != 0u ||
      layer_plan->position_base != diagnostic->position_base ||
      layer_plan->logical_query_count != diagnostic->logical_query_count) {
    return;
  }

  diagnostic->layer = layer_plan->layer;
  memcpy(diagnostic->kernel_name, kernel->name, strlen(kernel->name) + 1u);
  diagnostic->kernel = *kernel;
  diagnostic->kernel.name = diagnostic->kernel_name;
  diagnostic->geometry = *geometry;
  diagnostic->packet = *published_packet;
  diagnostic->dispatch_result = *dispatch_result;
  diagnostic->kernarg_size = source_kernarg_size;

  const void* ring_kernarg =
      (const void*)(uintptr_t)published_packet->kernarg_address;
  memcpy(diagnostic->kernarg, ring_kernarg, source_kernarg_size);
  diagnostic->source_kernarg_matches_ring =
      memcmp(source_kernarg, diagnostic->kernarg, source_kernarg_size) == 0
          ? 1u
          : 0u;
  diagnostic->attention_dispatch_captured = 1u;
  diagnostic->attention_dispatch_sequence = next_event_sequence(diagnostic);
}

void q16k_first_attention_diagnostic_finish_request(
    q16k_first_attention_diagnostic_t* diagnostic,
    uint32_t request_succeeded) {
  if (diagnostic == NULL || diagnostic->armed == 0u ||
      diagnostic->request_finished != 0u) {
    return;
  }
  diagnostic->request_succeeded = request_succeeded != 0u ? 1u : 0u;
  diagnostic->request_finished = 1u;
  diagnostic->request_finished_sequence = next_event_sequence(diagnostic);
}

int q16k_first_attention_diagnostic_is_complete(
    const q16k_first_attention_diagnostic_t* diagnostic) {
  return diagnostic != NULL && diagnostic->armed != 0u &&
         diagnostic->host_metadata_captured != 0u &&
         diagnostic->gpu_metadata_captured != 0u &&
         diagnostic->attention_dispatch_captured != 0u &&
         diagnostic->request_finished != 0u &&
         diagnostic->kernarg_size == Q16K_AITER_ATTENTION_KERNARG_SIZE &&
         diagnostic->host_metadata_sequence <
             diagnostic->gpu_metadata_sequence &&
         diagnostic->gpu_metadata_sequence <
             diagnostic->attention_dispatch_sequence &&
         diagnostic->attention_dispatch_sequence <
             diagnostic->request_finished_sequence;
}

static int write_json_string(FILE* file, const char* value) {
  if (fputc('"', file) == EOF) return 0;
  for (const unsigned char* p = (const unsigned char*)value; *p != '\0'; ++p) {
    switch (*p) {
      case '"':
        if (fputs("\\\"", file) == EOF) return 0;
        break;
      case '\\':
        if (fputs("\\\\", file) == EOF) return 0;
        break;
      case '\n':
        if (fputs("\\n", file) == EOF) return 0;
        break;
      case '\r':
        if (fputs("\\r", file) == EOF) return 0;
        break;
      case '\t':
        if (fputs("\\t", file) == EOF) return 0;
        break;
      default:
        if (*p < 0x20u) {
          if (fprintf(file, "\\u%04x", (unsigned)*p) < 0) return 0;
        } else if (fputc(*p, file) == EOF) {
          return 0;
        }
        break;
    }
  }
  return fputc('"', file) != EOF;
}

static int write_metadata(FILE* file,
                          const q16k_aiter_attention_metadata_t* metadata) {
  int32_t values[6];
  memcpy(values, metadata, sizeof(values));
  return fprintf(file,
                 "[%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32
                 ",%" PRId32 ",%" PRId32 "]",
                 values[0], values[1], values[2], values[3], values[4],
                 values[5]) >= 0;
}

static int write_kernarg_hex(FILE* file, const uint8_t* bytes, size_t size) {
  if (fputc('"', file) == EOF) return 0;
  for (size_t i = 0u; i < size; ++i) {
    if (fprintf(file, "%02x", bytes[i]) < 0) return 0;
  }
  return fputc('"', file) != EOF;
}

static int write_json(FILE* file,
                      const q16k_first_attention_diagnostic_t* diagnostic) {
  const uint8_t* kernarg = diagnostic->kernarg;
  const int complete = q16k_first_attention_diagnostic_is_complete(diagnostic);
  const int metadata_match =
      diagnostic->host_metadata_captured != 0u &&
      diagnostic->gpu_metadata_captured != 0u &&
      memcmp(&diagnostic->host_metadata, &diagnostic->gpu_metadata,
             sizeof(diagnostic->host_metadata)) == 0;
  const uint16_t packet_dimensions =
      (uint16_t)((diagnostic->packet.full_header >> 16) & 0x3u);

  if (fprintf(file,
              "{\n  \"schema\": \"%s\",\n"
              "  \"complete\": %s,\n"
              "  \"request\": {\"id\": %" PRIu64
              ", \"ordinal\": %" PRIu32
              ", \"finished\": %s, \"succeeded\": %s},\n"
              "  \"capture\": {\"position_base\": %" PRIu32
              ", \"query_count\": %" PRIu32
              ", \"layer\": %" PRIu32 "},\n"
              "  \"ordering\": {\"host_metadata\": %" PRIu32
              ", \"gpu_metadata_copy_complete\": %" PRIu32
              ", \"attention_packet_published\": %" PRIu32
              ", \"request_finished\": %" PRIu32 "},\n"
              "  \"metadata\": {\n"
              "    \"gpu_address\": \"0x%016" PRIx64 "\",\n"
              "    \"host_i32\": ",
              Q16K_FIRST_ATTENTION_DIAGNOSTIC_SCHEMA,
              complete ? "true" : "false", diagnostic->request_id,
              diagnostic->request_ordinal,
              diagnostic->request_finished ? "true" : "false",
              diagnostic->request_succeeded ? "true" : "false",
              diagnostic->position_base, diagnostic->logical_query_count,
              diagnostic->layer, diagnostic->host_metadata_sequence,
              diagnostic->gpu_metadata_sequence,
              diagnostic->attention_dispatch_sequence,
              diagnostic->request_finished_sequence,
              diagnostic->gpu_metadata_address) < 0) {
    return 0;
  }
  if (!write_metadata(file, &diagnostic->host_metadata) ||
      fputs(",\n    \"gpu_i32\": ", file) == EOF ||
      !write_metadata(file, &diagnostic->gpu_metadata) ||
      fprintf(file,
              ",\n    \"host_gpu_match\": %s\n  },\n"
              "  \"kernel\": {\"name\": ",
              metadata_match ? "true" : "false") < 0 ||
      !write_json_string(file, diagnostic->kernel_name) ||
      fprintf(file,
              ", \"object\": \"0x%016" PRIx64
              "\", \"kernarg_segment_size\": %" PRIu32
              ", \"kernarg_segment_alignment\": %" PRIu32
              ", \"group_segment_size\": %" PRIu32
              ", \"private_segment_size\": %" PRIu32 "},\n",
              diagnostic->kernel.kernel_object,
              diagnostic->kernel.kernarg_segment_size,
              diagnostic->kernel.kernarg_segment_alignment,
              diagnostic->kernel.group_segment_size,
              diagnostic->kernel.private_segment_size) < 0) {
    return 0;
  }

  if (fprintf(file,
              "  \"geometry\": {\"grid\": [%" PRIu32 ", %" PRIu32
              ", %" PRIu32 "], \"workgroup\": [%" PRIu16
              ", %" PRIu16 ", %" PRIu16 "]},\n"
              "  \"packet\": {\n"
              "    \"full_header\": \"0x%08" PRIx32 "\",\n"
              "    \"dimensions\": %" PRIu16 ",\n"
              "    \"workgroup\": [%" PRIu16 ", %" PRIu16
              ", %" PRIu16 "],\n"
              "    \"grid\": [%" PRIu32 ", %" PRIu32 ", %" PRIu32
              "],\n"
              "    \"private_segment_size\": %" PRIu32 ",\n"
              "    \"group_segment_size\": %" PRIu32 ",\n"
              "    \"kernel_object\": \"0x%016" PRIx64 "\",\n"
              "    \"kernarg_address\": \"0x%016" PRIx64 "\",\n"
              "    \"completion_signal\": \"0x%016" PRIx64 "\",\n"
              "    \"packet_id\": %" PRIu64 ",\n"
              "    \"kernarg_slot\": %" PRIu32 ",\n"
              "    \"doorbell_written\": %" PRIu32 "\n  },\n",
              diagnostic->geometry.grid_size[0],
              diagnostic->geometry.grid_size[1],
              diagnostic->geometry.grid_size[2],
              diagnostic->geometry.workgroup_size[0],
              diagnostic->geometry.workgroup_size[1],
              diagnostic->geometry.workgroup_size[2],
              diagnostic->packet.full_header, packet_dimensions,
              diagnostic->packet.workgroup_size_x,
              diagnostic->packet.workgroup_size_y,
              diagnostic->packet.workgroup_size_z,
              diagnostic->packet.grid_size_x,
              diagnostic->packet.grid_size_y,
              diagnostic->packet.grid_size_z,
              diagnostic->packet.private_segment_size,
              diagnostic->packet.group_segment_size,
              diagnostic->packet.kernel_object,
              diagnostic->packet.kernarg_address,
              diagnostic->packet.completion_signal,
              diagnostic->dispatch_result.packet_id,
              diagnostic->dispatch_result.kernarg_slot,
              diagnostic->dispatch_result.doorbell_written) < 0) {
    return 0;
  }

  if (fprintf(file,
              "  \"kernarg\": {\n"
              "    \"size\": %zu,\n"
              "    \"source_matches_ring\": %s,\n"
              "    \"hex\": ",
              diagnostic->kernarg_size,
              diagnostic->source_kernarg_matches_ring ? "true" : "false") <
          0 ||
      !write_kernarg_hex(file, kernarg, diagnostic->kernarg_size) ||
      fprintf(file,
              ",\n    \"decoded\": {\n"
              "      \"q\": \"0x%016" PRIx64 "\",\n"
              "      \"k\": \"0x%016" PRIx64 "\",\n"
              "      \"v\": \"0x%016" PRIx64 "\",\n"
              "      \"output\": \"0x%016" PRIx64 "\",\n"
              "      \"sequence_sentinels\": [%" PRId32 ", %" PRId32
              "],\n"
              "      \"key_end\": %" PRId32 ",\n"
              "      \"page_size\": %" PRId32 ",\n"
              "      \"kv_indptr\": \"0x%016" PRIx64 "\",\n"
              "      \"kv_page_indices\": \"0x%016" PRIx64 "\",\n"
              "      \"kv_last_page_lens\": \"0x%016" PRIx64 "\",\n"
              "      \"softmax_scale_log2\": %.9g,\n"
              "      \"window_left\": %" PRId32 ",\n"
              "      \"window_right\": %" PRId32 ",\n"
              "      \"mask_type\": %" PRId32 ",\n"
              "      \"cu_seqlens_q\": \"0x%016" PRIx64 "\",\n"
              "      \"hidden_block_counts\": [%" PRIu32 ", %" PRIu32
              ", %" PRIu32 "],\n"
              "      \"group_sizes\": [%" PRIu16 ", %" PRIu16
              ", %" PRIu16 "],\n"
              "      \"hidden_dimensions\": %" PRIu16 "\n"
              "    }\n  }\n}\n",
              load_u64(kernarg, 0u), load_u64(kernarg, 8u),
              load_u64(kernarg, 16u), load_u64(kernarg, 24u),
              load_i32(kernarg, 40u), load_i32(kernarg, 44u),
              load_i32(kernarg, 64u), load_i32(kernarg, 68u),
              load_u64(kernarg, 72u), load_u64(kernarg, 80u),
              load_u64(kernarg, 88u), (double)load_f32(kernarg, 96u),
              load_i32(kernarg, 132u), load_i32(kernarg, 136u),
              load_i32(kernarg, 144u), load_u64(kernarg, 152u),
              load_u32(kernarg, 168u), load_u32(kernarg, 172u),
              load_u32(kernarg, 176u), load_u16(kernarg, 180u),
              load_u16(kernarg, 182u), load_u16(kernarg, 184u),
              load_u16(kernarg, 232u)) < 0) {
    return 0;
  }
  return 1;
}

q16k_first_attention_diagnostic_status_t
q16k_first_attention_diagnostic_write_json_atomic(
    const q16k_first_attention_diagnostic_t* diagnostic,
    const char* output_path, char* error_message,
    size_t error_message_capacity) {
  if (diagnostic == NULL || output_path == NULL || output_path[0] != '/') {
    set_error(error_message, error_message_capacity,
              "diagnostic and an absolute output path are required");
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT;
  }

  const size_t path_length = strlen(output_path);
  if (path_length > SIZE_MAX - 64u) {
    set_error(error_message, error_message_capacity,
              "diagnostic output path is too long");
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_INVALID_ARGUMENT;
  }
  char* temporary_path = (char*)malloc(path_length + 64u);
  if (temporary_path == NULL) {
    set_error(error_message, error_message_capacity,
              "allocating temporary output path failed");
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR;
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
      return Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR;
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
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR;
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
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR;
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
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR;
  }
  if (link(temporary_path, output_path) != 0) {
    char message[256];
    snprintf(message, sizeof(message),
             "publishing diagnostic without replacement failed: %s",
             strerror(errno));
    unlink(temporary_path);
    set_error(error_message, error_message_capacity, message);
    free(temporary_path);
    return Q16K_FIRST_ATTENTION_DIAGNOSTIC_IO_ERROR;
  }
  unlink(temporary_path);
  free(temporary_path);
  if (error_message != NULL && error_message_capacity != 0u) {
    error_message[0] = '\0';
  }
  return q16k_first_attention_diagnostic_is_complete(diagnostic)
             ? Q16K_FIRST_ATTENTION_DIAGNOSTIC_OK
             : Q16K_FIRST_ATTENTION_DIAGNOSTIC_INCOMPLETE;
}
