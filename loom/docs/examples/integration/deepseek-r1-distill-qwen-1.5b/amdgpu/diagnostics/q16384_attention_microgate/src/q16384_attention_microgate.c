#define _GNU_SOURCE

#include "q16k_aiter_integration.h"
#include "reference.h"
#include "sha256.h"

#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/fs.h>
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
  kPositionBase = Q16K_AITER_LEGACY_PREFIX_ROWS,
  kQueryCount = Q16K_AITER_CHUNK_CAPACITY,
  kKeyEnd = Q16K_AITER_LEGACY_PREFIX_ROWS + Q16K_AITER_CHUNK_CAPACITY,
  kQueryHeads = 12,
  kKvHeads = 2,
  kHeadDimension = 128,
  kQueueSize = 64,
  kGuardBytes = 8192,
  kDispatchTimeoutSeconds = 180,
  kHipExplicitArgumentBytes = 168,
};

#define MICROGATE_SCHEMA "loom-q16384-first-attention-launch-abi-result-v2"
#define AUTHORIZATION_SCHEMA \
  "loom-q16384-first-attention-launch-abi-one-run-authorization-v1"
#define RECEIPT_SCHEMA \
  "loom-q16384-first-attention-launch-abi-consumption-v1"
#define AUTHORIZATION_SCOPE "q16384-first-attention-raw-hsa-vs-hip"
#define HSA_RUNTIME_PATH "/opt/rocm/lib/libhsa-runtime64.so.1"
#define HSA_RUNTIME_SHA256 \
  "d41abc620d2f228995f809b55c7e4183f513501cc070416316479f5eeaa58253"
#define HIP_RUNTIME_PATH "/opt/rocm/lib/libamdhip64.so"
#define HIP_RUNTIME_SHA256 \
  "1e9c69bed92a2cd458468e6af29707b4ebbbb94a8361adbb95d30d38b82074d4"
#define ROCPROFILER_REGISTER_PATH "/opt/rocm/lib/librocprofiler-register.so.0"
#define ROCPROFILER_REGISTER_SHA256 \
  "8faae1b02f834857d75b1f318bf826d371ff9eb30b3af5b2d26a279f4525a953"
#define Q16384_HSACO_SHA256 \
  "0e90cc246649934b27c886187dea4d7b960266e921edbb0b9b7d0b38eb5e4edc"

static const uint16_t kPoisonBf16 = UINT16_C(0x7fc1);
static const uint32_t kPoisonWord = UINT32_C(0x7fc17fc1);
static const uint32_t kGuardWord = UINT32_C(0x5a17c0de);
static const uint32_t kExpectedFullHeader = UINT32_C(0x00031502);

_Static_assert(kPositionBase == 384, "first suffix position changed");
_Static_assert(kQueryCount == 16384, "first suffix query count changed");
_Static_assert(kKeyEnd == 16768, "first suffix key end changed");
_Static_assert(Q16K_AITER_ATTENTION_KERNARG_SIZE == 424,
               "attention kernarg size changed");
_Static_assert(Q16K_AITER_KERNARG_STRIDE == 512,
               "production kernarg stride changed");
_Static_assert(sizeof(q16k_aiter_dispatch_packet_t) == 64,
               "AQL dispatch packet size changed");
_Static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64,
               "HSA dispatch packet size changed");

typedef struct error_state_s {
  char message[1024];
} error_state_t;

typedef struct consumption_receipt_s {
  char authorization_id[129];
  char authorization_sha256[65];
  char authorization_path[PATH_MAX];
  char microgate_root_sha256[65];
  char reservation_id[64];
  char host[256];
  char target[256];
  char consumption_receipt_path[PATH_MAX];
  char result_path[PATH_MAX];
  uint64_t reservation_start_epoch;
  uint64_t reservation_end_epoch;
  uint64_t authorization_issued_epoch;
  uint64_t authorization_expires_epoch;
  uint64_t consumed_epoch;
} consumption_receipt_t;

typedef struct runtime_state_s {
  bool initialized;
  hsa_agent_t cpu_agent;
  hsa_agent_t gpu_agent;
  char cpu_name[64];
  char gpu_name[64];
  char isa_name[256];
  uint32_t visible_gpu_count;
  uint64_t timestamp_frequency;
  hsa_amd_memory_pool_t host_pool;
  hsa_amd_memory_pool_t kernarg_pool;
  hsa_amd_memory_pool_t gpu_pool;
  uint64_t gpu_pool_max_allocation;
} runtime_state_t;

typedef struct loaded_module_s {
  uint8_t* image;
  size_t image_size;
  hsa_code_object_reader_t reader;
  hsa_executable_t executable;
  bool reader_created;
  bool executable_created;
  q16k_aiter_loaded_kernel_t kernel;
} loaded_module_t;

typedef struct guarded_buffer_s {
  void* base;
  void* data;
  size_t payload_bytes;
  size_t total_bytes;
  uint32_t guard_word;
} guarded_buffer_t;

typedef struct input_hashes_s {
  char q[65];
  char k[65];
  char v[65];
  char metadata[65];
  char page_indices[65];
} input_hashes_t;

typedef struct output_observation_s {
  char sha256[65];
  uint64_t poison_values_remaining;
  uint64_t nonfinite_values;
  uint64_t guard_mismatches;
  uint64_t sampled_elements_compared;
  uint64_t reference_violations;
  double max_absolute_error;
  double max_tolerance_ratio;
} output_observation_t;

typedef struct queue_adapter_s {
  hsa_queue_t* queue;
  q16k_aiter_dispatch_packet_t published_packet;
  bool packet_published;
} queue_adapter_t;

typedef struct queue_contract_observation_s {
  bool agent_min_query_completed;
  bool agent_max_query_completed;
  bool agent_type_query_completed;
  hsa_status_t agent_min_query_status;
  hsa_status_t agent_max_query_status;
  hsa_status_t agent_type_query_status;
  uint32_t agent_min_size;
  uint32_t agent_max_size;
  hsa_queue_type32_t agent_type;
  uint32_t requested_size;
  hsa_queue_type32_t requested_type;
  uint32_t expected_size;
  bool create_attempted;
  hsa_status_t create_status;
  uint64_t returned_pointer;
  uint64_t returned_base_address;
  uint64_t returned_doorbell_signal_handle;
  uint64_t returned_id;
  uint32_t returned_size;
  hsa_queue_type32_t returned_type;
  uint32_t returned_features;
  bool pointer_nonnull;
  bool base_nonnull;
  bool size_power_of_two;
  bool size_within_agent_bounds;
  bool size_matches_expected;
  bool type_ordinary;
  bool kernel_dispatch_supported;
  bool contract_satisfied;
} queue_contract_observation_t;

typedef int hip_error_t;
typedef void* hip_module_t;
typedef void* hip_function_t;
typedef void* hip_stream_t;

typedef struct hip_api_s {
  void* library;
  hip_error_t (*init)(unsigned int);
  hip_error_t (*set_device)(int);
  hip_error_t (*module_load)(hip_module_t*, const char*);
  hip_error_t (*module_get_function)(hip_function_t*, hip_module_t,
                                     const char*);
  hip_error_t (*module_launch_kernel)(hip_function_t, unsigned int,
                                      unsigned int, unsigned int,
                                      unsigned int, unsigned int,
                                      unsigned int, unsigned int,
                                      hip_stream_t, void**, void**);
  hip_error_t (*device_synchronize)(void);
  hip_error_t (*module_unload)(hip_module_t);
  const char* (*get_error_string)(hip_error_t);
} hip_api_t;

typedef struct run_resources_s {
  runtime_state_t runtime;
  loaded_module_t module;
  hsa_queue_t* queue;
  queue_contract_observation_t queue_contract;
  atomic_int queue_error;
  hsa_signal_t raw_completion;
  bool raw_completion_created;
  void* kernarg_ring;
  q16k_aiter_queue_state_t canonical_queue;
  queue_adapter_t queue_adapter;
  guarded_buffer_t q;
  guarded_buffer_t k;
  guarded_buffer_t v;
  guarded_buffer_t metadata;
  guarded_buffer_t page_indices;
  guarded_buffer_t raw_output;
  guarded_buffer_t hip_output;
  void* host_q;
  void* host_k;
  void* host_v;
  void* host_metadata;
  void* host_page_indices;
  void* host_raw_output;
  void* host_hip_output;
  void* host_guard;
  size_t q_bytes;
  size_t kv_bytes;
  size_t metadata_bytes;
  size_t page_index_bytes;
  size_t output_bytes;
} run_resources_t;

typedef struct run_result_s {
  input_hashes_t pre_inputs;
  input_hashes_t post_raw_inputs;
  input_hashes_t post_hip_inputs;
  output_observation_t raw_output;
  output_observation_t raw_output_after_hip;
  output_observation_t hip_output;
  q16k_aiter_dispatch_result_t raw_dispatch;
  q16k_aiter_dispatch_packet_t raw_packet;
  uint64_t raw_started_ns;
  uint64_t raw_completed_ns;
  uint64_t hip_started_ns;
  uint64_t hip_completed_ns;
  uint64_t mismatch_bytes;
  uint64_t first_mismatch_byte;
  uint64_t first_mismatch_element;
  uint32_t first_mismatch_row;
  uint32_t first_mismatch_head;
  uint32_t first_mismatch_dimension;
  bool byte_equal;
  bool inputs_unchanged_after_raw;
  bool inputs_unchanged_after_hip;
  bool raw_output_unchanged_after_hip;
  uint64_t input_guard_mismatches_after_raw;
  uint64_t input_guard_mismatches_after_hip;
  uint64_t hip_poison_values_after_raw;
  uint64_t hip_guard_mismatches_after_raw;
} run_result_t;

static bool fail(error_state_t* error, const char* format, ...) {
  if (error != NULL && error->message[0] == '\0') {
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error->message, sizeof(error->message), format,
                    arguments);
    va_end(arguments);
  }
  return false;
}

static bool checked_add_size(size_t left, size_t right, size_t* output) {
  if (left > SIZE_MAX - right) return false;
  *output = left + right;
  return true;
}

static bool checked_mul_size(size_t left, size_t right, size_t* output) {
  if (left != 0u && right > SIZE_MAX / left) return false;
  *output = left * right;
  return true;
}

static uint64_t monotonic_ns(void) {
  struct timespec timestamp;
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0 ||
      timestamp.tv_sec < 0 || timestamp.tv_nsec < 0 ||
      timestamp.tv_nsec >= 1000000000L) {
    return 0u;
  }
  const uint64_t seconds = (uint64_t)timestamp.tv_sec;
  const uint64_t nanoseconds = (uint64_t)timestamp.tv_nsec;
  if (seconds > (UINT64_MAX - nanoseconds) / UINT64_C(1000000000)) {
    return 0u;
  }
  return seconds * UINT64_C(1000000000) + nanoseconds;
}

static bool join_path(const char* root, const char* relative,
                      char output[PATH_MAX], error_state_t* error) {
  const int length = snprintf(output, PATH_MAX, "%s/%s", root, relative);
  return length >= 0 && length < PATH_MAX
             ? true
             : fail(error, "path is too long: %s/%s", root, relative);
}

static bool verify_hash(const char* path, const char* expected,
                        error_state_t* error) {
  char actual[65];
  if (sha256_file(path, actual) != 0) {
    return fail(error, "cannot hash %s: %s", path, strerror(errno));
  }
  return strcmp(actual, expected) == 0
             ? true
             : fail(error, "SHA-256 mismatch for %s: %s != %s", path,
                    actual, expected);
}

static void hash_bytes(const void* data, size_t size, char output[65]) {
  sha256_context_t context;
  uint8_t digest[32];
  sha256_init(&context);
  sha256_update(&context, data, size);
  sha256_final(&context, digest);
  sha256_hex(digest, output);
}

static bool read_first_digest(const char* path, char output[65],
                              error_state_t* error) {
  FILE* stream = fopen(path, "rb");
  if (stream == NULL) {
    return fail(error, "cannot open %s: %s", path, strerror(errno));
  }
  char line[160];
  const bool read_ok = fgets(line, sizeof(line), stream) != NULL;
  const int close_result = fclose(stream);
  if (!read_ok || close_result != 0 || strlen(line) < 64u) {
    return fail(error, "cannot read digest from %s", path);
  }
  for (size_t i = 0u; i < 64u; ++i) {
    const char value = line[i];
    if (!((value >= '0' && value <= '9') ||
          (value >= 'a' && value <= 'f'))) {
      return fail(error, "malformed digest in %s", path);
    }
    output[i] = value;
  }
  output[64] = '\0';
  return true;
}

static bool verify_static_inputs(const char* root, error_state_t* error) {
  char path[PATH_MAX];
  if (!join_path(root, "artifacts/aiter_prefill_q16384_gfx950.hsaco", path,
                 error) ||
      !verify_hash(path, Q16384_HSACO_SHA256, error)) {
    return false;
  }
  const q16k_aiter_asset_spec_t* asset =
      q16k_aiter_asset_spec(Q16K_AITER_ASSET_ATTENTION);
  const q16k_aiter_kernel_spec_t* kernel = q16k_attention_kernel_spec();
  if (asset == NULL || kernel == NULL ||
      strcmp(asset->sha256, Q16384_HSACO_SHA256) != 0 ||
      asset->byte_size != 30056u ||
      kernel->kernarg_segment_size != 424u ||
      kernel->kernarg_segment_alignment != 16u ||
      kernel->group_segment_size != 26112u ||
      kernel->private_segment_size != 0u) {
    return fail(error, "canonical q16k attention contract changed");
  }
  return true;
}

static bool verify_runtime_inputs(error_state_t* error) {
  return verify_hash(HSA_RUNTIME_PATH, HSA_RUNTIME_SHA256, error) &&
         verify_hash(HIP_RUNTIME_PATH, HIP_RUNTIME_SHA256, error) &&
         verify_hash(ROCPROFILER_REGISTER_PATH,
                     ROCPROFILER_REGISTER_SHA256, error);
}

static bool parse_u64_decimal(const char* text, uint64_t* output) {
  if (text == NULL || *text == '\0' ||
      (text[0] == '0' && text[1] != '\0')) {
    return false;
  }
  for (const char* cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') return false;
  }
  errno = 0;
  char* end = NULL;
  const unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return false;
  *output = (uint64_t)value;
  return true;
}

static bool valid_lower_hex(const char* text, size_t length) {
  if (strlen(text) != length) return false;
  for (size_t i = 0u; i < length; ++i) {
    if (!((text[i] >= '0' && text[i] <= '9') ||
          (text[i] >= 'a' && text[i] <= 'f'))) {
      return false;
    }
  }
  return true;
}

static bool valid_identifier(const char* text, size_t minimum,
                             size_t maximum) {
  const size_t length = strlen(text);
  if (length < minimum || length > maximum) return false;
  for (size_t i = 0u; i < length; ++i) {
    const char value = text[i];
    if (!((value >= 'a' && value <= 'z') ||
          (value >= 'A' && value <= 'Z') ||
          (value >= '0' && value <= '9') || value == '.' || value == '_' ||
          value == ':' || value == '-')) {
      return false;
    }
  }
  return true;
}

static bool valid_uuid(const char* text) {
  if (strlen(text) != 36u) return false;
  for (size_t i = 0u; i < 36u; ++i) {
    if (i == 8u || i == 13u || i == 18u || i == 23u) {
      if (text[i] != '-') return false;
    } else if (!((text[i] >= '0' && text[i] <= '9') ||
                 (text[i] >= 'a' && text[i] <= 'f'))) {
      return false;
    }
  }
  return true;
}

static bool valid_safe_absolute_path(const char* path) {
  const size_t length = strlen(path);
  if (length < 2u || length >= PATH_MAX || path[0] != '/' ||
      path[length - 1u] == '/') {
    return false;
  }
  size_t component_start = 1u;
  for (size_t index = 1u; index <= length; ++index) {
    const char value = path[index];
    if (value == '/' || value == '\0') {
      const size_t component_length = index - component_start;
      if (component_length == 0u ||
          (component_length == 1u && path[component_start] == '.') ||
          (component_length == 2u && path[component_start] == '.' &&
           path[component_start + 1u] == '.')) {
        return false;
      }
      component_start = index + 1u;
      continue;
    }
    if (!((value >= 'a' && value <= 'z') ||
          (value >= 'A' && value <= 'Z') ||
          (value >= '0' && value <= '9') || value == '.' || value == '_' ||
          value == '-')) {
      return false;
    }
  }
  return true;
}

static bool path_is_within(const char* root, const char* path) {
  const size_t root_length = strlen(root);
  return strcmp(root, path) == 0 ||
         (strncmp(root, path, root_length) == 0 && path[root_length] == '/');
}

static bool canonical_future_path(const char* path, error_state_t* error) {
  if (!valid_safe_absolute_path(path)) {
    return fail(error, "path is not a safe canonical absolute path: %s", path);
  }
  const char* slash = strrchr(path, '/');
  char parent[PATH_MAX];
  const size_t parent_length = slash == path ? 1u : (size_t)(slash - path);
  if (slash == NULL || slash[1] == '\0' || parent_length >= sizeof(parent)) {
    return fail(error, "path has an invalid parent: %s", path);
  }
  memcpy(parent, path, parent_length);
  parent[parent_length] = '\0';
  char parent_real[PATH_MAX];
  if (realpath(parent, parent_real) == NULL) {
    return fail(error, "cannot resolve path parent %s: %s", parent,
                strerror(errno));
  }
  struct stat status;
  if (stat(parent_real, &status) != 0 || !S_ISDIR(status.st_mode) ||
      status.st_uid != geteuid() || (status.st_mode & 0022) != 0) {
    return fail(error, "path parent is not an owned non-writable directory: %s",
                parent_real);
  }
  char reconstructed[PATH_MAX];
  const int length = strcmp(parent_real, "/") == 0
                         ? snprintf(reconstructed, sizeof(reconstructed),
                                    "/%s", slash + 1)
                         : snprintf(reconstructed, sizeof(reconstructed),
                                    "%s/%s", parent_real, slash + 1);
  return length >= 0 && (size_t)length < sizeof(reconstructed) &&
                 strcmp(reconstructed, path) == 0
             ? true
             : fail(error, "path is not canonical: %s", path);
}

static bool verify_live_path_contract(const char* root,
                                      const char* result_path,
                                      const char* receipt_path,
                                      const char* authorization_path,
                                      char claimed_path[PATH_MAX],
                                      error_state_t* error) {
  if (!canonical_future_path(receipt_path, error) ||
      !canonical_future_path(authorization_path, error) ||
      !canonical_future_path(result_path, error)) {
    return false;
  }
  const int length =
      snprintf(claimed_path, PATH_MAX, "%s.consumed", receipt_path);
  if (length < 0 || length >= PATH_MAX) {
    return fail(error, "claimed receipt path is too long");
  }
  if (strcmp(result_path, receipt_path) == 0 ||
      strcmp(result_path, claimed_path) == 0 ||
      strcmp(result_path, authorization_path) == 0 ||
      strcmp(receipt_path, authorization_path) == 0 ||
      strcmp(claimed_path, authorization_path) == 0) {
    return fail(error, "authorization, receipt, and result paths collide");
  }
  if (path_is_within(root, result_path) || path_is_within(root, receipt_path) ||
      path_is_within(root, claimed_path) ||
      path_is_within(root, authorization_path)) {
    return fail(error, "live mutable paths must be outside the sealed package");
  }
  return true;
}

static bool parse_receipt_text(char* text, consumption_receipt_t* receipt,
                               error_state_t* error) {
  static const char* const keys[] = {
      "schema",
      "authorization_id",
      "authorization_sha256",
      "authorization_path",
      "microgate_root_sha256",
      "authorized_cases",
      "authorized_case_count",
      "reservation_id",
      "consumption_receipt_path",
      "result_path",
      "reservation_start_epoch",
      "reservation_end_epoch",
      "authorization_issued_epoch",
      "authorization_expires_epoch",
      "host",
      "target",
      "maximum_gpu_invocations",
      "invocation_index",
      "invocation_count",
      "consumed_epoch",
  };
  memset(receipt, 0, sizeof(*receipt));
  char* cursor = text;
  for (size_t index = 0u; index < sizeof(keys) / sizeof(keys[0]); ++index) {
    char* newline = strchr(cursor, '\n');
    if (newline == NULL) {
      return fail(error, "authorization receipt has too few lines");
    }
    *newline = '\0';
    char* separator = strchr(cursor, '=');
    if (separator == NULL) {
      return fail(error, "authorization receipt line lacks separator");
    }
    *separator = '\0';
    const char* value = separator + 1;
    if (strcmp(cursor, keys[index]) != 0 || *value == '\0') {
      return fail(error, "authorization receipt field order or value changed");
    }
    uint64_t numeric = 0u;
    switch (index) {
      case 0:
        if (strcmp(value, RECEIPT_SCHEMA) != 0) {
          return fail(error, "authorization receipt schema changed");
        }
        break;
      case 1:
        if (!valid_identifier(value, 16u, 128u)) {
          return fail(error, "authorization ID is invalid");
        }
        (void)snprintf(receipt->authorization_id,
                       sizeof(receipt->authorization_id), "%s", value);
        break;
      case 2:
        if (!valid_lower_hex(value, 64u)) {
          return fail(error, "authorization SHA-256 is invalid");
        }
        memcpy(receipt->authorization_sha256, value, 65u);
        break;
      case 3:
        if (!valid_safe_absolute_path(value)) {
          return fail(error, "authorization path is invalid");
        }
        (void)snprintf(receipt->authorization_path,
                       sizeof(receipt->authorization_path), "%s", value);
        break;
      case 4:
        if (!valid_lower_hex(value, 64u)) {
          return fail(error, "microgate root SHA-256 is invalid");
        }
        memcpy(receipt->microgate_root_sha256, value, 65u);
        break;
      case 5:
        if (strcmp(value, "q16384-first-attention") != 0) {
          return fail(error, "authorized case set changed");
        }
        break;
      case 6:
      case 16:
      case 17:
      case 18:
        if (strcmp(value, "1") != 0) {
          return fail(error, "authorization invocation count is not one");
        }
        break;
      case 7:
        if (!valid_uuid(value)) {
          return fail(error, "reservation ID is invalid");
        }
        (void)snprintf(receipt->reservation_id,
                       sizeof(receipt->reservation_id), "%s", value);
        break;
      case 8:
        if (!valid_safe_absolute_path(value)) {
          return fail(error, "consumption receipt path is invalid");
        }
        (void)snprintf(receipt->consumption_receipt_path,
                       sizeof(receipt->consumption_receipt_path), "%s", value);
        break;
      case 9:
        if (!valid_safe_absolute_path(value)) {
          return fail(error, "result path is invalid");
        }
        (void)snprintf(receipt->result_path, sizeof(receipt->result_path),
                       "%s", value);
        break;
      case 10:
      case 11:
      case 12:
      case 13:
      case 19:
        if (!parse_u64_decimal(value, &numeric)) {
          return fail(error, "authorization receipt timestamp is invalid");
        }
        if (index == 10u) receipt->reservation_start_epoch = numeric;
        if (index == 11u) receipt->reservation_end_epoch = numeric;
        if (index == 12u) receipt->authorization_issued_epoch = numeric;
        if (index == 13u) receipt->authorization_expires_epoch = numeric;
        if (index == 19u) receipt->consumed_epoch = numeric;
        break;
      case 14:
        if (!valid_identifier(value, 1u, sizeof(receipt->host) - 1u)) {
          return fail(error, "authorization host is invalid");
        }
        (void)snprintf(receipt->host, sizeof(receipt->host), "%s", value);
        break;
      case 15:
        if (!valid_identifier(value, 1u, sizeof(receipt->target) - 1u)) {
          return fail(error, "authorization target is invalid");
        }
        (void)snprintf(receipt->target, sizeof(receipt->target), "%s", value);
        break;
      default:
        return fail(error, "internal receipt parser error");
    }
    cursor = newline + 1;
  }
  return *cursor == '\0'
             ? true
             : fail(error, "authorization receipt has unexpected extra fields");
}

static bool format_utc_epoch(uint64_t epoch, char output[21]) {
  const time_t value = (time_t)epoch;
  if ((uint64_t)value != epoch) return false;
  struct tm components;
  return gmtime_r(&value, &components) != NULL &&
         strftime(output, 21u, "%Y-%m-%dT%H:%M:%SZ", &components) == 20u;
}

static bool verify_authorization_file(const char* authorization_path,
                                      const consumption_receipt_t* receipt,
                                      error_state_t* error) {
  char authorization_real[PATH_MAX];
  if (realpath(authorization_path, authorization_real) == NULL ||
      strcmp(authorization_real, receipt->authorization_path) != 0) {
    return fail(error, "authorization file path is not the bound path");
  }
  const int descriptor =
      open(authorization_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  struct stat status;
  if (descriptor < 0 || fstat(descriptor, &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_nlink != 1 ||
      status.st_uid != geteuid() || (status.st_mode & 0077) != 0 ||
      status.st_size <= 0 || status.st_size > 16384) {
    if (descriptor >= 0) (void)close(descriptor);
    return fail(error,
                "authorization must be owned, mode-private, bounded, regular, "
                "non-symlink, and non-hardlinked");
  }
  char text[16385];
  size_t count = 0u;
  for (;;) {
    const ssize_t amount =
        read(descriptor, text + count, sizeof(text) - 1u - count);
    if (amount < 0 && errno == EINTR) continue;
    if (amount < 0) {
      (void)close(descriptor);
      return fail(error, "cannot read authorization file: %s",
                  strerror(errno));
    }
    if (amount == 0) break;
    count += (size_t)amount;
    if (count == sizeof(text) - 1u) {
      (void)close(descriptor);
      return fail(error, "authorization file is too large");
    }
  }
  if (close(descriptor) != 0) {
    return fail(error, "cannot close authorization file");
  }
  text[count] = '\0';
  if (strlen(text) != count) {
    return fail(error, "authorization contains an embedded NUL");
  }
  char digest_text[65];
  hash_bytes(text, count, digest_text);
  if (strcmp(digest_text, receipt->authorization_sha256) != 0) {
    return fail(error, "authorization file hash differs from receipt");
  }

  char issued_utc[21];
  char expires_utc[21];
  if (!format_utc_epoch(receipt->authorization_issued_epoch, issued_utc) ||
      !format_utc_epoch(receipt->authorization_expires_epoch, expires_utc)) {
    return fail(error, "authorization timestamps are not representable");
  }
  char expected[2 * PATH_MAX + 2048];
  const int expected_length = snprintf(
      expected, sizeof(expected),
      "{\"authorization_id\":\"%s\",\"authorized_case_count\":1,"
      "\"authorized_cases\":[\"q16384-first-attention\"],"
      "\"consumption_receipt_path\":\"%s\","
      "\"decision\":\"AUTHORIZE_ONE_GPU_INVOCATION\","
      "\"expires_utc\":\"%s\",\"host\":\"%s\","
      "\"invocation_count\":1,\"issued_utc\":\"%s\","
      "\"maximum_gpu_invocations\":1,\"microgate_root_sha256\":\"%s\","
      "\"pack_and_attention_contract_satisfied\":false,"
      "\"reservation_id\":\"%s\",\"result_path\":\"%s\","
      "\"schema\":\"%s\",\"scope\":\"%s\",\"target\":\"%s\"}\n",
      receipt->authorization_id, receipt->consumption_receipt_path,
      expires_utc, receipt->host, issued_utc, receipt->microgate_root_sha256,
      receipt->reservation_id, receipt->result_path, AUTHORIZATION_SCHEMA,
      AUTHORIZATION_SCOPE, receipt->target);
  if (expected_length < 0 || (size_t)expected_length >= sizeof(expected) ||
      (size_t)expected_length != count || memcmp(text, expected, count) != 0) {
    return fail(error,
                "authorization JSON is not the exact canonical 17-field grant");
  }
  return true;
}

static bool verify_consumption_receipt(const char* root,
                                       const char* receipt_path,
                                       const char* authorization_path,
                                       const char* result_path,
                                       char root_digest[65],
                                       consumption_receipt_t* receipt_output,
                                       struct stat* receipt_status,
                                       error_state_t* error) {
  const char* interlock = getenv("MICROGATE_AUTHORIZATION_CONSUMED");
  if (interlock == NULL || strcmp(interlock, "1") != 0) {
    return fail(error, "live runner requires consumed-authorization interlock");
  }
  char claimed_path[PATH_MAX];
  if (!verify_live_path_contract(root, result_path, receipt_path,
                                 authorization_path, claimed_path, error)) {
    return false;
  }
  const int descriptor = open(receipt_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  struct stat status;
  if (descriptor < 0 || fstat(descriptor, &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_nlink != 1 ||
      status.st_uid != geteuid() || (status.st_mode & 0777) != 0400 ||
      status.st_size <= 0 || status.st_size > 4096) {
    if (descriptor >= 0) (void)close(descriptor);
    return fail(error,
                "authorization receipt must be owned, mode-private, bounded, "
                "regular, non-symlink, and non-hardlinked");
  }
  char root_file[PATH_MAX];
  if (!join_path(root, "ROOT_SHA256", root_file, error) ||
      !read_first_digest(root_file, root_digest, error)) {
    (void)close(descriptor);
    return false;
  }
  char text[4097];
  size_t count = 0u;
  for (;;) {
    const ssize_t amount = read(descriptor, text + count,
                                sizeof(text) - 1u - count);
    if (amount < 0 && errno == EINTR) continue;
    if (amount < 0) {
      (void)close(descriptor);
      return fail(error, "cannot read authorization receipt: %s",
                  strerror(errno));
    }
    if (amount == 0) break;
    count += (size_t)amount;
    if (count == sizeof(text) - 1u) {
      (void)close(descriptor);
      return fail(error, "authorization receipt is too large");
    }
  }
  if (close(descriptor) != 0 || count == 0u || text[count - 1u] != '\n') {
    return fail(error, "cannot read authorization receipt");
  }
  text[count] = '\0';
  if (strlen(text) != count) {
    return fail(error, "authorization receipt contains an embedded NUL");
  }
  consumption_receipt_t receipt;
  if (!parse_receipt_text(text, &receipt, error) ||
      strcmp(receipt.microgate_root_sha256, root_digest) != 0 ||
      strcmp(receipt.authorization_path, authorization_path) != 0 ||
      strcmp(receipt.consumption_receipt_path, receipt_path) != 0 ||
      strcmp(receipt.result_path, result_path) != 0) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "authorization receipt binding changed");
  }
  if (receipt.reservation_start_epoch >= receipt.reservation_end_epoch) {
    return fail(error, "authorization receipt reservation window is invalid");
  }
  char receipt_real[PATH_MAX];
  if (realpath(receipt_path, receipt_real) == NULL ||
      strcmp(receipt_real, receipt.consumption_receipt_path) != 0) {
    return fail(error, "authorization receipt path is not the bound path");
  }
  struct stat collision;
  if (lstat(result_path, &collision) == 0 || errno != ENOENT ||
      lstat(claimed_path, &collision) == 0 || errno != ENOENT) {
    return fail(error, "result or claimed receipt path already exists");
  }
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    return fail(error, "cannot read hostname: %s", strerror(errno));
  }
  hostname[sizeof(hostname) - 1u] = '\0';
  if (strcmp(hostname, receipt.host) != 0 &&
      strcmp(hostname, receipt.target) != 0) {
    return fail(error, "live runner host mismatch: %s", hostname);
  }
  const time_t now_time = time(NULL);
  if (now_time < 0) return fail(error, "cannot read wall clock");
  const uint64_t now = (uint64_t)now_time;
  if (!(receipt.reservation_start_epoch <= now &&
        now < receipt.reservation_end_epoch &&
        receipt.authorization_issued_epoch >= receipt.reservation_start_epoch &&
        receipt.authorization_issued_epoch <= now &&
        now < receipt.authorization_expires_epoch &&
        receipt.authorization_expires_epoch <= receipt.reservation_end_epoch)) {
    return fail(error, "reservation or authorization time window is invalid");
  }
  if (receipt.reservation_end_epoch - now < 900u) {
    return fail(error, "reservation has under 900 seconds remaining");
  }
  if (receipt.consumed_epoch > now + 5u ||
      now - (receipt.consumed_epoch > now ? now : receipt.consumed_epoch) >
          60u ||
      receipt.consumed_epoch < receipt.authorization_issued_epoch ||
      receipt.consumed_epoch >= receipt.authorization_expires_epoch) {
    return fail(error, "authorization consumption receipt is stale");
  }
  if (!verify_authorization_file(authorization_path, &receipt, error)) {
    return false;
  }
  *receipt_output = receipt;
  *receipt_status = status;
  return true;
}

static bool claim_consumption_receipt(const char* receipt_path,
                                      const struct stat* opened_status,
                                      error_state_t* error) {
  struct stat current_status;
  if (lstat(receipt_path, &current_status) != 0 ||
      !S_ISREG(current_status.st_mode) || current_status.st_nlink != 1 ||
      current_status.st_dev != opened_status->st_dev ||
      current_status.st_ino != opened_status->st_ino) {
    return fail(error, "authorization receipt changed before atomic claim");
  }
  char claimed_path[PATH_MAX];
  const int length =
      snprintf(claimed_path, sizeof(claimed_path), "%s.consumed", receipt_path);
  if (length < 0 || (size_t)length >= sizeof(claimed_path)) {
    return fail(error, "claimed receipt path is too long");
  }
  if (renameat2(AT_FDCWD, receipt_path, AT_FDCWD, claimed_path,
                RENAME_NOREPLACE) != 0) {
    return fail(error, "atomic authorization receipt claim failed: %s",
                strerror(errno));
  }
  struct stat claimed_status;
  if (lstat(claimed_path, &claimed_status) != 0 ||
      !S_ISREG(claimed_status.st_mode) || claimed_status.st_nlink != 1 ||
      claimed_status.st_dev != opened_status->st_dev ||
      claimed_status.st_ino != opened_status->st_ino) {
    return fail(error, "claimed authorization receipt identity changed");
  }
  const char* slash = strrchr(receipt_path, '/');
  char parent[PATH_MAX];
  const size_t parent_length = slash == receipt_path ? 1u : (size_t)(slash - receipt_path);
  memcpy(parent, receipt_path, parent_length);
  parent[parent_length] = '\0';
  const int parent_descriptor =
      open(parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (parent_descriptor < 0) {
    return fail(error, "cannot persist authorization receipt claim: %s",
                strerror(errno));
  }
  if (fsync(parent_descriptor) != 0) {
    const int saved_errno = errno;
    (void)close(parent_descriptor);
    errno = saved_errno;
    return fail(error, "cannot persist authorization receipt claim: %s",
                strerror(errno));
  }
  if (close(parent_descriptor) != 0) {
    return fail(error, "cannot close authorization receipt directory: %s",
                strerror(errno));
  }
  return true;
}

static bool write_all(int descriptor, const char* data, size_t size,
                      error_state_t* error) {
  size_t written = 0u;
  while (written < size) {
    const ssize_t amount = write(descriptor, data + written, size - written);
    if (amount < 0 && errno == EINTR) continue;
    if (amount <= 0) {
      return fail(error, "cannot write reserved result: %s", strerror(errno));
    }
    written += (size_t)amount;
  }
  return true;
}

static bool reserve_result_path(const char* path, const char* root_digest,
                                const char* authorization_id,
                                int* output_descriptor,
                                error_state_t* error) {
  const int descriptor = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC |
                                        O_NOFOLLOW,
                              0600);
  if (descriptor < 0) {
    return fail(error, "cannot reserve result path %s: %s", path,
                strerror(errno));
  }
  if (fchmod(descriptor, 0600) != 0) {
    (void)close(descriptor);
    (void)unlink(path);
    return fail(error, "cannot set result mode: %s", strerror(errno));
  }
  char marker[1024];
  const int length = snprintf(
      marker, sizeof(marker),
      "{\"schema\":\"loom-q16384-first-attention-result-reservation-v1\","
      "\"status\":\"reserved\",\"gpu_execution_performed\":false,"
      "\"authorization_id\":\"%s\",\"microgate_root_sha256\":\"%s\"}\n",
      authorization_id, root_digest);
  if (length < 0 || (size_t)length >= sizeof(marker) ||
      !write_all(descriptor, marker, (size_t)length, error) ||
      fsync(descriptor) != 0) {
    (void)close(descriptor);
    (void)unlink(path);
    return error->message[0] != '\0'
               ? false
               : fail(error, "cannot persist result reservation: %s",
                      strerror(errno));
  }
  *output_descriptor = descriptor;
  return true;
}

static bool begin_reserved_result(int descriptor, const char* path,
                                  FILE** output, error_state_t* error) {
  struct stat descriptor_status;
  struct stat path_status;
  if (descriptor < 0 || fstat(descriptor, &descriptor_status) != 0 ||
      lstat(path, &path_status) != 0 ||
      !S_ISREG(descriptor_status.st_mode) || descriptor_status.st_nlink != 1 ||
      descriptor_status.st_uid != geteuid() ||
      (descriptor_status.st_mode & 0777) != 0600 ||
      descriptor_status.st_dev != path_status.st_dev ||
      descriptor_status.st_ino != path_status.st_ino ||
      lseek(descriptor, 0, SEEK_SET) < 0 || ftruncate(descriptor, 0) != 0) {
    return fail(error, "reserved result identity changed before publication");
  }
  FILE* stream = fdopen(descriptor, "wb");
  if (stream == NULL) {
    return fail(error, "cannot open reserved result stream: %s",
                strerror(errno));
  }
  *output = stream;
  return true;
}

static bool commit_reserved_result(FILE* stream, const char* path,
                                   error_state_t* error) {
  if (fflush(stream) != 0 || fsync(fileno(stream)) != 0) {
    const int saved_errno = errno;
    (void)fclose(stream);
    errno = saved_errno;
    return fail(error, "cannot flush result %s: %s", path, strerror(errno));
  }
  return fclose(stream) == 0
             ? true
             : fail(error, "cannot close result %s: %s", path,
                    strerror(errno));
}

static void json_string(FILE* stream, const char* value) {
  fputc('"', stream);
  for (const unsigned char* cursor = (const unsigned char*)value;
       *cursor != '\0'; ++cursor) {
    switch (*cursor) {
      case '"':
        fputs("\\\"", stream);
        break;
      case '\\':
        fputs("\\\\", stream);
        break;
      case '\b':
        fputs("\\b", stream);
        break;
      case '\f':
        fputs("\\f", stream);
        break;
      case '\n':
        fputs("\\n", stream);
        break;
      case '\r':
        fputs("\\r", stream);
        break;
      case '\t':
        fputs("\\t", stream);
        break;
      default:
        if (*cursor < 0x20u) {
          (void)fprintf(stream, "\\u%04x", (unsigned)*cursor);
        } else {
          fputc((int)*cursor, stream);
        }
        break;
    }
  }
  fputc('"', stream);
}

static const char* hsa_status_text(hsa_status_t status) {
  const char* text = NULL;
  if (hsa_status_string(status, &text) == HSA_STATUS_SUCCESS && text != NULL) {
    return text;
  }
  return "unknown HSA status";
}

static bool require_hsa(hsa_status_t status, const char* operation,
                        error_state_t* error) {
  return status == HSA_STATUS_SUCCESS
             ? true
             : fail(error, "%s failed: %s (%d)", operation,
                    hsa_status_text(status), (int)status);
}

typedef struct isa_search_s {
  bool gfx950;
  char name[256];
} isa_search_t;

static hsa_status_t inspect_isa(hsa_isa_t isa, void* user_data) {
  isa_search_t* search = (isa_search_t*)user_data;
  uint32_t length = 0u;
  hsa_status_t status =
      hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &length);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (length == 0u || length > sizeof(search->name)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  char name[sizeof(search->name)];
  memset(name, 0, sizeof(name));
  status = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (strstr(name, "gfx950") != NULL) {
    search->gfx950 = true;
    (void)snprintf(search->name, sizeof(search->name), "%s", name);
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

typedef struct agent_search_s {
  runtime_state_t* runtime;
  bool cpu_found;
  bool gpu_found;
} agent_search_t;

static hsa_status_t inspect_agent(hsa_agent_t agent, void* user_data) {
  agent_search_t* search = (agent_search_t*)user_data;
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (type == HSA_DEVICE_TYPE_CPU && !search->cpu_found) {
    memset(search->runtime->cpu_name, 0, sizeof(search->runtime->cpu_name));
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME,
                                search->runtime->cpu_name);
    if (status != HSA_STATUS_SUCCESS) return status;
    search->runtime->cpu_agent = agent;
    search->cpu_found = true;
  } else if (type == HSA_DEVICE_TYPE_GPU) {
    ++search->runtime->visible_gpu_count;
    isa_search_t isa = {0};
    status = hsa_agent_iterate_isas(agent, inspect_isa, &isa);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      return status;
    }
    if (isa.gfx950 && !search->gpu_found) {
      memset(search->runtime->gpu_name, 0,
             sizeof(search->runtime->gpu_name));
      status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME,
                                  search->runtime->gpu_name);
      if (status != HSA_STATUS_SUCCESS) return status;
      search->runtime->gpu_agent = agent;
      (void)snprintf(search->runtime->isa_name,
                     sizeof(search->runtime->isa_name), "%s", isa.name);
      search->gpu_found = true;
    }
  }
  return HSA_STATUS_SUCCESS;
}

typedef struct pool_search_s {
  hsa_agent_t access_agent;
  hsa_amd_memory_pool_global_flag_t required_flags;
  hsa_amd_memory_pool_location_t required_location;
  hsa_amd_memory_pool_t pool;
  uint64_t max_allocation;
  bool found;
} pool_search_t;

static hsa_status_t inspect_pool(hsa_amd_memory_pool_t pool,
                                 void* user_data) {
  pool_search_t* search = (pool_search_t*)user_data;
  hsa_amd_segment_t segment = HSA_AMD_SEGMENT_GLOBAL;
  bool allocation_allowed = false;
  hsa_amd_memory_pool_global_flag_t flags = 0;
  hsa_amd_memory_pool_location_t location =
      HSA_AMD_MEMORY_POOL_LOCATION_CPU;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL) {
    return status;
  }
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
      &allocation_allowed);
  if (status != HSA_STATUS_SUCCESS || !allocation_allowed) return status;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &location);
  if (status != HSA_STATUS_SUCCESS) return status;
  if ((flags & search->required_flags) != search->required_flags ||
      location != search->required_location) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_amd_memory_pool_access_t access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  status = hsa_amd_agent_memory_pool_get_info(
      search->access_agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
      &access);
  if (status != HSA_STATUS_SUCCESS ||
      access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    return status;
  }
  status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
      &search->max_allocation);
  if (status != HSA_STATUS_SUCCESS) return status;
  search->pool = pool;
  search->found = true;
  return HSA_STATUS_INFO_BREAK;
}

static bool find_pool(hsa_agent_t owner, hsa_agent_t access_agent,
                      hsa_amd_memory_pool_global_flag_t flags,
                      hsa_amd_memory_pool_location_t location,
                      const char* description, hsa_amd_memory_pool_t* output,
                      uint64_t* max_allocation, error_state_t* error) {
  pool_search_t search = {
      .access_agent = access_agent,
      .required_flags = flags,
      .required_location = location,
  };
  const hsa_status_t status =
      hsa_amd_agent_iterate_memory_pools(owner, inspect_pool, &search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    return require_hsa(status, "hsa_amd_agent_iterate_memory_pools", error);
  }
  if (!search.found) {
    return fail(error, "no suitable HSA memory pool for %s", description);
  }
  *output = search.pool;
  if (max_allocation != NULL) *max_allocation = search.max_allocation;
  return true;
}

static bool initialize_runtime(runtime_state_t* runtime,
                               error_state_t* error) {
  memset(runtime, 0, sizeof(*runtime));
  if (!require_hsa(hsa_init(), "hsa_init", error)) return false;
  runtime->initialized = true;
  if (!require_hsa(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                                       &runtime->timestamp_frequency),
                   "hsa_system_get_info(TIMESTAMP_FREQUENCY)", error) ||
      runtime->timestamp_frequency == 0u ||
      runtime->timestamp_frequency >
          UINT64_MAX / (uint64_t)kDispatchTimeoutSeconds) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "invalid HSA timestamp frequency");
  }
  agent_search_t agents = {.runtime = runtime};
  if (!require_hsa(hsa_iterate_agents(inspect_agent, &agents),
                   "hsa_iterate_agents", error) ||
      !agents.cpu_found || !agents.gpu_found) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "required CPU and gfx950 GPU agents not found");
  }
  if (runtime->visible_gpu_count != 1u) {
    return fail(error, "expected exactly one visible GPU, found %" PRIu32,
                runtime->visible_gpu_count);
  }
  return find_pool(runtime->cpu_agent, runtime->gpu_agent,
                   HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
                   HSA_AMD_MEMORY_POOL_LOCATION_CPU, "host staging",
                   &runtime->host_pool, NULL, error) &&
         find_pool(runtime->cpu_agent, runtime->gpu_agent,
                   HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                       HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
                   HSA_AMD_MEMORY_POOL_LOCATION_CPU, "kernarg ring",
                   &runtime->kernarg_pool, NULL, error) &&
         find_pool(runtime->gpu_agent, runtime->gpu_agent,
                   HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                   HSA_AMD_MEMORY_POOL_LOCATION_GPU, "GPU tensors",
                   &runtime->gpu_pool, &runtime->gpu_pool_max_allocation,
                   error);
}

static bool allocate_pool(hsa_amd_memory_pool_t pool, size_t bytes,
                          const char* description, void** output,
                          error_state_t* error) {
  void* pointer = NULL;
  const hsa_status_t status =
      hsa_amd_memory_pool_allocate(pool, bytes, 0u, &pointer);
  if (status != HSA_STATUS_SUCCESS || pointer == NULL) {
    return fail(error, "allocation for %s (%zu bytes) failed: %s (%d)",
                description, bytes, hsa_status_text(status), (int)status);
  }
  *output = pointer;
  return true;
}

static bool allocate_host(runtime_state_t* runtime, size_t bytes,
                          const char* description, void** output,
                          error_state_t* error) {
  if (!allocate_pool(runtime->host_pool, bytes, description, output, error)) {
    return false;
  }
  const hsa_status_t status = hsa_amd_agents_allow_access(
      1u, &runtime->gpu_agent, NULL, *output);
  if (status != HSA_STATUS_SUCCESS) {
    (void)hsa_amd_memory_pool_free(*output);
    *output = NULL;
    return fail(error, "GPU access for %s failed: %s (%d)", description,
                hsa_status_text(status), (int)status);
  }
  return true;
}

static void free_pool_pointer(void** pointer) {
  if (pointer == NULL || *pointer == NULL) return;
  (void)hsa_amd_memory_pool_free(*pointer);
  *pointer = NULL;
}

static bool initialize_guarded_buffer(runtime_state_t* runtime,
                                      size_t payload_bytes,
                                      const char* description,
                                      guarded_buffer_t* output,
                                      error_state_t* error) {
  memset(output, 0, sizeof(*output));
  size_t total = 0u;
  if (!checked_add_size(payload_bytes, 2u * (size_t)kGuardBytes, &total) ||
      total == 0u || total % sizeof(uint32_t) != 0u ||
      (uint64_t)total > runtime->gpu_pool_max_allocation) {
    return fail(error, "invalid guarded allocation size for %s", description);
  }
  if (!allocate_pool(runtime->gpu_pool, total, description, &output->base,
                     error)) {
    return false;
  }
  output->data = (uint8_t*)output->base + kGuardBytes;
  output->payload_bytes = payload_bytes;
  output->total_bytes = total;
  output->guard_word = kGuardWord;
  return require_hsa(
      hsa_amd_memory_fill(output->base, kGuardWord,
                          total / sizeof(uint32_t)),
      "hsa_amd_memory_fill(guards)", error);
}

static void free_guarded_buffer(guarded_buffer_t* buffer) {
  if (buffer == NULL) return;
  free_pool_pointer(&buffer->base);
  memset(buffer, 0, sizeof(*buffer));
}

static bool copy_memory(void* destination, const void* source, size_t bytes,
                        const char* description, error_state_t* error) {
  return require_hsa(hsa_memory_copy(destination, source, bytes), description,
                     error);
}

static bool read_file(const char* path, uint8_t** output, size_t* output_size,
                      error_state_t* error) {
  FILE* stream = fopen(path, "rb");
  if (stream == NULL) {
    return fail(error, "cannot open %s: %s", path, strerror(errno));
  }
  bool success = false;
  if (fseek(stream, 0, SEEK_END) != 0) goto cleanup;
  const long length = ftell(stream);
  if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) goto cleanup;
  uint8_t* bytes = (uint8_t*)malloc((size_t)length);
  if (bytes == NULL) goto cleanup;
  if (fread(bytes, 1u, (size_t)length, stream) != (size_t)length) {
    free(bytes);
    goto cleanup;
  }
  *output = bytes;
  *output_size = (size_t)length;
  success = true;
cleanup:
  if (fclose(stream) != 0 && success) success = false;
  return success ? true : fail(error, "cannot read complete file %s", path);
}

static bool load_module(loaded_module_t* module, hsa_agent_t gpu_agent,
                        const char* path, error_state_t* error) {
  memset(module, 0, sizeof(*module));
  const q16k_aiter_kernel_spec_t* spec = q16k_attention_kernel_spec();
  if (spec == NULL || !read_file(path, &module->image, &module->image_size,
                                 error) ||
      module->image_size != 30056u) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "q16384 HSACO size changed");
  }
  if (!require_hsa(hsa_code_object_reader_create_from_memory(
                       module->image, module->image_size, &module->reader),
                   "hsa_code_object_reader_create_from_memory", error)) {
    return false;
  }
  module->reader_created = true;
  if (!require_hsa(hsa_executable_create_alt(
                       HSA_PROFILE_FULL,
                       HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                       &module->executable),
                   "hsa_executable_create_alt", error)) {
    return false;
  }
  module->executable_created = true;
  hsa_loaded_code_object_t loaded = {0};
  if (!require_hsa(hsa_executable_load_agent_code_object(
                       module->executable, gpu_agent, module->reader, NULL,
                       &loaded),
                   "hsa_executable_load_agent_code_object", error) ||
      !require_hsa(hsa_executable_freeze(module->executable, NULL),
                   "hsa_executable_freeze", error)) {
    return false;
  }
  const size_t symbol_length = strlen(spec->symbol);
  char* descriptor_name = (char*)malloc(symbol_length + 4u);
  if (descriptor_name == NULL) return fail(error, "symbol allocation failed");
  (void)memcpy(descriptor_name, spec->symbol, symbol_length);
  (void)memcpy(descriptor_name + symbol_length, ".kd", 4u);
  hsa_executable_symbol_t symbol = {0};
  hsa_status_t status = hsa_executable_get_symbol_by_name(
      module->executable, descriptor_name, &gpu_agent, &symbol);
  if (status != HSA_STATUS_SUCCESS) {
    status = hsa_executable_get_symbol_by_name(
        module->executable, spec->symbol, &gpu_agent, &symbol);
  }
  free(descriptor_name);
  if (!require_hsa(status, "hsa_executable_get_symbol_by_name", error)) {
    return false;
  }
  module->kernel.name = spec->symbol;
  if (!require_hsa(hsa_executable_symbol_get_info(
                       symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                       &module->kernel.kernel_object),
                   "reflect kernel object", error) ||
      !require_hsa(hsa_executable_symbol_get_info(
                       symbol,
                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                       &module->kernel.kernarg_segment_size),
                   "reflect kernarg size", error) ||
      !require_hsa(hsa_executable_symbol_get_info(
                       symbol,
                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                       &module->kernel.kernarg_segment_alignment),
                   "reflect kernarg alignment", error) ||
      !require_hsa(hsa_executable_symbol_get_info(
                       symbol,
                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                       &module->kernel.group_segment_size),
                   "reflect group segment", error) ||
      !require_hsa(hsa_executable_symbol_get_info(
                       symbol,
                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                       &module->kernel.private_segment_size),
                   "reflect private segment", error)) {
    return false;
  }
  if (module->kernel.kernel_object == 0u ||
      module->kernel.kernarg_segment_size != spec->kernarg_segment_size ||
      module->kernel.kernarg_segment_alignment !=
          spec->kernarg_segment_alignment ||
      module->kernel.group_segment_size != spec->group_segment_size ||
      module->kernel.private_segment_size != spec->private_segment_size) {
    return fail(error, "reflected q16384 kernel contract changed");
  }
  return true;
}

static void destroy_module(loaded_module_t* module) {
  if (module->executable_created) {
    (void)hsa_executable_destroy(module->executable);
  }
  if (module->reader_created) {
    (void)hsa_code_object_reader_destroy(module->reader);
  }
  free(module->image);
  memset(module, 0, sizeof(*module));
}

static bool load_function_symbol(void* library, const char* name,
                                 void* destination, size_t destination_size,
                                 error_state_t* error) {
  dlerror();
  void* symbol = dlsym(library, name);
  const char* message = dlerror();
  if (message != NULL || symbol == NULL ||
      destination_size != sizeof(symbol)) {
    return fail(error, "cannot resolve HIP symbol %s: %s", name,
                message == NULL ? "invalid function pointer size" : message);
  }
  memcpy(destination, &symbol, destination_size);
  return true;
}

static bool load_hip_api(hip_api_t* api, error_state_t* error) {
  memset(api, 0, sizeof(*api));
  api->library = dlopen(HIP_RUNTIME_PATH, RTLD_NOW | RTLD_LOCAL);
  if (api->library == NULL) {
    return fail(error, "cannot load %s: %s", HIP_RUNTIME_PATH, dlerror());
  }
#define LOAD_HIP(member, symbol_name)                                      \
  do {                                                                      \
    if (!load_function_symbol(api->library, symbol_name, &api->member,      \
                              sizeof(api->member), error)) {                \
      return false;                                                         \
    }                                                                       \
  } while (0)
  LOAD_HIP(init, "hipInit");
  LOAD_HIP(set_device, "hipSetDevice");
  LOAD_HIP(module_load, "hipModuleLoad");
  LOAD_HIP(module_get_function, "hipModuleGetFunction");
  LOAD_HIP(module_launch_kernel, "hipModuleLaunchKernel");
  LOAD_HIP(device_synchronize, "hipDeviceSynchronize");
  LOAD_HIP(module_unload, "hipModuleUnload");
  LOAD_HIP(get_error_string, "hipGetErrorString");
#undef LOAD_HIP
  return true;
}

static const char* hip_error_text(const hip_api_t* api, hip_error_t status) {
  if (api != NULL && api->get_error_string != NULL) {
    const char* text = api->get_error_string(status);
    if (text != NULL) return text;
  }
  return "unknown HIP error";
}

static bool require_hip(const hip_api_t* api, hip_error_t status,
                        const char* operation, error_state_t* error) {
  return status == 0
             ? true
             : fail(error, "%s failed: %s (%d)", operation,
                    hip_error_text(api, status), status);
}

static void unload_hip_api(hip_api_t* api) {
  if (api->library != NULL) (void)dlclose(api->library);
  memset(api, 0, sizeof(*api));
}

static uint64_t queue_load_write_index(void* user_data, const void* queue) {
  (void)user_data;
  return hsa_queue_load_write_index_relaxed((const hsa_queue_t*)queue);
}

static uint64_t queue_load_read_index(void* user_data, const void* queue) {
  (void)user_data;
  return hsa_queue_load_read_index_scacquire((const hsa_queue_t*)queue);
}

static void queue_store_write_index(void* user_data, void* queue,
                                    uint64_t value) {
  (void)user_data;
  hsa_queue_store_write_index_screlease((hsa_queue_t*)queue, value);
}

static void queue_ring_doorbell(void* user_data, void* queue,
                                uint64_t packet_id) {
  (void)user_data;
  hsa_queue_t* hsa_queue = (hsa_queue_t*)queue;
  hsa_signal_store_screlease(hsa_queue->doorbell_signal,
                             (hsa_signal_value_t)packet_id);
}

static void queue_publish_packet(void* user_data,
                                 q16k_aiter_dispatch_packet_t* packet,
                                 uint32_t full_header) {
  queue_adapter_t* adapter = (queue_adapter_t*)user_data;
  __atomic_store_n(&packet->full_header, full_header, __ATOMIC_RELEASE);
  memcpy(&adapter->published_packet, packet, sizeof(*packet));
  adapter->packet_published = true;
}

static void queue_error_callback(hsa_status_t status, hsa_queue_t* queue,
                                 void* user_data) {
  (void)queue;
  run_resources_t* resources = (run_resources_t*)user_data;
  atomic_store_explicit(&resources->queue_error, (int)status,
                        memory_order_release);
  if (resources->raw_completion_created) {
    hsa_signal_store_screlease(resources->raw_completion, 0);
  }
}

static bool queue_size_is_power_of_two(uint32_t value) {
  return value != 0u && (value & (value - 1u)) == 0u;
}

static bool validate_returned_queue_contract(
    queue_contract_observation_t* observation, error_state_t* error) {
  observation->pointer_nonnull = observation->returned_pointer != 0u;
  observation->base_nonnull = observation->returned_base_address != 0u;
  observation->size_power_of_two =
      queue_size_is_power_of_two(observation->returned_size);
  observation->size_within_agent_bounds =
      observation->returned_size >= observation->agent_min_size &&
      observation->returned_size <= observation->agent_max_size;
  observation->size_matches_expected =
      observation->returned_size == observation->expected_size;
  observation->type_ordinary =
      observation->returned_type == HSA_QUEUE_TYPE_SINGLE ||
      observation->returned_type == HSA_QUEUE_TYPE_MULTI;
  observation->kernel_dispatch_supported =
      (observation->returned_features & HSA_QUEUE_FEATURE_KERNEL_DISPATCH) !=
      0u;
  observation->contract_satisfied =
      observation->pointer_nonnull && observation->base_nonnull &&
      observation->size_power_of_two &&
      observation->size_within_agent_bounds &&
      observation->size_matches_expected && observation->type_ordinary &&
      observation->kernel_dispatch_supported;

  if (!observation->pointer_nonnull) {
    return fail(error, "hsa_queue_create returned a null queue pointer");
  }
  if (!observation->base_nonnull) {
    return fail(error, "HSA returned a null queue base address");
  }
  if (!observation->size_power_of_two) {
    return fail(error,
                "HSA returned non-power-of-two queue size %" PRIu32,
                observation->returned_size);
  }
  if (!observation->size_within_agent_bounds) {
    return fail(error,
                "HSA returned queue size %" PRIu32
                " outside agent bounds [%" PRIu32 ", %" PRIu32 "]",
                observation->returned_size, observation->agent_min_size,
                observation->agent_max_size);
  }
  if (!observation->size_matches_expected) {
    return fail(error,
                "HSA returned queue size %" PRIu32
                " instead of expected clamped size %" PRIu32,
                observation->returned_size, observation->expected_size);
  }
  if (observation->returned_type == HSA_QUEUE_TYPE_COOPERATIVE) {
    return fail(error,
                "HSA returned cooperative queue type for ordinary SINGLE "
                "request");
  }
  if (!observation->type_ordinary) {
    return fail(error, "HSA returned unknown queue type %" PRIu32,
                observation->returned_type);
  }
  if (!observation->kernel_dispatch_supported) {
    return fail(error,
                "HSA returned queue features 0x%08" PRIx32
                " without KERNEL_DISPATCH",
                observation->returned_features);
  }
  return true;
}

static bool create_raw_queue(run_resources_t* resources,
                             error_state_t* error) {
  queue_contract_observation_t* observation = &resources->queue_contract;
  memset(observation, 0, sizeof(*observation));
  observation->requested_size = kQueueSize;
  observation->requested_type = HSA_QUEUE_TYPE_SINGLE;

  observation->agent_min_query_status = hsa_agent_get_info(
      resources->runtime.gpu_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
      &observation->agent_min_size);
  observation->agent_min_query_completed = true;
  if (!require_hsa(observation->agent_min_query_status,
                   "hsa_agent_get_info(QUEUE_MIN_SIZE)", error)) {
    return false;
  }
  observation->agent_max_query_status = hsa_agent_get_info(
      resources->runtime.gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
      &observation->agent_max_size);
  observation->agent_max_query_completed = true;
  if (!require_hsa(observation->agent_max_query_status,
                   "hsa_agent_get_info(QUEUE_MAX_SIZE)", error)) {
    return false;
  }
  observation->agent_type_query_status = hsa_agent_get_info(
      resources->runtime.gpu_agent, HSA_AGENT_INFO_QUEUE_TYPE,
      &observation->agent_type);
  observation->agent_type_query_completed = true;
  if (!require_hsa(observation->agent_type_query_status,
                   "hsa_agent_get_info(QUEUE_TYPE)", error)) {
    return false;
  }
  if (observation->agent_min_size == 0u ||
      !queue_size_is_power_of_two(observation->agent_min_size)) {
    return fail(error, "GPU queue minimum is invalid: %" PRIu32,
                observation->agent_min_size);
  }
  if (observation->agent_max_size < observation->agent_min_size) {
    return fail(error,
                "GPU queue bounds are invalid: min=%" PRIu32
                " max=%" PRIu32,
                observation->agent_min_size, observation->agent_max_size);
  }
  observation->expected_size =
      observation->requested_size < observation->agent_min_size
          ? observation->agent_min_size
          : observation->requested_size;
  if (!queue_size_is_power_of_two(observation->requested_size)) {
    return fail(error, "requested HSA queue size is not a power of two: %" PRIu32,
                observation->requested_size);
  }
  if (observation->expected_size > observation->agent_max_size) {
    return fail(error,
                "requested HSA queue size %" PRIu32
                " exceeds agent maximum %" PRIu32,
                observation->requested_size, observation->agent_max_size);
  }
  atomic_store_explicit(&resources->queue_error, (int)HSA_STATUS_SUCCESS,
                        memory_order_release);
  resources->queue = NULL;
  observation->create_attempted = true;
  observation->create_status = hsa_queue_create(
      resources->runtime.gpu_agent, observation->requested_size,
      observation->requested_type, queue_error_callback, resources, 0u,
      q16k_attention_kernel_spec()->group_segment_size, &resources->queue);
  observation->returned_pointer =
      (uint64_t)(uintptr_t)resources->queue;
  if (!require_hsa(observation->create_status, "hsa_queue_create", error)) {
    return false;
  }
  if (resources->queue != NULL) {
    observation->returned_base_address =
        (uint64_t)(uintptr_t)resources->queue->base_address;
    observation->returned_doorbell_signal_handle =
        resources->queue->doorbell_signal.handle;
    observation->returned_id = resources->queue->id;
    observation->returned_size = resources->queue->size;
    observation->returned_type = resources->queue->type;
    observation->returned_features = resources->queue->features;
  }
  if (!validate_returned_queue_contract(observation, error)) {
    return false;
  }
  resources->queue_adapter.queue = resources->queue;
  resources->canonical_queue = (q16k_aiter_queue_state_t){
      .queue = resources->queue,
      .packets = (q16k_aiter_dispatch_packet_t*)resources->queue->base_address,
      .queue_size = resources->queue->size,
      .kernarg_data = resources->kernarg_ring,
      .kernarg_stride = Q16K_AITER_KERNARG_STRIDE,
      .kernarg_capacity = 1u,
      .kernarg_cursor = 0u,
      .queue_depth_limit = 1u,
      .doorbell_batch_size = 0u,
      .batch_dispatch = 1u,
      .user_data = &resources->queue_adapter,
      .ops = {
          .load_write_index_relaxed = queue_load_write_index,
          .load_read_index_scacquire = queue_load_read_index,
          .store_write_index_screlease = queue_store_write_index,
          .ring_doorbell_screlease = queue_ring_doorbell,
          .publish_dispatch_packet = queue_publish_packet,
      },
  };
  return true;
}

static bool stop_raw_queue(run_resources_t* resources,
                           error_state_t* error) {
  if (resources->queue == NULL) return true;
  const hsa_status_t inactivate = hsa_queue_inactivate(resources->queue);
  const hsa_status_t destroy = hsa_queue_destroy(resources->queue);
  resources->queue = NULL;
  resources->queue_adapter.queue = NULL;
  if (inactivate != HSA_STATUS_SUCCESS) {
    return fail(error, "hsa_queue_inactivate failed: %s (%d)",
                hsa_status_text(inactivate), (int)inactivate);
  }
  return require_hsa(destroy, "hsa_queue_destroy", error);
}

static bool compute_buffer_sizes(run_resources_t* resources,
                                 error_state_t* error) {
  size_t elements = 0u;
  if (!checked_mul_size((size_t)kQueryCount, (size_t)kQueryHeads,
                        &elements) ||
      !checked_mul_size(elements, (size_t)kHeadDimension, &elements) ||
      !checked_mul_size(elements, sizeof(uint16_t), &resources->q_bytes)) {
    return fail(error, "Q tensor size overflow");
  }
  if (!checked_mul_size((size_t)Q16K_AITER_CONTEXT_CAPACITY,
                        (size_t)kKvHeads, &elements) ||
      !checked_mul_size(elements, (size_t)kHeadDimension, &elements) ||
      !checked_mul_size(elements, sizeof(uint16_t), &resources->kv_bytes)) {
    return fail(error, "KV tensor size overflow");
  }
  size_t page_count = 0u;
  if (!checked_add_size((size_t)Q16K_AITER_MAX_PROMPT_TOKEN_COUNT,
                        (size_t)Q16K_AITER_PAGE_INDEX_PADDING, &page_count) ||
      !checked_mul_size(page_count, sizeof(int32_t),
                        &resources->page_index_bytes)) {
    return fail(error, "page-index size overflow");
  }
  resources->metadata_bytes = sizeof(q16k_aiter_attention_metadata_t);
  resources->output_bytes = resources->q_bytes;
  return true;
}

static bool prepare_resources(run_resources_t* resources,
                              error_state_t* error) {
  if (!compute_buffer_sizes(resources, error) ||
      !initialize_guarded_buffer(&resources->runtime, resources->q_bytes,
                                 "guarded Q", &resources->q, error) ||
      !initialize_guarded_buffer(&resources->runtime, resources->kv_bytes,
                                 "guarded K", &resources->k, error) ||
      !initialize_guarded_buffer(&resources->runtime, resources->kv_bytes,
                                 "guarded V", &resources->v, error) ||
      !initialize_guarded_buffer(&resources->runtime,
                                 resources->metadata_bytes,
                                 "guarded metadata", &resources->metadata,
                                 error) ||
      !initialize_guarded_buffer(&resources->runtime,
                                 resources->page_index_bytes,
                                 "guarded page indices",
                                 &resources->page_indices, error) ||
      !initialize_guarded_buffer(&resources->runtime, resources->output_bytes,
                                 "guarded raw output",
                                 &resources->raw_output, error) ||
      !initialize_guarded_buffer(&resources->runtime, resources->output_bytes,
                                 "guarded HIP output",
                                 &resources->hip_output, error)) {
    return false;
  }
  if (!allocate_host(&resources->runtime, resources->q_bytes, "host Q",
                     &resources->host_q, error) ||
      !allocate_host(&resources->runtime, resources->kv_bytes, "host K",
                     &resources->host_k, error) ||
      !allocate_host(&resources->runtime, resources->kv_bytes, "host V",
                     &resources->host_v, error) ||
      !allocate_host(&resources->runtime, resources->metadata_bytes,
                     "host metadata", &resources->host_metadata, error) ||
      !allocate_host(&resources->runtime, resources->page_index_bytes,
                     "host page indices", &resources->host_page_indices,
                     error) ||
      !allocate_host(&resources->runtime, resources->output_bytes,
                     "host raw output", &resources->host_raw_output, error) ||
      !allocate_host(&resources->runtime, resources->output_bytes,
                     "host HIP output", &resources->host_hip_output, error) ||
      !allocate_host(&resources->runtime, kGuardBytes, "host guard staging",
                     &resources->host_guard, error) ||
      !allocate_pool(resources->runtime.kernarg_pool,
                     Q16K_AITER_KERNARG_STRIDE, "canonical kernarg ring",
                     &resources->kernarg_ring, error)) {
    return false;
  }
  if (((uintptr_t)resources->kernarg_ring &
       (Q16K_AITER_KERNARG_ALIGNMENT - 1u)) != 0u ||
      !require_hsa(hsa_amd_agents_allow_access(
                       1u, &resources->runtime.gpu_agent, NULL,
                       resources->kernarg_ring),
                   "allow GPU access to kernarg ring", error)) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "kernarg ring alignment changed");
  }
  memset(resources->kernarg_ring, 0, Q16K_AITER_KERNARG_STRIDE);
  microgate_fill_q((uint16_t*)resources->host_q, kQueryCount, kPositionBase);
  microgate_fill_kv((uint16_t*)resources->host_k,
                    (uint16_t*)resources->host_v, kPositionBase);
  q16k_aiter_attention_metadata_t* metadata =
      (q16k_aiter_attention_metadata_t*)resources->host_metadata;
  if (q16k_build_attention_metadata(kPositionBase, kQueryCount, metadata) !=
          Q16K_AITER_STATUS_OK ||
      q16k_fill_identity_page_indices(
          (int32_t*)resources->host_page_indices,
          resources->page_index_bytes / sizeof(int32_t)) !=
          Q16K_AITER_STATUS_OK) {
    return fail(error, "canonical attention input construction failed");
  }
  if (!copy_memory(resources->q.data, resources->host_q, resources->q_bytes,
                   "copy Q to GPU", error) ||
      !copy_memory(resources->k.data, resources->host_k, resources->kv_bytes,
                   "copy K to GPU", error) ||
      !copy_memory(resources->v.data, resources->host_v, resources->kv_bytes,
                   "copy V to GPU", error) ||
      !copy_memory(resources->metadata.data, resources->host_metadata,
                   resources->metadata_bytes, "copy metadata to GPU", error) ||
      !copy_memory(resources->page_indices.data,
                   resources->host_page_indices,
                   resources->page_index_bytes,
                   "copy page indices to GPU", error) ||
      !require_hsa(hsa_amd_memory_fill(
                       resources->raw_output.data, kPoisonWord,
                       resources->output_bytes / sizeof(uint32_t)),
                   "poison raw output", error) ||
      !require_hsa(hsa_amd_memory_fill(
                       resources->hip_output.data, kPoisonWord,
                       resources->output_bytes / sizeof(uint32_t)),
                   "poison HIP output", error) ||
      !require_hsa(hsa_signal_create(1, 0u, NULL,
                                     &resources->raw_completion),
                   "hsa_signal_create(raw completion)", error)) {
    return false;
  }
  resources->raw_completion_created = true;
  return create_raw_queue(resources, error);
}

static bool capture_input_hashes(run_resources_t* resources,
                                 input_hashes_t* hashes,
                                 error_state_t* error) {
  if (!copy_memory(resources->host_q, resources->q.data, resources->q_bytes,
                   "snapshot Q", error) ||
      !copy_memory(resources->host_k, resources->k.data, resources->kv_bytes,
                   "snapshot K", error) ||
      !copy_memory(resources->host_v, resources->v.data, resources->kv_bytes,
                   "snapshot V", error) ||
      !copy_memory(resources->host_metadata, resources->metadata.data,
                   resources->metadata_bytes, "snapshot metadata", error) ||
      !copy_memory(resources->host_page_indices,
                   resources->page_indices.data,
                   resources->page_index_bytes, "snapshot page indices",
                   error)) {
    return false;
  }
  hash_bytes(resources->host_q, resources->q_bytes, hashes->q);
  hash_bytes(resources->host_k, resources->kv_bytes, hashes->k);
  hash_bytes(resources->host_v, resources->kv_bytes, hashes->v);
  hash_bytes(resources->host_metadata, resources->metadata_bytes,
             hashes->metadata);
  hash_bytes(resources->host_page_indices, resources->page_index_bytes,
             hashes->page_indices);
  return true;
}

static bool input_hashes_equal(const input_hashes_t* left,
                               const input_hashes_t* right) {
  return strcmp(left->q, right->q) == 0 &&
         strcmp(left->k, right->k) == 0 &&
         strcmp(left->v, right->v) == 0 &&
         strcmp(left->metadata, right->metadata) == 0 &&
         strcmp(left->page_indices, right->page_indices) == 0;
}

static uint64_t count_word_mismatches(const uint32_t* words, size_t bytes,
                                      uint32_t expected) {
  uint64_t mismatches = 0u;
  const size_t count = bytes / sizeof(uint32_t);
  for (size_t i = 0u; i < count; ++i) {
    if (words[i] != expected) ++mismatches;
  }
  return mismatches;
}

static bool count_buffer_guard_mismatches(run_resources_t* resources,
                                          const guarded_buffer_t* buffer,
                                          uint64_t* mismatches,
                                          error_state_t* error) {
  if (!copy_memory(resources->host_guard, buffer->base, kGuardBytes,
                   "snapshot leading guard", error)) {
    return false;
  }
  *mismatches += count_word_mismatches(
      (const uint32_t*)resources->host_guard, kGuardBytes,
      buffer->guard_word);
  const void* trailing =
      (const uint8_t*)buffer->data + buffer->payload_bytes;
  if (!copy_memory(resources->host_guard, trailing, kGuardBytes,
                   "snapshot trailing guard", error)) {
    return false;
  }
  *mismatches += count_word_mismatches(
      (const uint32_t*)resources->host_guard, kGuardBytes,
      buffer->guard_word);
  return true;
}

static bool count_input_guard_mismatches(run_resources_t* resources,
                                         uint64_t* mismatches,
                                         error_state_t* error) {
  *mismatches = 0u;
  return count_buffer_guard_mismatches(resources, &resources->q, mismatches,
                                       error) &&
         count_buffer_guard_mismatches(resources, &resources->k, mismatches,
                                       error) &&
         count_buffer_guard_mismatches(resources, &resources->v, mismatches,
                                       error) &&
         count_buffer_guard_mismatches(resources, &resources->metadata,
                                       mismatches, error) &&
         count_buffer_guard_mismatches(resources, &resources->page_indices,
                                       mismatches, error);
}

static bool compare_reference(const uint16_t* output,
                              const microgate_reference_t* reference,
                              output_observation_t* observation,
                              error_state_t* error) {
  if (reference->position_base != kPositionBase ||
      reference->query_count != kQueryCount) {
    return fail(error, "CPU reference geometry changed");
  }
  for (size_t sample = 0u; sample < MICROGATE_SAMPLE_COUNT; ++sample) {
    const uint32_t row = reference->rows[sample];
    for (uint32_t head = 0u; head < kQueryHeads; ++head) {
      for (uint32_t dimension = 0u; dimension < kHeadDimension;
           ++dimension) {
        const size_t index =
            ((size_t)row * kQueryHeads + head) * kHeadDimension + dimension;
        const double actual =
            (double)microgate_bf16_to_float(output[index]);
        const double expected =
            (double)reference->values[sample][head][dimension];
        const double difference = fabs(actual - expected);
        const double tolerance =
            (double)MICROGATE_ABSOLUTE_TOLERANCE +
            (double)MICROGATE_RELATIVE_TOLERANCE * fabs(expected);
        const double ratio = tolerance == 0.0 ? INFINITY
                                              : difference / tolerance;
        ++observation->sampled_elements_compared;
        if (!isfinite(actual) || difference > tolerance) {
          ++observation->reference_violations;
        }
        if (difference > observation->max_absolute_error) {
          observation->max_absolute_error = difference;
        }
        if (ratio > observation->max_tolerance_ratio) {
          observation->max_tolerance_ratio = ratio;
        }
      }
    }
  }
  return true;
}

static bool observe_output(run_resources_t* resources,
                           const guarded_buffer_t* buffer, void* host_output,
                           const microgate_reference_t* reference,
                           output_observation_t* observation,
                           error_state_t* error) {
  memset(observation, 0, sizeof(*observation));
  if (!copy_memory(host_output, buffer->data, resources->output_bytes,
                   "snapshot attention output", error) ||
      !count_buffer_guard_mismatches(resources, buffer,
                                     &observation->guard_mismatches, error)) {
    return false;
  }
  hash_bytes(host_output, resources->output_bytes, observation->sha256);
  const uint16_t* values = (const uint16_t*)host_output;
  const size_t count = resources->output_bytes / sizeof(uint16_t);
  for (size_t i = 0u; i < count; ++i) {
    if (values[i] == kPoisonBf16) ++observation->poison_values_remaining;
    if ((values[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) {
      ++observation->nonfinite_values;
    }
  }
  return compare_reference(values, reference, observation, error);
}

static bool observe_poisoned_output(run_resources_t* resources,
                                    const guarded_buffer_t* buffer,
                                    void* host_output,
                                    uint64_t* poison_values,
                                    uint64_t* guard_mismatches,
                                    error_state_t* error) {
  *poison_values = 0u;
  *guard_mismatches = 0u;
  if (!copy_memory(host_output, buffer->data, resources->output_bytes,
                   "snapshot poisoned attention output", error) ||
      !count_buffer_guard_mismatches(resources, buffer, guard_mismatches,
                                     error)) {
    return false;
  }
  const uint16_t* values = (const uint16_t*)host_output;
  const size_t count = resources->output_bytes / sizeof(uint16_t);
  for (size_t i = 0u; i < count; ++i) {
    if (values[i] == kPoisonBf16) ++*poison_values;
  }
  return true;
}

static bool build_kernarg(run_resources_t* resources, void* output,
                          uint8_t kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE],
                          q16k_aiter_launch_geometry_t* geometry,
                          error_state_t* error) {
  const q16k_aiter_attention_request_t request = {
      .q = resources->q.data,
      .k = resources->k.data,
      .v = resources->v.data,
      .output = output,
      .kv_indptr = (const int32_t*)resources->metadata.data,
      .kv_page_indices = (const int32_t*)resources->page_indices.data,
      .kv_last_page_lens =
          (const int32_t*)((const uint8_t*)resources->metadata.data +
                           Q16K_AITER_METADATA_LAST_PAGE_LENGTH_OFFSET),
      .cu_seqlens_q =
          (const int32_t*)((const uint8_t*)resources->metadata.data +
                           Q16K_AITER_METADATA_CU_SEQLENS_Q_OFFSET),
      .position_base = kPositionBase,
      .logical_query_count = kQueryCount,
  };
  const q16k_aiter_status_t status = q16k_build_attention_kernarg(
      &request, kernarg, Q16K_AITER_ATTENTION_KERNARG_SIZE, geometry);
  if (status != Q16K_AITER_STATUS_OK) {
    return fail(error, "q16k_build_attention_kernarg failed: %s",
                q16k_aiter_status_string(status));
  }
  if (geometry->grid_size[0] != 3072u || geometry->grid_size[1] != 1u ||
      geometry->grid_size[2] != 128u ||
      geometry->workgroup_size[0] != 256u ||
      geometry->workgroup_size[1] != 1u ||
      geometry->workgroup_size[2] != 1u) {
    return fail(error, "canonical q16384 launch geometry changed");
  }
  return true;
}

static bool validate_raw_packet(const run_resources_t* resources,
                                const q16k_aiter_launch_geometry_t* geometry,
                                const uint8_t* kernarg,
                                const run_result_t* result,
                                error_state_t* error) {
  const q16k_aiter_dispatch_packet_t* packet =
      &resources->queue_adapter.published_packet;
  if (!resources->queue_adapter.packet_published ||
      packet->full_header != kExpectedFullHeader ||
      packet->workgroup_size_x != geometry->workgroup_size[0] ||
      packet->workgroup_size_y != geometry->workgroup_size[1] ||
      packet->workgroup_size_z != geometry->workgroup_size[2] ||
      packet->grid_size_x != geometry->grid_size[0] ||
      packet->grid_size_y != geometry->grid_size[1] ||
      packet->grid_size_z != geometry->grid_size[2] ||
      packet->private_segment_size != resources->module.kernel.private_segment_size ||
      packet->group_segment_size != resources->module.kernel.group_segment_size ||
      packet->kernel_object != resources->module.kernel.kernel_object ||
      packet->kernarg_address != (uint64_t)(uintptr_t)resources->kernarg_ring ||
      packet->completion_signal != resources->raw_completion.handle ||
      result->raw_dispatch.packet_id == UINT64_MAX ||
      result->raw_dispatch.kernarg_slot != 0u ||
      result->raw_dispatch.doorbell_written != 1u ||
      memcmp(resources->kernarg_ring, kernarg,
             Q16K_AITER_ATTENTION_KERNARG_SIZE) != 0) {
    return fail(error, "raw HSA dispatch packet or kernarg contract changed");
  }
  return true;
}

static bool dispatch_raw(run_resources_t* resources, run_result_t* result,
                         const microgate_reference_t* reference,
                         error_state_t* error) {
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t geometry;
  if (!build_kernarg(resources, resources->raw_output.data, kernarg,
                     &geometry, error)) {
    return false;
  }
  result->raw_started_ns = monotonic_ns();
  if (result->raw_started_ns == 0u) {
    return fail(error, "cannot read raw dispatch start time");
  }
  const q16k_aiter_status_t status = q16k_aiter_dispatch_raw(
      &resources->canonical_queue, &resources->module.kernel, &geometry,
      kernarg, sizeof(kernarg), resources->raw_completion.handle, 1u,
      &result->raw_dispatch);
  if (status != Q16K_AITER_STATUS_OK) {
    return fail(error, "q16k_aiter_dispatch_raw failed: %s",
                q16k_aiter_status_string(status));
  }
  const uint64_t timeout =
      resources->runtime.timestamp_frequency * kDispatchTimeoutSeconds;
  const hsa_signal_value_t completion = hsa_signal_wait_scacquire(
      resources->raw_completion, HSA_SIGNAL_CONDITION_EQ, 0, timeout,
      HSA_WAIT_STATE_BLOCKED);
  result->raw_completed_ns = monotonic_ns();
  const hsa_status_t queue_status = (hsa_status_t)atomic_load_explicit(
      &resources->queue_error, memory_order_acquire);
  if (completion != 0 || queue_status != HSA_STATUS_SUCCESS ||
      result->raw_completed_ns < result->raw_started_ns) {
    return fail(error, "raw HSA dispatch did not complete cleanly");
  }
  memcpy(&result->raw_packet, &resources->queue_adapter.published_packet,
         sizeof(result->raw_packet));
  return validate_raw_packet(resources, &geometry, kernarg, result, error) &&
         capture_input_hashes(resources, &result->post_raw_inputs, error) &&
         count_input_guard_mismatches(
             resources, &result->input_guard_mismatches_after_raw, error) &&
         observe_output(resources, &resources->raw_output,
                        resources->host_raw_output, reference,
                        &result->raw_output, error) &&
         observe_poisoned_output(
             resources, &resources->hip_output, resources->host_hip_output,
             &result->hip_poison_values_after_raw,
             &result->hip_guard_mismatches_after_raw, error);
}

static bool compare_explicit_kernargs(
    const uint8_t raw[Q16K_AITER_ATTENTION_KERNARG_SIZE],
    const uint8_t hip[Q16K_AITER_ATTENTION_KERNARG_SIZE]) {
  return memcmp(raw, hip, 24u) == 0 &&
         memcmp(raw + 32u, hip + 32u, kHipExplicitArgumentBytes - 32u) == 0;
}

static bool dispatch_hip(run_resources_t* resources, const char* hsaco_path,
                         run_result_t* result,
                         const microgate_reference_t* reference,
                         error_state_t* error) {
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t raw_kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t hip_kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t raw_geometry;
  q16k_aiter_launch_geometry_t hip_geometry;
  if (!build_kernarg(resources, resources->raw_output.data, raw_kernarg,
                     &raw_geometry, error) ||
      !build_kernarg(resources, resources->hip_output.data, hip_kernarg,
                     &hip_geometry, error) ||
      memcmp(&raw_geometry, &hip_geometry, sizeof(raw_geometry)) != 0 ||
      !compare_explicit_kernargs(raw_kernarg, hip_kernarg)) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "raw and HIP explicit argument objects differ");
  }
  uint64_t raw_output_pointer = 0u;
  uint64_t hip_output_pointer = 0u;
  memcpy(&raw_output_pointer, raw_kernarg + 24u, sizeof(raw_output_pointer));
  memcpy(&hip_output_pointer, hip_kernarg + 24u, sizeof(hip_output_pointer));
  if (raw_output_pointer == hip_output_pointer ||
      raw_output_pointer != (uint64_t)(uintptr_t)resources->raw_output.data ||
      hip_output_pointer != (uint64_t)(uintptr_t)resources->hip_output.data) {
    return fail(error, "raw and HIP outputs are not distinct bound buffers");
  }

  hip_api_t api;
  hip_module_t module = NULL;
  bool success = load_hip_api(&api, error);
  hip_function_t function = NULL;
  if (success) success = require_hip(&api, api.init(0u), "hipInit", error);
  if (success) {
    success = require_hip(&api, api.set_device(0), "hipSetDevice", error);
  }
  if (success) {
    success = require_hip(&api, api.module_load(&module, hsaco_path),
                          "hipModuleLoad", error);
  }
  if (success) {
    success = require_hip(
        &api,
        api.module_get_function(&function, module,
                                q16k_attention_kernel_spec()->symbol),
        "hipModuleGetFunction", error);
  }
  size_t explicit_size = kHipExplicitArgumentBytes;
  void* launch_config[] = {
      (void*)(uintptr_t)1u, hip_kernarg, (void*)(uintptr_t)2u,
      &explicit_size, (void*)(uintptr_t)3u,
  };
  if (success) {
    result->hip_started_ns = monotonic_ns();
    if (result->hip_started_ns == 0u) {
      success = fail(error, "cannot read HIP dispatch start time");
    }
  }
  if (success) {
    success = require_hip(
        &api,
        api.module_launch_kernel(function, 12u, 1u, 128u, 256u, 1u, 1u,
                                 0u, NULL, NULL, launch_config),
        "hipModuleLaunchKernel", error);
  }
  if (success) {
    success = require_hip(&api, api.device_synchronize(),
                          "hipDeviceSynchronize", error);
    result->hip_completed_ns = monotonic_ns();
    if (success && result->hip_completed_ns < result->hip_started_ns) {
      success = fail(error, "HIP dispatch timestamps are invalid");
    }
  }
  if (module != NULL && api.module_unload != NULL) {
    const hip_error_t unload_status = api.module_unload(module);
    if (unload_status != 0 && success) {
      success = fail(error, "hipModuleUnload failed: %s (%d)",
                     hip_error_text(&api, unload_status), unload_status);
    }
  }
  unload_hip_api(&api);
  if (!success) return false;
  return capture_input_hashes(resources, &result->post_hip_inputs, error) &&
         count_input_guard_mismatches(
             resources, &result->input_guard_mismatches_after_hip, error) &&
         observe_output(resources, &resources->hip_output,
                        resources->host_hip_output, reference,
                        &result->hip_output, error);
}

static void compare_outputs(run_resources_t* resources,
                            run_result_t* result) {
  result->byte_equal = true;
  result->first_mismatch_byte = UINT64_MAX;
  result->first_mismatch_element = UINT64_MAX;
  result->first_mismatch_row = UINT32_MAX;
  result->first_mismatch_head = UINT32_MAX;
  result->first_mismatch_dimension = UINT32_MAX;
  const uint8_t* raw = (const uint8_t*)resources->host_raw_output;
  const uint8_t* hip = (const uint8_t*)resources->host_hip_output;
  for (size_t byte = 0u; byte < resources->output_bytes; ++byte) {
    if (raw[byte] == hip[byte]) continue;
    if (result->byte_equal) {
      const uint64_t element = byte / sizeof(uint16_t);
      const uint64_t elements_per_row =
          (uint64_t)kQueryHeads * kHeadDimension;
      result->first_mismatch_byte = byte;
      result->first_mismatch_element = element;
      result->first_mismatch_row = (uint32_t)(element / elements_per_row);
      const uint64_t within_row = element % elements_per_row;
      result->first_mismatch_head =
          (uint32_t)(within_row / kHeadDimension);
      result->first_mismatch_dimension =
          (uint32_t)(within_row % kHeadDimension);
      result->byte_equal = false;
    }
    ++result->mismatch_bytes;
  }
}

static bool result_passes(const run_result_t* result) {
  const uint64_t output_elements =
      (uint64_t)kQueryCount * kQueryHeads * kHeadDimension;
  return result->byte_equal && result->mismatch_bytes == 0u &&
         result->hip_poison_values_after_raw == output_elements &&
         result->hip_guard_mismatches_after_raw == 0u &&
         result->raw_output.poison_values_remaining == 0u &&
         result->raw_output_after_hip.poison_values_remaining == 0u &&
         result->hip_output.poison_values_remaining == 0u &&
         result->raw_output.nonfinite_values == 0u &&
         result->raw_output_after_hip.nonfinite_values == 0u &&
         result->hip_output.nonfinite_values == 0u &&
         result->raw_output.guard_mismatches == 0u &&
         result->raw_output_after_hip.guard_mismatches == 0u &&
         result->hip_output.guard_mismatches == 0u &&
         result->raw_output.reference_violations == 0u &&
         result->raw_output_after_hip.reference_violations == 0u &&
         result->hip_output.reference_violations == 0u &&
         result->raw_output.sampled_elements_compared ==
             (uint64_t)MICROGATE_SAMPLE_COUNT * kQueryHeads * kHeadDimension &&
         result->hip_output.sampled_elements_compared ==
             result->raw_output.sampled_elements_compared &&
         result->raw_output_after_hip.sampled_elements_compared ==
             result->raw_output.sampled_elements_compared &&
         result->inputs_unchanged_after_raw &&
         result->inputs_unchanged_after_hip &&
         result->raw_output_unchanged_after_hip &&
         result->input_guard_mismatches_after_raw == 0u &&
         result->input_guard_mismatches_after_hip == 0u &&
         result->raw_packet.full_header == kExpectedFullHeader;
}

static void release_resources(run_resources_t* resources) {
  if (resources == NULL) return;
  if (resources->queue != NULL) {
    (void)hsa_queue_inactivate(resources->queue);
    (void)hsa_queue_destroy(resources->queue);
    resources->queue = NULL;
  }
  if (resources->raw_completion_created) {
    (void)hsa_signal_destroy(resources->raw_completion);
    resources->raw_completion_created = false;
  }
  free_pool_pointer(&resources->kernarg_ring);
  free_pool_pointer(&resources->host_guard);
  free_pool_pointer(&resources->host_hip_output);
  free_pool_pointer(&resources->host_raw_output);
  free_pool_pointer(&resources->host_page_indices);
  free_pool_pointer(&resources->host_metadata);
  free_pool_pointer(&resources->host_v);
  free_pool_pointer(&resources->host_k);
  free_pool_pointer(&resources->host_q);
  free_guarded_buffer(&resources->hip_output);
  free_guarded_buffer(&resources->raw_output);
  free_guarded_buffer(&resources->page_indices);
  free_guarded_buffer(&resources->metadata);
  free_guarded_buffer(&resources->v);
  free_guarded_buffer(&resources->k);
  free_guarded_buffer(&resources->q);
  destroy_module(&resources->module);
}

static bool execute_launch_parity(run_resources_t* resources,
                                  const char* hsaco_path,
                                  run_result_t* result,
                                  bool* gpu_execution_performed,
                                  error_state_t* error) {
  memset(result, 0, sizeof(*result));
  microgate_reference_t reference;
  if (!microgate_build_reference(kQueryCount, &reference) ||
      !load_module(&resources->module, resources->runtime.gpu_agent,
                   hsaco_path, error) ||
      !prepare_resources(resources, error) ||
      !capture_input_hashes(resources, &result->pre_inputs, error)) {
    return error->message[0] != '\0'
               ? false
               : fail(error, "pre-dispatch preparation failed");
  }
  *gpu_execution_performed = true;
  if (!dispatch_raw(resources, result, &reference, error)) return false;
  const uint64_t output_elements =
      (uint64_t)kQueryCount * kQueryHeads * kHeadDimension;
  if (result->hip_poison_values_after_raw != output_elements ||
      result->hip_guard_mismatches_after_raw != 0u) {
    return fail(error, "raw HSA modified the reserved HIP output region");
  }
  result->inputs_unchanged_after_raw =
      input_hashes_equal(&result->pre_inputs, &result->post_raw_inputs);
  if (!stop_raw_queue(resources, error)) return false;
  if (!dispatch_hip(resources, hsaco_path, result, &reference, error)) {
    return false;
  }
  result->inputs_unchanged_after_hip =
      input_hashes_equal(&result->pre_inputs, &result->post_hip_inputs);
  if (!observe_output(resources, &resources->raw_output,
                      resources->host_raw_output, &reference,
                      &result->raw_output_after_hip, error)) {
    return false;
  }
  result->raw_output_unchanged_after_hip =
      strcmp(result->raw_output.sha256,
             result->raw_output_after_hip.sha256) == 0;
  if (!result->raw_output_unchanged_after_hip ||
      result->raw_output_after_hip.guard_mismatches != 0u) {
    return fail(error, "HIP modified the completed raw-HSA output region");
  }
  compare_outputs(resources, result);
  return result_passes(result)
             ? true
             : fail(error, "raw-HSA and HIP launch-ABI parity checks failed");
}

static void write_input_hashes(FILE* stream, const input_hashes_t* hashes,
                               unsigned indentation) {
  const int width = (int)indentation;
  (void)fprintf(stream,
                "%*s{\"q\":\"%s\",\"k\":\"%s\",\"v\":\"%s\","
                "\"metadata\":\"%s\",\"page_indices\":\"%s\"}",
                width, "", hashes->q, hashes->k, hashes->v,
                hashes->metadata, hashes->page_indices);
}

static void write_output_observation(FILE* stream,
                                     const output_observation_t* observation,
                                     unsigned indentation) {
  const int width = (int)indentation;
  (void)fprintf(
      stream,
      "%*s{\"sha256\":\"%s\",\"poison_values_remaining\":%" PRIu64
      ",\"nonfinite_values\":%" PRIu64
      ",\"guard_mismatches\":%" PRIu64
      ",\"sampled_elements_compared\":%" PRIu64
      ",\"reference_violations\":%" PRIu64
      ",\"max_absolute_error\":%.17g,\"max_tolerance_ratio\":%.17g}",
      width, "", observation->sha256,
      observation->poison_values_remaining,
      observation->nonfinite_values, observation->guard_mismatches,
      observation->sampled_elements_compared,
      observation->reference_violations,
      observation->max_absolute_error,
      observation->max_tolerance_ratio);
}

static void write_queue_contract(
    FILE* stream, const queue_contract_observation_t* observation) {
  if (observation == NULL) {
    fputs("null", stream);
    return;
  }
  (void)fprintf(
      stream,
      "{\"agent\":{\"min_query_completed\":%s,\"min_query_status\":%d,"
      "\"min_size\":%" PRIu32 ",\"max_query_completed\":%s,"
      "\"max_query_status\":%d,\"max_size\":%" PRIu32 ","
      "\"type_query_completed\":%s,\"type_query_status\":%d,"
      "\"type\":%" PRIu32 "},"
      "\"request\":{\"size\":%" PRIu32 ",\"type\":%" PRIu32 "},"
      "\"expected_size\":%" PRIu32 ","
      "\"create\":{\"attempted\":%s,\"status\":%d},"
      "\"returned\":{\"pointer\":\"0x%016" PRIx64 "\","
      "\"base_address\":\"0x%016" PRIx64 "\","
      "\"doorbell_signal_handle\":\"0x%016" PRIx64 "\","
      "\"id\":%" PRIu64 ",\"size\":%" PRIu32 ","
      "\"type\":%" PRIu32 ",\"features\":%" PRIu32 "},"
      "\"predicates\":{\"pointer_nonnull\":%s,\"base_nonnull\":%s,"
      "\"size_power_of_two\":%s,\"size_within_agent_bounds\":%s,"
      "\"size_matches_expected\":%s,\"type_ordinary\":%s,"
      "\"kernel_dispatch_supported\":%s,\"contract_satisfied\":%s}}",
      observation->agent_min_query_completed ? "true" : "false",
      (int)observation->agent_min_query_status,
      observation->agent_min_size,
      observation->agent_max_query_completed ? "true" : "false",
      (int)observation->agent_max_query_status,
      observation->agent_max_size,
      observation->agent_type_query_completed ? "true" : "false",
      (int)observation->agent_type_query_status, observation->agent_type,
      observation->requested_size, observation->requested_type,
      observation->expected_size,
      observation->create_attempted ? "true" : "false",
      (int)observation->create_status, observation->returned_pointer,
      observation->returned_base_address,
      observation->returned_doorbell_signal_handle,
      observation->returned_id, observation->returned_size,
      observation->returned_type, observation->returned_features,
      observation->pointer_nonnull ? "true" : "false",
      observation->base_nonnull ? "true" : "false",
      observation->size_power_of_two ? "true" : "false",
      observation->size_within_agent_bounds ? "true" : "false",
      observation->size_matches_expected ? "true" : "false",
      observation->type_ordinary ? "true" : "false",
      observation->kernel_dispatch_supported ? "true" : "false",
      observation->contract_satisfied ? "true" : "false");
}

static bool write_success_result(int descriptor, const char* path,
                                 const char* root_digest,
                                 const char* authorization_id,
                                 const runtime_state_t* runtime,
                                 const queue_contract_observation_t* queue,
                                 const run_result_t* result,
                                 error_state_t* error) {
  FILE* stream = NULL;
  if (!begin_reserved_result(descriptor, path, &stream, error)) return false;
  fputs("{\n  \"schema\": \"" MICROGATE_SCHEMA "\",\n"
        "  \"status\": \"pass\",\n"
        "  \"gpu_execution_performed\": true,\n"
        "  \"authorization_id\": ", stream);
  json_string(stream, authorization_id);
  fputs(",\n  \"microgate_root_sha256\": ", stream);
  json_string(stream, root_digest);
  fputs(",\n  \"scope\": {\"case\":\"q16384-first-attention\","
        "\"attention_only\":true,\"launch_abi_cleared\":true,"
        "\"pack_semantics_resolved\":false,"
        "\"dependency_ordering_resolved\":false,"
        "\"live_tensor_semantics_resolved\":false},\n"
        "  \"runtime\": {\"cpu_agent\":", stream);
  json_string(stream, runtime->cpu_name);
  fputs(",\"gpu_agent\":", stream);
  json_string(stream, runtime->gpu_name);
  fputs(",\"isa\":", stream);
  json_string(stream, runtime->isa_name);
  (void)fprintf(stream,
                ",\"visible_gpu_count\":%" PRIu32
                ",\"hsa_runtime_sha256\":\"%s\","
                "\"hip_runtime_sha256\":\"%s\","
                "\"rocprofiler_register_sha256\":\"%s\"},\n",
                runtime->visible_gpu_count, HSA_RUNTIME_SHA256,
                HIP_RUNTIME_SHA256, ROCPROFILER_REGISTER_SHA256);
  fputs("  \"queue_contract\": ", stream);
  write_queue_contract(stream, queue);
  fputs(",\n", stream);
  fputs("  \"geometry\": {\"position_base\":384,\"query_count\":16384,"
        "\"key_end\":16768,\"query_heads\":12,\"kv_heads\":2,"
        "\"head_dimension\":128,\"metadata_i32\":[0,16768,1,0,0,16384],"
        "\"raw_grid_workitems\":[3072,1,128],"
        "\"hip_grid_blocks\":[12,1,128],\"workgroup\":[256,1,1],"
        "\"group_segment_bytes\":26112,\"private_segment_bytes\":0,"
        "\"kernarg_segment_bytes\":424,\"hip_explicit_argument_bytes\":168},\n",
        stream);
  (void)fprintf(stream,
                "  \"dispatch\": {\"raw_hsa_first\":true,"
                "\"q512_code_object_loaded\":false,"
                "\"q512_dispatches\":0,\"warmup_dispatches\":0,"
                "\"raw_completion_waits\":1,\"hip_device_synchronizes\":1,"
                "\"raw_packet_id\":%" PRIu64
                ",\"raw_kernarg_slot\":%" PRIu32
                ",\"raw_doorbell_written\":%" PRIu32
                ",\"raw_full_header\":\"0x%08" PRIx32
                "\",\"raw_started_ns\":%" PRIu64
                ",\"raw_completed_ns\":%" PRIu64
                ",\"hip_started_ns\":%" PRIu64
                ",\"hip_completed_ns\":%" PRIu64 "},\n",
                result->raw_dispatch.packet_id,
                result->raw_dispatch.kernarg_slot,
                result->raw_dispatch.doorbell_written,
                result->raw_packet.full_header, result->raw_started_ns,
                result->raw_completed_ns, result->hip_started_ns,
                result->hip_completed_ns);
  fputs("  \"inputs\": {\n    \"pre_raw\":", stream);
  write_input_hashes(stream, &result->pre_inputs, 0u);
  fputs(",\n    \"post_raw\":", stream);
  write_input_hashes(stream, &result->post_raw_inputs, 0u);
  fputs(",\n    \"post_hip\":", stream);
  write_input_hashes(stream, &result->post_hip_inputs, 0u);
  (void)fprintf(stream,
                ",\n    \"unchanged_after_raw\":%s,"
                "\"unchanged_after_hip\":%s,"
                "\"guard_mismatches_after_raw\":%" PRIu64
                ",\"guard_mismatches_after_hip\":%" PRIu64 "\n  },\n",
                result->inputs_unchanged_after_raw ? "true" : "false",
                result->inputs_unchanged_after_hip ? "true" : "false",
                result->input_guard_mismatches_after_raw,
                result->input_guard_mismatches_after_hip);
  (void)fprintf(stream,
                "  \"outputs\": {\n"
                "    \"hip_before_launch\":{"
                "\"expected_poison_values\":%" PRIu64
                ",\"poison_values_present\":%" PRIu64
                ",\"guard_mismatches\":%" PRIu64 "},\n"
                "    \"raw_hsa\":",
                (uint64_t)kQueryCount * kQueryHeads * kHeadDimension,
                result->hip_poison_values_after_raw,
                result->hip_guard_mismatches_after_raw);
  write_output_observation(stream, &result->raw_output, 0u);
  fputs(",\n    \"raw_hsa_after_hip\":", stream);
  write_output_observation(stream, &result->raw_output_after_hip, 0u);
  fputs(",\n    \"hip\":", stream);
  write_output_observation(stream, &result->hip_output, 0u);
  (void)fprintf(stream, ",\n    \"raw_hsa_unchanged_after_hip\":%s\n  },\n",
                result->raw_output_unchanged_after_hip ? "true" : "false");
  (void)fprintf(stream,
                "  \"comparison\": {\"byte_equal\":%s,"
                "\"bytes_compared\":%zu,\"mismatch_bytes\":%" PRIu64,
                result->byte_equal ? "true" : "false",
                (size_t)((uint64_t)kQueryCount * kQueryHeads *
                         kHeadDimension * sizeof(uint16_t)),
                result->mismatch_bytes);
  if (result->byte_equal) {
    fputs(",\"first_mismatch_byte\":null,"
          "\"first_mismatch_element\":null,"
          "\"first_mismatch_row\":null,"
          "\"first_mismatch_head\":null,"
          "\"first_mismatch_dimension\":null}\n}\n",
          stream);
  } else {
    (void)fprintf(stream,
                  ",\"first_mismatch_byte\":%" PRIu64
                  ",\"first_mismatch_element\":%" PRIu64
                  ",\"first_mismatch_row\":%" PRIu32
                  ",\"first_mismatch_head\":%" PRIu32
                  ",\"first_mismatch_dimension\":%" PRIu32 "}\n}\n",
                  result->first_mismatch_byte,
                  result->first_mismatch_element,
                  result->first_mismatch_row,
                  result->first_mismatch_head,
                  result->first_mismatch_dimension);
  }
  return commit_reserved_result(stream, path, error);
}

static bool write_failure_result(int descriptor, const char* path,
                                 const char* root_digest,
                                 const char* authorization_id,
                                 bool gpu_execution_performed,
                                 const queue_contract_observation_t* queue,
                                 const char* message,
                                 error_state_t* error) {
  FILE* stream = NULL;
  if (!begin_reserved_result(descriptor, path, &stream, error)) return false;
  fputs("{\n  \"schema\": \"" MICROGATE_SCHEMA "\",\n"
        "  \"status\": \"fail\",\n"
        "  \"gpu_execution_performed\": ", stream);
  fputs(gpu_execution_performed ? "true" : "false", stream);
  fputs(",\n  \"authorization_id\": ", stream);
  json_string(stream, authorization_id);
  fputs(",\n  \"microgate_root_sha256\": ", stream);
  json_string(stream, root_digest);
  fputs(",\n  \"scope\": {\"attention_only\":true,"
        "\"launch_abi_cleared\":false,"
        "\"pack_semantics_resolved\":false,"
        "\"dependency_ordering_resolved\":false,"
        "\"live_tensor_semantics_resolved\":false},\n"
        "  \"queue_contract\": ", stream);
  write_queue_contract(stream, queue);
  fputs(",\n  \"error\": ", stream);
  json_string(stream, message);
  fputs("\n}\n", stream);
  return commit_reserved_result(stream, path, error);
}

static bool self_check_contract(error_state_t* error) {
  q16k_aiter_attention_metadata_t metadata;
  if (q16k_build_attention_metadata(kPositionBase, kQueryCount, &metadata) !=
          Q16K_AITER_STATUS_OK ||
      metadata.kv_indptr[0] != 0 || metadata.kv_indptr[1] != kKeyEnd ||
      metadata.kv_last_page_lens[0] != 1 || metadata.reserved_12 != 0 ||
      metadata.cu_seqlens_q[0] != 0 ||
      metadata.cu_seqlens_q[1] != kQueryCount) {
    return fail(error, "first-attention metadata contract changed");
  }
  _Alignas(Q16K_AITER_KERNARG_ALIGNMENT)
      uint8_t kernarg[Q16K_AITER_ATTENTION_KERNARG_SIZE];
  q16k_aiter_launch_geometry_t geometry;
  const q16k_aiter_attention_request_t request = {
      .q = (const void*)(uintptr_t)0x1000u,
      .k = (const void*)(uintptr_t)0x2000u,
      .v = (const void*)(uintptr_t)0x3000u,
      .output = (void*)(uintptr_t)0x4000u,
      .kv_indptr = (const int32_t*)(uintptr_t)0x5000u,
      .kv_page_indices = (const int32_t*)(uintptr_t)0x6000u,
      .kv_last_page_lens = (const int32_t*)(uintptr_t)0x7000u,
      .cu_seqlens_q = (const int32_t*)(uintptr_t)0x8000u,
      .position_base = kPositionBase,
      .logical_query_count = kQueryCount,
  };
  if (q16k_build_attention_kernarg(&request, kernarg, sizeof(kernarg),
                                   &geometry) != Q16K_AITER_STATUS_OK ||
      geometry.grid_size[0] != 3072u || geometry.grid_size[1] != 1u ||
      geometry.grid_size[2] != 128u ||
      geometry.workgroup_size[0] != 256u ||
      geometry.workgroup_size[1] != 1u ||
      geometry.workgroup_size[2] != 1u) {
    return fail(error, "canonical attention kernarg construction changed");
  }
  int32_t key_end = 0;
  uint32_t hidden_blocks_x = 0u;
  uint32_t hidden_blocks_z = 0u;
  uint16_t hidden_group_x = 0u;
  uint16_t hidden_dimensions = 0u;
  memcpy(&key_end, kernarg + 64u, sizeof(key_end));
  memcpy(&hidden_blocks_x, kernarg + 168u, sizeof(hidden_blocks_x));
  memcpy(&hidden_blocks_z, kernarg + 176u, sizeof(hidden_blocks_z));
  memcpy(&hidden_group_x, kernarg + 180u, sizeof(hidden_group_x));
  memcpy(&hidden_dimensions, kernarg + 232u,
         sizeof(hidden_dimensions));
  if (key_end != kKeyEnd || hidden_blocks_x != 12u ||
      hidden_blocks_z != 128u || hidden_group_x != 256u ||
      hidden_dimensions != 3u) {
    return fail(error, "canonical attention hidden launch fields changed");
  }
  microgate_reference_t reference;
  return microgate_build_reference(kQueryCount, &reference) &&
                 reference.position_base == kPositionBase &&
                 reference.query_count == kQueryCount &&
                 reference.rows[MICROGATE_SAMPLE_COUNT - 1u] ==
                     kQueryCount - 1u &&
                 reference.zero_control_violations != 0u &&
                 reference.full_noncausal_control_violations != 0u &&
                 reference.top_left_control_violations != 0u
             ? true
             : fail(error, "first-attention CPU reference changed");
}

static int run_live(const char* root, const char* result_path,
                    const char* receipt_path,
                    const char* authorization_path) {
  error_state_t error = {{0}};
  char root_digest[65] = {0};
  consumption_receipt_t receipt;
  struct stat receipt_status = {0};
  if (!verify_static_inputs(root, &error) ||
      !verify_runtime_inputs(&error) || !self_check_contract(&error) ||
      !verify_consumption_receipt(root, receipt_path, authorization_path,
                                  result_path, root_digest, &receipt,
                                  &receipt_status, &error)) {
    fprintf(stderr, "pre-HSA validation failed: %s\n", error.message);
    return 2;
  }

  int result_descriptor = -1;
  if (!reserve_result_path(result_path, root_digest,
                           receipt.authorization_id, &result_descriptor,
                           &error)) {
    fprintf(stderr, "pre-HSA result reservation failed: %s\n",
            error.message);
    return 2;
  }
  if (!claim_consumption_receipt(receipt_path, &receipt_status, &error)) {
    error_state_t publication_error = {{0}};
    if (!write_failure_result(result_descriptor, result_path, root_digest,
                              receipt.authorization_id, false,
                              NULL, error.message, &publication_error)) {
      fprintf(stderr, "result publication failed: %s\n",
              publication_error.message);
      return 3;
    }
    fprintf(stderr, "pre-HSA receipt claim failed: %s\n", error.message);
    return 2;
  }

  run_resources_t resources;
  memset(&resources, 0, sizeof(resources));
  run_result_t result;
  memset(&result, 0, sizeof(result));
  bool gpu_execution_performed = false;
  bool success = initialize_runtime(&resources.runtime, &error);
  char hsaco_path[PATH_MAX];
  if (success) {
    success = join_path(root, "artifacts/aiter_prefill_q16384_gfx950.hsaco",
                        hsaco_path, &error);
  }
  if (success) {
    success = execute_launch_parity(&resources, hsaco_path, &result,
                                    &gpu_execution_performed, &error);
  }
  release_resources(&resources);
  if (resources.runtime.initialized) {
    const hsa_status_t shutdown_status = hsa_shut_down();
    resources.runtime.initialized = false;
    if (shutdown_status != HSA_STATUS_SUCCESS && success) {
      success = fail(&error, "hsa_shut_down failed: %s (%d)",
                     hsa_status_text(shutdown_status),
                     (int)shutdown_status);
    }
  }

  error_state_t publication_error = {{0}};
  const queue_contract_observation_t* queue_contract =
      resources.queue_contract.agent_min_query_completed
          ? &resources.queue_contract
          : NULL;
  const bool published =
      success
          ? write_success_result(result_descriptor, result_path, root_digest,
                                 receipt.authorization_id, &resources.runtime,
                                 &resources.queue_contract, &result,
                                 &publication_error)
          : write_failure_result(result_descriptor, result_path, root_digest,
                                 receipt.authorization_id,
                                 gpu_execution_performed,
                                 queue_contract, error.message,
                                 &publication_error);
  if (!published) {
    fprintf(stderr, "result publication failed: %s\n",
            publication_error.message);
    return 3;
  }
  if (!success) {
    fprintf(stderr, "q16384 launch-ABI microgate failed: %s\n",
            error.message);
    return 1;
  }
  puts("q16384 first-attention launch-ABI microgate: PASS");
  return 0;
}

static int run_self_check(const char* root) {
  error_state_t error = {{0}};
  if (!verify_static_inputs(root, &error) || !self_check_contract(&error)) {
    fprintf(stderr, "self-check failed: %s\n", error.message);
    return 1;
  }
  puts("{\"schema\":\"loom-q16384-first-attention-self-check-v1\","
       "\"status\":\"pass\",\"gpu_execution_performed\":false,"
       "\"hsa_initialized\":false,"
       "\"case\":\"q16384-first-attention\"}");
  return 0;
}

static void usage(const char* program) {
  fprintf(stderr,
          "usage: %s --self-check ROOT\n"
          "       %s --run ROOT RESULT_JSON CONSUMPTION_RECEIPT "
          "AUTHORIZATION_JSON\n",
          program, program);
}

#ifndef MICROGATE_NO_MAIN
int main(int argc, char** argv) {
  if (argc == 3 && strcmp(argv[1], "--self-check") == 0) {
    char root[PATH_MAX];
    if (realpath(argv[2], root) == NULL) {
      fprintf(stderr, "cannot resolve root %s: %s\n", argv[2],
              strerror(errno));
      return 2;
    }
    return run_self_check(root);
  }
  if (argc == 6 && strcmp(argv[1], "--run") == 0) {
    if (argv[2][0] != '/' || argv[3][0] != '/' || argv[4][0] != '/' ||
        argv[5][0] != '/') {
      fputs("live-run paths must be absolute\n", stderr);
      return 2;
    }
    char root[PATH_MAX];
    if (realpath(argv[2], root) == NULL) {
      fprintf(stderr, "cannot resolve root %s: %s\n", argv[2],
              strerror(errno));
      return 2;
    }
    return run_live(root, argv[3], argv[4], argv[5]);
  }
  usage(argv[0]);
  return 2;
}
#endif
