#define main q16384_attention_microgate_program_main
#include "../src/q16384_attention_microgate.c"
#undef main

#include <stdio.h>

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

static int create_private_file(const char* path, mode_t mode,
                               const char* contents) {
  const int descriptor =
      open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode);
  if (descriptor < 0) return -1;
  const size_t length = strlen(contents);
  const bool ok = fchmod(descriptor, mode) == 0 &&
                  write(descriptor, contents, length) == (ssize_t)length &&
                  close(descriptor) == 0;
  return ok ? 0 : -1;
}

static void initialize_passing_result(run_result_t* result) {
  memset(result, 0, sizeof(*result));
  result->byte_equal = true;
  result->hip_poison_values_after_raw =
      (uint64_t)kQueryCount * kQueryHeads * kHeadDimension;
  result->raw_output.sampled_elements_compared =
      (uint64_t)MICROGATE_SAMPLE_COUNT * kQueryHeads * kHeadDimension;
  result->raw_output_after_hip.sampled_elements_compared =
      result->raw_output.sampled_elements_compared;
  result->hip_output.sampled_elements_compared =
      result->raw_output.sampled_elements_compared;
  result->inputs_unchanged_after_raw = true;
  result->inputs_unchanged_after_hip = true;
  result->raw_output_unchanged_after_hip = true;
  result->raw_packet.full_header = kExpectedFullHeader;
}

static int test_result_isolation_gates(void) {
  run_result_t result;
  initialize_passing_result(&result);
  CHECK(result_passes(&result));

  --result.hip_poison_values_after_raw;
  CHECK(!result_passes(&result));
  initialize_passing_result(&result);
  result.hip_guard_mismatches_after_raw = 1u;
  CHECK(!result_passes(&result));
  initialize_passing_result(&result);
  result.raw_output_unchanged_after_hip = false;
  CHECK(!result_passes(&result));
  initialize_passing_result(&result);
  result.raw_output_after_hip.guard_mismatches = 1u;
  CHECK(!result_passes(&result));
  return 0;
}

static int test_explicit_argument_boundary(void) {
  uint8_t raw[Q16K_AITER_ATTENTION_KERNARG_SIZE] = {0};
  uint8_t hip[Q16K_AITER_ATTENTION_KERNARG_SIZE] = {0};
  CHECK(compare_explicit_kernargs(raw, hip));
  hip[24] = 1u;
  CHECK(compare_explicit_kernargs(raw, hip));
  hip[24] = 0u;
  hip[167] = 1u;
  CHECK(!compare_explicit_kernargs(raw, hip));
  hip[167] = 0u;
  hip[168] = 1u;
  CHECK(compare_explicit_kernargs(raw, hip));
  return 0;
}

static void initialize_valid_queue_contract(
    queue_contract_observation_t* observation) {
  memset(observation, 0, sizeof(*observation));
  observation->agent_min_query_completed = true;
  observation->agent_max_query_completed = true;
  observation->agent_type_query_completed = true;
  observation->agent_min_query_status = HSA_STATUS_SUCCESS;
  observation->agent_max_query_status = HSA_STATUS_SUCCESS;
  observation->agent_type_query_status = HSA_STATUS_SUCCESS;
  observation->agent_min_size = 64u;
  observation->agent_max_size = 131072u;
  observation->agent_type = HSA_QUEUE_TYPE_MULTI;
  observation->requested_size = 64u;
  observation->requested_type = HSA_QUEUE_TYPE_SINGLE;
  observation->expected_size = 64u;
  observation->create_attempted = true;
  observation->create_status = HSA_STATUS_SUCCESS;
  observation->returned_pointer = UINT64_C(0x1000);
  observation->returned_base_address = UINT64_C(0x2000);
  observation->returned_doorbell_signal_handle = UINT64_C(0x3000);
  observation->returned_id = 7u;
  observation->returned_size = 64u;
  observation->returned_type = HSA_QUEUE_TYPE_MULTI;
  observation->returned_features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
}

static int test_queue_contract_accepts_ordinary_runtime_results(void) {
  queue_contract_observation_t observation;
  error_state_t error = {{0}};
  initialize_valid_queue_contract(&observation);
  CHECK(validate_returned_queue_contract(&observation, &error));
  CHECK(observation.contract_satisfied);
  CHECK(error.message[0] == '\0');

  initialize_valid_queue_contract(&observation);
  observation.returned_type = HSA_QUEUE_TYPE_SINGLE;
  observation.returned_features =
      HSA_QUEUE_FEATURE_KERNEL_DISPATCH | UINT32_C(0x80000000);
  memset(&error, 0, sizeof(error));
  CHECK(validate_returned_queue_contract(&observation, &error));
  CHECK(observation.contract_satisfied);

  initialize_valid_queue_contract(&observation);
  observation.agent_min_size = 128u;
  observation.expected_size = 128u;
  observation.returned_size = 128u;
  memset(&error, 0, sizeof(error));
  CHECK(validate_returned_queue_contract(&observation, &error));
  CHECK(observation.contract_satisfied);
  return 0;
}

static int test_queue_contract_rejects_bad_fields(void) {
  queue_contract_observation_t observation;
  error_state_t error = {{0}};

  initialize_valid_queue_contract(&observation);
  observation.returned_pointer = 0u;
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "null queue pointer") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_base_address = 0u;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "null queue base address") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_size = 96u;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "non-power-of-two queue size") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_size = 32u;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "outside agent bounds") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_size = 128u;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "expected clamped size") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_features = 0u;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "without KERNEL_DISPATCH") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_type = HSA_QUEUE_TYPE_COOPERATIVE;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "cooperative queue type") != NULL);

  initialize_valid_queue_contract(&observation);
  observation.returned_type = UINT32_MAX;
  memset(&error, 0, sizeof(error));
  CHECK(!validate_returned_queue_contract(&observation, &error));
  CHECK(strstr(error.message, "unknown queue type") != NULL);
  return 0;
}

int main(void) {
  CHECK(test_result_isolation_gates() == 0);
  CHECK(test_explicit_argument_boundary() == 0);
  CHECK(test_queue_contract_accepts_ordinary_runtime_results() == 0);
  CHECK(test_queue_contract_rejects_bad_fields() == 0);

  uint64_t parsed = 0u;
  CHECK(parse_u64_decimal("0", &parsed) && parsed == 0u);
  CHECK(parse_u64_decimal("1788195600", &parsed) &&
        parsed == UINT64_C(1788195600));
  CHECK(!parse_u64_decimal("01", &parsed));
  CHECK(!parse_u64_decimal("+1", &parsed));

  char directory[] = "/tmp/loom-q16384-interlocks-XXXXXX";
  CHECK(mkdtemp(directory) != NULL);
  CHECK(chmod(directory, 0700) == 0);
  char authorization[PATH_MAX];
  char receipt[PATH_MAX];
  char consumed[PATH_MAX];
  char result[PATH_MAX];
  CHECK(snprintf(authorization, sizeof(authorization), "%s/authorization.json",
                 directory) > 0);
  CHECK(snprintf(receipt, sizeof(receipt), "%s/receipt", directory) > 0);
  CHECK(snprintf(consumed, sizeof(consumed), "%s.consumed", receipt) > 0);
  CHECK(snprintf(result, sizeof(result), "%s/result.json", directory) > 0);

  error_state_t error = {{0}};
  char claimed[PATH_MAX];
  CHECK(verify_live_path_contract("/sealed/package", result, receipt,
                                  authorization, claimed, &error));
  CHECK(strcmp(claimed, consumed) == 0);
  memset(&error, 0, sizeof(error));
  CHECK(!verify_live_path_contract("/sealed/package", authorization, receipt,
                                   authorization, claimed, &error));
  CHECK(strstr(error.message, "collide") != NULL);

  consumption_receipt_t grant;
  memset(&grant, 0, sizeof(grant));
  snprintf(grant.authorization_id, sizeof(grant.authorization_id), "%s",
           "q16384-review-20260831-001");
  snprintf(grant.authorization_path, sizeof(grant.authorization_path), "%s",
           authorization);
  snprintf(grant.consumption_receipt_path,
           sizeof(grant.consumption_receipt_path), "%s", receipt);
  snprintf(grant.result_path, sizeof(grant.result_path), "%s", result);
  snprintf(grant.microgate_root_sha256,
           sizeof(grant.microgate_root_sha256), "%s",
           "7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f"
           "7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f");
  snprintf(grant.reservation_id, sizeof(grant.reservation_id), "%s",
           "06a94f67-275d-78f6-8000-3680fb507fd5");
  snprintf(grant.host, sizeof(grant.host), "%s", "asrock-gbs6a-wb13c");
  snprintf(grant.target, sizeof(grant.target), "%s",
           "asrock-gbs6a-wb13c.png-odc.dcgpu");
  grant.authorization_issued_epoch = UINT64_C(1788192000);
  grant.authorization_expires_epoch = UINT64_C(1788206400);

  char canonical[2 * PATH_MAX + 2048];
  const int canonical_length = snprintf(
      canonical, sizeof(canonical),
      "{\"authorization_id\":\"q16384-review-20260831-001\","
      "\"authorized_case_count\":1,"
      "\"authorized_cases\":[\"q16384-first-attention\"],"
      "\"consumption_receipt_path\":\"%s\","
      "\"decision\":\"AUTHORIZE_ONE_GPU_INVOCATION\","
      "\"expires_utc\":\"2026-08-31T20:00:00Z\","
      "\"host\":\"asrock-gbs6a-wb13c\",\"invocation_count\":1,"
      "\"issued_utc\":\"2026-08-31T16:00:00Z\","
      "\"maximum_gpu_invocations\":1,\"microgate_root_sha256\":\"%s\","
      "\"pack_and_attention_contract_satisfied\":false,"
      "\"reservation_id\":\"06a94f67-275d-78f6-8000-3680fb507fd5\","
      "\"result_path\":\"%s\","
      "\"schema\":\"loom-q16384-first-attention-launch-abi-one-run-authorization-v1\","
      "\"scope\":\"q16384-first-attention-raw-hsa-vs-hip\","
      "\"target\":\"asrock-gbs6a-wb13c.png-odc.dcgpu\"}\n",
      receipt, grant.microgate_root_sha256, result);
  CHECK(canonical_length > 0 && (size_t)canonical_length < sizeof(canonical));
  CHECK(create_private_file(authorization, 0600, canonical) == 0);
  CHECK(sha256_file(authorization, grant.authorization_sha256) == 0);
  memset(&error, 0, sizeof(error));
  CHECK(verify_authorization_file(authorization, &grant, &error));

  int result_descriptor = -1;
  memset(&error, 0, sizeof(error));
  CHECK(reserve_result_path(result, grant.microgate_root_sha256,
                            grant.authorization_id, &result_descriptor, &error));
  struct stat result_status;
  CHECK(fstat(result_descriptor, &result_status) == 0);
  CHECK(S_ISREG(result_status.st_mode) && result_status.st_nlink == 1);
  CHECK((result_status.st_mode & 0777) == 0600);
  CHECK(close(result_descriptor) == 0);

  CHECK(create_private_file(receipt, 0400, "receipt\n") == 0);
  const int receipt_descriptor = open(receipt, O_RDONLY | O_CLOEXEC);
  CHECK(receipt_descriptor >= 0);
  struct stat receipt_status;
  CHECK(fstat(receipt_descriptor, &receipt_status) == 0);
  CHECK(close(receipt_descriptor) == 0);
  CHECK(create_private_file(consumed, 0400, "collision\n") == 0);
  memset(&error, 0, sizeof(error));
  CHECK(!claim_consumption_receipt(receipt, &receipt_status, &error));
  CHECK(unlink(consumed) == 0);
  memset(&error, 0, sizeof(error));
  CHECK(claim_consumption_receipt(receipt, &receipt_status, &error));
  CHECK(access(receipt, F_OK) != 0 && errno == ENOENT);
  CHECK(access(consumed, F_OK) == 0);

  CHECK(unlink(consumed) == 0);
  CHECK(unlink(result) == 0);
  CHECK(unlink(authorization) == 0);
  CHECK(rmdir(directory) == 0);
  puts("interlock tests: PASS (no HSA initialization)");
  return 0;
}
