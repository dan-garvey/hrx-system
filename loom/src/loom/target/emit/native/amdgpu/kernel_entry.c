// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_entry.h"

#include <string.h>

#include "iree/base/alignment.h"

// This entry profile uses an unclaused VMEM followed by V_NOP. Replay mode
// establishes the multi-group XNACK behavior assumed by XCNT wait insertion
// before the scheduled body performs any VGPR-MSB transitions.
static const char loom_amdgpu_initial_vmem_replay_entry_assembly[] =
    "  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE\n"
    "  v_nop\n"
    "  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1\n";

static const uint8_t loom_amdgpu_initial_vmem_replay_entry_text[] = {
    0x00, 0x40, 0x17, 0xee, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x7e, 0x41, 0x06, 0x80, 0xb9, 0x01, 0x00, 0x00, 0x00,
};

static const loom_amdgpu_kernel_entry_envelope_t
    loom_amdgpu_initial_vmem_replay_entry_envelope = {
        .assembly =
            {
                .data = loom_amdgpu_initial_vmem_replay_entry_assembly,
                .size =
                    sizeof(loom_amdgpu_initial_vmem_replay_entry_assembly) - 1u,
            },
        .text =
            {
                .data = loom_amdgpu_initial_vmem_replay_entry_text,
                .data_length =
                    sizeof(loom_amdgpu_initial_vmem_replay_entry_text),
            },
        .instruction_count = 3u,
        .minimum_sgpr_count = 2u,
        .minimum_vgpr_count = 1u,
};

static const loom_amdgpu_kernel_entry_envelope_t
    loom_amdgpu_empty_entry_envelope = {0};

enum {
  LOOM_AMDGPU_KERNARG_PRELOAD_LOAD_BYTE_COUNT = 8u,
  LOOM_AMDGPU_KERNARG_PRELOAD_WAIT_BYTE_COUNT = 4u,
  LOOM_AMDGPU_KERNARG_PRELOAD_INSTRUCTION_BYTE_COUNT = 4u,
};

static const uint32_t kLoomAmdgpuKernargPreloadLoadDword0 =
    UINT32_C(0xC0060080);
static const uint32_t kLoomAmdgpuKernargPreloadWait = UINT32_C(0xBF8CC07F);
static const uint32_t kLoomAmdgpuSNop0 = UINT32_C(0xBF800000);

bool loom_amdgpu_kernel_entry_supports_kernarg_preload(
    uint32_t kernarg_preload_dword_count,
    uint32_t kernarg_preload_dword_offset) {
  return kernarg_preload_dword_offset == 0u &&
         (kernarg_preload_dword_count == 8u ||
          kernarg_preload_dword_count == 10u ||
          kernarg_preload_dword_count == 14u ||
          kernarg_preload_dword_count == 26u);
}

uint32_t loom_amdgpu_kernel_entry_kernarg_preload_nop_count(
    uint32_t kernarg_preload_dword_count) {
  if (!loom_amdgpu_kernel_entry_supports_kernarg_preload(
          kernarg_preload_dword_count, 0u)) {
    return 0u;
  }
  const uint32_t load_count = kernarg_preload_dword_count / 2u;
  const uint32_t used_byte_count =
      load_count * LOOM_AMDGPU_KERNARG_PRELOAD_LOAD_BYTE_COUNT +
      LOOM_AMDGPU_KERNARG_PRELOAD_WAIT_BYTE_COUNT;
  IREE_ASSERT_LE(used_byte_count, LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES);
  const uint32_t remaining_byte_count =
      LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES - used_byte_count;
  IREE_ASSERT_EQ(
      remaining_byte_count % LOOM_AMDGPU_KERNARG_PRELOAD_INSTRUCTION_BYTE_COUNT,
      0u);
  return remaining_byte_count /
         LOOM_AMDGPU_KERNARG_PRELOAD_INSTRUCTION_BYTE_COUNT;
}

uint32_t loom_amdgpu_kernel_entry_kernarg_preload_instruction_count(
    uint32_t kernarg_preload_dword_count) {
  if (!loom_amdgpu_kernel_entry_supports_kernarg_preload(
          kernarg_preload_dword_count, 0u)) {
    return 0u;
  }
  return kernarg_preload_dword_count / 2u + 1u +
         loom_amdgpu_kernel_entry_kernarg_preload_nop_count(
             kernarg_preload_dword_count);
}

const loom_amdgpu_kernel_entry_envelope_t*
loom_amdgpu_kernel_entry_envelope_for_properties(
    const loom_amdgpu_processor_properties_t* properties) {
  switch (properties->kernel_entry.profile) {
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE:
      return &loom_amdgpu_empty_entry_envelope;
    case LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY:
      return &loom_amdgpu_initial_vmem_replay_entry_envelope;
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU kernel entry profile");
      return &loom_amdgpu_empty_entry_envelope;
  }
}

iree_status_t loom_amdgpu_kernel_entry_prepend_text(
    const loom_amdgpu_kernel_entry_envelope_t* envelope,
    uint32_t kernarg_preload_dword_count, uint32_t kernarg_preload_dword_offset,
    iree_const_byte_span_t body_text,
    const loom_amdgpu_hsaco_text_fixup_t* body_fixups,
    iree_host_size_t body_fixup_count, iree_const_byte_span_t* out_text,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups,
    iree_arena_allocator_t* arena) {
  *out_text = iree_const_byte_span_empty();
  *out_fixups = NULL;
  if (envelope == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel entry envelope is required");
  }
  if (body_text.data_length != 0 && body_text.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel body has no text storage");
  }
  if (body_fixup_count != 0 && body_fixups == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel body has no text fixup storage");
  }
  if (envelope->text.data_length != 0 && envelope->text.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel entry has no text storage");
  }
  if (kernarg_preload_dword_count != 0 &&
      !loom_amdgpu_kernel_entry_supports_kernarg_preload(
          kernarg_preload_dword_count, kernarg_preload_dword_offset)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel entry supports only an eight-, ten-, fourteen-, or "
        "twenty-six-dword preload at offset zero");
  }
  if (kernarg_preload_dword_count == 0 && kernarg_preload_dword_offset != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel preload offset requires a nonzero length");
  }
  if (kernarg_preload_dword_count != 0 && envelope->text.data_length != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernarg preload is incompatible with a target entry envelope");
  }
  const iree_host_size_t prefix_length =
      kernarg_preload_dword_count != 0
          ? LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES
          : envelope->text.data_length;
  if (prefix_length == 0) {
    *out_text = body_text;
    *out_fixups = body_fixups;
    return iree_ok_status();
  }

  iree_host_size_t text_length = 0;
  if (!iree_host_size_checked_add(prefix_length, body_text.data_length,
                                  &text_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU kernel entry text length overflowed");
  }
  uint8_t* text = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, text_length, (void**)&text));
  if (kernarg_preload_dword_count != 0) {
    const uint32_t load_count = kernarg_preload_dword_count / 2u;
    IREE_ASSERT_LE(load_count,
                   LOOM_AMDGPU_KERNARG_PRELOAD_COMPAT_LOAD_COUNT_MAX);
    iree_host_size_t offset = 0;
    for (uint32_t i = 0; i < load_count; ++i) {
      iree_unaligned_store_le_u32(
          text + offset,
          kLoomAmdgpuKernargPreloadLoadDword0 + i * UINT32_C(0x80));
      iree_unaligned_store_le_u32(text + offset + sizeof(uint32_t), i * 8u);
      offset += LOOM_AMDGPU_KERNARG_PRELOAD_LOAD_BYTE_COUNT;
    }
    iree_unaligned_store_le_u32(text + offset, kLoomAmdgpuKernargPreloadWait);
    offset += LOOM_AMDGPU_KERNARG_PRELOAD_WAIT_BYTE_COUNT;
    const uint32_t nop_count =
        loom_amdgpu_kernel_entry_kernarg_preload_nop_count(
            kernarg_preload_dword_count);
    for (uint32_t i = 0; i < nop_count; ++i) {
      iree_unaligned_store_le_u32(text + offset, kLoomAmdgpuSNop0);
      offset += LOOM_AMDGPU_KERNARG_PRELOAD_INSTRUCTION_BYTE_COUNT;
    }
    IREE_ASSERT_EQ(offset, prefix_length);
  } else {
    memcpy(text, envelope->text.data, envelope->text.data_length);
  }
  if (body_text.data_length != 0) {
    memcpy(text + prefix_length, body_text.data, body_text.data_length);
  }

  loom_amdgpu_hsaco_text_fixup_t* fixups = NULL;
  if (body_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, body_fixup_count, sizeof(fixups[0]), (void**)&fixups));
    for (iree_host_size_t i = 0; i < body_fixup_count; ++i) {
      fixups[i] = body_fixups[i];
      if (!iree_checked_add_u64(fixups[i].literal_byte_offset, prefix_length,
                                &fixups[i].literal_byte_offset) ||
          !iree_checked_add_u64(fixups[i].base_pc_byte_offset, prefix_length,
                                &fixups[i].base_pc_byte_offset)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU kernel entry text fixup offset overflowed");
      }
    }
  }

  *out_text = iree_make_const_byte_span(text, text_length);
  *out_fixups = fixups;
  return iree_ok_status();
}
