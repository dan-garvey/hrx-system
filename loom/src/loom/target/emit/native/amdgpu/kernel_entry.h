// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned AMDGPU hardware kernel-entry envelopes.
//
// Scheduled target-low instructions describe the semantic kernel body. Some
// processors additionally require native instructions at the hardware entry
// point. This layer owns those instructions, their resource floors, and the
// displacement of body-relative code-object fixups.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_kernel_entry_envelope_t {
  // Assembly instructions inserted immediately after the entry label.
  iree_string_view_t assembly;
  // Encoded instructions inserted immediately before the kernel body.
  iree_const_byte_span_t text;
  // Number of native instructions in |assembly| and |text|.
  uint32_t instruction_count;
  // Minimum scalar register count required by the entry instructions.
  uint32_t minimum_sgpr_count;
  // Minimum vector register count required by the entry instructions.
  uint32_t minimum_vgpr_count;
} loom_amdgpu_kernel_entry_envelope_t;

enum {
  // Compatible CP firmware enters this many bytes after the descriptor entry
  // when kernarg preloading is enabled.
  LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES = 256,
  // Compatibility firmware loads up to thirteen 64-bit resource pointers in
  // naturally aligned register pairs.
  LOOM_AMDGPU_KERNARG_PRELOAD_COMPAT_LOAD_COUNT_MAX = 13,
};

// Returns true for an exact compatibility-entry preload shape. A zero-length
// preload is represented by the ordinary empty entry path, not by this helper.
bool loom_amdgpu_kernel_entry_supports_kernarg_preload(
    uint32_t kernarg_preload_dword_count,
    uint32_t kernarg_preload_dword_offset);

// Returns the number of NOPs required to fill the 256-byte firmware entry
// window for a supported preload count, or zero for an unsupported count.
uint32_t loom_amdgpu_kernel_entry_kernarg_preload_nop_count(
    uint32_t kernarg_preload_dword_count);

// Returns the number of native load/wait/NOP instructions in a supported
// preload entry, or zero when preloading is disabled or unsupported.
uint32_t loom_amdgpu_kernel_entry_kernarg_preload_instruction_count(
    uint32_t kernarg_preload_dword_count);

// Returns the immutable hardware-entry envelope selected by |properties|.
// Targets without an entry profile return an empty record.
const loom_amdgpu_kernel_entry_envelope_t*
loom_amdgpu_kernel_entry_envelope_for_properties(
    const loom_amdgpu_processor_properties_t* properties);

// Prepends |envelope| to |body_text| and displaces all body-relative fixups.
// Empty envelopes return the original body storage and fixup array directly.
iree_status_t loom_amdgpu_kernel_entry_prepend_text(
    const loom_amdgpu_kernel_entry_envelope_t* envelope,
    uint32_t kernarg_preload_dword_count, uint32_t kernarg_preload_dword_offset,
    iree_const_byte_span_t body_text,
    const loom_amdgpu_hsaco_text_fixup_t* body_fixups,
    iree_host_size_t body_fixup_count, iree_const_byte_span_t* out_text,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ENTRY_H_
