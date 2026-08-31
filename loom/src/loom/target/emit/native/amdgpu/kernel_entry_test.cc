// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_entry.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transformed entry text and fixups.
  iree_arena_allocator_t arena_ = {0};
};

uint32_t LoadLeU32(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

void ExpectKernargPreloadWindow(iree_const_byte_span_t text,
                                uint32_t dword_count) {
  ASSERT_GE(text.data_length, LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES);
  const uint32_t load_count = dword_count / 2u;
  iree_host_size_t offset = 0;
  for (uint32_t i = 0; i < load_count; ++i) {
    EXPECT_EQ(LoadLeU32(text.data + offset),
              UINT32_C(0xC0060080) + i * UINT32_C(0x80))
        << "load " << i;
    EXPECT_EQ(LoadLeU32(text.data + offset + 4u), i * 8u) << "load " << i;
    offset += 8u;
  }
  EXPECT_EQ(LoadLeU32(text.data + offset), UINT32_C(0xBF8CC07F));
  offset += 4u;
  const uint32_t nop_count =
      loom_amdgpu_kernel_entry_kernarg_preload_nop_count(dword_count);
  for (uint32_t i = 0; i < nop_count; ++i) {
    EXPECT_EQ(LoadLeU32(text.data + offset), UINT32_C(0xBF800000))
        << "nop " << i;
    offset += 4u;
  }
  EXPECT_EQ(offset, LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES);
}

TEST(AmdgpuKernelEntryTest, SelectsInitialVmemReplayEnvelope) {
  static const uint8_t kExpectedText[] = {
      0x00, 0x40, 0x17, 0xee, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x7e, 0x41, 0x06, 0x80, 0xb9, 0x01, 0x00, 0x00, 0x00,
  };
  static constexpr char kExpectedAssembly[] =
      "  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE\n"
      "  v_nop\n"
      "  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1\n";
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile =
      LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY;

  const loom_amdgpu_kernel_entry_envelope_t* envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties);
  EXPECT_EQ(std::string(envelope->assembly.data, envelope->assembly.size),
            kExpectedAssembly);
  ASSERT_EQ(envelope->text.data_length, sizeof(kExpectedText));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(envelope->text.data),
                        envelope->text.data_length),
            std::string(reinterpret_cast<const char*>(kExpectedText),
                        sizeof(kExpectedText)));
  EXPECT_EQ(envelope->instruction_count, 3u);
  EXPECT_EQ(envelope->minimum_sgpr_count, 2u);
  EXPECT_EQ(envelope->minimum_vgpr_count, 1u);
}

TEST(AmdgpuKernelEntryTest, PrependsTextAndDisplacesBodyFixups) {
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile =
      LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY;
  const loom_amdgpu_kernel_entry_envelope_t* envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties);
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd, 0x00, 0x00, 0x00, 0x00};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
      /*.literal_byte_offset=*/4,
      /*.base_pc_byte_offset=*/0,
      /*.target_symbol=*/IREE_SV("target_data"),
      /*.target_symbol_byte_offset=*/12,
  };

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      envelope, /*kernarg_preload_dword_count=*/0,
      /*kernarg_preload_dword_offset=*/0,
      iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1, &text,
      &fixups, arena.arena()));

  ASSERT_EQ(text.data_length, envelope->text.data_length + sizeof(body));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(text.data),
                        envelope->text.data_length),
            std::string(reinterpret_cast<const char*>(envelope->text.data),
                        envelope->text.data_length));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(
                            text.data + envelope->text.data_length),
                        sizeof(body)),
            std::string(reinterpret_cast<const char*>(body), sizeof(body)));
  ASSERT_NE(fixups, nullptr);
  EXPECT_EQ(fixups[0].literal_byte_offset, 28u);
  EXPECT_EQ(fixups[0].base_pc_byte_offset, 24u);
  EXPECT_TRUE(
      iree_string_view_equal(fixups[0].target_symbol, IREE_SV("target_data")));
  EXPECT_EQ(fixups[0].target_symbol_byte_offset, 12u);
}

TEST(AmdgpuKernelEntryTest,
     PrependsKernargPreloadCompatibilityWindowAndDisplacesFixups) {
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
      /*.literal_byte_offset=*/0,
      /*.base_pc_byte_offset=*/4,
      /*.target_symbol=*/IREE_SV("target_data"),
      /*.target_symbol_byte_offset=*/12,
  };
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties),
      /*kernarg_preload_dword_count=*/8,
      /*kernarg_preload_dword_offset=*/0,
      iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1, &text,
      &fixups, arena.arena()));

  ASSERT_EQ(text.data_length,
            LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES + sizeof(body));
  ExpectKernargPreloadWindow(text, 8u);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(
                      text.data + LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES),
                  sizeof(body)),
      std::string(reinterpret_cast<const char*>(body), sizeof(body)));
  ASSERT_NE(fixups, nullptr);
  EXPECT_EQ(fixups[0].literal_byte_offset, 256u);
  EXPECT_EQ(fixups[0].base_pc_byte_offset, 260u);
}

TEST(AmdgpuKernelEntryTest, PrependsTenDwordKernargPreloadCompatibilityWindow) {
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd};
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties),
      /*kernarg_preload_dword_count=*/10,
      /*kernarg_preload_dword_offset=*/0,
      iree_make_const_byte_span(body, sizeof(body)), nullptr, 0, &text, &fixups,
      arena.arena()));

  ASSERT_EQ(text.data_length,
            LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES + sizeof(body));
  ExpectKernargPreloadWindow(text, 10u);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(
                      text.data + LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES),
                  sizeof(body)),
      std::string(reinterpret_cast<const char*>(body), sizeof(body)));
  EXPECT_EQ(fixups, nullptr);
}

TEST(AmdgpuKernelEntryTest,
     PrependsTwentySixDwordKernargPreloadCompatibilityWindow) {
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd};
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties),
      /*kernarg_preload_dword_count=*/26,
      /*kernarg_preload_dword_offset=*/0,
      iree_make_const_byte_span(body, sizeof(body)), nullptr, 0, &text, &fixups,
      arena.arena()));

  ASSERT_EQ(text.data_length,
            LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES + sizeof(body));
  ExpectKernargPreloadWindow(text, 26u);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(
                      text.data + LOOM_AMDGPU_KERNARG_PRELOAD_ENTRY_SKIP_BYTES),
                  sizeof(body)),
      std::string(reinterpret_cast<const char*>(body), sizeof(body)));
  EXPECT_EQ(fixups, nullptr);
}

TEST(AmdgpuKernelEntryTest, ComputesCompatibilityWindowInstructionCounts) {
  EXPECT_TRUE(loom_amdgpu_kernel_entry_supports_kernarg_preload(8u, 0u));
  EXPECT_TRUE(loom_amdgpu_kernel_entry_supports_kernarg_preload(10u, 0u));
  EXPECT_TRUE(loom_amdgpu_kernel_entry_supports_kernarg_preload(14u, 0u));
  EXPECT_TRUE(loom_amdgpu_kernel_entry_supports_kernarg_preload(26u, 0u));
  EXPECT_FALSE(loom_amdgpu_kernel_entry_supports_kernarg_preload(30u, 0u));
  EXPECT_FALSE(loom_amdgpu_kernel_entry_supports_kernarg_preload(26u, 1u));
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_nop_count(8u), 55u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_nop_count(10u), 53u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_nop_count(14u), 49u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_nop_count(26u), 37u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_instruction_count(8u),
            60u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_instruction_count(10u),
            59u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_instruction_count(14u),
            57u);
  EXPECT_EQ(loom_amdgpu_kernel_entry_kernarg_preload_instruction_count(26u),
            51u);
}

TEST(AmdgpuKernelEntryTest, RejectsUnsupportedKernargPreloadCount) {
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd};
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_amdgpu_kernel_entry_prepend_text(
          loom_amdgpu_kernel_entry_envelope_for_properties(&properties),
          /*kernarg_preload_dword_count=*/30,
          /*kernarg_preload_dword_offset=*/0,
          iree_make_const_byte_span(body, sizeof(body)), nullptr, 0, &text,
          &fixups, arena.arena()));
}

TEST(AmdgpuKernelEntryTest, EmptyEnvelopePreservesBodyStorage) {
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;
  const loom_amdgpu_kernel_entry_envelope_t* envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties);
  const uint8_t body[] = {0x00, 0x00, 0xb0, 0xbf};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
  };

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      envelope, /*kernarg_preload_dword_count=*/0,
      /*kernarg_preload_dword_offset=*/0,
      iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1, &text,
      &fixups, arena.arena()));

  EXPECT_EQ(text.data, body);
  EXPECT_EQ(text.data_length, sizeof(body));
  EXPECT_EQ(fixups, &body_fixup);
}

}  // namespace
}  // namespace loom
