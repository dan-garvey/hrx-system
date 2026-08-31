#!/usr/bin/env python3
"""Validates one Loom q16K first-attention diagnostic record."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any


SCHEMA = "loom.q16k.first_attention.v1"
KERNARG_SIZE = 424
ATTENTION_KERNEL_NAME = (
    "_ZN7ck_tile6kentryILi2ENS_38FmhaBatchPrefillWithPagedKVCacheKernelINS_"
    "40BlockFmhaBatchPrefillPipelineQRKSVSAsyncINS_36BlockFmhaBatchPrefill"
    "PipelineProblemIDF16bDF16bDF16bffDF16bhfDF16bfDF16bNS_13TileFmhaShape"
    "INS_8sequenceIJLi128ELi128ELi32ELi128ELi32ELi128EEEENS5_IJLi4ELi1ELi1"
    "EEEENS5_IJLi32ELi32ELi16EEEES7_S8_Lb1EEELb1ENS_17ComposedAttentionILj0E"
    "Lb1EEENS_30SimplifiedGenericAttentionMaskILb1EEELb0ELi1ENS_26TileFmhaBat"
    "chPrefillTraitsILb1ELb1ELb1ELb1ELb0ELNS_22BlockAttentionBiasEnumE0ELb0"
    "ELb0ELb0ELNS_28BlockAttentionQuantScaleEnumE0ELin1ELb0ELb0ELi1ELNS_37B"
    "lockAttentionKVCacheMemoryLayoutEnumE1ELNS_36BlockAttentionKVCacheLookup"
    "TableEnumE1ELNS_33BlockAttentionKVCacheLoadModeEnumE0EEEEENS_53BlockFmha"
    "BatchPrefillPipelineQRKSVSAsyncDefaultPolicyEEENS_17Default2DEpilogueIN"
    "S_24Default2DEpilogueProblemIfDF16bLb1ELb1ELb1EEEvEEEEJNSS_21FmhaFwdGr"
    "oupModeKargsEEEEvDpT1_"
)
ATTENTION_KERNARG_ALIGNMENT = 16
ATTENTION_GROUP_SEGMENT_SIZE = 26112
SOFTMAX_SCALE_LOG2_BITS = 0x3E0293EE


def parse_hex(value: str) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"expected hexadecimal string, got {value!r}")
    return int(value, 16)


def unpack(fmt: str, data: bytes, offset: int) -> int | float:
    return struct.unpack_from("<" + fmt, data, offset)[0]


def analyze(record: dict[str, Any], expected_position: int,
            expected_query: int, expected_layer: int) -> list[str]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def exact_keys(value: Any, expected: set[str], description: str) -> None:
        require(isinstance(value, dict), f"{description} is not an object")
        if isinstance(value, dict):
            require(set(value) == expected,
                    f"{description} keys mismatch: {sorted(value)!r}")

    exact_keys(record, {"schema", "complete", "request", "capture", "ordering",
                        "metadata", "kernel", "geometry", "packet", "kernarg"},
               "record")
    require(record.get("schema") == SCHEMA, "schema mismatch")
    require(record.get("complete") is True, "capture is incomplete")

    request = record.get("request", {})
    exact_keys(request, {"id", "ordinal", "finished", "succeeded"}, "request")
    require(type(request.get("id")) is int and request.get("id") > 0,
            "request ID is not positive")
    require(request.get("ordinal") == 1, "request ordinal is not exactly one")
    require(request.get("finished") is True, "request did not finish")

    capture = record.get("capture", {})
    exact_keys(capture, {"position_base", "query_count", "layer"}, "capture")
    require(capture.get("position_base") == expected_position,
            "position_base mismatch")
    require(capture.get("query_count") == expected_query,
            "query_count mismatch")
    require(capture.get("layer") == expected_layer, "layer mismatch")

    ordering = record.get("ordering", {})
    exact_keys(ordering, {"host_metadata", "gpu_metadata_copy_complete",
                          "attention_packet_published", "request_finished"},
               "ordering")
    order = [
        ordering.get("host_metadata"),
        ordering.get("gpu_metadata_copy_complete"),
        ordering.get("attention_packet_published"),
        ordering.get("request_finished"),
    ]
    require(order == [1, 2, 3, 4],
            f"capture ordering mismatch: observed {order!r}")

    metadata = record.get("metadata", {})
    exact_keys(metadata, {"gpu_address", "host_i32", "gpu_i32",
                          "host_gpu_match"}, "metadata")
    key_end = expected_position + expected_query
    expected_metadata = [0, key_end, 1, 0, 0, expected_query]
    require(metadata.get("host_i32") == expected_metadata,
            "host metadata values mismatch")
    require(metadata.get("gpu_i32") == expected_metadata,
            "GPU copy-back metadata values mismatch")
    require(metadata.get("host_gpu_match") is True,
            "host and copied-back GPU metadata differ")

    geometry = record.get("geometry", {})
    exact_keys(geometry, {"grid", "workgroup"}, "geometry")
    require(geometry.get("grid") == [3072, 1, 128],
            "attention launch grid is not [3072, 1, 128]")
    require(geometry.get("workgroup") == [256, 1, 1],
            "attention workgroup is not [256, 1, 1]")

    kernel = record.get("kernel", {})
    packet = record.get("packet", {})
    exact_keys(kernel, {"name", "object", "kernarg_segment_size",
                        "kernarg_segment_alignment", "group_segment_size",
                        "private_segment_size"}, "kernel")
    exact_keys(packet, {"full_header", "dimensions", "workgroup", "grid",
                        "private_segment_size", "group_segment_size",
                        "kernel_object", "kernarg_address", "completion_signal",
                        "packet_id", "kernarg_slot", "doorbell_written"},
               "packet")
    kernel_name = kernel.get("name")
    require(isinstance(kernel_name, str), "kernel name is not a string")
    if isinstance(kernel_name, str):
        require(kernel_name == ATTENTION_KERNEL_NAME,
                "attention kernel symbol mismatch")
    try:
        kernel_object = parse_hex(kernel.get("object"))
        packet_kernel_object = parse_hex(packet.get("kernel_object"))
        full_header = parse_hex(packet.get("full_header"))
        completion_signal = parse_hex(packet.get("completion_signal"))
        packet_kernarg_address = parse_hex(packet.get("kernarg_address"))
    except (TypeError, ValueError) as exc:
        failures.append(str(exc))
        kernel_object = packet_kernel_object = full_header = completion_signal = 0
        packet_kernarg_address = 0
    require(kernel_object != 0, "kernel object is zero")
    require(packet_kernel_object == kernel_object,
            "packet kernel object differs from reflected kernel")
    require(kernel.get("kernarg_segment_size") == KERNARG_SIZE,
            "reflected kernarg size is not 424")
    require(kernel.get("kernarg_segment_alignment") == ATTENTION_KERNARG_ALIGNMENT,
            "reflected kernarg alignment is not 16")
    require(kernel.get("group_segment_size") == ATTENTION_GROUP_SEGMENT_SIZE,
            "reflected group segment size is not 26112")
    require(kernel.get("private_segment_size") == 0,
            "reflected private segment size is not zero")
    require(full_header == 0x00031502,
            f"AQL full_header is {full_header:#010x}, expected 0x00031502")
    require(packet.get("dimensions") == 3, "AQL packet dimensions are not 3")
    require(packet.get("grid") == [3072, 1, 128],
            "AQL packet grid is not [3072, 1, 128]")
    require(packet.get("workgroup") == [256, 1, 1],
            "AQL packet workgroup is not [256, 1, 1]")
    require(packet.get("private_segment_size") == 0,
            "AQL private segment size is not zero")
    require(packet.get("group_segment_size") == ATTENTION_GROUP_SEGMENT_SIZE,
            "AQL group segment size is not 26112")
    require(packet_kernarg_address != 0 and
            packet_kernarg_address % ATTENTION_KERNARG_ALIGNMENT == 0,
            "AQL kernarg address is zero or misaligned")
    require(completion_signal == 0,
            "first attention completion signal is unexpectedly nonzero")
    require(type(packet.get("packet_id")) is int and
            packet.get("packet_id") >= 0, "AQL packet ID is invalid")
    require(type(packet.get("kernarg_slot")) is int and
            packet.get("kernarg_slot") >= 0, "AQL kernarg slot is invalid")
    require(packet.get("doorbell_written") in (0, 1),
            "AQL doorbell_written is not binary")

    kernarg = record.get("kernarg", {})
    exact_keys(kernarg, {"size", "source_matches_ring", "hex", "decoded"},
               "kernarg")
    require(kernarg.get("size") == KERNARG_SIZE, "kernarg size is not 424")
    require(kernarg.get("source_matches_ring") is True,
            "source kernarg differs from launched ring bytes")
    raw_hex = kernarg.get("hex", "")
    try:
        raw = bytes.fromhex(raw_hex)
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid kernarg hex: {exc}")
        raw = b""
    require(len(raw) == KERNARG_SIZE,
            f"kernarg hex decodes to {len(raw)} bytes, expected 424")
    if len(raw) != KERNARG_SIZE:
        return failures

    try:
        gpu_metadata_address = parse_hex(metadata.get("gpu_address"))
    except (TypeError, ValueError) as exc:
        failures.append(str(exc))
        gpu_metadata_address = 0
    pointer_offsets = (0, 8, 16, 24, 72, 80, 88, 152)
    pointer_values = {offset: unpack("Q", raw, offset) for offset in pointer_offsets}
    require(all(value != 0 for value in pointer_values.values()),
            "attention kernarg contains a null pointer")
    require(all(value % 2 == 0 for value in pointer_values.values()),
            "attention kernarg contains a misaligned pointer")
    require(pointer_values[72] == gpu_metadata_address,
            "kv_indptr does not point at GPU metadata + 0")
    require(pointer_values[88] == gpu_metadata_address + 8,
            "kv_last_page_lens does not point at GPU metadata + 8")
    require(pointer_values[152] == gpu_metadata_address + 16,
            "cu_seqlens_q does not point at GPU metadata + 16")

    expected_raw = bytearray(KERNARG_SIZE)
    known = bytearray(b"\x01" * KERNARG_SIZE)
    for offset in pointer_offsets:
        known[offset:offset + 8] = b"\x00" * 8

    def expected(fmt: str, offset: int, value: int) -> None:
        struct.pack_into("<" + fmt, expected_raw, offset, value)

    for offset, value in {
        40: -1, 44: -1, 48: 128, 52: 128, 56: 12, 60: 6,
        64: key_end, 68: 1, 100: 1536, 104: 128, 108: 128,
        112: 1536, 116: 128, 120: 128, 124: 128, 128: 128,
        132: -1, 136: 0, 140: 0, 144: 2, 160: 256, 164: 256,
    }.items():
        expected("i", offset, value)
    expected("I", 96, SOFTMAX_SCALE_LOG2_BITS)
    for offset, value in ((168, 12), (172, 1), (176, 128)):
        expected("I", offset, value)
    for offset, value in ((180, 256), (182, 1), (184, 1), (232, 3)):
        expected("H", offset, value)
    mismatched_offsets = [
        offset for offset, (actual, expected_byte, is_known) in enumerate(
            zip(raw, expected_raw, known)
        ) if is_known and actual != expected_byte
    ]
    require(not mismatched_offsets,
            f"fixed kernarg ABI bytes mismatch at offsets {mismatched_offsets[:16]}")

    decoded = kernarg.get("decoded", {})
    exact_keys(decoded, {"q", "k", "v", "output", "sequence_sentinels",
                         "key_end", "page_size", "kv_indptr",
                         "kv_page_indices", "kv_last_page_lens",
                         "softmax_scale_log2", "window_left", "window_right",
                         "mask_type", "cu_seqlens_q", "hidden_block_counts",
                         "group_sizes", "hidden_dimensions"}, "decoded kernarg")
    for name, offset in (("q", 0), ("k", 8), ("v", 16), ("output", 24),
                         ("kv_indptr", 72), ("kv_page_indices", 80),
                         ("kv_last_page_lens", 88), ("cu_seqlens_q", 152)):
        try:
            decoded_pointer = parse_hex(decoded.get(name))
        except (TypeError, ValueError) as exc:
            failures.append(f"decoded {name}: {exc}")
            decoded_pointer = 0
        require(decoded_pointer == pointer_values[offset],
                f"serialized {name} differs from raw kernarg")
    require(decoded.get("sequence_sentinels") == [-1, -1],
            "serialized sequence sentinels mismatch")
    require(decoded.get("key_end") == key_end,
            "serialized kernarg decode has the wrong key_end")
    require(decoded.get("page_size") == 1, "serialized page size mismatch")
    require(decoded.get("window_left") == -1 and
            decoded.get("window_right") == 0, "serialized window mismatch")
    require(decoded.get("mask_type") == 2, "serialized mask type mismatch")
    require(decoded.get("hidden_block_counts") == [12, 1, 128],
            "serialized hidden block counts mismatch")
    require(decoded.get("group_sizes") == [256, 1, 1],
            "serialized group sizes mismatch")
    require(decoded.get("hidden_dimensions") == 3,
            "serialized hidden dimensions mismatch")
    try:
        decoded_softmax = struct.pack("<f", decoded.get("softmax_scale_log2"))
    except (OverflowError, struct.error, TypeError):
        decoded_softmax = b""
    require(decoded_softmax == struct.pack("<I", SOFTMAX_SCALE_LOG2_BITS),
            "serialized softmax scale mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--position-base", type=int, default=384)
    parser.add_argument("--query-count", type=int, default=16384)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    with args.record.open("r", encoding="utf-8") as file:
        record = json.load(file)
    failures = analyze(record, args.position_base, args.query_count,
                       args.layer)
    result = {
        "schema": "loom.q16k.first_attention.analysis.v1",
        "pass": not failures,
        "failures": failures,
        "record": str(args.record),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
