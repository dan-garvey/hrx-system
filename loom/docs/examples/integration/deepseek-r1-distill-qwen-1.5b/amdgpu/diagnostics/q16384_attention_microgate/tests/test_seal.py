#!/usr/bin/env python3
"""Host-only tests for deterministic seal parsing and tree safety."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import seal_lib  # noqa: E402


class SealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="loom_attention_seal_")
        self.root = Path(self.temporary.name).resolve()
        (self.root / "payload.txt").write_text("payload\n", encoding="ascii")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_trip_and_mutation_detection(self) -> None:
        digest, count = seal_lib.seal_package(self.root)
        self.assertEqual(seal_lib.verify_tree(self.root), (digest, count))
        with self.assertRaisesRegex(RuntimeError, "already sealed"):
            seal_lib.seal_package(self.root)
        (self.root / "payload.txt").write_text("changed\n", encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            seal_lib.verify_tree(self.root)

    def test_malformed_and_duplicate_manifest_entries_are_rejected(self) -> None:
        valid = "0" * 64 + "  payload.txt\n"
        cases = (
            "not-a-manifest\n",
            valid + valid,
            "0" * 64 + "  ../payload.txt\n",
            valid.rstrip("\n"),
        )
        for raw in cases:
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(RuntimeError):
                    seal_lib.parse_manifest(raw.encode("ascii"))

    def test_symlink_special_file_and_hardlink_are_rejected(self) -> None:
        symlink = self.root / "link"
        symlink.symlink_to(self.root / "payload.txt")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            seal_lib.collect_files(self.root)
        symlink.unlink()

        hardlink = self.root / "hardlink"
        os.link(self.root / "payload.txt", hardlink)
        with self.assertRaisesRegex(RuntimeError, "hardlink"):
            seal_lib.collect_files(self.root)
        hardlink.unlink()

        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(RuntimeError, "special"):
            seal_lib.collect_files(self.root)

    def test_bytecode_cache_is_rejected(self) -> None:
        cache = self.root / "__pycache__"
        cache.mkdir()
        (cache / "payload.cpython-310.pyc").write_bytes(b"not bytecode")
        with self.assertRaisesRegex(RuntimeError, "bytecode cache"):
            seal_lib.collect_files(self.root)

    def test_build_directory_is_outside_the_sealed_closure(self) -> None:
        build = self.root / "build"
        build.mkdir()
        (build / "object.o").write_bytes(b"temporary")
        files = seal_lib.collect_files(self.root)
        self.assertEqual(files, [Path("payload.txt")])

    def test_tree_digest_binds_sorted_file_names_and_bytes(self) -> None:
        (self.root / "second.txt").write_text("second\n", encoding="ascii")
        digest, count = seal_lib.tree_digest(self.root)
        first_hash = hashlib.sha256(b"payload\n").hexdigest()
        second_hash = hashlib.sha256(b"second\n").hexdigest()
        expected_manifest = (
            f"{first_hash}  payload.txt\n"
            f"{second_hash}  second.txt\n"
        ).encode("ascii")
        self.assertEqual(digest, hashlib.sha256(expected_manifest).hexdigest())
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
