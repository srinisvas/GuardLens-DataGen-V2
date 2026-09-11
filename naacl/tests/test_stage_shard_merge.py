#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import unittest

THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legacy"))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from merge_stage_shards import merge_stage_shards  # noqa: E402


def rec(cid):
    return {"conversation_id": cid}


class StageShardMergeTests(unittest.TestCase):
    def setUp(self):
        self.source = [rec(f"c{i}") for i in range(6)]

    def test_valid_merge_preserves_source_order(self):
        shards = [
            [rec("c0"), rec("c2"), rec("c4")],
            [rec("c1"), rec("c3"), rec("c5")],
        ]
        out = merge_stage_shards(source_records=self.source, shard_records=shards)
        self.assertEqual([r["conversation_id"] for r in out], [f"c{i}" for i in range(6)])

    def test_duplicate_across_shards_is_rejected(self):
        shards = [
            [rec("c0"), rec("c2"), rec("c4")],
            [rec("c1"), rec("c3"), rec("c0")],
        ]
        with self.assertRaises(RuntimeError):
            merge_stage_shards(source_records=self.source, shard_records=shards)

    def test_wrong_shard_record_is_rejected(self):
        shards = [
            [rec("c0"), rec("c1"), rec("c4")],
            [rec("c2"), rec("c3"), rec("c5")],
        ]
        with self.assertRaises(RuntimeError):
            merge_stage_shards(source_records=self.source, shard_records=shards)

    def test_unexpected_record_is_rejected(self):
        shards = [
            [rec("c0"), rec("c2"), rec("extra")],
            [rec("c1"), rec("c3"), rec("c5")],
        ]
        with self.assertRaises(RuntimeError):
            merge_stage_shards(source_records=self.source, shard_records=shards)

    def test_missing_record_is_rejected(self):
        shards = [
            [rec("c0"), rec("c2")],
            [rec("c1"), rec("c3"), rec("c5")],
        ]
        with self.assertRaises(RuntimeError):
            merge_stage_shards(source_records=self.source, shard_records=shards)


if __name__ == "__main__":
    unittest.main(verbosity=2)
