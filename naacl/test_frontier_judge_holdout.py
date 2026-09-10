#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from sample_frontier_judge_holdout import (
    assert_score_blind_b1,
    largest_remainder_alloc,
    load_excluded_ids,
)


class HoldoutProtocolTests(unittest.TestCase):
    def test_rejects_judged_input(self):
        record = {
            "conversation_id": "x",
            "rollout_status": "complete",
            "frontier_behavioral_validation": {"protocol": "anything"},
            "turns": [],
        }
        with self.assertRaises(RuntimeError):
            assert_score_blind_b1(record)

    def test_rejects_turn_level_judgment(self):
        record = {
            "conversation_id": "x",
            "rollout_status": "complete",
            "turns": [{"role": "assistant", "behavioral_judgment": {}}],
        }
        with self.assertRaises(RuntimeError):
            assert_score_blind_b1(record)

    def test_largest_remainder_exact_total(self):
        counts = {
            ("context_required", 0): 60,
            ("context_required", 1): 40,
            ("surface_control", 0): 20,
        }
        alloc = largest_remainder_alloc(counts, 37)
        self.assertEqual(sum(alloc.values()), 37)
        for key, n in alloc.items():
            self.assertLessEqual(n, counts[key])

    def test_load_excluded_ids_accepts_id_keyed_design_manifest(self):
        payload = {
            "records": {
                "cid-a": {"expected_realized_behavior": "unsafe"},
                "cid-b": {"expected_realized_behavior": "safe"},
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            self.assertEqual(load_excluded_ids(path), {"cid-a", "cid-b"})
        finally:
            os.unlink(path)

    def test_load_excluded_ids_accepts_list_manifest(self):
        payload = {"records": [{"conversation_id": "cid-a"}, {"conversation_id": "cid-b"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            self.assertEqual(load_excluded_ids(path), {"cid-a", "cid-b"})
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
