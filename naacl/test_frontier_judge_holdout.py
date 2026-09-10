#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from sample_frontier_judge_holdout import assert_score_blind_b1, largest_remainder_alloc


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
        counts = {("context_required", 0): 60, ("context_required", 1): 40, ("surface_control", 0): 20}
        alloc = largest_remainder_alloc(counts, 37)
        self.assertEqual(sum(alloc.values()), 37)
        for key, n in alloc.items():
            self.assertLessEqual(n, counts[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
