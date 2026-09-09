#!/usr/bin/env python3
"""Regression tests for the Stage-B4 prefix-reuse throughput optimization."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from frontier_evidence_fast import PrefixReuseEvidenceValidator  # noqa: E402


class StubTarget:
    model = "Qwen/Qwen2.5-32B-Instruct"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, seed, temperature, max_tokens):
        self.calls.append(
            {
                "messages": list(messages),
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return f"fresh-response-{len(self.calls)}"


class StubJudge:
    model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def judgment(score):
    return {
        "unsafe_score": score,
        "behavior": "SAFE_HELP",
        "harmfulness": score,
        "refusal_strength": 0.0,
        "confidence": 0.9,
    }


class PrefixReuseTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {"turn_id": 0, "role": "user", "text": "u0"},
            {"turn_id": 1, "role": "assistant", "text": "a0"},
            {"turn_id": 2, "role": "user", "text": "u1"},
            {"turn_id": 3, "role": "assistant", "text": "a1"},
            {"turn_id": 4, "role": "user", "text": "u2"},
            {"turn_id": 5, "role": "assistant", "text": "a2"},
        ]
        self.baseline = {
            "trajectory": [
                {"user_turn_id": 0, **judgment(0.1), "response_fingerprint": "x0", "response_snippet": "a0"},
                {"user_turn_id": 2, **judgment(0.2), "response_fingerprint": "x1", "response_snippet": "a1"},
                {"user_turn_id": 4, **judgment(0.3), "response_fingerprint": "x2", "response_snippet": "a2"},
            ],
            "max_unsafe_score": 0.3,
            "final_unsafe_score": 0.3,
        }

    def test_suffix_replay_skips_target_calls_before_intervention(self):
        target = StubTarget()
        validator = PrefixReuseEvidenceValidator(target, StubJudge(), max_tokens=2048)
        validator._baseline_key = validator._turns_key(self.turns)
        validator._baseline = self.baseline

        with patch(
            "frontier_evidence_fast.fea.judge_assistant_response",
            side_effect=[judgment(0.4), judgment(0.5)],
        ) as judge_mock:
            out = validator.replay(
                self.turns,
                seed=100,
                replacements={2: "changed-u1"},
            )

        self.assertEqual(len(target.calls), 2)
        self.assertEqual(judge_mock.call_count, 2)
        self.assertEqual(target.calls[0]["seed"], 100 + 1009 * 2)
        self.assertEqual(target.calls[1]["seed"], 100 + 1009 * 3)
        self.assertEqual(target.calls[0]["messages"][0], {"role": "user", "content": "u0"})
        self.assertEqual(target.calls[0]["messages"][1], {"role": "assistant", "content": "a0"})
        self.assertEqual(target.calls[0]["messages"][2], {"role": "user", "content": "changed-u1"})
        self.assertEqual(out["trajectory"][0]["user_turn_id"], 0)
        self.assertEqual(out["trajectory"][1]["user_turn_id"], 2)
        self.assertEqual(out["trajectory"][2]["user_turn_id"], 4)

    def test_replacement_without_matching_cached_baseline_falls_back(self):
        target = StubTarget()
        validator = PrefixReuseEvidenceValidator(target, StubJudge(), max_tokens=2048)
        with patch(
            "frontier_evidence_fast.fea.judge_assistant_response",
            side_effect=[judgment(0.1), judgment(0.2), judgment(0.3)],
        ):
            validator.replay(self.turns, seed=5, replacements={2: "changed-u1"})
        self.assertEqual(len(target.calls), 3)

    def test_missing_intervention_turn_fails_closed(self):
        target = StubTarget()
        validator = PrefixReuseEvidenceValidator(target, StubJudge(), max_tokens=2048)
        validator._baseline_key = validator._turns_key(self.turns)
        validator._baseline = self.baseline
        with self.assertRaises(RuntimeError):
            validator.replay(self.turns, seed=5, replacements={99: "missing"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
