#!/usr/bin/env python3
"""Regression tests for the Stage-B4 production throughput wrapper."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import frontier_evidence_analysis as fea  # noqa: E402
from frontier_evidence_fast import (  # noqa: E402
    PrefixReuseEvidenceValidator,
    build_record_seed_map,
)
from frontier_seed_policy import (  # noqa: E402
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)


class StubTarget:
    model = "Qwen/Qwen2.5-32B-Instruct"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, seed, temperature, max_tokens):
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return f"fresh-response-{len(self.calls)}"


class DeterministicTarget:
    model = "Qwen/Qwen2.5-32B-Instruct"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, seed, temperature, max_tokens):
        snapshot = [dict(m) for m in messages]
        self.calls.append(snapshot)
        user_messages = [m for m in snapshot if m["role"] == "user"]
        idx = len(user_messages) - 1
        last_user = user_messages[-1]["content"]
        if last_user == f"u{idx}":
            return f"a{idx}"
        return f"cf{idx}:{last_user}"


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


def deterministic_judgment(_judge, prefix, *, seed, max_context_chars):
    text = "\n".join(str(m.get("content", "")) for m in prefix)
    score = 0.8 if "changed-u1" in text else 0.2
    return judgment(score)


def seeded_record(cid, pair_id, base_seed=42):
    record = {"conversation_id": cid, "pair_id": pair_id}
    seed = experiment_record_seed(base_seed, record)
    record["rollout_provenance"] = {
        "base_seed": base_seed,
        "seed_policy": SEED_POLICY,
        "record_seed": seed,
        "seed_key": experiment_seed_key(record),
    }
    record["frontier_behavioral_validation"] = {
        "seed_policy": SEED_POLICY,
        "record_seed": seed,
    }
    return record


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
                {
                    "user_turn_id": 0,
                    **judgment(0.1),
                    "response_fingerprint": fea.json_fingerprint("a0"),
                    "response_snippet": "a0",
                },
                {
                    "user_turn_id": 2,
                    **judgment(0.2),
                    "response_fingerprint": fea.json_fingerprint("a1"),
                    "response_snippet": "a1",
                },
                {
                    "user_turn_id": 4,
                    **judgment(0.3),
                    "response_fingerprint": fea.json_fingerprint("a2"),
                    "response_snippet": "a2",
                },
            ],
            "max_unsafe_score": 0.3,
            "final_unsafe_score": 0.3,
        }

    def test_suffix_replay_skips_target_calls_before_intervention(self):
        target = StubTarget()
        validator = PrefixReuseEvidenceValidator(target, StubJudge(), max_tokens=2048)
        validator._baseline_key = validator._turns_key(self.turns)
        validator._baseline = self.baseline
        validator._baseline_prefix_safe = True

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

    def test_fresh_baseline_mismatch_disables_prefix_reuse(self):
        target = StubTarget()
        validator = PrefixReuseEvidenceValidator(target, StubJudge(), max_tokens=2048)
        with patch(
            "frontier_evidence_fast.fea.judge_assistant_response",
            side_effect=[judgment(0.1)] * 6,
        ):
            validator.replay(self.turns, seed=5)
            self.assertFalse(validator._baseline_prefix_safe)
            validator.replay(self.turns, seed=5, replacements={2: "changed-u1"})
        self.assertEqual(len(target.calls), 6)

    def test_fast_replay_is_identical_to_full_replay_when_prefix_is_proven(self):
        full_target = DeterministicTarget()
        fast_target = DeterministicTarget()
        full = fea.FrontierEvidenceValidator(full_target, StubJudge(), max_tokens=2048)
        fast = PrefixReuseEvidenceValidator(fast_target, StubJudge(), max_tokens=2048)

        with patch(
            "frontier_evidence_fast.fea.judge_assistant_response",
            side_effect=deterministic_judgment,
        ):
            baseline = fast.replay(self.turns, seed=100)
            self.assertTrue(fast._baseline_prefix_safe)
            self.assertEqual(
                [x["response_fingerprint"] for x in baseline["trajectory"]],
                [
                    fea.json_fingerprint("a0"),
                    fea.json_fingerprint("a1"),
                    fea.json_fingerprint("a2"),
                ],
            )
            optimized = fast.replay(
                self.turns,
                seed=100,
                replacements={2: "changed-u1"},
            )
            reference = full.replay(
                self.turns,
                seed=100,
                replacements={2: "changed-u1"},
            )

        self.assertEqual(optimized, reference)
        self.assertEqual(len(fast_target.calls), 5)
        self.assertEqual(len(full_target.calls), 3)

    def test_missing_intervention_turn_fails_closed(self):
        target = StubTarget()
        validator = PrefixReuseEvidenceValidator(target, StubJudge(), max_tokens=2048)
        validator._baseline_key = validator._turns_key(self.turns)
        validator._baseline = self.baseline
        validator._baseline_prefix_safe = True
        with self.assertRaises(RuntimeError):
            validator.replay(self.turns, seed=5, replacements={99: "missing"})


class ProductionSeedBindingTests(unittest.TestCase):
    def test_b4_seed_map_preserves_pair_shared_seed(self):
        a = seeded_record("adv", "pair-z")
        b = seeded_record("ben", "pair-z")
        seed_map = build_record_seed_map([a, b], expected_base_seed=42)
        self.assertEqual(seed_map["adv"], seed_map["ben"])

    def test_b4_rejects_b2_seed_drift(self):
        record = seeded_record("adv", "pair-z")
        record["frontier_behavioral_validation"]["record_seed"] += 1
        with self.assertRaises(RuntimeError):
            build_record_seed_map([record], expected_base_seed=42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
