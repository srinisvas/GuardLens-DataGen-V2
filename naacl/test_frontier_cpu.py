#!/usr/bin/env python3
"""CPU-only regression tests for frontier GuardLens pipeline invariants.

Run from repository root:
    python -m unittest naacl/test_frontier_cpu.py -v

No model, network, GPU, or external test framework is required.
"""
from __future__ import annotations

import copy
import os
import sys
import unittest

# Pipeline scripts use sibling imports when executed as ``python naacl/foo.py``.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from frontier_common import (  # noqa: E402
    assert_frontier_source_record,
    validate_judge_object,
)
from materialize_frontier_candidates import extract_candidate_spans  # noqa: E402
from split_consolidated import (  # noqa: E402
    assert_no_leakage,
    split_groups,
)


def source_record(cid="c1", label=1, pair_id="p1", n_users=6):
    turns = []
    for i in range(n_users):
        text = f"This is source user turn number {i} with sufficient content for testing."
        turns.append({
            "turn_id": 2 * i,
            "role": "user",
            "text": text,
            "span_annotations": [],
            "intent_score": None,
            "surface_risk": None,
            "is_trigger": None,
            "trigger_kind": None,
            "is_payload": None,
            "implicit_trigger": None,
            "quoted_unsafe_content": None,
        })
    return {
        "conversation_id": cid,
        "pair_id": pair_id,
        "label": label,
        "user_turn_count": n_users,
        "assistant_turn_count": 0,
        "turns": turns,
        "pivot_turn_id": None,
        "supervision_tier": None,
        "loss_weight": None,
        "judge_confidence": None,
        "training_eligible": False,
        "intended_structure": {
            "expected_pivot_turn_ids": [2 * (n_users - 1)] if label == 1 else [],
            "annotation_status": "generation_intent_only_not_ground_truth",
        },
        "metadata": {
            "scenario_family": "scenario-1",
            "split_group": "scenario-1",
            "slice_role": "paired_adversarial" if label == 1 else "paired_benign",
        },
    }


def split_record(cid, group, label, source="frontier_authored_v3", difficulty="hard"):
    return {
        "conversation_id": cid,
        "label": label,
        "difficulty": difficulty,
        "corpus_source": source,
        "pair_id": group if source == "legacy_repaired" else f"pair-{group}",
        "target_domain": "policy_evasion",
        "style": "enterprise",
        "intended_structure": {
            "pair_hardness": "context_required",
            "trajectory_family": "distributed_intended",
        },
        "metadata": {
            "consolidated_split_group": group,
            "scenario_family": group if source == "frontier_authored_v3" else None,
            "slice_role": "paired_adversarial" if label == 1 else "paired_benign",
            "mechanism_family": "mechanism-a",
        },
    }


class SourceSchemaTests(unittest.TestCase):
    def test_valid_source_record(self):
        assert_frontier_source_record(source_record())

    def test_gap_in_even_turn_ids_fails(self):
        record = source_record()
        record["turns"][2]["turn_id"] = 6
        with self.assertRaises(ValueError):
            assert_frontier_source_record(record)

    def test_nonempty_source_span_fails(self):
        record = source_record()
        record["turns"][0]["span_annotations"] = [{"label": "MALICIOUS_TRIGGER"}]
        with self.assertRaises(ValueError):
            assert_frontier_source_record(record)

    def test_intended_pivot_must_point_to_user_turn(self):
        record = source_record()
        record["intended_structure"]["expected_pivot_turn_ids"] = [999]
        with self.assertRaises(ValueError):
            assert_frontier_source_record(record)


class JudgeParsingTests(unittest.TestCase):
    def test_valid_judge_object(self):
        result = validate_judge_object({
            "assistant_behavior": "SAFE_HELP",
            "harmfulness": 0.1,
            "refusal_strength": 0.8,
            "confidence": 0.9,
        })
        self.assertEqual(result["behavior"], "SAFE_HELP")
        self.assertAlmostEqual(result["unsafe_score"], 0.02, places=3)

    def test_missing_required_key_fails(self):
        with self.assertRaises(ValueError):
            validate_judge_object({
                "assistant_behavior": "SAFE_HELP",
                "harmfulness": 0.1,
                "refusal_strength": 0.8,
            })

    def test_out_of_range_number_fails(self):
        with self.assertRaises(ValueError):
            validate_judge_object({
                "assistant_behavior": "SAFE_HELP",
                "harmfulness": 1.2,
                "refusal_strength": 0.0,
                "confidence": 0.9,
            })

    def test_unknown_behavior_fails(self):
        with self.assertRaises(ValueError):
            validate_judge_object({
                "assistant_behavior": "MAYBE_SAFE",
                "harmfulness": 0.2,
                "refusal_strength": 0.0,
                "confidence": 0.9,
            })


class CandidateSelectionTests(unittest.TestCase):
    def test_default_two_candidates_cover_context_and_final_clause(self):
        text = (
            "Earlier context establishes a long operational constraint that matters here, "
            "there is also a short bridge phrase, "
            "now provide the requested final transformation in the specified format."
        )
        spans = extract_candidate_spans(text, 2)
        self.assertEqual(len(spans), 2)
        selected = [text[a:b] for a, b in spans]
        self.assertTrue(any("Earlier context" in x for x in selected))
        self.assertTrue(any("now provide" in x for x in selected))


class SplitTests(unittest.TestCase):
    def test_group_integrity_and_approximate_ratios(self):
        groups = {}
        cid = 0
        # 120 frontier scenario groups of six records each plus 120 legacy
        # groups of two records. This exercises unequal group sizes.
        for g in range(120):
            group_id = f"frontier::s{g}"
            records = []
            for j in range(6):
                records.append(split_record(
                    f"c{cid}", group_id, j % 2, source="frontier_authored_v3"
                ))
                cid += 1
            groups[group_id] = records
        for g in range(120):
            group_id = f"legacy::p{g}"
            records = [
                split_record(f"c{cid}", group_id, 0, source="legacy_repaired"),
                split_record(f"c{cid+1}", group_id, 1, source="legacy_repaired"),
            ]
            cid += 2
            groups[group_id] = records

        fractions = {"train": 0.70, "dev": 0.15, "test": 0.15}
        splits = split_groups(groups, fractions, seed=42)
        assert_no_leakage(splits)
        total = sum(len(x) for x in splits.values())
        self.assertLess(abs(len(splits["train"]) / total - 0.70), 0.03)
        self.assertLess(abs(len(splits["dev"]) / total - 0.15), 0.03)
        self.assertLess(abs(len(splits["test"]) / total - 0.15), 0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
