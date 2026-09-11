#!/usr/bin/env python3
"""CPU regression tests for the multi-author frontier source path."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import Counter

THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legacy"))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from frontier_common import assert_frontier_source_record, write_jsonl  # noqa: E402
from merge_frontier_sources import sequence_hash, validate_one  # noqa: E402
from select_frontier_smoke_subset import (  # noqa: E402
    pair_features,
    select_pairs,
    validate_pairs,
)


def make_record(author, pair_idx, label, *, hardness="context_required"):
    pair_id = f"{author}-pair-{pair_idx}"
    scenario = f"{author}::scenario::{pair_idx // 3}"
    first = f"Shared opening for {author} pair {pair_idx}."
    branch = (
        f"Adversarial branch {pair_idx} for {author}."
        if label == 1
        else f"Benign branch {pair_idx} for {author}."
    )
    final = (
        f"Shared final request for {author} pair {pair_idx}."
        if hardness == "context_required"
        else branch
    )
    texts = [first, branch, final]
    turns = []
    for i, text in enumerate(texts):
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
        "conversation_id": f"{author}-c-{pair_idx}-{label}",
        "pair_id": pair_id,
        "label": label,
        "family": "frontier_authored_adversarial" if label == 1 else "frontier_authored_benign",
        "subtype": "test" if label == 1 else "test_benign_twin",
        "difficulty": "hard" if pair_idx % 2 else "medium",
        "difficulty_score": 0.8,
        "target_domain": "policy_evasion" if pair_idx % 2 else "prompt_injection",
        "style": "enterprise" if pair_idx % 2 else "technical",
        "user_turn_count": len(turns),
        "assistant_turn_count": 0,
        "turns": turns,
        "pivot_turn_id": None,
        "supervision_tier": None,
        "loss_weight": None,
        "pivot_kind": None,
        "judge_confidence": None,
        "training_eligible": False,
        "seed_source": author,
        "metadata": {
            "corpus_version": author,
            "generator": author,
            "scenario_family": scenario,
            "split_group": scenario,
            "mechanism_family": f"m-{pair_idx}",
            "slice_role": "paired_adversarial" if label == 1 else "paired_benign",
        },
        "intended_structure": {
            "trajectory_family": "distributed_intended",
            "expected_pivot_turn_ids": [2, 4] if label == 1 else [],
            "pair_hardness": hardness,
            "pair_hardness_rationale": "test",
            "pivot_self_sufficient": hardness == "surface_control",
            "requires_context": hardness == "context_required",
            "annotation_status": "generation_intent_only_not_ground_truth",
        },
    }


def make_pair(author, idx, hardness="context_required"):
    return [
        make_record(author, idx, 1, hardness=hardness),
        make_record(author, idx, 0, hardness=hardness),
    ]


class MultiAuthorTests(unittest.TestCase):
    def test_records_satisfy_source_schema(self):
        for r in make_pair("author-a", 0):
            assert_frontier_source_record(r)

    def test_smoke_selection_balances_two_authors_exactly(self):
        records = []
        for author in ("author-a", "author-b"):
            for idx in range(6):
                hardness = "surface_control" if idx == 0 else "context_required"
                records.extend(make_pair(author, idx, hardness))
        groups = validate_pairs(records)
        selected = select_pairs(groups, 6, seed=44)
        counts = Counter(pair_features(group)["source_author"] for _, group in selected)
        self.assertEqual(counts, Counter({"author-a": 3, "author-b": 3}))
        hardness = {pair_features(group)["pair_hardness"] for _, group in selected}
        self.assertEqual(hardness, {"context_required", "surface_control"})

    def test_sequence_hash_distinguishes_authored_trajectories(self):
        a = make_record("author-a", 1, 1)
        b = make_record("author-b", 1, 1)
        self.assertNotEqual(sequence_hash(a), sequence_hash(b))

    def test_validate_one_rejects_duplicate_normalized_trajectory(self):
        pair = make_pair("author-a", 1)
        duplicate = dict(pair[1])
        duplicate["conversation_id"] = "duplicate-cid"
        duplicate["pair_id"] = "duplicate-pair"
        duplicate["turns"] = [dict(t) for t in pair[0]["turns"]]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "source.jsonl")
            write_jsonl(pair + [duplicate], path)
            with self.assertRaises(RuntimeError):
                validate_one(path, expected_records=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
