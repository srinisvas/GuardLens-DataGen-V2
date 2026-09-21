#!/usr/bin/env python3
"""Regression tests for Dataset A detection-only auxiliary materialization."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_legacy_detection_aux as aux


def make_record(
    cid: str,
    *,
    pair_id: str,
    label: int,
    user_turns: int,
    detection_weight: float | None = None,
):
    turns = []
    for idx in range(user_turns):
        turns.append({
            "turn_id": 2 * idx,
            "role": "user",
            "text": f"user {idx}",
            "span_annotations": [],
        })
        turns.append({
            "turn_id": 2 * idx + 1,
            "role": "assistant",
            "text": f"assistant {idx}",
            "span_annotations": [],
        })
    record = {
        "conversation_id": cid,
        "pair_id": pair_id,
        "label": label,
        "family": "interactive_benign_twin" if label == 0 else "interactive_adversarial",
        "turns": turns,
        "conversation_length": len(turns),
        "validation_status": "validated",
        "training_eligible": True,
        "supervision_tier": "benign_validated" if label == 0 else "llm_confirmed",
        "loss_weight": 1.0,
        "metadata": {},
    }
    if detection_weight is not None:
        record.update({
            "auxiliary_detection_only": True,
            "use_as": "auxiliary_detection_only",
            "detection_label": label,
            "detection_loss_weight": detection_weight,
            "localization_supervision_ignore": True,
            "pivot_supervision_ignore": True,
            "pivot_loss_weight": 0.0,
            "span_loss_weight": 0.0,
        })
    return record


class LegacyAuxiliaryTests(unittest.TestCase):
    def test_auxiliary_copy_preserves_observable_text_and_masks_localization(self):
        record = make_record(
            "benign-1", pair_id="standalone-1", label=0, user_turns=4
        )
        before = aux.observable_turn_hash(record)
        out = aux.auxiliary_copy(record, 1.0)
        self.assertEqual(before, aux.observable_turn_hash(out))
        self.assertEqual(out["detection_label"], 0)
        self.assertEqual(out["detection_loss_weight"], 1.0)
        self.assertTrue(out["auxiliary_detection_only"])
        self.assertTrue(out["localization_supervision_ignore"])
        self.assertTrue(out["pivot_supervision_ignore"])
        self.assertEqual(out["pivot_loss_weight"], 0.0)
        self.assertEqual(out["span_loss_weight"], 0.0)
        self.assertEqual(out["loss_weight"], 1.0)

    def test_primary_pair_contract_accepts_one_exact_pair(self):
        mal = make_record("m1", pair_id="p1", label=1, user_turns=4)
        ben = make_record("b1", pair_id="p1", label=0, user_turns=3)
        aux.validate_primary(
            [mal, ben],
            1,
            expected_sha256="",
            actual_sha256="anything",
        )

    def test_primary_pair_contract_rejects_duplicate_label_pair(self):
        one = make_record("b1", pair_id="p1", label=0, user_turns=3)
        two = make_record("b2", pair_id="p1", label=0, user_turns=3)
        with self.assertRaises(RuntimeError):
            aux.validate_primary(
                [one, two],
                1,
                expected_sha256="",
                actual_sha256="anything",
            )

    def test_trajectory_contract_rejects_unmatched_user_turn(self):
        record = make_record("x", pair_id="x", label=0, user_turns=2)
        record["turns"] = record["turns"][:-1]
        record["conversation_length"] = len(record["turns"])
        with self.assertRaises(RuntimeError):
            aux.validate_trajectory(record, source_name="test")

    def test_source_contract_rejects_length_matched_derivative(self):
        record = make_record("x", pair_id="x", label=0, user_turns=2)
        record["metadata"]["naacl_length_match"] = {"method": "trim"}
        with self.assertRaises(RuntimeError):
            aux.validate_broad_benign_source(
                [record],
                1,
                expected_git_blob="",
                actual_git_blob="anything",
            )

    def test_weighted_diagnostic_exposes_downweighted_auxiliary(self):
        primary_benign = make_record(
            "pb", pair_id="p1", label=0, user_turns=2
        )
        primary_malicious = make_record(
            "pm", pair_id="p1", label=1, user_turns=8
        )
        long_aux = make_record(
            "ab", pair_id="aux", label=0, user_turns=8, detection_weight=0.25
        )
        stats = aux.turn_count_diagnostic(
            [primary_benign, primary_malicious, long_aux]
        )
        self.assertNotEqual(
            stats["unweighted"]["turn_count_auc"],
            stats["effective_detection_weighted"]["turn_count_auc"],
        )

    def test_weight_one_makes_weighted_and_unweighted_auc_identical(self):
        primary_benign = make_record(
            "pb", pair_id="p1", label=0, user_turns=2
        )
        primary_malicious = make_record(
            "pm", pair_id="p1", label=1, user_turns=8
        )
        long_aux = make_record(
            "ab", pair_id="aux", label=0, user_turns=8, detection_weight=1.0
        )
        stats = aux.turn_count_diagnostic(
            [primary_benign, primary_malicious, long_aux]
        )
        self.assertAlmostEqual(
            stats["unweighted"]["turn_count_auc"],
            stats["effective_detection_weighted"]["turn_count_auc"],
        )

    def test_git_blob_hash_matches_git_object_contract(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"abc\n")
            path = handle.name
        try:
            # printf 'abc\n' | git hash-object --stdin
            self.assertEqual(
                aux.git_blob_sha1(path),
                "8baef1b4abc478178b004d62031cf7fe6db6f903",
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
