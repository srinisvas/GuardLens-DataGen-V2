import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAACL = HERE.parent
if str(NAACL) not in sys.path:
    sys.path.insert(0, str(NAACL))

from audit_frontier_auxiliary import audit, audit_record
from prepare_frontier_auxiliary import (
    AUXILIARY_WEIGHT,
    build_auxiliary_record,
    prepare_auxiliary,
    split_auxiliary,
)
from split_consolidated import (
    assert_no_leakage,
    group_records,
    split_groups,
    training_label,
)


def source(cid, *, label, rejected=True, unsafe=None, scenario="scenario::one"):
    if unsafe is None:
        unsafe = 0.9 if label == 0 else 0.0
    status = "rejected" if rejected else "validated"
    return {
        "conversation_id": cid,
        "pair_id": f"pair::{scenario}",
        "label": label,
        "validation_status": status,
        "training_eligible": False,
        "turns": [
            {"turn_id": 0, "role": "user", "text": f"context {scenario}"},
            {"turn_id": 1, "role": "assistant", "text": "response"},
            {"turn_id": 2, "role": "user", "text": f"request {cid}"},
            {"turn_id": 3, "role": "assistant", "text": "response"},
        ],
        "metadata": {
            "scenario_family": scenario,
            "corpus_version": "fixture-v3",
            "slice_role": "paired_benign" if label == 0 else "paired_adversarial",
            "mechanism_family": "fixture",
        },
        "intended_structure": {"trajectory_family": "fixture", "pair_hardness": "context_required"},
        "frontier_behavioral_validation": {
            "status": status,
            "unsafe_threshold": 0.5,
            "ambiguous_threshold": 0.35,
            "min_confidence_threshold": 0.55,
            "max_unsafe_score": unsafe,
            "max_risk_confidence": 0.9,
            "uncertain_risk": False,
            "near_boundary": False,
            "trajectory": [
                {
                    "unsafe_score": unsafe,
                    "confidence": 0.9,
                }
            ],
        },
    }


class AuxiliaryOutcomeTests(unittest.TestCase):
    def test_rejected_benign_becomes_unsafe_detection_only(self):
        row = build_auxiliary_record(source("benign-unsafe", label=0))
        self.assertEqual(row["authoring_intent_label"], 0)
        self.assertEqual(row["label"], 0)
        self.assertEqual(row["detection_label"], 1)
        self.assertEqual(row["observed_behavior_label"], 1)
        self.assertEqual(row["loss_weight"], AUXILIARY_WEIGHT)
        self.assertEqual(row["detection_loss_weight"], AUXILIARY_WEIGHT)
        self.assertEqual(row["pivot_loss_weight"], 0.0)
        self.assertEqual(row["span_loss_weight"], 0.0)
        self.assertTrue(row["pivot_supervision_ignore"])
        self.assertTrue(row["localization_supervision_ignore"])
        self.assertEqual(row["evidence_turn_ids"], [])
        audit_record(row)

    def test_rejected_malicious_becomes_safe_detection_only(self):
        row = build_auxiliary_record(source("mal-safe", label=1))
        self.assertEqual(row["authoring_intent_label"], 1)
        self.assertEqual(row["label"], 1)
        self.assertEqual(row["detection_label"], 0)
        self.assertEqual(row["observed_behavior_label"], 0)
        audit_record(row)

    def test_validated_records_are_not_auxiliary(self):
        self.assertIsNone(build_auxiliary_record(source("validated", label=1, rejected=False, unsafe=0.8)))

    def test_fail_closed_if_rejection_does_not_match_realized_behavior(self):
        with self.assertRaisesRegex(RuntimeError, "benign-authored rejection is not unsafe"):
            build_auxiliary_record(source("bad-benign", label=0, unsafe=0.0))
        with self.assertRaisesRegex(RuntimeError, "malicious-authored rejected record still qualifies unsafe"):
            build_auxiliary_record(source("bad-mal", label=1, unsafe=0.9))

    def test_prepare_and_audit_preserve_detection_only_contract(self):
        rows = prepare_auxiliary([
            source("a", label=0, scenario="s1"),
            source("b", label=1, scenario="s1"),
            source("c", label=1, rejected=False, unsafe=0.9, scenario="s2"),
        ])
        self.assertEqual(len(rows), 2)
        report = audit(rows)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["detection_labels"], {"1": 1, "0": 1})
        self.assertEqual({r["metadata"]["consolidated_split_group"] for r in rows}, {"frontier::s1"})

    def test_auxiliary_split_keeps_scenario_families_together(self):
        rows = []
        for i in range(18):
            scenario = f"s{i:02d}"
            rows.append(build_auxiliary_record(source(f"b{i}", label=0, scenario=scenario)))
            rows.append(build_auxiliary_record(source(f"m{i}", label=1, scenario=scenario)))
        splits = split_auxiliary(rows, fractions={"train": 0.7, "dev": 0.15, "test": 0.15}, seed=42)
        owner = {}
        for split, subset in splits.items():
            self.assertTrue(subset)
            for row in subset:
                group = row["metadata"]["consolidated_split_group"]
                self.assertIn(owner.setdefault(group, split), {split})
                self.assertEqual(owner[group], split)

    def test_joint_split_balances_on_detection_label_and_shares_frontier_group(self):
        combined = []
        for i in range(18):
            scenario = f"joint::{i:02d}"
            primary = source(f"p{i}", label=i % 2, rejected=False, unsafe=0.8 if i % 2 else 0.0, scenario=scenario)
            primary["corpus_source"] = "frontier_authored_v3"
            primary["training_eligible"] = True
            primary["supervision_tier"] = "benign_validated" if primary["label"] == 0 else "llm_confirmed"
            primary["metadata"]["consolidated_split_group"] = f"frontier::{scenario}"
            primary["metadata"]["normalized_user_trajectory_hash"] = f"primary-{i}"
            aux = build_auxiliary_record(source(f"a{i}", label=i % 2, scenario=scenario))
            self.assertEqual(training_label(aux), 1 - (i % 2))
            combined.extend([primary, aux])
        groups = group_records(combined)
        splits = split_groups(groups, {"train": 0.7, "dev": 0.15, "test": 0.15}, 42)
        assert_no_leakage(splits)
        owner = {}
        for split, subset in splits.items():
            self.assertTrue(subset)
            for row in subset:
                scenario = row["metadata"]["scenario_family"]
                self.assertEqual(owner.setdefault(scenario, split), split)

    def test_audit_rejects_exact_user_trajectory_across_independent_groups(self):
        one = build_auxiliary_record(source("dup-a", label=0, scenario="group-a"))
        two = build_auxiliary_record(source("dup-b", label=0, scenario="group-b"))
        two["turns"] = copy.deepcopy(one["turns"])
        from prepare_frontier_auxiliary import user_trajectory_hash
        two["metadata"]["normalized_user_trajectory_hash"] = user_trajectory_hash(two)
        with self.assertRaisesRegex(RuntimeError, "independent scenario groups"):
            audit([one, two])

    def test_recomputed_status_rejects_inconsistent_stored_b2_status(self):
        row = source("bad-status", label=1)
        row["frontier_behavioral_validation"]["trajectory"][0]["unsafe_score"] = 0.9
        row["frontier_behavioral_validation"]["max_unsafe_score"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "does not recompute"):
            build_auxiliary_record(row)

    def test_audit_rejects_nonzero_localization_weight(self):
        row = build_auxiliary_record(source("bad-weight", label=0))
        row["span_loss_weight"] = 0.25
        with self.assertRaisesRegex(RuntimeError, "span_loss_weight"):
            audit_record(row)


if __name__ == "__main__":
    unittest.main()
