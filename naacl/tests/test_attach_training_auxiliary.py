import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAACL = HERE.parent
if str(NAACL) not in sys.path:
    sys.path.insert(0, str(NAACL))

from attach_training_auxiliary import attach_training_auxiliary
from prepare_frontier_auxiliary import build_auxiliary_record


def source(cid, *, label, scenario):
    unsafe = 0.9 if label == 0 else 0.0
    return {
        "conversation_id": cid,
        "pair_id": f"pair::{scenario}",
        "label": label,
        "validation_status": "rejected",
        "turns": [
            {"turn_id": 0, "role": "user", "text": f"context {scenario}"},
            {"turn_id": 1, "role": "assistant", "text": "response"},
        ],
        "metadata": {
            "scenario_family": scenario,
            "corpus_version": "fixture-v3",
            "slice_role": "paired_benign" if label == 0 else "paired_adversarial",
            "mechanism_family": "fixture",
        },
        "intended_structure": {"trajectory_family": "fixture", "pair_hardness": "context_required"},
        "frontier_behavioral_validation": {
            "status": "rejected",
            "unsafe_threshold": 0.5,
            "min_confidence_threshold": 0.55,
            "max_unsafe_score": unsafe,
            "max_risk_confidence": 0.9,
            "uncertain_risk": False,
        },
    }


def primary(cid, group):
    return {
        "conversation_id": cid,
        "label": 0,
        "training_eligible": True,
        "metadata": {"consolidated_split_group": group},
        "turns": [{"turn_id": 0, "role": "user", "text": cid}],
    }


class AttachTrainingAuxiliaryTests(unittest.TestCase):
    def test_primary_eval_splits_stay_unchanged_and_eval_families_are_withheld(self):
        train = [primary("p-train", "frontier::train-family")]
        dev = [primary("p-dev", "frontier::dev-family")]
        test = [primary("p-test", "frontier::test-family")]

        aux_train = build_auxiliary_record(source("a-train", label=0, scenario="train-family"))
        aux_dev = build_auxiliary_record(source("a-dev", label=1, scenario="dev-family"))
        aux_test = build_auxiliary_record(source("a-test", label=0, scenario="test-family"))
        aux_only = build_auxiliary_record(source("a-only", label=1, scenario="aux-only-family"))

        splits, stats = attach_training_auxiliary(
            train, dev, test, [aux_train, aux_dev, aux_test, aux_only]
        )
        self.assertEqual(splits["dev"], dev)
        self.assertEqual(splits["test"], test)
        self.assertEqual(
            {r["conversation_id"] for r in splits["train"]},
            {"p-train", "a-train", "a-only"},
        )
        self.assertEqual(stats["auxiliary_included"], 2)
        self.assertEqual(stats["auxiliary_disposition"]["withheld_primary_dev_family"], 1)
        self.assertEqual(stats["auxiliary_disposition"]["withheld_primary_test_family"], 1)
        self.assertTrue(stats["dev_test_primary_only"])

    def test_primary_group_leakage_fails_closed(self):
        train = [primary("p1", "frontier::same")]
        dev = [primary("p2", "frontier::same")]
        with self.assertRaisesRegex(RuntimeError, "primary split-group leakage"):
            attach_training_auxiliary(train, dev, [], [])

    def test_conversation_id_collision_fails_closed(self):
        train = [primary("duplicate", "frontier::train")]
        aux = build_auxiliary_record(source("duplicate", label=0, scenario="aux"))
        with self.assertRaisesRegex(RuntimeError, "collides with primary"):
            attach_training_auxiliary(train, [], [], [aux])


if __name__ == "__main__":
    unittest.main()
