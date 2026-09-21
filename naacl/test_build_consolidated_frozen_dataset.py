#!/usr/bin/env python3
"""Regression tests for the restored-A + frozen-B final freeze builder."""

from __future__ import annotations

import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_consolidated_frozen_dataset as freeze


def primary_record(cid: str, *, source: str, group: str, text: str) -> dict:
    pair_id = f"pair::{cid}"
    scenario = group.removeprefix("frontier::") if group.startswith("frontier::") else ""
    record = {
        "conversation_id": cid,
        "pair_id": pair_id,
        "label": 0,
        "corpus_source": source,
        "turns": [
            {"turn_id": 0, "role": "user", "text": text},
            {"turn_id": 1, "role": "assistant", "text": "ok"},
        ],
        "metadata": {
            "consolidated_split_group": group,
        },
    }
    if scenario:
        record["metadata"]["scenario_family"] = scenario
    record["metadata"]["normalized_user_trajectory_hash"] = freeze.normalized_user_hash(record)
    return record


def a_aux_record(cid: str, *, text: str) -> dict:
    return {
        "conversation_id": cid,
        "label": 0,
        "detection_label": 0,
        "validation_status": "validated",
        "auxiliary_detection_only": True,
        "use_as": "auxiliary_detection_only",
        "supervision_tier": "auxiliary_detection",
        "detection_loss_weight": 1.0,
        "localization_supervision_ignore": True,
        "pivot_supervision_ignore": True,
        "pivot_loss_weight": 0.0,
        "span_loss_weight": 0.0,
        "turns": [
            {"turn_id": 0, "role": "user", "text": text},
            {"turn_id": 1, "role": "assistant", "text": "ok"},
        ],
    }


def b_aux_record(
    cid: str,
    *,
    scenario: str,
    text: str,
    detection_label: int = 0,
) -> dict:
    record = {
        "conversation_id": cid,
        "pair_id": f"aux-pair::{cid}",
        "label": 1 if detection_label == 0 else 0,
        "authoring_intent_label": 1 if detection_label == 0 else 0,
        "observed_behavior_label": detection_label,
        "detection_label": detection_label,
        "validation_status": "rejected",
        "training_eligible": True,
        "auxiliary_detection_only": True,
        "use_as": "auxiliary_detection_only",
        "primary_pair_complete": False,
        "supervision_tier": "auxiliary_detection",
        "loss_weight": 0.25,
        "detection_loss_weight": 0.25,
        "pivot_loss_weight": 0.0,
        "span_loss_weight": 0.0,
        "pivot_supervision_ignore": True,
        "localization_supervision_ignore": True,
        "pivot_turn_id": None,
        "evidence_turn_ids": [],
        "corpus_source": freeze.B_AUX_SOURCE,
        "turns": [
            {"turn_id": 0, "role": "user", "text": text},
            {"turn_id": 1, "role": "assistant", "text": "ok"},
        ],
        "metadata": {
            "scenario_family": scenario,
            "consolidated_split_group": f"frontier::{scenario}",
        },
    }
    record["metadata"]["normalized_user_trajectory_hash"] = freeze.normalized_user_hash(record)
    return record


class FinalFreezeAttachmentTests(unittest.TestCase):
    def test_b_auxiliary_recomputes_train_membership_from_primary_ownership(self):
        train = [
            primary_record(
                "p-train",
                source="frontier_authored_v3",
                group="frontier::train-family",
                text="train primary",
            )
        ]
        dev = [
            primary_record(
                "p-dev",
                source="frontier_authored_v3",
                group="frontier::dev-family",
                text="dev primary",
            )
        ]
        test = [
            primary_record(
                "p-test",
                source="frontier_authored_v3",
                group="frontier::test-family",
                text="test primary",
            )
        ]

        b_train = b_aux_record(
            "b-train", scenario="train-family", text="aux train"
        )
        b_dev = b_aux_record(
            "b-dev", scenario="dev-family", text="aux dev"
        )
        b_test = b_aux_record(
            "b-test", scenario="test-family", text="aux test"
        )
        b_only = b_aux_record(
            "b-only", scenario="aux-only-family", text="aux only"
        )

        candidate, included, withheld, stats = freeze.attach_auxiliary(
            {"train": train, "dev": dev, "test": test},
            [],
            [b_train, b_dev, b_test, b_only],
        )

        self.assertEqual(
            {r["conversation_id"] for r in candidate["train"]},
            {"p-train", "b-train", "b-only"},
        )
        self.assertEqual(
            {r["conversation_id"] for r in included},
            {"b-train", "b-only"},
        )
        self.assertEqual(
            {r["conversation_id"] for r in withheld},
            {"b-dev", "b-test"},
        )
        self.assertEqual(
            stats["b_auxiliary_disposition"]["withheld_primary_dev_family"], 1
        )
        self.assertEqual(
            stats["b_auxiliary_disposition"]["withheld_primary_test_family"], 1
        )
        self.assertEqual(candidate["dev"], dev)
        self.assertEqual(candidate["test"], test)

    def test_b_aux_exact_dev_trajectory_is_withheld_even_for_aux_only_family(self):
        b = b_aux_record(
            "b-shared", scenario="aux-only-family", text="shared user trajectory"
        )
        shared_hash = b["metadata"]["normalized_user_trajectory_hash"]

        train = [
            primary_record(
                "p-train",
                source="frontier_authored_v3",
                group="frontier::train-family",
                text="different train",
            )
        ]
        dev = [
            primary_record(
                "p-dev",
                source="frontier_authored_v3",
                group="frontier::dev-family",
                text="shared user trajectory",
            )
        ]
        self.assertEqual(
            dev[0]["metadata"]["normalized_user_trajectory_hash"],
            shared_hash,
        )

        candidate, included, withheld, stats = freeze.attach_auxiliary(
            {"train": train, "dev": dev, "test": []},
            [],
            [b],
        )
        self.assertEqual(len(included), 0)
        self.assertEqual(len(withheld), 1)
        self.assertEqual(
            stats["b_auxiliary_disposition"][
                "withheld_primary_dev_exact_user_trajectory"
            ],
            1,
        )
        self.assertEqual(
            {r["conversation_id"] for r in candidate["train"]},
            {"p-train"},
        )

    def test_a_auxiliary_is_train_only_and_kept_when_independent(self):
        train = [
            primary_record(
                "p-train",
                source="legacy_restored_primary",
                group="legacy::pair::one",
                text="primary",
            )
        ]
        aux = a_aux_record("a-aux", text="independent benign")

        candidate, included_b, withheld_b, stats = freeze.attach_auxiliary(
            {"train": train, "dev": [], "test": []},
            [aux],
            [],
        )
        self.assertEqual(len(included_b), 0)
        self.assertEqual(len(withheld_b), 0)
        self.assertEqual(stats["a_auxiliary_included"], 1)
        self.assertEqual(
            {r["conversation_id"] for r in candidate["train"]},
            {"p-train", "a-aux"},
        )
        aux_out = next(r for r in candidate["train"] if r["conversation_id"] == "a-aux")
        self.assertEqual(aux_out["corpus_source"], "legacy_detection_aux")
        self.assertTrue(
            aux_out["metadata"]["consolidated_split_group"].startswith(
                "legacy_aux::conversation::"
            )
        )

    def test_a_auxiliary_exact_primary_overlap_fails_closed(self):
        primary = primary_record(
            "p",
            source="legacy_restored_primary",
            group="legacy::pair::p",
            text="same user text",
        )
        aux = a_aux_record("a", text="same user text")

        with self.assertRaisesRegex(RuntimeError, "overlaps primary"):
            freeze.attach_auxiliary(
                {"train": [primary], "dev": [], "test": []},
                [aux],
                [],
            )

    def test_cross_a_b_auxiliary_exact_overlap_fails_closed(self):
        a = a_aux_record("a", text="same aux text")
        b = b_aux_record("b", scenario="b-family", text="same aux text")

        with self.assertRaisesRegex(RuntimeError, "duplicates A auxiliary"):
            freeze.attach_auxiliary(
                {"train": [], "dev": [], "test": []},
                [a],
                [b],
            )

    def test_primary_group_leakage_fails_closed(self):
        train = [
            primary_record(
                "p1",
                source="frontier_authored_v3",
                group="frontier::same",
                text="one",
            )
        ]
        dev = [
            primary_record(
                "p2",
                source="frontier_authored_v3",
                group="frontier::same",
                text="two",
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "group ownership conflict"):
            freeze.index_primary_ownership(
                {"train": train, "dev": dev, "test": []}
            )

    def test_source_diagnostics_do_not_hide_paired_a_length_signal(self):
        a_ben = primary_record(
            "a-ben",
            source="legacy_restored_primary",
            group="legacy::pair::x",
            text="short",
        )
        a_ben["label"] = 0

        a_mal = copy.deepcopy(a_ben)
        a_mal["conversation_id"] = "a-mal"
        a_mal["label"] = 1
        a_mal["turns"] = [
            {"turn_id": 0, "role": "user", "text": "u0"},
            {"turn_id": 1, "role": "assistant", "text": "a0"},
            {"turn_id": 2, "role": "user", "text": "u1"},
            {"turn_id": 3, "role": "assistant", "text": "a1"},
        ]

        stats = freeze.describe_training_view([a_ben, a_mal])
        self.assertIn("A", stats["source_diagnostics"])
        self.assertGreater(
            stats["source_diagnostics"]["A"]["turn_count_auc_unweighted"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
