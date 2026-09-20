import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAACL = HERE.parent
if str(NAACL) not in sys.path:
    sys.path.insert(0, str(NAACL))

from audit_semantic_turn_consistency import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    classify_masked_turn,
    inspect_splits,
)
from semantic_span_policy import ADJUDICATION_VERSION
from prepare_frontier_dataset import (
    SEMANTIC_TURN_REPAIR_VERSION,
    reconcile_semantically_masked_evidence_turns,
)


def masked_span(text="shared context rather than restating the process"):
    return {
        "label": "EVIDENCE_CANDIDATE",
        "text": text,
        "char_start": 0,
        "char_end": len(text),
        "evidence_status": "supported_strong",
        "counterfactual_delta": 1.0,
        "semantic_adjudication": ADJUDICATION_VERSION,
        "semantic_token_supervision_ignore": True,
        "semantic_original_supervision_tier": "cf_strong",
        "supervision_tier": "ignore",
        "causal_type": "unvalidated",
    }


def positive_span(text="substantive supported evidence"):
    return {
        "label": "EVIDENCE_CANDIDATE",
        "text": text,
        "char_start": 0,
        "char_end": len(text),
        "evidence_status": "supported_strong",
        "counterfactual_delta": 1.0,
        "supervision_tier": "cf_strong",
        "causal_type": "causal",
    }


def record(*, evidence_ids, turn_status=None, extra_spans=None, cid="c1"):
    spans = [masked_span()] + list(extra_spans or [])
    analysis = {"evidence_turn_ids": list(evidence_ids), "turn_interventions": []}
    if turn_status is not None:
        analysis["turn_interventions"].append({
            "turn_id": 0,
            "status": turn_status,
            "delta": 1.0 if turn_status == "supported_strong" else 0.0,
        })
    return {
        "conversation_id": cid,
        "evidence_turn_ids": list(evidence_ids),
        "frontier_evidence_analysis": analysis,
        "turns": [{
            "turn_id": 0,
            "role": "user",
            "text": "x" * 200,
            "span_annotations": spans,
        }],
    }


class SemanticTurnConsistencyTests(unittest.TestCase):
    def test_case_a_explicit_turn_negative_with_only_masked_support(self):
        r = record(evidence_ids=[0], turn_status="not_supported")
        item = classify_masked_turn(r, r["turns"][0])
        self.assertEqual(item["case"], CASE_A)
        self.assertEqual(
            item["recommended_turn_effect"],
            "remove_positive_membership_and_keep_explicit_negative",
        )

    def test_case_b_orphan_evidence_id_after_mask(self):
        r = record(evidence_ids=[0], turn_status=None)
        item = classify_masked_turn(r, r["turns"][0])
        self.assertEqual(item["case"], CASE_B)
        self.assertEqual(
            item["recommended_turn_effect"],
            "requires_policy_review_before_repair",
        )

    def test_case_c_whole_turn_support_is_independent(self):
        r = record(evidence_ids=[0], turn_status="supported_strong")
        item = classify_masked_turn(r, r["turns"][0])
        self.assertEqual(item["case"], CASE_C)

    def test_case_c_nonmasked_supported_span_is_independent(self):
        r = record(
            evidence_ids=[0],
            turn_status="not_supported",
            extra_spans=[positive_span()],
        )
        item = classify_masked_turn(r, r["turns"][0])
        self.assertEqual(item["case"], CASE_C)
        self.assertEqual(item["eligible_nonmasked_positive_span_count"], 1)

    def test_case_d_masked_span_without_turn_membership_dependency(self):
        r = record(evidence_ids=[], turn_status="not_supported")
        item = classify_masked_turn(r, r["turns"][0])
        self.assertEqual(item["case"], CASE_D)

    def test_independent_support_missing_from_evidence_ids_fails_closed(self):
        r = record(evidence_ids=[], turn_status="supported_strong")
        with self.assertRaisesRegex(RuntimeError, "missing from raw B4 evidence_turn_ids"):
            classify_masked_turn(r, r["turns"][0])

    def test_case_a_repair_is_recognized_and_raw_b4_ids_stay_intact(self):
        r = record(evidence_ids=[0], turn_status="not_supported")
        prepared = reconcile_semantically_masked_evidence_turns(r)
        self.assertEqual(prepared, [])
        r["evidence_turn_ids"] = prepared

        self.assertEqual(
            r["frontier_evidence_analysis"]["evidence_turn_ids"],
            [0],
        )
        repair = r["semantic_turn_supervision_repair"]
        self.assertEqual(repair["version"], SEMANTIC_TURN_REPAIR_VERSION)
        self.assertEqual(repair["removed_evidence_turn_ids"], [0])

        item = classify_masked_turn(r, r["turns"][0])
        self.assertEqual(item["case"], CASE_A)
        self.assertTrue(item["repair_applied"])
        self.assertFalse(item["prepared_in_evidence_turn_ids"])
        self.assertTrue(item["raw_in_evidence_turn_ids"])

    def test_reconcile_keeps_independently_supported_masked_turn(self):
        r = record(evidence_ids=[0], turn_status="supported_strong")
        prepared = reconcile_semantically_masked_evidence_turns(r)
        self.assertEqual(prepared, [0])
        self.assertNotIn("semantic_turn_supervision_repair", r)

    def test_reconcile_keeps_nonmasked_span_supported_turn(self):
        r = record(
            evidence_ids=[0],
            turn_status="not_supported",
            extra_spans=[positive_span()],
        )
        prepared = reconcile_semantically_masked_evidence_turns(r)
        self.assertEqual(prepared, [0])
        self.assertNotIn("semantic_turn_supervision_repair", r)

    def test_reconcile_fails_closed_on_orphan_case_b(self):
        r = record(evidence_ids=[0], turn_status=None)
        with self.assertRaisesRegex(ValueError, "without independent supported"):
            reconcile_semantically_masked_evidence_turns(r)

    def test_record_and_analysis_evidence_ids_must_match(self):
        r = record(evidence_ids=[0], turn_status="supported_strong")
        r["frontier_evidence_analysis"]["evidence_turn_ids"] = []
        with self.assertRaisesRegex(RuntimeError, "differs"):
            classify_masked_turn(r, r["turns"][0])

    def test_inspection_counts_spans_turns_and_records_separately(self):
        r1 = record(
            evidence_ids=[0],
            turn_status="not_supported",
            extra_spans=[masked_span("which makes the boundary subtle")],
            cid="a",
        )
        r2 = record(evidence_ids=[], turn_status="not_supported", cid="b")
        report = inspect_splits({"train": [r1], "dev": [r2]})
        self.assertEqual(report["masked_spans"], 3)
        self.assertEqual(report["masked_turns"], 2)
        self.assertEqual(report["masked_records"], 2)
        self.assertEqual(report["case_counts_by_turn"][CASE_A], 1)
        self.assertEqual(report["case_counts_by_span"][CASE_A], 2)
        self.assertEqual(report["case_counts_by_turn"][CASE_D], 1)
        self.assertFalse(report["held_out_test_accessed"])
        self.assertEqual(report["repaired_case_A_turns"], 0)
        self.assertEqual(report["unrepaired_case_A_turns"], 1)

    def test_test_split_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "held-out test"):
            inspect_splits({"test": []})


if __name__ == "__main__":
    unittest.main()
