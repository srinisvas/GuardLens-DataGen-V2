import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAACL = HERE.parent
if str(NAACL) not in sys.path:
    sys.path.insert(0, str(NAACL))

from audit_semantic_span_masking import audit as audit_semantic_masking
from semantic_span_policy import (
    ADJUDICATION_VERSION,
    apply_semantic_span_policy,
)


def span(*, label="EVIDENCE_CANDIDATE", status="supported_strong", text="", tier="cf_strong"):
    return {
        "label": label,
        "evidence_status": status,
        "text": text,
        "supervision_tier": tier,
        "causal_type": "causal",
        "counterfactual_delta": 0.8,
    }


class SemanticSpanPolicyTests(unittest.TestCase):
    def test_supported_construction_span_is_masked_without_rewriting_evidence(self):
        row = span(text="Intent and prior context should matter more than obvious risk vocabulary.")
        self.assertTrue(apply_semantic_span_policy(row))
        self.assertEqual(row["label"], "EVIDENCE_CANDIDATE")
        self.assertEqual(row["evidence_status"], "supported_strong")
        self.assertEqual(row["counterfactual_delta"], 0.8)
        self.assertEqual(row["semantic_adjudication"], ADJUDICATION_VERSION)
        self.assertTrue(row["semantic_token_supervision_ignore"])
        self.assertEqual(row["semantic_original_supervision_tier"], "cf_strong")
        self.assertEqual(row["supervision_tier"], "ignore")
        self.assertEqual(row["causal_type"], "unvalidated")

    def test_weak_supported_construction_span_is_also_masked(self):
        row = span(
            status="supported_weak",
            tier="cf_weak",
            text="The final request should look ordinary if read by itself.",
        )
        self.assertTrue(apply_semantic_span_policy(row))
        self.assertEqual(row["evidence_status"], "supported_weak")
        self.assertEqual(row["semantic_original_supervision_tier"], "cf_weak")
        self.assertEqual(row["supervision_tier"], "ignore")

    def test_unsupported_construction_candidate_is_not_changed(self):
        row = span(
            status="not_supported",
            tier="ignore",
            text="Which makes the boundary subtle.",
        )
        before = dict(row)
        self.assertFalse(apply_semantic_span_policy(row))
        self.assertEqual(row, before)

    def test_supported_substantive_span_remains_positive(self):
        row = span(text="The payment change has not been independently verified.")
        before = dict(row)
        self.assertFalse(apply_semantic_span_policy(row))
        self.assertEqual(row, before)

    def test_audit_requires_every_reviewed_span_retained_in_primary_to_be_masked(self):
        raw_span = span(
            text="The final request should look ordinary if read by itself."
        )
        raw_span.update({"char_start": 0, "char_end": 57})
        raw = [{
            "conversation_id": "c1",
            "turns": [{
                "turn_id": 0,
                "role": "user",
                "text": raw_span["text"],
                "span_annotations": [dict(raw_span)],
            }],
        }]
        prepared = [{
            "conversation_id": "c1",
            "turns": [{
                "turn_id": 0,
                "role": "user",
                "text": raw_span["text"],
                "span_annotations": [dict(raw_span)],
            }],
        }]
        with self.assertRaisesRegex(RuntimeError, "masking is incomplete"):
            audit_semantic_masking(raw, prepared)

        apply_semantic_span_policy(prepared[0]["turns"][0]["span_annotations"][0])
        report = audit_semantic_masking(raw, prepared)
        self.assertEqual(report["prepared_masked_spans"], 1)

    def test_control_candidate_is_never_semantically_masked(self):
        row = span(
            label="CONTROL_CANDIDATE",
            text="The conversation should sound like ordinary work rather than a safety demonstration.",
        )
        before = dict(row)
        self.assertFalse(apply_semantic_span_policy(row))
        self.assertEqual(row, before)


if __name__ == "__main__":
    unittest.main()
