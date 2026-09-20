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

from audit_semantic_turn_repair_delta import (
    audit_artifact_pair,
    expected_repaired_record,
)
from semantic_span_policy import ADJUDICATION_VERSION


def case_a_record(cid="c1"):
    text = "shared context rather than restating the process"
    return {
        "conversation_id": cid,
        "evidence_turn_ids": [0, 2],
        "pivot_turn_id": 0,
        "pivot_kind": "distributed",
        "pivot_supervision_ignore": False,
        "frontier_evidence_analysis": {
            "evidence_turn_ids": [0, 2],
            "turn_interventions": [
                {"turn_id": 0, "status": "not_supported", "delta": 0.0},
                {"turn_id": 2, "status": "supported_strong", "delta": 1.0},
            ],
        },
        "turns": [
            {
                "turn_id": 0,
                "role": "user",
                "text": text,
                "span_annotations": [{
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
                }],
            },
            {
                "turn_id": 2,
                "role": "user",
                "text": "substantive evidence",
                "span_annotations": [{
                    "label": "EVIDENCE_CANDIDATE",
                    "text": "substantive evidence",
                    "char_start": 0,
                    "char_end": 20,
                    "evidence_status": "supported_strong",
                    "counterfactual_delta": 1.0,
                    "supervision_tier": "cf_strong",
                    "causal_type": "causal",
                }],
            },
        ],
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class SemanticTurnRepairDeltaTests(unittest.TestCase):
    def test_expected_record_changes_only_derived_turn_fields_and_provenance(self):
        before = case_a_record()
        after, tids = expected_repaired_record(before)
        self.assertEqual(tids, [0])
        self.assertEqual(after["evidence_turn_ids"], [2])
        self.assertEqual(after["pivot_turn_id"], 2)
        self.assertEqual(after["pivot_kind"], "contextual_pivot")
        self.assertEqual(
            after["frontier_evidence_analysis"],
            before["frontier_evidence_analysis"],
        )
        self.assertEqual(
            after["semantic_turn_supervision_repair"]["removed_evidence_turn_ids"],
            [0],
        )

    def test_artifact_pair_rejects_any_extra_change(self):
        before = case_a_record()
        after, _ = expected_repaired_record(before)
        with tempfile.TemporaryDirectory() as tmp:
            before_path = os.path.join(tmp, "before.jsonl")
            after_path = os.path.join(tmp, "after.jsonl")
            write_jsonl(before_path, [before])
            write_jsonl(after_path, [after])
            report = audit_artifact_pair(
                before_path,
                after_path,
                where="test",
                expected_total_repaired_turns=1,
            )
            self.assertEqual(report["changed_records"], 1)

            after["turns"][1]["text"] = "unexpected mutation"
            write_jsonl(after_path, [after])
            with self.assertRaisesRegex(RuntimeError, "differs beyond"):
                audit_artifact_pair(
                    before_path,
                    after_path,
                    where="test",
                    expected_total_repaired_turns=1,
                )


if __name__ == "__main__":
    unittest.main()
