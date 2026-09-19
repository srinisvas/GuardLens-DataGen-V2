import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
NAACL = HERE.parent
if str(NAACL) not in sys.path:
    sys.path.insert(0, str(NAACL))

import audit_review_export as review_export
from audit_final_data_prep import (
    audit_frontier_excluded_contract,
    audit_frontier_stress_contract,
    expected_frontier_membership,
)
from prepare_frontier_dataset import add_common_provenance


def raw_record(
    cid,
    *,
    label,
    validation,
    b4_status,
    pair_id=None,
    scenario=None,
):
    return {
        "conversation_id": cid,
        "label": label,
        "validation_status": validation,
        "pair_id": pair_id,
        "metadata": {"scenario_family": scenario} if scenario is not None else {},
        "frontier_evidence_analysis": {"status": b4_status},
        "turns": [],
    }


class ReviewExportContractTests(unittest.TestCase):
    def _write_fixture(self, directory):
        rows = [
            raw_record(
                "r1",
                label=1,
                validation="validated",
                b4_status="complete",
                pair_id="p1",
                scenario="s1",
            ),
            raw_record(
                "r2",
                label=0,
                validation="validated",
                b4_status="not_applicable",
                pair_id="p1",
                scenario="s1",
            ),
        ]
        jsonl = os.path.join(directory, "review.jsonl")
        with open(jsonl, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        digest = hashlib.sha256(Path(jsonl).read_bytes()).hexdigest()
        manifest = {
            "export_protocol": "fixture_export",
            "records_exported": 2,
            "jsonl_sha256": digest,
            "contract_fingerprint": "fixture-contract",
            "snapshot_digest": "fixture-snapshot",
            "input_order_preserved": True,
            "record_fields_unchanged": True,
            "statuses": {"complete": 1, "not_applicable": 1},
            "missing_records": [{"conversation_id": "missing-fixture"}],
        }
        manifest_path = os.path.join(directory, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        return jsonl, manifest_path, digest

    def test_exact_review_contract_accepts_matching_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl, manifest, digest = self._write_fixture(tmp)
            patches = [
                mock.patch.object(review_export, "EXPECTED_EXPORT_PROTOCOL", "fixture_export"),
                mock.patch.object(review_export, "EXPECTED_JSONL_SHA256", digest),
                mock.patch.object(review_export, "EXPECTED_CONTRACT_FINGERPRINT", "fixture-contract"),
                mock.patch.object(review_export, "EXPECTED_SNAPSHOT_DIGEST", "fixture-snapshot"),
                mock.patch.object(review_export, "EXPECTED_RECORDS", 2),
                mock.patch.object(
                    review_export,
                    "EXPECTED_B4_STATUSES",
                    Counter({"complete": 1, "not_applicable": 1}),
                ),
                mock.patch.object(review_export, "EXPECTED_LABELS", Counter({0: 1, 1: 1})),
                mock.patch.object(
                    review_export,
                    "EXPECTED_B2_STATUSES",
                    Counter({"validated": 2}),
                ),
                mock.patch.object(
                    review_export,
                    "EXPECTED_MISSING_RECORD",
                    "missing-fixture",
                ),
            ]
            for patcher in patches:
                patcher.start()
                self.addCleanup(patcher.stop)
            report = review_export.audit_review_export(jsonl, manifest)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["jsonl_sha256"], digest)

    def test_review_contract_rejects_byte_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl, manifest, _ = self._write_fixture(tmp)
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                review_export.audit_review_export(jsonl, manifest)


class FinalMembershipContractTests(unittest.TestCase):
    def test_raw_membership_is_reconstructed_exactly(self):
        raw = [
            raw_record(
                "primary-mal",
                label=1,
                validation="validated",
                b4_status="complete",
                pair_id="p-primary",
                scenario="scenario-primary",
            ),
            raw_record(
                "primary-ben",
                label=0,
                validation="validated",
                b4_status="not_applicable",
                pair_id="p-primary",
                scenario="scenario-primary",
            ),
            raw_record(
                "excluded-mal",
                label=1,
                validation="validated",
                b4_status="complete",
                pair_id="p-rejected-twin",
                scenario="scenario-rejected-twin",
            ),
            raw_record(
                "excluded-ben-rejected",
                label=0,
                validation="rejected",
                b4_status="not_applicable",
                pair_id="p-rejected-twin",
                scenario="scenario-rejected-twin",
            ),
            raw_record(
                "stress-ben",
                label=0,
                validation="validated",
                b4_status="not_applicable",
            ),
            raw_record(
                "excluded-standalone",
                label=0,
                validation="rejected",
                b4_status="not_applicable",
            ),
        ]
        primary, stress, excluded, auxiliary = expected_frontier_membership(raw)
        self.assertEqual(primary, {"primary-mal", "primary-ben"})
        self.assertEqual(stress, {"stress-ben"})
        self.assertEqual(
            excluded,
            {"excluded-mal", "excluded-ben-rejected", "excluded-standalone"},
        )
        self.assertEqual(auxiliary, {"excluded-ben-rejected", "excluded-standalone"})

    def test_inconsistent_pair_scenario_is_not_primary(self):
        raw = [
            raw_record(
                "mal",
                label=1,
                validation="validated",
                b4_status="complete",
                pair_id="p",
                scenario="s1",
            ),
            raw_record(
                "ben",
                label=0,
                validation="validated",
                b4_status="not_applicable",
                pair_id="p",
                scenario="s2",
            ),
        ]
        primary, _, excluded, _ = expected_frontier_membership(raw)
        self.assertEqual(primary, set())
        self.assertEqual(excluded, {"mal", "ben"})

    def test_stress_and_excluded_contracts_fail_closed(self):
        stress = {
            "conversation_id": "stress",
            "label": 0,
            "pair_id": None,
            "training_eligible": False,
            "primary_pair_complete": False,
            "use_as": "benign_stress_evaluation_only",
        }
        audit_frontier_stress_contract([stress])
        bad_stress = dict(stress, training_eligible=True)
        with self.assertRaisesRegex(RuntimeError, "training eligible"):
            audit_frontier_stress_contract([bad_stress])

        excluded = {
            "conversation_id": "excluded",
            "training_eligible": False,
            "use_as": "excluded_from_primary_frontier_corpus",
            "exclusion_reason": "fixture",
        }
        audit_frontier_excluded_contract([excluded])
        bad_excluded = dict(excluded, exclusion_reason="")
        with self.assertRaisesRegex(RuntimeError, "lacks exclusion_reason"):
            audit_frontier_excluded_contract([bad_excluded])

    def test_pair_role_is_explicit_in_canonical_frontier_records(self):
        malicious = add_common_provenance(
            {"conversation_id": "m", "pair_id": "p", "label": 1}
        )
        benign = add_common_provenance(
            {"conversation_id": "b", "pair_id": "p", "label": 0}
        )
        self.assertEqual(malicious["pair_role"], "malicious")
        self.assertEqual(benign["pair_role"], "benign")


if __name__ == "__main__":
    unittest.main()
