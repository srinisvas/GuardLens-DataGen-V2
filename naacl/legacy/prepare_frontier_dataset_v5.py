#!/usr/bin/env python3
"""Stage B5 preparation wrapper locked to B2-v5 and B4-v5 provenance.

The canonical preparation/sanitization logic stays in prepare_frontier_dataset.py.
This wrapper replaces only its protocol-chain validator so v3/v4 judge/evidence
artifacts cannot enter the final Dataset B.
"""
from __future__ import annotations

import prepare_frontier_dataset as pfd
from audit_frontier_evidence_protocol_v5 import audit_record as audit_b4_v5_record
from frontier_evidence_v5 import EVIDENCE_PROTOCOL
from frontier_judge_v5 import PROTOCOL as VALIDATION_PROTOCOL

EXPECTED_TARGET_MAX_TOKENS = 2048
EXPECTED_TARGET_MAX_MODEL_LEN = 16384
EXPECTED_JUDGE_MAX_MODEL_LEN = 32768
EXPECTED_JUDGE_MAX_CONTEXT_CHARS = 100000


def assert_expected_provenance_v5(
    record,
    *,
    expected_target: str,
    expected_judge: str,
    require_evidence: bool,
) -> None:
    cid = str(record.get("conversation_id", ""))
    audit_b4_v5_record(
        record,
        target_model=expected_target,
        judge_model=expected_judge,
        max_tokens=EXPECTED_TARGET_MAX_TOKENS,
        target_max_model_len=EXPECTED_TARGET_MAX_MODEL_LEN,
        judge_max_model_len=EXPECTED_JUDGE_MAX_MODEL_LEN,
        judge_max_context_chars=EXPECTED_JUDGE_MAX_CONTEXT_CHARS,
    )
    analysis = record.get("frontier_evidence_analysis", {}) or {}
    if require_evidence:
        if analysis.get("status") != "complete":
            raise ValueError(f"{cid}: malicious evidence analysis is not complete")
        if analysis.get("baseline_reproduced_stored_rollout") is not True:
            raise ValueError(f"{cid}: fresh B4 baseline did not reproduce B1/B2-v5")
    else:
        if analysis.get("status") != "not_applicable":
            raise ValueError(
                f"{cid}: benign evidence status={analysis.get('status')!r}; expected not_applicable"
            )


def main() -> None:
    original_assert = pfd.assert_expected_provenance
    original_validation_protocol = pfd.VALIDATION_PROTOCOL
    original_evidence_protocol = pfd.EVIDENCE_PROTOCOL
    original_target_tokens = pfd.EXPECTED_TARGET_MAX_TOKENS
    original_target_context = pfd.EXPECTED_TARGET_MAX_MODEL_LEN
    original_judge_context = pfd.EXPECTED_JUDGE_MAX_MODEL_LEN
    original_judge_chars = pfd.EXPECTED_JUDGE_MAX_CONTEXT_CHARS
    try:
        pfd.assert_expected_provenance = assert_expected_provenance_v5
        pfd.VALIDATION_PROTOCOL = VALIDATION_PROTOCOL
        pfd.EVIDENCE_PROTOCOL = EVIDENCE_PROTOCOL
        pfd.EXPECTED_TARGET_MAX_TOKENS = EXPECTED_TARGET_MAX_TOKENS
        pfd.EXPECTED_TARGET_MAX_MODEL_LEN = EXPECTED_TARGET_MAX_MODEL_LEN
        pfd.EXPECTED_JUDGE_MAX_MODEL_LEN = EXPECTED_JUDGE_MAX_MODEL_LEN
        pfd.EXPECTED_JUDGE_MAX_CONTEXT_CHARS = EXPECTED_JUDGE_MAX_CONTEXT_CHARS
        pfd.main()
    finally:
        pfd.assert_expected_provenance = original_assert
        pfd.VALIDATION_PROTOCOL = original_validation_protocol
        pfd.EVIDENCE_PROTOCOL = original_evidence_protocol
        pfd.EXPECTED_TARGET_MAX_TOKENS = original_target_tokens
        pfd.EXPECTED_TARGET_MAX_MODEL_LEN = original_target_context
        pfd.EXPECTED_JUDGE_MAX_MODEL_LEN = original_judge_context
        pfd.EXPECTED_JUDGE_MAX_CONTEXT_CHARS = original_judge_chars


if __name__ == "__main__":
    main()
