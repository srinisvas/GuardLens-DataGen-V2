#!/usr/bin/env python3
"""Fail-closed post-merge audit for deterministic Stage-B4 evidence v6."""
from __future__ import annotations

import audit_frontier_evidence_protocol_v5 as base
from audit_frontier_validation_protocol_v5_deterministic import audit_record as audit_b2_v5_deterministic
from frontier_evidence_v6 import EVIDENCE_PROTOCOL, build_v6_evidence_config
from frontier_runtime_determinism import assert_runtime_fields

_BASE_AUDIT_RECORD = base.audit_record


def audit_record(
    record,
    *,
    target_model: str,
    judge_model: str,
    max_tokens: int,
    target_max_model_len: int,
    judge_max_model_len: int,
    judge_max_context_chars: int,
):
    _BASE_AUDIT_RECORD(
        record,
        target_model=target_model,
        judge_model=judge_model,
        max_tokens=max_tokens,
        target_max_model_len=target_max_model_len,
        judge_max_model_len=judge_max_model_len,
        judge_max_context_chars=judge_max_context_chars,
    )
    cid = str(record.get("conversation_id", ""))
    analysis = record.get("frontier_evidence_analysis", {}) or {}
    try:
        assert_runtime_fields(analysis, "target")
        assert_runtime_fields(analysis, "judge")
    except Exception as exc:
        raise RuntimeError(f"{cid}: B4 deterministic runtime provenance invalid: {exc}") from exc


def main() -> None:
    base.audit_b2_v5_record = audit_b2_v5_deterministic
    base.EVIDENCE_PROTOCOL = EVIDENCE_PROTOCOL
    base.build_v5_evidence_config = build_v6_evidence_config
    base.audit_record = audit_record
    base.main()


if __name__ == "__main__":
    main()
