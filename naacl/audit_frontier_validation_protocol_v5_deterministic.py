#!/usr/bin/env python3
"""Fail-closed B2-v5 audit including deterministic B1/judge runtime provenance."""
from __future__ import annotations

import audit_frontier_validation_protocol_v5 as base
from frontier_runtime_determinism import assert_runtime_fields
from rollout_frontier_source_v3 import PROTOCOL as ROLLOUT_PROTOCOL
from validate_frontier_rollout_v5_deterministic import validation_config

_BASE_AUDIT_RECORD = base.audit_record


def audit_record(record, *, target_model, judge_model, judge_max_model_len, judge_max_context_chars):
    # The base audit reconstructs its config through a module-global function.
    # Patch it locally as well as in CLI main so this audit remains correct when
    # imported by B4 rather than executed as a standalone script.
    original_validation_config = base.validation_config
    try:
        base.validation_config = validation_config
        _BASE_AUDIT_RECORD(
            record,
            target_model=target_model,
            judge_model=judge_model,
            judge_max_model_len=judge_max_model_len,
            judge_max_context_chars=judge_max_context_chars,
        )
    finally:
        base.validation_config = original_validation_config

    cid = str(record.get("conversation_id", ""))
    rollout = record.get("rollout_provenance", {}) or {}
    if rollout.get("protocol") != ROLLOUT_PROTOCOL:
        raise RuntimeError(
            f"{cid}: B2 source rollout protocol={rollout.get('protocol')!r} != {ROLLOUT_PROTOCOL!r}"
        )
    try:
        assert_runtime_fields(rollout)
    except Exception as exc:
        raise RuntimeError(f"{cid}: B1 deterministic runtime provenance invalid: {exc}") from exc

    validation = record.get("frontier_behavioral_validation", {}) or {}
    try:
        assert_runtime_fields(validation, "judge")
    except Exception as exc:
        raise RuntimeError(f"{cid}: B2 judge deterministic runtime provenance invalid: {exc}") from exc


def main() -> None:
    original_validation_config = base.validation_config
    original_audit_record = base.audit_record
    try:
        base.validation_config = validation_config
        base.audit_record = audit_record
        base.main()
    finally:
        base.validation_config = original_validation_config
        base.audit_record = original_audit_record


if __name__ == "__main__":
    main()
