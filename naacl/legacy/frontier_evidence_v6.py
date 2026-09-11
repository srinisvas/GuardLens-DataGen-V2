#!/usr/bin/env python3
"""Stage B4 evidence v6 under deterministic target and judge runtimes.

The counterfactual method and frozen v5 judge semantics are unchanged. v6 marks
only the new execution contract: B1 factual rollouts, B2 judge calls, and B4
factual/counterfactual replays all use vLLM batch invariance plus eager execution.
"""
from __future__ import annotations

import frontier_evidence_v5 as base
from audit_frontier_validation_protocol_v5_deterministic import audit_record as audit_b2_v5_deterministic
from frontier_runtime_determinism import assert_batch_invariant_env, runtime_provenance

EVIDENCE_PROTOCOL = "frontier_context_paired_counterfactual_v6"
_BASE_BUILD_CONFIG = base.build_v5_evidence_config


def build_v6_evidence_config(**kwargs):
    cfg = _BASE_BUILD_CONFIG(**kwargs)
    cfg["protocol"] = EVIDENCE_PROTOCOL
    cfg.update(runtime_provenance("target"))
    cfg.update(runtime_provenance("judge"))
    return cfg


def install() -> None:
    base.EVIDENCE_PROTOCOL = EVIDENCE_PROTOCOL
    base.build_v5_evidence_config = build_v6_evidence_config
    base.audit_b2_v5_record = audit_b2_v5_deterministic


def main() -> None:
    assert_batch_invariant_env()
    install()
    base.main()


if __name__ == "__main__":
    main()
