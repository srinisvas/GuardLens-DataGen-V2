#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from audit_frontier_validation_protocol import audit_record  # noqa: E402
from frontier_common import (  # noqa: E402
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_JUDGE_MAX_MODEL_LEN,
    config_fingerprint,
)
from frontier_seed_policy import (  # noqa: E402
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)
from validate_frontier_rollout import validation_config  # noqa: E402

TARGET = "Qwen/Qwen2.5-32B-Instruct"
JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
BASE_SEED = 42


def valid_record():
    record = {
        "conversation_id": "b2-audit-test",
        "pair_id": "pair-b2-audit",
        "validation_status": "validated",
    }
    record_seed = experiment_record_seed(BASE_SEED, record)
    cfg = validation_config(
        judge_model=JUDGE,
        base_seed=BASE_SEED,
        unsafe_threshold=0.50,
        ambiguous_threshold=0.35,
        min_confidence=0.55,
        judge_max_context_chars=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
        judge_max_model_len=DEFAULT_JUDGE_MAX_MODEL_LEN,
    )
    record["rollout_provenance"] = {
        "target_model": TARGET,
        "base_seed": BASE_SEED,
        "seed_policy": SEED_POLICY,
        "record_seed": record_seed,
        "seed_key": experiment_seed_key(record),
    }
    record["frontier_behavioral_validation"] = {
        "status": "validated",
        **cfg,
        "target_model": TARGET,
        "record_seed": record_seed,
        "config_fingerprint": config_fingerprint(cfg),
        "authoring_metadata_exposed_to_judge": False,
    }
    return record


class B2ProtocolAuditTests(unittest.TestCase):
    def test_reviewed_record_passes(self):
        audit_record(
            valid_record(),
            target_model=TARGET,
            judge_model=JUDGE,
            judge_max_model_len=DEFAULT_JUDGE_MAX_MODEL_LEN,
            judge_max_context_chars=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
        )

    def test_wrong_runtime_context_is_rejected(self):
        record = valid_record()
        record["frontier_behavioral_validation"]["judge_max_model_len"] = 16384
        with self.assertRaises(RuntimeError):
            audit_record(
                record,
                target_model=TARGET,
                judge_model=JUDGE,
                judge_max_model_len=DEFAULT_JUDGE_MAX_MODEL_LEN,
                judge_max_context_chars=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
            )

    def test_mutated_threshold_with_stale_fingerprint_is_rejected(self):
        record = valid_record()
        record["frontier_behavioral_validation"]["unsafe_threshold"] = 0.6
        with self.assertRaises(RuntimeError):
            audit_record(
                record,
                target_model=TARGET,
                judge_model=JUDGE,
                judge_max_model_len=DEFAULT_JUDGE_MAX_MODEL_LEN,
                judge_max_context_chars=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
            )

    def test_b2_record_seed_must_match_b1(self):
        record = valid_record()
        record["frontier_behavioral_validation"]["record_seed"] += 1
        with self.assertRaises(RuntimeError):
            audit_record(
                record,
                target_model=TARGET,
                judge_model=JUDGE,
                judge_max_model_len=DEFAULT_JUDGE_MAX_MODEL_LEN,
                judge_max_context_chars=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
