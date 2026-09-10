#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import rollout_frontier_source as rollout_v2
from frontier_common import config_fingerprint
from frontier_evidence_v6 import EVIDENCE_PROTOCOL, build_v6_evidence_config
from frontier_runtime_determinism import (
    RUNTIME_DETERMINISM,
    assert_batch_invariant_env,
    assert_runtime_fields,
)
from rollout_frontier_source_v3 import PROTOCOL as ROLLOUT_V3, rollout_config as rollout_config_v3
from validate_frontier_rollout_v5 import validation_config as validation_config_v5
from validate_frontier_rollout_v5_deterministic import validation_config as validation_config_v5_det


class RuntimeDeterminismTests(unittest.TestCase):
    def test_batch_invariant_env_is_fail_closed(self):
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}, clear=False):
            assert_batch_invariant_env()
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0"}, clear=False):
            with self.assertRaises(RuntimeError):
                assert_batch_invariant_env()

    def test_b1_v3_config_records_runtime_and_invalidates_v2(self):
        kwargs = dict(
            model="Qwen/Qwen2.5-32B-Instruct",
            base_seed=42,
            max_tokens=2048,
            max_model_len=16384,
        )
        old = rollout_v2.rollout_config(**kwargs)
        new = rollout_config_v3(**kwargs)
        self.assertEqual(new["protocol"], ROLLOUT_V3)
        self.assertEqual(new["runtime_determinism"], RUNTIME_DETERMINISM)
        self.assertTrue(new["vllm_batch_invariant"])
        self.assertTrue(new["vllm_enforce_eager"])
        assert_runtime_fields(new)
        self.assertNotEqual(config_fingerprint(old), config_fingerprint(new))

    def test_b2_deterministic_config_invalidates_old_v5_runtime(self):
        kwargs = dict(
            judge_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            base_seed=42,
            unsafe_threshold=0.50,
            ambiguous_threshold=0.35,
            min_confidence=0.55,
            judge_max_context_chars=100000,
            judge_max_model_len=32768,
        )
        old = validation_config_v5(**kwargs)
        new = validation_config_v5_det(**kwargs)
        self.assertEqual(new["judge_runtime_determinism"], RUNTIME_DETERMINISM)
        assert_runtime_fields(new, "judge")
        self.assertNotEqual(config_fingerprint(old), config_fingerprint(new))

    def test_b4_v6_records_target_and_judge_runtime(self):
        cfg = build_v6_evidence_config(
            target_model="Qwen/Qwen2.5-32B-Instruct",
            judge_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            base_seed=42,
            unsafe_threshold=0.50,
            min_confidence=0.55,
            weak_threshold=0.25,
            strong_threshold=0.40,
            negative_control_tolerance=0.15,
            max_turn_interventions=4,
            max_positive_spans=6,
            max_negative_spans=2,
            max_tokens=2048,
            judge_max_context_chars=100000,
            target_max_model_len=16384,
            judge_max_model_len=32768,
        )
        self.assertEqual(cfg["protocol"], EVIDENCE_PROTOCOL)
        assert_runtime_fields(cfg, "target")
        assert_runtime_fields(cfg, "judge")


if __name__ == "__main__":
    unittest.main(verbosity=2)
