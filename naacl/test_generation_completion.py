#!/usr/bin/env python3
"""CPU regression tests for frontier completion and protocol-chain contracts."""
from __future__ import annotations

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from frontier_common import (  # noqa: E402
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_JUDGE_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_TOKENS,
    parse_chat_completion_data,
    transcript_text,
)
from frontier_seed_policy import (  # noqa: E402
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)
from prepare_frontier_dataset import assert_expected_provenance  # noqa: E402
from validate_frontier_rollout import assert_realized_rollout  # noqa: E402

TARGET = "Qwen/Qwen2.5-32B-Instruct"
JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
BASE_SEED = 42


def completion_payload(*, finish_reason="stop", completion_tokens=42):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "A complete answer."},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"completion_tokens": completion_tokens},
    }


def realized_record(
    *,
    finish_reason="stop",
    completion_tokens=42,
    max_tokens=2048,
    max_model_len=16384,
    pair_id=None,
    conversation_id="completion-test",
):
    record = {
        "conversation_id": conversation_id,
        "pair_id": pair_id,
        "user_turn_count": 1,
        "assistant_turn_count": 1,
        "conversation_length": 2,
        "rollout_status": "complete",
        "turns": [
            {"turn_id": 0, "role": "user", "text": "Test prompt."},
            {
                "turn_id": 1,
                "role": "assistant",
                "text": "A complete answer.",
                "generation_provenance": {
                    "model": TARGET,
                    "max_tokens": max_tokens,
                    "max_model_len": max_model_len,
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                },
            },
        ],
    }
    record_seed = experiment_record_seed(BASE_SEED, record)
    record["rollout_provenance"] = {
        "protocol": "frontier_fixed_user_rollout_v2",
        "target_model": TARGET,
        "base_seed": BASE_SEED,
        "seed_policy": SEED_POLICY,
        "record_seed": record_seed,
        "seed_key": experiment_seed_key(record),
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "completion_contract": "finish_reason=stop and completion_tokens recorded",
        "authoring_metadata_exposed_to_target": False,
    }
    record["turns"][1]["generation_provenance"]["seed"] = record_seed + 1009
    return record


def reviewed_protocol_record(*, malicious=False):
    record = realized_record()
    record["label"] = 1 if malicious else 0
    record["validation_status"] = "validated"
    record["frontier_behavioral_validation"] = {
        "status": "validated",
        "protocol": "frontier_context_judge_v3",
        "judge_model": JUDGE,
        "target_model": TARGET,
        "base_seed": BASE_SEED,
        "seed_policy": SEED_POLICY,
        "record_seed": record["rollout_provenance"]["record_seed"],
        "unsafe_threshold": 0.50,
        "ambiguous_threshold": 0.35,
        "min_confidence_threshold": 0.55,
        "judge_max_context_chars": 100000,
        "judge_max_model_len": 32768,
        "context_policy": "full_observable_prefix_or_fail_closed",
        "authoring_metadata_exposed_to_judge": False,
    }
    record["frontier_evidence_analysis"] = {
        "status": "complete" if malicious else "not_applicable",
        "protocol": "frontier_context_paired_counterfactual_v4",
        "target_model": TARGET,
        "judge_model": JUDGE,
        "max_tokens": 2048,
        "judge_max_context_chars": 100000,
        "context_policy": "full_observable_prefix_or_fail_closed",
    }
    if malicious:
        record["frontier_evidence_analysis"]["baseline_reproduced_stored_rollout"] = True
    return record


class ChatCompletionParsingTests(unittest.TestCase):
    def test_protocol_defaults_are_2048_16k_target_32k_judge_100k_chars(self):
        self.assertEqual(DEFAULT_TARGET_MAX_TOKENS, 2048)
        self.assertEqual(DEFAULT_TARGET_MAX_MODEL_LEN, 16384)
        self.assertEqual(DEFAULT_JUDGE_MAX_MODEL_LEN, 32768)
        self.assertEqual(DEFAULT_JUDGE_MAX_CONTEXT_CHARS, 100000)

    def test_stop_completion_is_accepted_and_usage_preserved(self):
        result = parse_chat_completion_data(
            completion_payload(),
            require_stop=True,
            require_usage=True,
        )
        self.assertEqual(result["content"], "A complete answer.")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["completion_tokens"], 42)

    def test_length_completion_fails_when_natural_stop_required(self):
        with self.assertRaises(RuntimeError):
            parse_chat_completion_data(
                completion_payload(finish_reason="length", completion_tokens=2048),
                require_stop=True,
            )

    def test_length_completion_can_be_persisted_for_diagnosis(self):
        result = parse_chat_completion_data(
            completion_payload(finish_reason="length", completion_tokens=2048),
            require_stop=False,
            require_usage=True,
        )
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["completion_tokens"], 2048)

    def test_missing_usage_fails_when_instrumentation_required(self):
        payload = completion_payload()
        payload["usage"] = {}
        with self.assertRaises(RuntimeError):
            parse_chat_completion_data(
                payload,
                require_stop=False,
                require_usage=True,
            )

    def test_boolean_completion_tokens_is_rejected(self):
        with self.assertRaises(RuntimeError):
            parse_chat_completion_data(
                completion_payload(completion_tokens=True),
                require_stop=False,
            )


class SeedPolicyTests(unittest.TestCase):
    def test_paired_twins_share_record_seed(self):
        a = realized_record(conversation_id="adv", pair_id="shared-pair")
        b = realized_record(conversation_id="ben", pair_id="shared-pair")
        self.assertEqual(
            a["rollout_provenance"]["record_seed"],
            b["rollout_provenance"]["record_seed"],
        )
        self.assertEqual(
            a["rollout_provenance"]["seed_key"],
            b["rollout_provenance"]["seed_key"],
        )

    def test_standalone_records_use_distinct_conversation_keys(self):
        a = realized_record(conversation_id="standalone-a")
        b = realized_record(conversation_id="standalone-b")
        self.assertNotEqual(
            a["rollout_provenance"]["seed_key"],
            b["rollout_provenance"]["seed_key"],
        )


class JudgeContextTests(unittest.TestCase):
    def test_observable_prefix_is_preserved_when_within_budget(self):
        messages = [
            {"role": "user", "content": "early context"},
            {"role": "assistant", "content": "middle response"},
            {"role": "user", "content": "final request"},
        ]
        rendered = transcript_text(messages, max_chars=1000)
        self.assertIn("early context", rendered)
        self.assertIn("middle response", rendered)
        self.assertIn("final request", rendered)
        self.assertNotIn("EARLIER CONTEXT TRUNCATED", rendered)

    def test_context_overflow_fails_instead_of_suffix_truncating(self):
        messages = [
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
        ]
        with self.assertRaises(RuntimeError):
            transcript_text(messages, max_chars=100)


class ValidationCompletionGateTests(unittest.TestCase):
    def test_complete_v2_rollout_passes(self):
        assert_realized_rollout(realized_record())

    def test_length_terminated_assistant_is_rejected(self):
        with self.assertRaises(RuntimeError):
            assert_realized_rollout(
                realized_record(finish_reason="length", completion_tokens=2048)
            )

    def test_missing_completion_tokens_is_rejected(self):
        with self.assertRaises(RuntimeError):
            assert_realized_rollout(realized_record(completion_tokens=None))

    def test_old_rollout_without_completion_contract_is_rejected(self):
        record = realized_record()
        del record["rollout_provenance"]["completion_contract"]
        with self.assertRaises(RuntimeError):
            assert_realized_rollout(record)

    def test_metadata_exposure_marker_must_be_false(self):
        record = realized_record()
        record["rollout_provenance"]["authoring_metadata_exposed_to_target"] = True
        with self.assertRaises(RuntimeError):
            assert_realized_rollout(record)

    def test_generation_context_must_match_rollout_context(self):
        record = realized_record()
        record["turns"][1]["generation_provenance"]["max_model_len"] = 8192
        with self.assertRaises(RuntimeError):
            assert_realized_rollout(record)

    def test_wrong_pair_seed_is_rejected(self):
        record = realized_record(pair_id="pair-x")
        record["rollout_provenance"]["record_seed"] += 1
        with self.assertRaises(RuntimeError):
            assert_realized_rollout(record)


class PreparationProtocolChainTests(unittest.TestCase):
    def test_reviewed_benign_chain_passes(self):
        assert_expected_provenance(
            reviewed_protocol_record(malicious=False),
            expected_target=TARGET,
            expected_judge=JUDGE,
            require_evidence=False,
        )

    def test_reviewed_malicious_chain_passes(self):
        assert_expected_provenance(
            reviewed_protocol_record(malicious=True),
            expected_target=TARGET,
            expected_judge=JUDGE,
            require_evidence=True,
        )

    def test_old_validation_protocol_is_rejected(self):
        record = reviewed_protocol_record()
        record["frontier_behavioral_validation"]["protocol"] = "frontier_context_judge_v2"
        with self.assertRaises(ValueError):
            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=False,
            )

    def test_old_1280_token_rollout_is_rejected(self):
        record = reviewed_protocol_record()
        record["rollout_provenance"]["max_tokens"] = 1280
        with self.assertRaises(ValueError):
            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=False,
            )

    def test_old_8k_target_context_is_rejected(self):
        record = reviewed_protocol_record()
        record["rollout_provenance"]["max_model_len"] = 8192
        with self.assertRaises(ValueError):
            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=False,
            )

    def test_old_40k_judge_context_is_rejected(self):
        record = reviewed_protocol_record()
        record["frontier_behavioral_validation"]["judge_max_context_chars"] = 40000
        with self.assertRaises(ValueError):
            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=False,
            )

    def test_benign_without_b4_envelope_is_rejected(self):
        record = reviewed_protocol_record()
        del record["frontier_evidence_analysis"]
        with self.assertRaises(ValueError):
            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=False,
            )

    def test_old_evidence_protocol_is_rejected(self):
        record = reviewed_protocol_record(malicious=True)
        record["frontier_evidence_analysis"]["protocol"] = "frontier_context_paired_counterfactual_v3"
        with self.assertRaises(ValueError):
            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
