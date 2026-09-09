#!/usr/bin/env python3
"""CPU regression tests for the frontier generation-completion contract."""
from __future__ import annotations

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from frontier_common import (  # noqa: E402
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_TARGET_MAX_TOKENS,
    parse_chat_completion_data,
    transcript_text,
)
from validate_frontier_rollout import assert_realized_rollout  # noqa: E402


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


def realized_record(*, finish_reason="stop", completion_tokens=42, max_tokens=640):
    return {
        "conversation_id": "completion-test",
        "user_turn_count": 1,
        "assistant_turn_count": 1,
        "conversation_length": 2,
        "rollout_status": "complete",
        "rollout_provenance": {
            "protocol": "frontier_fixed_user_rollout_v2",
            "target_model": "Qwen/Qwen2.5-32B-Instruct",
            "max_tokens": max_tokens,
            "completion_contract": "finish_reason=stop and completion_tokens recorded",
            "authoring_metadata_exposed_to_target": False,
        },
        "turns": [
            {"turn_id": 0, "role": "user", "text": "Test prompt."},
            {
                "turn_id": 1,
                "role": "assistant",
                "text": "A complete answer.",
                "generation_provenance": {
                    "model": "Qwen/Qwen2.5-32B-Instruct",
                    "max_tokens": max_tokens,
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                },
            },
        ],
    }


class ChatCompletionParsingTests(unittest.TestCase):
    def test_protocol_defaults_are_640_and_40k(self):
        self.assertEqual(DEFAULT_TARGET_MAX_TOKENS, 640)
        self.assertEqual(DEFAULT_JUDGE_MAX_CONTEXT_CHARS, 40000)

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
                completion_payload(finish_reason="length", completion_tokens=640),
                require_stop=True,
            )

    def test_length_completion_can_be_persisted_for_diagnosis(self):
        result = parse_chat_completion_data(
            completion_payload(finish_reason="length", completion_tokens=640),
            require_stop=False,
            require_usage=True,
        )
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["completion_tokens"], 640)

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
                realized_record(finish_reason="length", completion_tokens=640)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
