#!/usr/bin/env python3
"""CPU regression tests for the frontier generation-completion contract."""
from __future__ import annotations

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from frontier_common import parse_chat_completion_data  # noqa: E402
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
            "target_model": "Qwen/Qwen2.5-32B-Instruct",
            "max_tokens": max_tokens,
            "completion_contract": "finish_reason=stop and completion_tokens recorded",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
