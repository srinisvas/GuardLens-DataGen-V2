#!/usr/bin/env python3
"""Stage B1: execute frontier-authored user trajectories against a local target.

Only observable user text is sent to the target. Authoring metadata, including
``intended_structure``, is preserved in the output but never enters the prompt.
Checkpoint resume is fail-closed against source/configuration drift.

A target turn is experimentally complete only when the OpenAI-compatible server
reports ``finish_reason=stop`` and usage includes ``completion_tokens``. A length-
truncated turn is persisted for diagnosis, but the record stops immediately and
cannot proceed to behavioral validation.
"""
from __future__ import annotations

import copy

from frontier_common import DEFAULT_TARGET_MAX_MODEL_LEN, DEFAULT_TARGET_MAX_TOKENS, VLLMClient, assert_frontier_source_record, config_fingerprint, json_fingerprint
from frontier_seed_policy import (
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)

PROTOCOL = "frontier_fixed_user_rollout_v3"
DEFAULT_MAX_TOKENS = DEFAULT_TARGET_MAX_TOKENS
DEFAULT_MAX_MODEL_LEN = DEFAULT_TARGET_MAX_MODEL_LEN
COMPLETION_CONTRACT = "finish_reason=stop and completion_tokens recorded"


def _base_rollout_config(*, model: str, base_seed: int, max_tokens: int, max_model_len: int):
    return {
        "protocol": PROTOCOL,
        "target_model": model,
        "base_seed": int(base_seed),
        "seed_policy": SEED_POLICY,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "max_model_len": int(max_model_len),
        "completion_contract": COMPLETION_CONTRACT,
    }


def finalize_partial_rollout(
    r,
    *,
    realized_turns,
    assistant_count: int,
    cfg,
    record_seed: int,
    seed_key: str,
    input_fp: str,
    status: str,
    error: str,
):
    r["turns"] = realized_turns
    r["assistant_turn_count"] = assistant_count
    r["conversation_length"] = len(realized_turns)
    r["source_stage"] = "local_target_rollout_incomplete"
    r["use_as"] = "not_valid_for_behavioral_validation"
    r["rollout_status"] = status
    r["rollout_error"] = error
    r["rollout_provenance"] = {
        **cfg,
        "record_seed": record_seed,
        "seed_key": seed_key,
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fingerprint(cfg),
        "authoring_metadata_exposed_to_target": False,
    }
    r.setdefault("metadata", {})["assistant_responses_present"] = assistant_count > 0
    r["pivot_turn_id"] = None
    r["pivot_kind"] = None
    r["supervision_tier"] = None
    r["loss_weight"] = None
    r["judge_confidence"] = None
    r["training_eligible"] = False
    r["validation_status"] = "unvalidated"
    return r


def rollout_record(
    record,
    client: VLLMClient,
    *,
    base_seed: int,
    max_tokens: int,
    max_model_len: int,
):
    assert_frontier_source_record(record)
    r = copy.deepcopy(record)
    cid = str(r["conversation_id"])
    record_seed = experiment_record_seed(base_seed, r)
    seed_key = experiment_seed_key(r)
    input_fp = json_fingerprint(record)
    cfg = rollout_config(
        model=client.model,
        base_seed=base_seed,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
    )

    messages = []
    realized_turns = []
    user_index = 0
    used_turn_ids = set()

    for source_turn in r.get("turns", []):
        turn = copy.deepcopy(source_turn)
        user_turn_id = int(turn["turn_id"])
        if user_turn_id in used_turn_ids:
            raise RuntimeError(f"{cid}: duplicate source turn_id={user_turn_id}")
        if user_turn_id % 2 != 0:
            raise RuntimeError(f"{cid}: source user turn_id={user_turn_id} is not even")
        used_turn_ids.add(user_turn_id)
        realized_turns.append(turn)

        text = str(turn.get("text", "")).strip()
        if not text:
            raise RuntimeError(f"{cid}: empty user text at turn_id={user_turn_id}")
        messages.append({"role": "user", "content": text})

        response_seed = record_seed + 1009 * (user_index + 1)
        result = client.chat_result(
            messages,
            seed=response_seed,
            temperature=0.0,
            max_tokens=max_tokens,
            require_stop=False,
            require_usage=False,
        )
        response = result["content"]
        finish_reason = result.get("finish_reason")
        completion_tokens = result.get("completion_tokens")

        assistant_turn_id = user_turn_id + 1
        if assistant_turn_id in used_turn_ids:
            raise RuntimeError(f"{cid}: assistant turn_id collision at {assistant_turn_id}")
        used_turn_ids.add(assistant_turn_id)
        assistant_turn = {
            "turn_id": assistant_turn_id,
            "role": "assistant",
            "text": response,
            "semantic_role": "local_target_response",
            "span_annotations": [],
            "generation_provenance": {
                "model": client.model,
                "seed": response_seed,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "max_model_len": max_model_len,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
            },
        }
        realized_turns.append(assistant_turn)
        user_index += 1

        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
        ):
            return finalize_partial_rollout(
                r,
                realized_turns=realized_turns,
                assistant_count=user_index,
                cfg=cfg,
                record_seed=record_seed,
                seed_key=seed_key,
                input_fp=input_fp,
                status="instrumentation_incomplete",
                error=(
                    f"assistant turn {assistant_turn_id} has invalid "
                    f"usage.completion_tokens={completion_tokens!r}; trajectory is not "
                    "scientifically complete"
                ),
            )
        if finish_reason != "stop":
            return finalize_partial_rollout(
                r,
                realized_turns=realized_turns,
                assistant_count=user_index,
                cfg=cfg,
                record_seed=record_seed,
                seed_key=seed_key,
                input_fp=input_fp,
                status="incomplete_generation",
                error=(
                    f"assistant turn {assistant_turn_id} ended with "
                    f"finish_reason={finish_reason!r}, completion_tokens={completion_tokens}; "
                    "later fixed user turns were not executed"
                ),
            )

        messages.append({"role": "assistant", "content": response})

    if user_index != int(r.get("user_turn_count", -1)):
        raise RuntimeError(f"{cid}: realized assistant count does not match user_turn_count")

    r["turns"] = realized_turns
    r["assistant_turn_count"] = user_index
    r["conversation_length"] = len(realized_turns)
    r["source_stage"] = "local_target_rollout"
    r["use_as"] = "input_for_independent_behavioral_validation"
    r["rollout_status"] = "complete"
    r.pop("rollout_error", None)
    r["rollout_provenance"] = {
        **cfg,
        "record_seed": record_seed,
        "seed_key": seed_key,
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fingerprint(cfg),
        "authoring_metadata_exposed_to_target": False,
    }
    r.setdefault("metadata", {})["assistant_responses_present"] = True

    r["pivot_turn_id"] = None
    r["pivot_kind"] = None
    r["supervision_tier"] = None
    r["loss_weight"] = None
    r["judge_confidence"] = None
    r["training_eligible"] = False
    r["validation_status"] = "unvalidated"
    return r


def cached_rollout_is_reusable(cached, source_record, cfg) -> bool:
    if not cached or cached.get("rollout_status") != "complete":
        return False
    provenance = cached.get("rollout_provenance", {}) or {}
    expected_seed = experiment_record_seed(cfg["base_seed"], source_record)
    expected_key = experiment_seed_key(source_record)
    if not (
        provenance.get("protocol") == PROTOCOL
        and provenance.get("seed_policy") == SEED_POLICY
        and provenance.get("completion_contract") == COMPLETION_CONTRACT
        and provenance.get("input_fingerprint") == json_fingerprint(source_record)
        and provenance.get("config_fingerprint") == config_fingerprint(cfg)
        and provenance.get("target_model") == cfg["target_model"]
        and provenance.get("max_tokens") == cfg["max_tokens"]
        and provenance.get("max_model_len") == cfg["max_model_len"]
        and provenance.get("record_seed") == expected_seed
        and provenance.get("seed_key") == expected_key
        and provenance.get("authoring_metadata_exposed_to_target") is False
    ):
        return False
    assistant_turns = [
        t for t in cached.get("turns", [])
        if str(t.get("role", "")).lower() == "assistant"
    ]
    if len(assistant_turns) != int(source_record.get("user_turn_count", -1)):
        return False
    for user_index, turn in enumerate(assistant_turns):
        generation = turn.get("generation_provenance", {}) or {}
        tokens = generation.get("completion_tokens")
        expected_response_seed = expected_seed + 1009 * (user_index + 1)
        if generation.get("finish_reason") != "stop":
            return False
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            return False
        if generation.get("max_tokens") != cfg["max_tokens"]:
            return False
        if generation.get("max_model_len") != cfg["max_model_len"]:
            return False
        if generation.get("model") != cfg["target_model"]:
            return False
        if generation.get("seed") != expected_response_seed:
            return False
    return True


from frontier_runtime_determinism import runtime_provenance

def rollout_config(**kwargs):
    cfg = _base_rollout_config(**kwargs)
    cfg.update(runtime_provenance())
    return cfg
