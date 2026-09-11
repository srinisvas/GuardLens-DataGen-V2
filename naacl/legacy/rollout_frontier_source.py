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

import argparse
import copy
import json
import os
import time
from collections import Counter

from frontier_common import (
    DEFAULT_TARGET_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_TOKENS,
    VLLMClient,
    assert_frontier_source_record,
    config_fingerprint,
    json_fingerprint,
    load_completed,
    load_jsonl,
    write_jsonl,
)
from frontier_seed_policy import (
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)

PROTOCOL = "frontier_fixed_user_rollout_v2"
DEFAULT_MAX_TOKENS = DEFAULT_TARGET_MAX_TOKENS
DEFAULT_MAX_MODEL_LEN = DEFAULT_TARGET_MAX_MODEL_LEN
COMPLETION_CONTRACT = "finish_reason=stop and completion_tokens recorded"


def rollout_config(*, model: str, base_seed: int, max_tokens: int, max_model_len: int):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.max_model_len <= 0:
        raise ValueError("max-model-len must be positive")

    client = VLLMClient(args.model, args.base_url, args.api_key)
    if not client.health_check():
        raise RuntimeError(f"target vLLM server not ready at {args.base_url}")

    all_records = load_jsonl(args.input)
    ids = [str(r.get("conversation_id", "")) for r in all_records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("input contains duplicate conversation_id values")
    records = [r for i, r in enumerate(all_records) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        records = records[: args.limit]
    checkpoint = args.checkpoint or args.output + ".checkpoint.jsonl"
    completed = load_completed(checkpoint)
    cfg = rollout_config(
        model=args.model,
        base_seed=args.seed,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
    )

    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
    processed = 0
    reused = 0
    started = time.time()
    with open(checkpoint, "a", encoding="utf-8") as handle:
        for record in records:
            cid = str(record.get("conversation_id", ""))
            cached = completed.get(cid)
            if cached_rollout_is_reusable(cached, record, cfg):
                reused += 1
                continue
            try:
                out = rollout_record(
                    record,
                    client,
                    base_seed=args.seed,
                    max_tokens=args.max_tokens,
                    max_model_len=args.max_model_len,
                )
            except Exception as exc:
                out = copy.deepcopy(record)
                out["rollout_status"] = "error"
                out["rollout_error"] = repr(exc)
                out["training_eligible"] = False
                out["rollout_provenance"] = {
                    **cfg,
                    "record_seed": experiment_record_seed(args.seed, record),
                    "seed_key": experiment_seed_key(record),
                    "input_fingerprint": json_fingerprint(record),
                    "config_fingerprint": config_fingerprint(cfg),
                    "authoring_metadata_exposed_to_target": False,
                }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            completed[cid] = out
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} new records in {(time.time()-started)/60:.1f} min")

    ordered = [completed.get(str(r.get("conversation_id", "")), r) for r in records]
    write_jsonl(ordered, args.output)
    statuses = Counter(r.get("rollout_status", "missing") for r in ordered)
    finish_reasons = Counter()
    completion_tokens = []
    for record in ordered:
        for turn in record.get("turns", []):
            if str(turn.get("role", "")).lower() != "assistant":
                continue
            provenance = turn.get("generation_provenance", {}) or {}
            finish_reasons[str(provenance.get("finish_reason", "missing"))] += 1
            value = provenance.get("completion_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                completion_tokens.append(value)

    print(f"Shard {args.shard_index}/{args.num_shards}: {len(ordered)} records")
    print(f"Checkpoint records reused: {reused}")
    print(f"Rollout statuses: {dict(statuses)}")
    print(f"Assistant finish reasons: {dict(finish_reasons)}")
    if completion_tokens:
        print(
            "Completion tokens: "
            f"min={min(completion_tokens)} max={max(completion_tokens)} "
            f"mean={sum(completion_tokens)/len(completion_tokens):.1f}"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
