#!/usr/bin/env python3
"""Fail-closed audit for Stage-B1 frontier target rollouts.

A scientifically usable rollout must contain every requested record and every
assistant response must terminate naturally with ``finish_reason=stop``. The
target envelope, pair-shared seed policy, completion instrumentation, and prompt
metadata-exposure marker are all rechecked before downstream use.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter, defaultdict

from frontier_common import config_fingerprint, load_jsonl, transcript_text
from frontier_seed_policy import (
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)
from rollout_frontier_source import rollout_config

ROLLOUT_PROTOCOL = "frontier_fixed_user_rollout_v2"
COMPLETION_CONTRACT = "finish_reason=stop and completion_tokens recorded"


def percentile(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument("--expected-model", default=None)
    parser.add_argument("--expected-max-tokens", type=int, default=0)
    parser.add_argument("--expected-max-model-len", type=int, default=0)
    parser.add_argument("--max-transcript-chars", type=int, default=0)
    parser.add_argument("--near-cap-fraction", type=float, default=0.90)
    args = parser.parse_args()

    if args.expected_records < 0:
        raise ValueError("expected-records must be nonnegative")
    if args.expected_max_tokens < 0:
        raise ValueError("expected-max-tokens must be nonnegative")
    if args.expected_max_model_len < 0:
        raise ValueError("expected-max-model-len must be nonnegative")
    if args.max_transcript_chars < 0:
        raise ValueError("max-transcript-chars must be nonnegative")
    if not (0.0 < args.near_cap_fraction <= 1.0):
        raise ValueError("near-cap-fraction must be in (0,1]")

    records = load_jsonl(args.input)
    errors = []
    warnings = []
    seen = set()
    statuses = Counter()
    finish_reasons = Counter()
    completion_tokens = []
    near_cap = []
    transcript_lengths = []
    pair_seeds = defaultdict(set)

    if args.expected_records and len(records) != args.expected_records:
        errors.append(f"expected {args.expected_records} records, got {len(records)}")

    for record in records:
        cid = str(record.get("conversation_id", ""))
        if not cid:
            errors.append("record missing conversation_id")
            continue
        if cid in seen:
            errors.append(f"duplicate conversation_id: {cid}")
        seen.add(cid)

        status = str(record.get("rollout_status", "missing"))
        statuses[status] += 1
        if status != "complete":
            errors.append(
                f"{cid}: rollout_status={status!r}; error={record.get('rollout_error')!r}"
            )
            continue

        rollout = record.get("rollout_provenance", {}) or {}
        if rollout.get("protocol") != ROLLOUT_PROTOCOL:
            errors.append(
                f"{cid}: rollout protocol={rollout.get('protocol')!r} != {ROLLOUT_PROTOCOL!r}"
            )
        target_model = rollout.get("target_model")
        if args.expected_model and target_model != args.expected_model:
            errors.append(
                f"{cid}: target_model={target_model!r} != expected {args.expected_model!r}"
            )
        if rollout.get("completion_contract") != COMPLETION_CONTRACT:
            errors.append(f"{cid}: missing/invalid completion contract")
        if rollout.get("seed_policy") != SEED_POLICY:
            errors.append(f"{cid}: seed_policy={rollout.get('seed_policy')!r} != {SEED_POLICY!r}")
        if rollout.get("authoring_metadata_exposed_to_target") is not False:
            errors.append(f"{cid}: target metadata-exposure marker is not false")
        if float(rollout.get("temperature", -1.0)) != 0.0:
            errors.append(f"{cid}: rollout temperature is not 0.0")

        base_seed = rollout.get("base_seed")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            errors.append(f"{cid}: missing/invalid base_seed={base_seed!r}")
            base_seed = None
        record_seed = rollout.get("record_seed")
        if isinstance(record_seed, bool) or not isinstance(record_seed, int):
            errors.append(f"{cid}: missing/invalid record_seed={record_seed!r}")
            record_seed = None
        if base_seed is not None and record_seed is not None:
            expected_record_seed = experiment_record_seed(base_seed, record)
            expected_seed_key = experiment_seed_key(record)
            if record_seed != expected_record_seed:
                errors.append(
                    f"{cid}: record_seed={record_seed} != locked-policy seed {expected_record_seed}"
                )
            if rollout.get("seed_key") != expected_seed_key:
                errors.append(
                    f"{cid}: seed_key={rollout.get('seed_key')!r} != {expected_seed_key!r}"
                )
            pair_id = record.get("pair_id")
            if pair_id not in (None, ""):
                pair_seeds[str(pair_id)].add(record_seed)

        if not isinstance(rollout.get("input_fingerprint"), str):
            errors.append(f"{cid}: missing input_fingerprint")
        if not isinstance(rollout.get("config_fingerprint"), str):
            errors.append(f"{cid}: missing config_fingerprint")

        max_tokens = rollout.get("max_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            errors.append(f"{cid}: invalid rollout max_tokens={max_tokens!r}")
            continue
        if args.expected_max_tokens and max_tokens != args.expected_max_tokens:
            errors.append(
                f"{cid}: rollout max_tokens={max_tokens} != expected {args.expected_max_tokens}"
            )

        max_model_len = rollout.get("max_model_len")
        if (
            isinstance(max_model_len, bool)
            or not isinstance(max_model_len, int)
            or max_model_len <= 0
        ):
            errors.append(f"{cid}: invalid rollout max_model_len={max_model_len!r}")
            continue
        if args.expected_max_model_len and max_model_len != args.expected_max_model_len:
            errors.append(
                f"{cid}: rollout max_model_len={max_model_len} != expected "
                f"{args.expected_max_model_len}"
            )

        if base_seed is not None and target_model:
            expected_cfg = rollout_config(
                model=str(target_model),
                base_seed=base_seed,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
            )
            if rollout.get("config_fingerprint") != config_fingerprint(expected_cfg):
                errors.append(f"{cid}: rollout config fingerprint mismatch")

        turns = record.get("turns", [])
        if len(turns) != int(record.get("conversation_length", -1)):
            errors.append(f"{cid}: conversation_length does not match physical turns")
        ids = [int(t.get("turn_id", -1)) for t in turns]
        if ids != list(range(len(turns))):
            errors.append(f"{cid}: realized turn IDs are not contiguous 0..N-1")

        for idx, turn in enumerate(turns):
            expected_role = "user" if idx % 2 == 0 else "assistant"
            if str(turn.get("role", "")).lower() != expected_role:
                errors.append(
                    f"{cid}: turn {idx} role={turn.get('role')!r}, expected {expected_role!r}"
                )

        users = [t for t in turns if str(t.get("role", "")).lower() == "user"]
        assistants = [t for t in turns if str(t.get("role", "")).lower() == "assistant"]
        if not (
            len(users)
            == len(assistants)
            == int(record.get("user_turn_count", -1))
            == int(record.get("assistant_turn_count", -1))
        ):
            errors.append(f"{cid}: realized user/assistant counts are inconsistent")

        observable = [
            {
                "role": str(t.get("role", "")).lower(),
                "content": str(t.get("text", "")),
            }
            for t in turns
            if str(t.get("role", "")).lower() in {"user", "assistant"}
        ]
        if args.max_transcript_chars:
            try:
                rendered = transcript_text(observable, max_chars=args.max_transcript_chars)
                transcript_lengths.append(len(rendered))
            except Exception as exc:
                errors.append(f"{cid}: judge transcript budget failure: {exc}")

        for assistant_index, turn in enumerate(assistants):
            tid = int(turn.get("turn_id", -1))
            generation = turn.get("generation_provenance", {}) or {}
            finish = generation.get("finish_reason")
            finish_reasons[str(finish)] += 1
            if finish != "stop":
                errors.append(f"{cid}: assistant turn {tid} finish_reason={finish!r}")
            if generation.get("model") != target_model:
                errors.append(f"{cid}: assistant turn {tid} model provenance mismatch")
            if generation.get("max_tokens") != max_tokens:
                errors.append(f"{cid}: assistant turn {tid} max_tokens provenance mismatch")
            if generation.get("max_model_len") != max_model_len:
                errors.append(f"{cid}: assistant turn {tid} max_model_len provenance mismatch")
            if float(generation.get("temperature", -1.0)) != 0.0:
                errors.append(f"{cid}: assistant turn {tid} temperature is not 0.0")
            if record_seed is not None:
                expected_seed = record_seed + 1009 * (assistant_index + 1)
                if generation.get("seed") != expected_seed:
                    errors.append(
                        f"{cid}: assistant turn {tid} seed={generation.get('seed')!r} "
                        f"!= expected {expected_seed}"
                    )

            tokens = generation.get("completion_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                errors.append(
                    f"{cid}: assistant turn {tid} invalid completion_tokens={tokens!r}"
                )
                continue
            if tokens > max_tokens:
                errors.append(
                    f"{cid}: assistant turn {tid} completion_tokens={tokens} exceeds max_tokens={max_tokens}"
                )
            completion_tokens.append(tokens)
            if tokens >= args.near_cap_fraction * max_tokens:
                near_cap.append((cid, tid, tokens, max_tokens))

    for pair_id, seeds in pair_seeds.items():
        if len(seeds) != 1:
            errors.append(
                f"pair_id={pair_id}: paired records do not share one record seed: {sorted(seeds)}"
            )

    print("=== Frontier rollout completion audit ===")
    print(f"Records: {len(records)}")
    print(f"Rollout statuses: {dict(statuses)}")
    print(f"Assistant finish reasons: {dict(finish_reasons)}")
    print(f"Seed policy: {SEED_POLICY}; paired groups checked={len(pair_seeds)}")
    if completion_tokens:
        print(
            "Completion tokens: "
            f"n={len(completion_tokens)} min={min(completion_tokens)} "
            f"median={statistics.median(completion_tokens):.1f} "
            f"p95={percentile(completion_tokens, 0.95):.1f} "
            f"max={max(completion_tokens)} "
            f"mean={statistics.mean(completion_tokens):.1f}"
        )
    if args.max_transcript_chars and transcript_lengths:
        print(
            "Full observable transcript chars: "
            f"max={max(transcript_lengths)} "
            f"p95={percentile(transcript_lengths, 0.95):.1f} "
            f"budget={args.max_transcript_chars}"
        )
    print(
        f"Near-cap assistant turns (>= {args.near_cap_fraction:.0%} of max_tokens): "
        f"{len(near_cap)}"
    )
    if near_cap:
        for cid, tid, tokens, max_tokens in near_cap[:20]:
            warnings.append(f"{cid} turn {tid}: completion_tokens={tokens}/{max_tokens}")
        for warning in warnings:
            print(f"WARN: {warning}")

    if errors:
        print("ROLLOUT AUDIT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors", file=sys.stderr)
        sys.exit(2)
    print("ROLLOUT AUDIT PASSED")


if __name__ == "__main__":
    main()
