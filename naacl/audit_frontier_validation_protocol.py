#!/usr/bin/env python3
"""Fail-closed audit of Stage-B2 validation protocol/provenance."""
from __future__ import annotations

import argparse
from collections import Counter

from frontier_common import config_fingerprint, load_jsonl
from validate_frontier_rollout import (
    PROTOCOL,
    TERMINAL_STATUSES,
    validation_config,
)

CONTEXT_POLICY = "full_observable_prefix_or_fail_closed"


def audit_record(
    record,
    *,
    target_model: str,
    judge_model: str,
    judge_max_model_len: int,
    judge_max_context_chars: int,
):
    cid = str(record.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("validation record missing conversation_id")
    status = str(record.get("validation_status", ""))
    if status not in TERMINAL_STATUSES:
        raise RuntimeError(f"{cid}: validation_status={status!r} is not terminal")

    rollout = record.get("rollout_provenance", {}) or {}
    validation = record.get("frontier_behavioral_validation", {}) or {}
    if rollout.get("target_model") != target_model:
        raise RuntimeError(f"{cid}: rollout target model mismatch")
    if validation.get("protocol") != PROTOCOL:
        raise RuntimeError(f"{cid}: validation protocol mismatch")
    if validation.get("target_model") != target_model:
        raise RuntimeError(f"{cid}: validation target model mismatch")
    if validation.get("judge_model") != judge_model:
        raise RuntimeError(f"{cid}: validation judge model mismatch")
    if int(validation.get("judge_max_model_len", -1)) != int(judge_max_model_len):
        raise RuntimeError(f"{cid}: judge runtime context mismatch")
    if int(validation.get("judge_max_context_chars", -1)) != int(judge_max_context_chars):
        raise RuntimeError(f"{cid}: judge observable-prefix budget mismatch")
    if validation.get("context_policy") != CONTEXT_POLICY:
        raise RuntimeError(f"{cid}: judge context policy mismatch")
    if validation.get("authoring_metadata_exposed_to_judge") is not False:
        raise RuntimeError(f"{cid}: judge metadata exposure marker is not false")

    # Reconstruct exactly the B2 configuration dictionary whose fingerprint was
    # written by validate_frontier_rollout.py. This catches stale/mutated
    # thresholds, seeds, context windows, or judge identity before B4.
    cfg = validation_config(
        judge_model=judge_model,
        base_seed=int(validation.get("base_seed", -1)),
        unsafe_threshold=float(validation.get("unsafe_threshold", -1)),
        ambiguous_threshold=float(validation.get("ambiguous_threshold", -1)),
        min_confidence=float(validation.get("min_confidence_threshold", -1)),
        judge_max_context_chars=judge_max_context_chars,
        judge_max_model_len=judge_max_model_len,
    )
    if validation.get("config_fingerprint") != config_fingerprint(cfg):
        raise RuntimeError(f"{cid}: validation config fingerprint mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument(
        "--target-model", default="Qwen/Qwen2.5-32B-Instruct"
    )
    parser.add_argument(
        "--judge-model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    parser.add_argument("--judge-max-model-len", type=int, default=16384)
    parser.add_argument("--judge-max-context-chars", type=int, default=40000)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.expected_records and len(records) != args.expected_records:
        raise RuntimeError(
            f"record count={len(records)} expected={args.expected_records}"
        )
    ids = [str(r.get("conversation_id", "")) for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate conversation_id values in validation file")

    statuses = Counter()
    for record in records:
        audit_record(
            record,
            target_model=args.target_model,
            judge_model=args.judge_model,
            judge_max_model_len=args.judge_max_model_len,
            judge_max_context_chars=args.judge_max_context_chars,
        )
        statuses[str(record.get("validation_status"))] += 1

    print("=== Frontier B2 protocol audit ===")
    print(f"Records: {len(records)}")
    print(f"Validation statuses: {dict(statuses)}")
    print(f"Target: {args.target_model}")
    print(f"Judge: {args.judge_model}")
    print(
        f"Judge context: max_model_len={args.judge_max_model_len} "
        f"max_context_chars={args.judge_max_context_chars}"
    )
    print("B2 PROTOCOL AUDIT PASSED")


if __name__ == "__main__":
    main()
