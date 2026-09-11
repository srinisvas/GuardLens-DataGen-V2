#!/usr/bin/env python3
"""Fail-closed post-merge audit for production Stage-B4 evidence artifacts."""
from __future__ import annotations

import argparse
from collections import Counter

import frontier_evidence_analysis as fea
from audit_frontier_validation_protocol import audit_record as audit_b2_record
from frontier_common import (
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_JUDGE_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_TOKENS,
    config_fingerprint,
    load_jsonl,
)
from frontier_evidence_fast import (
    EXECUTION_OPTIMIZATION,
    RECORD_SEED_SOURCE,
)
from frontier_seed_policy import SEED_POLICY

ALLOWED_EVIDENCE_STATUSES = {"complete", "baseline_not_unsafe", "not_applicable"}
CONTEXT_POLICY = "full_observable_prefix_or_fail_closed"


def reconstructed_config(analysis, *, target_max_model_len: int, judge_max_model_len: int):
    cfg = fea.evidence_config(
        target_model=str(analysis.get("target_model", "")),
        judge_model=str(analysis.get("judge_model", "")),
        base_seed=int(analysis.get("base_seed", -1)),
        unsafe_threshold=float(analysis.get("unsafe_threshold", -1)),
        min_confidence=float(analysis.get("min_confidence_threshold", -1)),
        weak_threshold=float(analysis.get("weak_delta_threshold", -1)),
        strong_threshold=float(analysis.get("strong_delta_threshold", -1)),
        negative_control_tolerance=float(
            analysis.get("negative_control_tolerance", -1)
        ),
        max_turn_interventions=int(analysis.get("max_turn_interventions", -1)),
        max_positive_spans=int(analysis.get("max_positive_spans", -1)),
        max_negative_spans=int(analysis.get("max_negative_spans", -1)),
        max_tokens=int(analysis.get("max_tokens", -1)),
        judge_max_context_chars=int(analysis.get("judge_max_context_chars", -1)),
    )
    cfg.update(
        {
            "seed_policy": SEED_POLICY,
            "record_seed_source": RECORD_SEED_SOURCE,
            "target_max_model_len": int(target_max_model_len),
            "judge_max_model_len": int(judge_max_model_len),
            "execution_optimization": EXECUTION_OPTIMIZATION,
        }
    )
    return cfg


def audit_record(
    record,
    *,
    target_model: str,
    judge_model: str,
    max_tokens: int,
    target_max_model_len: int,
    judge_max_model_len: int,
    judge_max_context_chars: int,
):
    cid = str(record.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("evidence record missing conversation_id")

    audit_b2_record(
        record,
        target_model=target_model,
        judge_model=judge_model,
        judge_max_model_len=judge_max_model_len,
        judge_max_context_chars=judge_max_context_chars,
    )

    analysis = record.get("frontier_evidence_analysis", {}) or {}
    status = str(analysis.get("status", ""))
    if status not in ALLOWED_EVIDENCE_STATUSES:
        raise RuntimeError(f"{cid}: non-terminal evidence status={status!r}")
    if analysis.get("protocol") != fea.PROTOCOL:
        raise RuntimeError(f"{cid}: evidence protocol mismatch")
    if analysis.get("target_model") != target_model:
        raise RuntimeError(f"{cid}: evidence target model mismatch")
    if analysis.get("judge_model") != judge_model:
        raise RuntimeError(f"{cid}: evidence judge model mismatch")
    if int(analysis.get("max_tokens", -1)) != int(max_tokens):
        raise RuntimeError(f"{cid}: evidence max_tokens mismatch")
    if int(analysis.get("target_max_model_len", -1)) != int(target_max_model_len):
        raise RuntimeError(f"{cid}: evidence target runtime context mismatch")
    if int(analysis.get("judge_max_model_len", -1)) != int(judge_max_model_len):
        raise RuntimeError(f"{cid}: evidence judge runtime context mismatch")
    if int(analysis.get("judge_max_context_chars", -1)) != int(judge_max_context_chars):
        raise RuntimeError(f"{cid}: evidence judge prefix budget mismatch")
    if analysis.get("context_policy") != CONTEXT_POLICY:
        raise RuntimeError(f"{cid}: evidence context policy mismatch")
    if analysis.get("seed_policy") != SEED_POLICY:
        raise RuntimeError(f"{cid}: evidence seed policy mismatch")
    if analysis.get("record_seed_source") != RECORD_SEED_SOURCE:
        raise RuntimeError(f"{cid}: evidence record-seed source mismatch")
    if analysis.get("execution_optimization") != EXECUTION_OPTIMIZATION:
        raise RuntimeError(f"{cid}: evidence execution optimization mismatch")

    cfg = reconstructed_config(
        analysis,
        target_max_model_len=target_max_model_len,
        judge_max_model_len=judge_max_model_len,
    )
    if analysis.get("config_fingerprint") != config_fingerprint(cfg):
        raise RuntimeError(f"{cid}: evidence config fingerprint mismatch")

    label = int(record.get("label", -1))
    validation_status = str(record.get("validation_status", ""))
    should_run_evidence = label == 1 and validation_status == "validated"
    if should_run_evidence:
        if status not in {"complete", "baseline_not_unsafe"}:
            raise RuntimeError(
                f"{cid}: validated malicious record has evidence status={status!r}"
            )
        if analysis.get("baseline_reproduced_stored_rollout") is not True:
            raise RuntimeError(f"{cid}: fresh B4 baseline was not proven reproducible")
    elif status != "not_applicable":
        raise RuntimeError(
            f"{cid}: non-evidence record has unexpected evidence status={status!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument(
        "--judge-model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_TARGET_MAX_TOKENS)
    parser.add_argument(
        "--target-max-model-len", type=int, default=DEFAULT_TARGET_MAX_MODEL_LEN
    )
    parser.add_argument(
        "--judge-max-model-len", type=int, default=DEFAULT_JUDGE_MAX_MODEL_LEN
    )
    parser.add_argument(
        "--judge-max-context-chars", type=int, default=DEFAULT_JUDGE_MAX_CONTEXT_CHARS
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.expected_records and len(records) != args.expected_records:
        raise RuntimeError(
            f"record count={len(records)} expected={args.expected_records}"
        )
    ids = [str(r.get("conversation_id", "")) for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate conversation_id values in evidence file")

    statuses = Counter()
    for record in records:
        audit_record(
            record,
            target_model=args.target_model,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            target_max_model_len=args.target_max_model_len,
            judge_max_model_len=args.judge_max_model_len,
            judge_max_context_chars=args.judge_max_context_chars,
        )
        statuses[str((record.get("frontier_evidence_analysis", {}) or {}).get("status"))] += 1

    print("=== Frontier B4 protocol audit ===")
    print(f"Records: {len(records)}")
    print(f"Evidence statuses: {dict(statuses)}")
    print(f"Seed policy: {SEED_POLICY}")
    print(f"Execution optimization: {EXECUTION_OPTIMIZATION}")
    print(
        f"Runtime: target_tokens={args.max_tokens} "
        f"target_context={args.target_max_model_len} "
        f"judge_context={args.judge_max_model_len} "
        f"judge_prefix_chars={args.judge_max_context_chars}"
    )
    print("B4 PROTOCOL AUDIT PASSED")


if __name__ == "__main__":
    main()
