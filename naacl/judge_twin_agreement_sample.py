#!/usr/bin/env python3
"""Judge a deterministic Dataset A twin subset for cross-judge agreement.

Uses the same full-prefix bridge protocol as judge_stored_twin_responses.py but
writes only compact results for a deterministic subset. Intended for an
apples-to-apples 7B-vs-24B judge agreement audit on the same stored responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List

from frontier_common import (
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_JUDGE_MAX_MODEL_LEN,
    VLLMClient,
)
from judge_stored_twin_responses import (
    BRIDGE_PROTOCOL,
    DEFAULT_AMBIGUOUS_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_UNSAFE_THRESHOLD,
    is_original_benign_twin,
    judge_stored_trajectory,
    load_jsonl,
    turn_text_hash,
    write_jsonl,
)


def sample_key(seed: int, record: Dict) -> str:
    raw = f"{seed}:{record.get('pair_id','')}:{record.get('conversation_id','')}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def infer_target_models(records: List[Dict]) -> Dict[str, str]:
    out = {}
    for record in records:
        if record.get("label") != 1:
            continue
        pair_id = str(record.get("pair_id", ""))
        validation = record.get("llama_validation", {}) or {}
        model = validation.get("model_used") or validation.get("target_model")
        if pair_id and model:
            out[pair_id] = str(model)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--sample-seed", type=int, default=20260920)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--unsafe-threshold", type=float, default=DEFAULT_UNSAFE_THRESHOLD)
    parser.add_argument("--ambiguous-threshold", type=float, default=DEFAULT_AMBIGUOUS_THRESHOLD)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        "--judge-max-context-chars",
        type=int,
        default=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    )
    parser.add_argument(
        "--judge-max-model-len",
        type=int,
        default=DEFAULT_JUDGE_MAX_MODEL_LEN,
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    targets = infer_target_models(records)
    twins = [r for r in records if is_original_benign_twin(r)]
    twins.sort(key=lambda r: sample_key(args.sample_seed, r))
    selected = twins[: min(args.sample_size, len(twins))]
    if not selected:
        raise RuntimeError("no original benign twins available for agreement audit")

    judge = VLLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    output = []
    for index, record in enumerate(selected, 1):
        pair_id = str(record.get("pair_id", ""))
        target_model = targets.get(pair_id)
        if not target_model:
            raise RuntimeError(
                f"{record.get('conversation_id')}: missing target-model provenance"
            )
        validation = judge_stored_trajectory(
            record,
            judge,
            target_model,
            base_seed=args.judge_seed,
            unsafe_threshold=args.unsafe_threshold,
            ambiguous_threshold=args.ambiguous_threshold,
            min_confidence=args.min_confidence,
            judge_max_context_chars=args.judge_max_context_chars,
            judge_max_model_len=args.judge_max_model_len,
            judge_model_revision=args.model_revision,
        )
        output.append({
            "conversation_id": record.get("conversation_id"),
            "pair_id": pair_id,
            "source_turn_text_sha256": turn_text_hash(record),
            "sample_seed": args.sample_seed,
            "sample_size_requested": args.sample_size,
            "judge_model": args.model,
            "judge_model_revision": args.model_revision,
            "bridge_protocol": BRIDGE_PROTOCOL,
            "validation": validation,
        })
        if index % 10 == 0:
            print(f"Agreement sample judged: {index}/{len(selected)}")

    write_jsonl(output, args.output)
    print(json.dumps({
        "records": len(output),
        "judge_model": args.model,
        "judge_model_revision": args.model_revision,
        "sample_seed": args.sample_seed,
        "sample_size_requested": args.sample_size,
        "bridge_protocol": BRIDGE_PROTOCOL,
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
