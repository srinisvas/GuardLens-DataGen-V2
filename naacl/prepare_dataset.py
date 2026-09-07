#!/usr/bin/env python3
"""Prepare the repaired GuardLens dataset for NAACL training/evaluation.

The goal is to remove two avoidable shortcuts without rebuilding the attacks:

1. Unassessed construction-only spans must not become positive attribution
   targets through the legacy label-name fallback in the Transformer loader.
2. Benign trajectories are deterministically prefix-trimmed so their user-turn
   count follows the empirical malicious distribution instead of carrying a
   strong length cue.

The original full-length benign pool is written separately as a stress set so
we can report that length matching did not merely hide the old distribution.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Optional


SUPPORTED = {"supported_strong", "supported_weak"}
NEGATIVE_SUPPORTED = {"negative_control_supported"}


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def n_user_turns(record: Dict) -> int:
    return sum(
        1 for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    )


def truncate_to_user_turns(record: Dict, target_user_turns: int) -> Dict:
    """Keep a benign prefix containing target_user_turns user messages.

    If an assistant response immediately follows the final retained user turn,
    it is retained too. We stop before the next user turn.
    """
    r = copy.deepcopy(record)
    original = list(r.get("turns", []))
    if target_user_turns <= 0 or not original:
        return r

    kept: List[Dict] = []
    seen_users = 0
    reached = False

    for turn in original:
        role = str(turn.get("role", "")).lower()
        if role == "user":
            if reached:
                break
            seen_users += 1
            kept.append(turn)
            if seen_users >= target_user_turns:
                reached = True
        else:
            if not reached or kept:
                kept.append(turn)

    # Remove any extra assistant chain after the retained user response. One
    # immediate assistant reply is sufficient and avoids creating a new length
    # cue from trailing assistant-only turns.
    if reached:
        last_user_index = max(
            (i for i, t in enumerate(kept)
             if str(t.get("role", "")).lower() == "user"),
            default=len(kept) - 1,
        )
        tail = kept[last_user_index + 1:]
        first_assistant = tail[:1]
        kept = kept[:last_user_index + 1] + first_assistant

    r["turns"] = kept
    r["conversation_length"] = len(kept)
    r.setdefault("metadata", {})["naacl_length_match"] = {
        "original_user_turns": n_user_turns(record),
        "target_user_turns": target_user_turns,
        "final_user_turns": n_user_turns(r),
        "method": "benign_prefix_trim_to_malicious_empirical_distribution",
    }
    return r


def choose_target_length(rng: random.Random, malicious_lengths: List[int], benign_len: int) -> int:
    feasible = [x for x in malicious_lengths if x <= benign_len]
    if feasible:
        return rng.choice(feasible)
    return min(benign_len, min(malicious_lengths)) if malicious_lengths else benign_len


def sanitize_attribution_targets(record: Dict) -> Dict:
    """Make only tested evidence visible to the legacy attribution loader."""
    r = copy.deepcopy(record)

    for turn in r.get("turns", []):
        for span in turn.get("span_annotations", []):
            status = span.get("evidence_status", "unassessed")
            if status in SUPPORTED:
                # Keep original semantic label and legacy compatibility fields.
                continue
            if status in NEGATIVE_SUPPORTED:
                span["causal_type"] = "incidental"
                span["supervision_tier"] = "incidental"
                continue

            original_label = span.get("label", "")
            span.setdefault("original_label", original_label)
            span["label"] = "EVIDENCE_CANDIDATE"
            span["causal_type"] = "unvalidated"
            span["supervision_tier"] = "ignore"
            # Important: null means not established, not a measured zero effect.
            if status == "unassessed":
                span["counterfactual_delta"] = None

    # Use evidence-bearing turns for the existing pivot head. Preserve the old
    # value explicitly for reproducibility.
    r.setdefault("legacy_pre_naacl_pivot_turn_id", r.get("pivot_turn_id"))
    evidence_turns = list(r.get("evidence_turn_ids", []))
    r["pivot_turn_id"] = evidence_turns[0] if evidence_turns else None
    if not evidence_turns:
        r["pivot_kind"] = "none"
    elif len(evidence_turns) > 1:
        r["pivot_kind"] = "distributed"

    return r


def describe(name: str, records: List[Dict]) -> Dict:
    user_lengths = [n_user_turns(r) for r in records]
    total_lengths = [len(r.get("turns", [])) for r in records]
    labels = Counter(r.get("label", -1) for r in records)
    out = {
        "name": name,
        "n": len(records),
        "labels": dict(labels),
        "user_turns": {
            "mean": statistics.mean(user_lengths) if user_lengths else 0.0,
            "median": statistics.median(user_lengths) if user_lengths else 0.0,
            "min": min(user_lengths) if user_lengths else 0,
            "max": max(user_lengths) if user_lengths else 0,
            "histogram": dict(Counter(user_lengths)),
        },
        "total_turns": {
            "mean": statistics.mean(total_lengths) if total_lengths else 0.0,
            "median": statistics.median(total_lengths) if total_lengths else 0.0,
            "min": min(total_lengths) if total_lengths else 0,
            "max": max(total_lengths) if total_lengths else 0,
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-input", required=True,
                        help="Output of naacl/evidence_analysis.py")
    parser.add_argument("--benign-input", required=True,
                        help="Clean benign pool validated by both model families")
    parser.add_argument("--output", required=True,
                        help="Combined repaired dataset before train/dev/test split")
    parser.add_argument("--benign-stress-output", required=True,
                        help="Untrimmed benign pool retained for length-stress evaluation")
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evidence_records = load_jsonl(args.evidence_input)
    malicious = [
        sanitize_attribution_targets(r)
        for r in evidence_records
        if r.get("label") == 1 and r.get("validation_status") == "validated"
    ]
    benign_original = [
        copy.deepcopy(r)
        for r in load_jsonl(args.benign_input)
        if r.get("label") == 0 and r.get("validation_status", "validated") == "validated"
    ]

    if not malicious:
        raise RuntimeError("No validated malicious records were found")
    if not benign_original:
        raise RuntimeError("No validated benign records were found")

    malicious_lengths = [n_user_turns(r) for r in malicious]
    rng = random.Random(args.seed)
    benign_matched: List[Dict] = []

    for record in benign_original:
        original_len = n_user_turns(record)
        target = choose_target_length(rng, malicious_lengths, original_len)
        matched = truncate_to_user_turns(record, target)
        matched["benign_status"] = "clean_benign_length_matched"
        matched["source_dataset"] = matched.get("source_dataset", "separate_benign_pool")
        matched["supervision_tier"] = "benign_validated"
        matched["loss_weight"] = 1.0
        matched["training_eligible"] = True
        matched["pivot_turn_id"] = None
        matched["pivot_kind"] = "none"
        benign_matched.append(matched)

    combined = malicious + benign_matched
    rng.shuffle(combined)

    write_jsonl(combined, args.output)
    write_jsonl(benign_original, args.benign_stress_output)

    stats = {
        "malicious": describe("malicious", malicious),
        "benign_original": describe("benign_original", benign_original),
        "benign_length_matched": describe("benign_length_matched", benign_matched),
        "combined": describe("combined", combined),
        "method": {
            "seed": args.seed,
            "attribution_policy": "counterfactual-supported spans only; unassessed candidates ignored",
            "length_policy": "benign prefixes sampled from empirical malicious user-turn distribution",
        },
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2))
    print(f"Wrote combined dataset: {args.output}")
    print(f"Wrote full benign stress set: {args.benign_stress_output}")
    print(f"Wrote stats: {args.stats_output}")


if __name__ == "__main__":
    main()
