#!/usr/bin/env python3
"""Prepare the repaired GuardLens dataset for NAACL training/evaluation.

The preparation step is fail-closed. It recomputes sample-level supervision
from repaired evidence statuses instead of trusting any legacy v11 tier and it
prevents benign records from leaking positive attribution labels through the
legacy Transformer label-name fallback.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
from collections import Counter
from typing import Dict, Iterable, List


SUPPORTED_STRONG = "supported_strong"
SUPPORTED_WEAK = "supported_weak"
SUPPORTED = {SUPPORTED_STRONG, SUPPORTED_WEAK}
NEGATIVE_SUPPORTED = {"negative_control_supported"}
DETECTION_TIERS = {"transfer_success", "target_only", "cross_only"}
LOSS_WEIGHTS = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "benign_validated": 1.00,
    "ignore": 0.00,
}


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
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
    retain only that first assistant response and stop before the next user.
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

    if reached:
        last_user_index = max(
            (
                i for i, t in enumerate(kept)
                if str(t.get("role", "")).lower() == "user"
            ),
            default=len(kept) - 1,
        )
        tail = kept[last_user_index + 1:]
        kept = kept[:last_user_index + 1] + tail[:1]

    r["turns"] = kept
    r["conversation_length"] = len(kept)
    r.setdefault("metadata", {})["naacl_length_match"] = {
        "original_user_turns": n_user_turns(record),
        "target_user_turns": target_user_turns,
        "final_user_turns": n_user_turns(r),
        "method": "benign_prefix_trim_to_malicious_empirical_distribution",
    }
    return r


def choose_target_length(
    rng: random.Random, malicious_lengths: List[int], benign_len: int
) -> int:
    feasible = [x for x in malicious_lengths if x <= benign_len]
    if feasible:
        return rng.choice(feasible)
    return min(benign_len, min(malicious_lengths)) if malicious_lengths else benign_len


def _all_spans(record: Dict):
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            yield span


def sanitize_attribution_targets(record: Dict) -> Dict:
    """Expose only repaired, tested evidence to the legacy attribution loader."""
    r = copy.deepcopy(record)

    if r.get("evidence_analysis", {}).get("status") == "error":
        raise RuntimeError(
            f"{r.get('conversation_id', '<missing>')}: evidence analysis error reached preparation"
        )

    has_strong = False
    has_weak = False
    for span in _all_spans(r):
        status = span.get("evidence_status", "unassessed")
        if status == SUPPORTED_STRONG:
            has_strong = True
            span["causal_type"] = "causal"
            span["supervision_tier"] = "cf_strong"
            continue
        if status == SUPPORTED_WEAK:
            has_weak = True
            span["causal_type"] = "causal"
            span["supervision_tier"] = "cf_weak"
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
        if status.startswith("not_assessable") or status == "unassessed":
            span["counterfactual_delta"] = None

    # Recompute sample-level supervision from repaired evidence rather than
    # trusting any legacy tier that may have survived an interrupted old run.
    if has_strong:
        r["supervision_tier"] = "cf_strong"
    elif has_weak:
        r["supervision_tier"] = "cf_weak"
    elif (
        r.get("validation_status") == "validated"
        and r.get("transfer_tier") in DETECTION_TIERS
    ):
        r["supervision_tier"] = "llm_confirmed"
    else:
        r["supervision_tier"] = "ignore"

    r["loss_weight"] = LOSS_WEIGHTS[r["supervision_tier"]]
    r["training_eligible"] = r["supervision_tier"] != "ignore"

    r.setdefault("legacy_pre_naacl_pivot_turn_id", r.get("pivot_turn_id"))
    evidence_turns = sorted({int(x) for x in r.get("evidence_turn_ids", [])})
    r["evidence_turn_ids"] = evidence_turns
    r["pivot_turn_id"] = evidence_turns[0] if evidence_turns else None
    if not evidence_turns:
        r["pivot_kind"] = "none"
    elif len(evidence_turns) > 1:
        r["pivot_kind"] = "distributed"

    return r


def sanitize_benign_targets(record: Dict) -> Dict:
    """Make any annotated benign span an explicit negative, never a positive fallback."""
    r = copy.deepcopy(record)
    for span in _all_spans(r):
        span["causal_type"] = "incidental"
        span["supervision_tier"] = "incidental"
        span["evidence_status"] = span.get("evidence_status", "benign_negative")
        span["counterfactual_delta"] = None
    r["supervision_tier"] = "benign_validated"
    r["loss_weight"] = LOSS_WEIGHTS["benign_validated"]
    r["training_eligible"] = True
    r["pivot_turn_id"] = None
    r["pivot_kind"] = "none"
    return r


def describe(name: str, records: List[Dict]) -> Dict:
    user_lengths = [n_user_turns(r) for r in records]
    total_lengths = [len(r.get("turns", [])) for r in records]
    labels = Counter(r.get("label", -1) for r in records)
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-input", required=True,
                        help="Output of the repaired paired evidence analysis")
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
    evidence_errors = [
        r.get("conversation_id", "")
        for r in evidence_records
        if r.get("evidence_analysis", {}).get("status") == "error"
    ]
    if evidence_errors:
        raise RuntimeError(
            f"Evidence analysis contains {len(evidence_errors)} errors. "
            "Fix/rerun evidence analysis before preparing the dataset."
        )

    malicious = [
        sanitize_attribution_targets(r)
        for r in evidence_records
        if r.get("label") == 1 and r.get("validation_status") == "validated"
    ]

    raw_benign = load_jsonl(args.benign_input)
    benign_original = [
        sanitize_benign_targets(r)
        for r in raw_benign
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
        matched["source_dataset"] = matched.get(
            "source_dataset", "separate_benign_pool"
        )
        matched["supervision_tier"] = "benign_validated"
        matched["loss_weight"] = LOSS_WEIGHTS["benign_validated"]
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
        "sample_tiers": dict(Counter(r.get("supervision_tier", "unknown") for r in combined)),
        "method": {
            "seed": args.seed,
            "attribution_policy": (
                "counterfactual-supported malicious spans only; unestablished "
                "candidates ignored; annotated benign spans are explicit negatives"
            ),
            "length_policy": (
                "benign prefixes sampled from empirical malicious user-turn distribution"
            ),
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
