#!/usr/bin/env python3
"""Prepare the repaired GuardLens dataset for NAACL training/evaluation.

Fail-closed policies:
- only repaired counterfactual-supported spans become positive attribution labels;
- unestablished malicious spans are hidden from the legacy label-name fallback;
- annotated benign spans are explicit negatives;
- a malicious record is trainable only if it has non-Qwen repaired behavioral
  evidence: either a fresh Llama replay judged unsafe by Mistral, or successful
  independent Mistral transfer validation;
- evidence execution errors never enter training.
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
LOSS_WEIGHTS = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "benign_validated": 1.00,
    "ignore": 0.00,
}


def load_jsonl(path: str) -> List[Dict]:
    records = []
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
    return sum(1 for t in record.get("turns", []) if str(t.get("role", "")).lower() == "user")


def _all_spans(record: Dict):
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            yield span


def has_repaired_detection_evidence(record: Dict) -> bool:
    """Require behavioral evidence that does not depend only on the Qwen in-loop judge."""
    analysis = record.get("evidence_analysis", {}) or {}
    fresh_target_unsafe = analysis.get("status") == "complete"
    independent_success = bool(
        record.get("validation_provenance", {}).get("independent_success", False)
    )
    return fresh_target_unsafe or independent_success


def sanitize_attribution_targets(record: Dict) -> Dict:
    r = copy.deepcopy(record)
    analysis_status = str(r.get("evidence_analysis", {}).get("status", "missing"))
    if analysis_status in {"error", "missing"}:
        raise RuntimeError(
            f"{r.get('conversation_id', '<missing>')}: invalid evidence status {analysis_status}"
        )

    has_strong = False
    has_weak = False
    for span in _all_spans(r):
        status = str(span.get("evidence_status", "unassessed"))
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

    detection_ok = has_repaired_detection_evidence(r)
    if has_strong:
        r["supervision_tier"] = "cf_strong"
    elif has_weak:
        r["supervision_tier"] = "cf_weak"
    elif detection_ok:
        r["supervision_tier"] = "llm_confirmed"
    else:
        r["supervision_tier"] = "ignore"

    r["loss_weight"] = LOSS_WEIGHTS[r["supervision_tier"]]
    r["training_eligible"] = r["supervision_tier"] != "ignore"
    r["repaired_detection_evidence"] = {
        "fresh_target_unsafe": r.get("evidence_analysis", {}).get("status") == "complete",
        "independent_success": bool(
            r.get("validation_provenance", {}).get("independent_success", False)
        ),
        "eligible": detection_ok,
    }

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


def truncate_to_user_turns(record: Dict, target_user_turns: int) -> Dict:
    """Keep a benign prefix with the requested number of user turns plus one reply."""
    r = copy.deepcopy(record)
    original = list(r.get("turns", []))
    if target_user_turns <= 0 or not original:
        return r

    kept = []
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
            kept.append(turn)

    if reached:
        last_user_index = max(
            (i for i, t in enumerate(kept) if str(t.get("role", "")).lower() == "user"),
            default=len(kept) - 1,
        )
        kept = kept[:last_user_index + 1] + kept[last_user_index + 1:last_user_index + 2]

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
    feasible = [x for x in malicious_lengths if 0 < x <= benign_len]
    return rng.choice(feasible) if feasible else benign_len


def describe(name: str, records: List[Dict]) -> Dict:
    user_lengths = [n_user_turns(r) for r in records]
    total_lengths = [len(r.get("turns", [])) for r in records]
    return {
        "name": name,
        "n": len(records),
        "labels": dict(Counter(r.get("label", -1) for r in records)),
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
    parser.add_argument("--evidence-input", required=True)
    parser.add_argument("--benign-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benign-stress-output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evidence_records = load_jsonl(args.evidence_input)
    evidence_errors = [
        r.get("conversation_id", "")
        for r in evidence_records
        if r.get("label") == 1
        and r.get("validation_status") == "validated"
        and str(r.get("evidence_analysis", {}).get("status", "missing")) in {"error", "missing"}
    ]
    if evidence_errors:
        raise RuntimeError(
            f"Evidence analysis contains {len(evidence_errors)} validated-malicious errors. "
            "Fix/rerun those records before preparing the dataset."
        )

    sanitized_malicious = [
        sanitize_attribution_targets(r)
        for r in evidence_records
        if r.get("label") == 1 and r.get("validation_status") == "validated"
    ]
    malicious = [r for r in sanitized_malicious if r.get("training_eligible")]
    excluded_malicious = [r for r in sanitized_malicious if not r.get("training_eligible")]

    benign_original = [
        sanitize_benign_targets(r)
        for r in load_jsonl(args.benign_input)
        if r.get("label") == 0
        and r.get("validation_status", "validated") == "validated"
        and r.get("training_eligible", True)
    ]

    if not malicious:
        raise RuntimeError("No malicious records survived repaired behavioral validation")
    if not benign_original:
        raise RuntimeError("No validated benign records were found")

    malicious_lengths = [n_user_turns(r) for r in malicious]
    rng = random.Random(args.seed)
    benign_matched = []
    for record in benign_original:
        target = choose_target_length(rng, malicious_lengths, n_user_turns(record))
        matched = truncate_to_user_turns(record, target)
        matched["benign_status"] = "clean_benign_length_matched"
        matched["source_dataset"] = matched.get("source_dataset", "separate_benign_pool")
        benign_matched.append(matched)

    combined = malicious + benign_matched
    rng.shuffle(combined)
    write_jsonl(combined, args.output)
    write_jsonl(benign_original, args.benign_stress_output)

    stats = {
        "malicious": describe("malicious", malicious),
        "malicious_excluded_no_repaired_behavioral_evidence": describe(
            "malicious_excluded", excluded_malicious
        ),
        "benign_original": describe("benign_original", benign_original),
        "benign_length_matched": describe("benign_length_matched", benign_matched),
        "combined": describe("combined", combined),
        "sample_tiers": dict(Counter(r.get("supervision_tier", "unknown") for r in combined)),
        "repaired_behavioral_gate": {
            "fresh_target_unsafe": sum(
                bool(r.get("repaired_detection_evidence", {}).get("fresh_target_unsafe"))
                for r in sanitized_malicious
            ),
            "independent_success": sum(
                bool(r.get("repaired_detection_evidence", {}).get("independent_success"))
                for r in sanitized_malicious
            ),
            "excluded": len(excluded_malicious),
        },
        "method": {
            "seed": args.seed,
            "attribution_policy": "counterfactual-supported malicious spans only",
            "behavioral_policy": (
                "fresh Llama replay judged unsafe by Mistral OR independent Mistral transfer success; "
                "legacy Qwen-only target judgment is insufficient"
            ),
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
