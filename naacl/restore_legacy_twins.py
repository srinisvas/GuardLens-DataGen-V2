#!/usr/bin/env python3
"""Build a candidate Dataset A that restores original validated twin controls.

This script does not alter conversation text. It consumes the judge-only artifact
from judge_stored_twin_responses.py and constructs a paired candidate using only
malicious records that survive the repaired evidence gate and their own original
benign twin when that twin passes both stored-target and independent validation.

The existing frozen dataset is not modified by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from prepare_dataset import (
    describe,
    load_jsonl,
    sanitize_attribution_targets,
    sanitize_benign_targets,
    write_jsonl,
)


def final_malicious(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    sanitized = [
        sanitize_attribution_targets(record)
        for record in records
        if record.get("label") == 1 and record.get("validation_status") == "validated"
    ]
    kept = [record for record in sanitized if record.get("training_eligible")]
    excluded = [record for record in sanitized if not record.get("training_eligible")]
    return kept, excluded


def eligible_twin(record: Dict) -> bool:
    restoration = record.get("twin_restoration", {}) or {}
    stored = record.get("stored_target_validation", {}) or {}
    independent = record.get("causal_validation", {}) or {}

    if record.get("label") != 0:
        return False
    if str(record.get("family", "")) != "interactive_benign_twin":
        return False
    if not restoration.get("eligible", False):
        return False
    if stored.get("validated") is not True or stored.get("jailbreak_detected"):
        return False
    if independent.get("validated") is not True or independent.get("jailbreak_detected"):
        return False
    if not restoration.get("independent_replay_reused", False):
        return False
    if restoration.get("conversation_text_modified") is not False:
        return False
    if restoration.get("target_replayed") is not False:
        return False
    if restoration.get("independent_model_replayed") is not False:
        return False

    stored_traj = stored.get("compliance_trajectory", []) or []
    independent_traj = independent.get("compliance_trajectory", []) or []
    if not stored_traj or not independent_traj:
        return False
    if any(item.get("method") != "structured_judge" for item in stored_traj):
        return False
    if any(item.get("method") != "structured_judge" for item in independent_traj):
        return False
    return True


def pair_index(records: List[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        if pair_id:
            out[pair_id].append(record)
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--excluded-output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    malicious, evidence_excluded = final_malicious(records)
    groups = pair_index(records)

    restored_malicious = []
    restored_benign = []
    unmatched = []
    rejection_reasons = Counter()

    for mal in malicious:
        pair_id = str(mal.get("pair_id", ""))
        siblings = groups.get(pair_id, [])
        benign_siblings = [r for r in siblings if r.get("label") == 0]

        if len(benign_siblings) != 1:
            unmatched.append(mal)
            rejection_reasons[f"benign_sibling_count={len(benign_siblings)}"] += 1
            continue

        benign = benign_siblings[0]
        if not eligible_twin(benign):
            unmatched.append(mal)
            restoration = benign.get("twin_restoration", {}) or {}
            reason = (
                restoration.get("stored_target_reason", "unknown")
                + "|"
                + restoration.get("independent_reason", "unknown")
            )
            rejection_reasons[reason] += 1
            continue

        restored_malicious.append(mal)
        restored_benign.append(sanitize_benign_targets(benign))

    if not restored_malicious:
        raise RuntimeError(
            "No repaired malicious records have a restorable original benign twin"
        )
    if len(restored_malicious) != len(restored_benign):
        raise RuntimeError("Twin restoration produced class imbalance")

    mal_pairs = Counter(str(r.get("pair_id", "")) for r in restored_malicious)
    ben_pairs = Counter(str(r.get("pair_id", "")) for r in restored_benign)
    if mal_pairs != ben_pairs:
        raise RuntimeError("Restored malicious and benign pair IDs do not match")
    if any(n != 1 for n in mal_pairs.values()):
        raise RuntimeError("A restored pair_id occurs more than once per class")

    combined = restored_malicious + restored_benign
    rng = random.Random(args.seed)
    rng.shuffle(combined)
    write_jsonl(combined, args.output)

    if args.excluded_output:
        write_jsonl(evidence_excluded + unmatched, args.excluded_output)

    stats = {
        "input_records": len(records),
        "repaired_malicious_before_pair_gate": len(malicious),
        "malicious_excluded_by_repaired_evidence_gate": len(evidence_excluded),
        "restored_pairs": len(restored_malicious),
        "primary_records": len(combined),
        "malicious_without_valid_original_twin": len(unmatched),
        "pair_rejection_reasons": dict(rejection_reasons),
        "malicious": describe("malicious", restored_malicious),
        "benign_original_twin": describe("benign_original_twin", restored_benign),
        "sample_tiers": dict(
            Counter(r.get("supervision_tier", "unknown") for r in combined)
        ),
        "policy": {
            "conversation_text_modified": False,
            "malicious_text_modified": False,
            "benign_text_modified": False,
            "target_replayed": False,
            "independent_model_replayed": False,
            "benign_control": "original interactive benign twin only",
            "pairing": "exact original pair_id",
            "attribution_policy": "counterfactual-supported malicious spans only",
            "behavioral_policy": (
                "malicious repaired evidence gate retained; benign twin requires "
                "judge-only validation of stored Llama responses plus the existing "
                "independent replay to remain safe"
            ),
        },
    }

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2))
    print(f"Wrote candidate restored Dataset A: {args.output}")
    print(f"Wrote stats: {args.stats_output}")
    if args.excluded_output:
        print(f"Wrote excluded/unmatched records: {args.excluded_output}")


if __name__ == "__main__":
    main()
