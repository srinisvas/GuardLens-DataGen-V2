#!/usr/bin/env python3
"""Compare Qwen primary and Gemma cross-target behavioral validation outputs.

Robustness statistics are descriptive only and never overwrite canonical Qwen
training labels. Unsafe-anchor agreement is computed only for authoring-intent
malicious records that are independently validated unsafe on both targets.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from frontier_common import load_jsonl


def val(record):
    return record.get("frontier_behavioral_validation", {}) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--cross-target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary = {str(r.get("conversation_id")): r for r in load_jsonl(args.primary)}
    cross = {str(r.get("conversation_id")): r for r in load_jsonl(args.cross_target)}
    shared = sorted(set(primary) & set(cross))
    if not shared:
        raise RuntimeError("primary and cross-target inputs contain no shared conversation IDs")

    attack_primary = 0
    attack_transfer = 0
    benign_primary = 0
    benign_stable = 0
    anchor_both_malicious = 0
    anchor_exact = 0
    anchor_within_one_user_turn = 0
    transitions = Counter()
    label_counts = Counter()

    for cid in shared:
        a, b = primary[cid], cross[cid]
        if a.get("label") != b.get("label"):
            raise RuntimeError(f"{cid}: label differs between target outputs")
        label = a.get("label")
        label_counts[str(label)] += 1
        sa, sb = a.get("validation_status"), b.get("validation_status")
        transitions[(str(label), str(sa), str(sb))] += 1

        if label == 1 and sa == "validated":
            attack_primary += 1
            if sb == "validated":
                attack_transfer += 1
        if label == 0 and sa == "validated":
            benign_primary += 1
            if sb == "validated":
                benign_stable += 1

        # Anchor agreement is meaningful for successful malicious transfer, not
        # for benign safety failures that happen to become unsafe on both models.
        if label == 1 and sa == "validated" and sb == "validated":
            aa = val(a).get("earliest_unsafe_user_turn_id")
            ab = val(b).get("earliest_unsafe_user_turn_id")
            if aa is None or ab is None:
                raise RuntimeError(
                    f"{cid}: validated malicious transfer lacks an unsafe anchor"
                )
            anchor_both_malicious += 1
            if int(aa) == int(ab):
                anchor_exact += 1
            # Source user IDs increase by two, so <=2 means within one user turn.
            if abs(int(aa) - int(ab)) <= 2:
                anchor_within_one_user_turn += 1

    result = {
        "shared_records": len(shared),
        "shared_labels": dict(label_counts),
        "primary_validated_attacks": attack_primary,
        "cross_target_attack_transfer": attack_transfer,
        "cross_target_attack_transfer_rate": round(attack_transfer / attack_primary, 4) if attack_primary else None,
        "primary_validated_benign": benign_primary,
        "cross_target_benign_stability": benign_stable,
        "cross_target_benign_stability_rate": round(benign_stable / benign_primary, 4) if benign_primary else None,
        "malicious_validated_on_both_targets": anchor_both_malicious,
        "unsafe_anchor_exact_agreement": anchor_exact,
        "unsafe_anchor_exact_rate": round(anchor_exact / anchor_both_malicious, 4) if anchor_both_malicious else None,
        "unsafe_anchor_within_one_user_turn": anchor_within_one_user_turn,
        "unsafe_anchor_within_one_user_turn_rate": round(anchor_within_one_user_turn / anchor_both_malicious, 4) if anchor_both_malicious else None,
        "validation_transitions": {"|".join(k): v for k, v in transitions.items()},
        "note": (
            "Cross-target results are robustness measurements only and never overwrite canonical "
            "Qwen-derived training labels. Anchor agreement is restricted to malicious records "
            "validated unsafe on both targets."
        ),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
