#!/usr/bin/env python3
"""Stage B5: convert evidence-analyzed frontier records to canonical GuardLens records."""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from collections import Counter
from typing import Dict, List

from frontier_common import load_jsonl, write_jsonl

LOSS_WEIGHTS = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "benign_validated": 1.00,
    "ignore": 0.00,
}


def iter_spans(record: Dict):
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            yield span


def sanitize_malicious(record: Dict) -> Dict:
    r = copy.deepcopy(record)
    analysis = r.get("frontier_evidence_analysis", {}) or {}
    if r.get("validation_status") != "validated":
        raise ValueError("malicious record is not behaviorally validated")
    if analysis.get("status") != "complete":
        raise ValueError(f"malicious evidence status is {analysis.get('status', 'missing')}")

    has_strong = False
    has_weak = False
    for span in iter_spans(r):
        status = str(span.get("evidence_status", "unassessed"))
        if status == "supported_strong":
            has_strong = True
            span["causal_type"] = "causal"
            span["supervision_tier"] = "cf_strong"
        elif status == "supported_weak":
            has_weak = True
            span["causal_type"] = "causal"
            span["supervision_tier"] = "cf_weak"
        elif status == "negative_control_supported":
            span["causal_type"] = "incidental"
            span["supervision_tier"] = "incidental"
        else:
            span.setdefault("original_label", span.get("label", ""))
            span["label"] = "EVIDENCE_CANDIDATE"
            span["causal_type"] = "unvalidated"
            span["supervision_tier"] = "ignore"
            if status.startswith("not_assessable") or status == "unassessed":
                span["counterfactual_delta"] = None

    if has_strong:
        tier = "cf_strong"
    elif has_weak:
        tier = "cf_weak"
    else:
        tier = "llm_confirmed"
    r["supervision_tier"] = tier
    r["loss_weight"] = LOSS_WEIGHTS[tier]
    r["training_eligible"] = True

    evidence_turns = sorted({int(x) for x in r.get("evidence_turn_ids", [])})
    r["evidence_turn_ids"] = evidence_turns
    r["pivot_turn_id"] = evidence_turns[0] if evidence_turns else None
    r["pivot_supervision_ignore"] = not bool(evidence_turns)
    if len(evidence_turns) > 1:
        r["pivot_kind"] = "distributed"
    elif len(evidence_turns) == 1:
        r["pivot_kind"] = "contextual_pivot"
    else:
        r["pivot_kind"] = "none"

    r["repaired_detection_evidence"] = {
        "fresh_target_unsafe": True,
        "independent_success": True,
        "eligible": True,
        "protocol": "frontier_context_paired_counterfactual_v1",
    }
    return r


def sanitize_benign(record: Dict) -> Dict:
    r = copy.deepcopy(record)
    if r.get("validation_status") != "validated":
        raise ValueError("benign record is not behaviorally validated")
    for span in iter_spans(r):
        span["causal_type"] = "incidental"
        span["supervision_tier"] = "incidental"
        span["evidence_status"] = span.get("evidence_status", "benign_negative")
        span["counterfactual_delta"] = None
    r["supervision_tier"] = "benign_validated"
    r["loss_weight"] = LOSS_WEIGHTS["benign_validated"]
    r["training_eligible"] = True
    r["pivot_turn_id"] = None
    r["pivot_kind"] = "none"
    r["pivot_supervision_ignore"] = False
    r["evidence_turn_ids"] = []
    return r


def describe(records: List[Dict]) -> Dict:
    user_counts = [sum(str(t.get("role", "")).lower() == "user" for t in r.get("turns", [])) for r in records]
    return {
        "n": len(records),
        "labels": dict(Counter(r.get("label") for r in records)),
        "validation": dict(Counter(r.get("validation_status") for r in records)),
        "supervision_tiers": dict(Counter(r.get("supervision_tier") for r in records)),
        "pivot_modes": {
            "supported_malicious": sum(r.get("label") == 1 and bool(r.get("evidence_turn_ids")) for r in records),
            "unknown_malicious": sum(r.get("label") == 1 and bool(r.get("pivot_supervision_ignore")) for r in records),
            "benign_true_no_pivot": sum(r.get("label") == 0 and not r.get("pivot_supervision_ignore", False) for r in records),
        },
        "user_turns": {
            "mean": statistics.mean(user_counts) if user_counts else 0.0,
            "min": min(user_counts) if user_counts else 0,
            "max": max(user_counts) if user_counts else 0,
            "histogram": dict(Counter(user_counts)),
        },
        "scenario_families": len({(r.get("metadata", {}) or {}).get("scenario_family") for r in records}),
        "mechanism_families": len({(r.get("metadata", {}) or {}).get("mechanism_family") for r in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded-output", required=True)
    parser.add_argument("--stats-output", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    prepared: List[Dict] = []
    excluded: List[Dict] = []

    for record in records:
        r = copy.deepcopy(record)
        r["authoring_intent_label"] = r.get("label")
        r["corpus_source"] = "frontier_authored_v3"
        try:
            if r.get("label") == 1:
                out = sanitize_malicious(r)
            elif r.get("label") == 0:
                out = sanitize_benign(r)
            else:
                raise ValueError("unsupported label")
            out["source_stage"] = "canonical_training_record"
            out["use_as"] = "eligible_for_consolidated_merge"
            prepared.append(out)
        except Exception as exc:
            r["training_eligible"] = False
            r["exclusion_reason"] = str(exc)
            excluded.append(r)

    write_jsonl(prepared, args.output)
    write_jsonl(excluded, args.excluded_output)
    stats = {
        "prepared": describe(prepared),
        "excluded": {
            "n": len(excluded),
            "by_label": dict(Counter(r.get("label") for r in excluded)),
            "by_validation": dict(Counter(r.get("validation_status") for r in excluded)),
            "by_reason": dict(Counter(r.get("exclusion_reason") for r in excluded)),
        },
        "method": {
            "authoring_intent_is_ground_truth": False,
            "malicious_gate": "validated source rollout AND fresh paired evidence baseline remains unsafe",
            "benign_gate": "independent context-aware judge validates safe realized trajectory",
            "pivot_policy": "only counterfactually supported evidence turns populate pivot_turn_id",
            "span_policy": "only counterfactually supported spans receive positive attribution supervision",
        },
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote prepared: {args.output}")
    print(f"Wrote excluded: {args.excluded_output}")


if __name__ == "__main__":
    main()
